import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

from basicsr.archs.arch_util import LayerNorm2d
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.deformable_self_attention import MSDeformableNeighborhoodAttention

##########################################################################
# Layer Norm


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# 无偏置层归一化
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


# 有偏置层归一化
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


# 统一接口
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.norm_type = LayerNorm_type
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            # self.body = WithBias_LayerNorm(dim)
            self.body = LayerNorm2d(dim, eps=1e-5)
            # eps default 1e-6 consistent to 1e-5 in Restormer

    def forward(self, x):

        if self.norm_type == 'BiasFree':
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)

        else:
            return self.body(x)


##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
# Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads,
                 ffn_expansion_factor,
                 bias, LayerNorm_type, rel_pos_bias,
                 kernel_size=3, dilation=1):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MSDeformableNeighborhoodAttention(
            dim, num_heads, kernel_size=kernel_size, dilation=dilation, rel_pos_bias=rel_pos_bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
# Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
# ---------- Restormer -----------------------


@ARCH_REGISTRY.register()
class MSDeformableNAIR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[4, 6, 6, 8],
                 kernel_size=[3, 3, 3, 3],
                 dilation=[1, 1, 1, 1],
                 num_refinement_blocks=4,
                 heads=[2, 4, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 rel_pos_bias=True,
                 dual_pixel_task=False,
                 global_residual=True,
                 ):

        super(MSDeformableNAIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias, LayerNorm_type=LayerNorm_type, rel_pos_bias=rel_pos_bias,
            kernel_size=kernel_size[0], dilation=dilation[0]) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], rel_pos_bias=rel_pos_bias,
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                                            kernel_size=kernel_size[1], dilation=dilation[1]) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1))  # From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], rel_pos_bias=rel_pos_bias,
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                                            kernel_size=kernel_size[2], dilation=dilation[2]) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))  # From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], rel_pos_bias=rel_pos_bias,
                                    ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                                    kernel_size=kernel_size[3], dilation=dilation[3]) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3))  # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], rel_pos_bias=rel_pos_bias,
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                                            kernel_size=kernel_size[2], dilation=dilation[2]) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], rel_pos_bias=rel_pos_bias,
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                                            kernel_size=kernel_size[1], dilation=dilation[1]) for i in range(num_blocks[1])])

        # From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_1 = Upsample(int(dim*2**1))

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], rel_pos_bias=rel_pos_bias,
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                                            kernel_size=kernel_size[0], dilation=dilation[0]) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                        bias=bias, LayerNorm_type=LayerNorm_type, rel_pos_bias=rel_pos_bias,
                                        kernel_size=kernel_size[0], dilation=dilation[0]) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(
                dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(
            int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.global_residual = global_residual

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        if self.global_residual:
            out_dec_level1 = out_dec_level1 + inp_img

        return out_dec_level1
    

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    
    # 基础配置
    inp_channels = 3        # RGB输入
    out_channels = 3        # RGB输出
    dim = 36               # 基础特征维度（能被6和12整除）
    height = 160           # 图像高度
    width = 160            # 图像宽度
    
    # 网络结构配置
    num_blocks = [4, 6, 6, 8]           # 每个阶段的块数
    kernel_size = [3, 3, 3, 3]          # 每个阶段的卷积核大小
    dilation = [1, 1, 1, 1]             # 每个阶段的膨胀率
    heads = [3, 3, 6, 6]                # 每个阶段的注意力头数（能被3整除）
    num_refinement_blocks = 4            # 精细化块数量
    ffn_expansion_factor = 2             # FFN扩展因子
    
    # 高级配置（注释掉的备选方案）
    # dim = 48
    # height = 256
    # width = 256
    # heads = [3, 6, 12, 12]
    # kernel_size = [5, 5, 5, 5]
    
    # 创建MSDeformableNAIR模型
    net = MSDeformableNAIR(
        inp_channels=inp_channels,
        out_channels=out_channels,
        dim=dim,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        dilation=dilation,
        num_refinement_blocks=num_refinement_blocks,
        heads=heads,
        ffn_expansion_factor=ffn_expansion_factor,
        bias=False,
        LayerNorm_type='WithBias',
        rel_pos_bias=True,
        dual_pixel_task=False,      # 测试普通任务
        global_residual=True,       # 启用全局残差
    )
    
    # 输入形状（用于复杂度计算）
    inp_shape = (inp_channels, height, width)
    
    # 计算模型复杂度
    print("🔍 计算模型复杂度中...")
    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    
    params = float(params[:-3])
    macs = float(macs[:-4])
    
    print(f"📊 模型统计信息:")
    print(f"   MACs: {macs:.2f}G")
    print(f"   Params: {params:.2f}M")
    
    # # 测试输入输出维度
    # print("\n🚀 测试网络前向传播...")
    # test_input = torch.randn(2, inp_channels, height, width)  # 批量大小为2
    
    # # 设置为评估模式
    # net.eval()
    
    # with torch.no_grad():
    #     test_output = net(test_input)
    
    # print(f"✅ 前向传播测试结果:")
    # print(f"   Input shape: {test_input.shape}")
    # print(f"   Output shape: {test_output.shape}")
    # print(f"   Shape preserved: {test_input.shape == test_output.shape}")
    
    # # 测试双像素任务配置
    # print("\n🔄 测试双像素任务配置...")
    # net_dual = MSDeformableNAIR(
    #     inp_channels=inp_channels,
    #     out_channels=out_channels,
    #     dim=dim,
    #     num_blocks=num_blocks,
    #     kernel_size=kernel_size,
    #     dilation=dilation,
    #     num_refinement_blocks=num_refinement_blocks,
    #     heads=heads,
    #     ffn_expansion_factor=ffn_expansion_factor,
    #     bias=False,
    #     LayerNorm_type='WithBias',
    #     rel_pos_bias=True,
    #     dual_pixel_task=True,       # 启用双像素任务
    #     global_residual=True,
    # )
    
    # net_dual.eval()
    # with torch.no_grad():
    #     test_output_dual = net_dual(test_input)
    
    # print(f"📱 双像素任务测试结果:")
    # print(f"   Input shape: {test_input.shape}")
    # print(f"   Output shape: {test_output_dual.shape}")
    # print(f"   Shape preserved: {test_input.shape == test_output_dual.shape}")
    
    # # 测试不同分辨率的适应性
    # print("\n📐 测试不同分辨率适应性...")
    # test_sizes = [(128, 128), (192, 192), (224, 224)]
    
    # for h, w in test_sizes:
    #     test_input_var = torch.randn(1, inp_channels, h, w)
    #     with torch.no_grad():
    #         test_output_var = net(test_input_var)
    #     print(f"   {h}x{w}: {test_input_var.shape} → {test_output_var.shape} ✅")
    
    # 输出网络结构信息
    print(f"\n🏗️ 网络结构信息:")
    print(f"   阶段数: {len(num_blocks)}")
    print(f"   每阶段块数: {num_blocks}")
    print(f"   注意力头数: {heads}")
    print(f"   特征维度变化: {dim} → {dim*2} → {dim*4} → {dim*8}")
    print(f"   精细化块数: {num_refinement_blocks}")
    print(f"   全局残差: {'✅' if net.global_residual else '❌'}")
    
    print("\n🎉 所有测试完成！网络运行正常~")

