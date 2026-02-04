from math import sqrt, pow, log2
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .modules import *
from mamba_ssm import Mamba





class CGNL1DATTN(nn.Module):
    r"""The CGNL1DATTN block
    It computes the non-local attentions along 1D dimensions with dot production kernel.
    When groups=1, the correlations will be computed among channels.
    Input: (B, C, D)
    Output: (B, C, D)
    """
    def __init__(self, inplanes, ch_ds=False, attn_scale=1, groups=1, resid = 0):
        self.attn_scale = attn_scale
        self.groups = groups
        self.residual = resid

        super().__init__()
        self.in_chans = inplanes
        if ch_ds:
            ch_tmp = self.in_chans //4
        else:
            ch_tmp = self.in_chans
        # conv theta
        self.t = nn.Conv1d(self.in_chans, ch_tmp, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.t.weight, 0, 'fan_in', 'relu')
        # conv phi
        self.p = nn.Conv1d(self.in_chans, ch_tmp, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.p.weight, 0, 'fan_in', 'relu')
        # conv g
        self.g = nn.Conv1d(self.in_chans, ch_tmp, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.g.weight, 0, 'fan_in', 'relu')
        # conv z
        self.z = nn.Conv1d(ch_tmp, self.in_chans, kernel_size=1, stride=1, groups=self.groups, bias=False)
        init.kaiming_normal_(self.z.weight, 0, 'fan_in', 'relu')

        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=self.in_chans)
        init.constant_(self.gn.weight, 0); nn.init.constant_(self.gn.bias, 0)

        print("\033[0;33;40m=> WARN: 1D CGNL Attention block uses SCALE= {}\033[0m".format(self.attn_scale))
        if self.groups > 1:
            print("\033[0;33;40m=> WARN: Temporal CGNL block uses '{}' groups\033[0m".format(self.groups))

    def kernel(self, t, p, g, b, c, d):
        """The linear kernel (dot production).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            d: depth channels
        """
        t = t.view(b, 1, c * d)
        p = p.view(b, 1, c * d)
        g = g.view(b, c * d, 1)

        att = torch.bmm(p, g)* self.attn_scale

        x = torch.bmm(att, t)
        x = x.view(b, c, d)

        return x

    def forward(self, inp_q, inp_k, inp_v):
        residual = inp_v* self.residual

        t = self.t(inp_v)
        p = self.p(inp_q)
        g = self.g(inp_k)

        b, c, d = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, d)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, d)

        x = self.z(x)
        x = self.gn(x) + residual

        return x

    def flops(self, N):
        # N: number of depth dimension
        flops = 0
        # self.t(x), self.p(x), and self.g(x)
        flops += 3* self.in_chans* self.in_chans//4 * N
        # self.kernel(t, p, g, b, c, d)
        flops += self.in_chans//4 * N* 2

        return flops

class CGNL1DMHAttention(nn.Module):
    r"""The CGNL1DMHAttention block => a kind of replacement to nn.MultiHeadAttention(), which requires less GPU memory
    It is just a multi-head attention block with CGNLATTN to commpute attention.
    *Since the CGNLATTN has the codings in channel, please mind use it or not.*
    Input:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: False
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    Output: attention output
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, proj_drop=0., resi=0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.resi = resi

        self.attn = CGNL1DATTN(self.dim, attn_scale=self.scale, groups=self.num_heads, resid=self.resi)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, inp_q, inp_k, inp_v):
        r"""
        Input:
            inputs: the three input features with shape of (B, H*W, C) as Q, K, and V. Note that we should reshape the input tensor.
            H, W (int): height and width of the input tensor.
        Output: out
        """

        inter_q = inp_q.permute(1,2,0) #Dkp, Datt, C
        inter_k = inp_k.permute(1,2,0) #Dkp, Datt, C
        inter_v = inp_v.permute(1,2,0) #Dkp, Datt, C
        out_att = self.attn(inter_q, inter_k, inter_v) # Datt, Dkp, C
        # out = self.proj(out)
        out = self.proj_drop(out_att)
        return out

    def extra_repr(self):
        return f"dim={self.dim}, num_head={self.num_heads}"

    def flops(self, N):
        """The function for calculating flops for attention
        Input: N: the length of the token, that is: B, <N>, C
        output: calculating flops
        """
        flops = 0
        # qkv = self.qkv(inputs)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.attn.flops(N* self.dim)

        return flops



class CrossDecoder(nn.Module):
    r"""The CrossDecoder block
    It consists of several CGNL1DMHAttention to generate high integrated features.
    It is the one block and will integrate information with others
    Input:
        ch_in (int): Number of input channels.
        input_resolution (tuple[int, int, int]): Input resulotion of H, W, and D.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.1
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
    Output: Integrated output with resolution of (B, C, D, H, W).
    """

    def __init__(self, ch_in=64, input_resolution=[32,32,64], num_heads=8, res_con = 0,
                mlp_ratio=4, mlp_drop=0., attn_drop=0.1, 
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_chans = ch_in
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropnum = mlp_drop
        self.resid = res_con

        # decoder block (we can use built-in MHSA to replace it)
        self.norm1 = norm_layer(self.in_chans)
        self.norm12 = norm_layer(self.in_chans)
        self.attn_hw = CGNL1DMHAttention(self.in_chans, self.num_heads, resi=self.resid)
        self.attn_t = CGNL1DMHAttention(self.in_chans, self.num_heads, resi=self.resid)
        self.drop1 = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(self.in_chans)
        mlp_hidden_dim_loc = int(self.in_chans* self.mlp_ratio)
        self.mlp = MLP(self.in_chans, mlp_hidden_dim_loc,act_layer=act_layer, drop=self.dropnum)
        self.drop2 = DropPath(drop_path) if drop_path>0. else nn.Identity()
        attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        print("\033[0;33;40m=> WARN: CrossDecoder block in/out channel sizes, num_head and resolution {} / {} / {} / {},{},{} \033[0m".format(self.in_chans, self.in_chans, self.num_heads, self.input_resolution[0],self.input_resolution[1],self.input_resolution[2]))

    def forward(self, inpt_kv, inpt_q):
        r"""
        Input: two input features with shape of (B, C, D, H, W).
        Output: integrated features with shape of (B, C, D, H, W).
        """

        B, C, D, H, W = inpt_kv.shape
        shortcut = inpt_kv.reshape(B, C, D*H*W).permute(0, 2, 1) # B, L, C

        # decoder for spatial and temporal independently
        k = v = self.norm1(inpt_kv.permute(0,2,3,4,1)).reshape(B, D, H* W, C)  # B, D, HW, C
        q = self.norm12(inpt_q.permute(0,2,3,4,1)).reshape(B, D, H* W, C)  # B, D, HW, C
        q_in_hw = q.contiguous().view(-1, H* W, C).permute(1,0,2) # HW, BD, C
        q_in_t = q.permute(1,0,2,3).contiguous().view(D, -1, C) # D,BHW,C
        k_in = k.contiguous().view(-1, H* W, C).permute(1,0,2) # HW, BD, C
        v_in = v.contiguous().view(-1, H* W, C).permute(1,0,2) # HW, BD, C
        attn_hw = self.attn_hw(q_in_hw, k_in, v_in) # HW, BD, C
        attn_hw = attn_hw.reshape(H* W, B, D, C).contiguous().view(-1, D, C).permute(1, 0, 2) # D, BHW, C
        attn_t = self.attn_t(q_in_t, attn_hw, attn_hw) # D, BHW, C
        attn = attn_t.reshape(D, B, H, W, C).permute(1,0,2,3,4).contiguous().view(B, -1, C) # B, L, C

        # FFN
        out_drop = shortcut + self.drop1(attn)
        out_ffn = out_drop+ self.drop2(self.mlp(self.norm2(out_drop)))
        out = out_ffn.reshape(B, D, H, W, C).permute(0,4,1,2,3) # B, C, D, H, W

        return out

    def flops(self, N):
        flops = 0
        H, W, D = self.input_resolution
        # local and global attension
        flops += 2* self.in_chans* H* W* D
        flops += 2* (4* H*W* pow(self.in_chans,2)+ pow(H*W,2)*self.in_chans) + (4* D* pow(self.in_chans,2)+ pow(D,2)*self.in_chans)
        flops += 2* 2* H* W* D* self.in_chans* self.in_chans* self.mlp_ratio
        flops += 2* self.in_chans* H* W* D

        return flops





       
class transient_fusion(nn.Module):
    r"""The LocGlbInteNBlks block
    It integrates the information from local patches and global images with several numbers (can assign) of transformer decoders.
    *It use our CGNLMHAttention instead of built-in MHA in pytorch.*
    Note that these decoders do NOT use the positional encodings. And the number of decoders can be assigned.
    For local pathes, its resolution is: B, C, D, H, W.
    For global images, its resolution is: B, C, D, H/2, W/2.
    Workflow: local patches-> as Q to refine the global images; global images-> as Q to refine the local pathes.
    Input:
        ch_in (int): Number of input channels.
        input_resolution (tuple[int, int, int]): Input resulotion of H, W, and D.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.1
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
    Output: Integrated output, two tensor with resolution of (B, C, D, H, W).
    """

    def __init__(self, ch_in=64, input_resolution=[32,32,64], num_heads=8, num_ders = 3, res_con = 0,
                mlp_ratio=4, mlp_drop=0., attn_drop=0.1, 
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_chans = ch_in
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.num_ders = num_ders
        self.mlp_ratio = mlp_ratio
        self.drop = mlp_drop
        self.resid = res_con

        # local and global pathes decoders
        # self.loc_decs = modelClone(CrossDecoder_spatial(self.in_chans, self.input_resolution, self.num_heads, self.resid), self.num_ders)
        self.loc_decs = modelClone(CrossDecoder(self.in_chans, self.input_resolution, self.num_heads, self.resid), self.num_ders)
        # self.glb_decs = modelClone(CrossDecoder_spatial(self.in_chans, self.input_resolution, self.num_heads, self.resid), self.num_ders)
        self.glb_decs = modelClone(CrossDecoder(self.in_chans, self.input_resolution, self.num_heads, self.resid), self.num_ders)
            
        attn_mask_loc = None
        attn_mask_glb = None

        self.register_buffer("attn_mask_loc", attn_mask_loc)
        self.register_buffer("attn_mask_glb", attn_mask_glb)

        # print("\033[0;33;40m=> WARN: LocGlbInteNBlks block in/out channel sizes, num_head, num_decoders, and resolution {} / {} / {} / {} / {},{},{} \033[0m".format(self.in_chans, self.in_chans, self.num_heads, self.num_ders, self.input_resolution[0],self.input_resolution[1],self.input_resolution[2]))

    def forward(self, inpt_loc, inpt_glb):
        r"""
        Input:
            inputs: input local and global features with shape of (B, C, D, H, W). Use the local patches as Q to process the global images; use the global images as Q to process the local patches.
            workflow: local patches: Q-> global images;
            global images: Q-> local patches
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None, defult: None.
        Output: out: two tensor with resolution of (B, C, D, H, W).
        MIND THE SHAPE CHANGES!!!
        """
        # up_loc = self.upop_L(inpt_loc)    # B,C,D,H,W
        # up_glb = self.upop_G(inpt_glb)    # B,C,D,H,W
        up_glb = inpt_glb
        up_loc = inpt_loc

        # cross correlations between decoders
        for loc_dec, glb_dec in zip(self.loc_decs, self.glb_decs):

            #version 3 by yl  self and cross attention lcgs
            out_loc = loc_dec(up_loc, up_loc)
            out_glb = glb_dec(up_loc, up_glb)
            
            up_loc = out_loc
            up_glb = out_glb 

        return up_loc, up_glb
        # return out_loc + up_loc, out_glb + up_glb
        
    def flops(self):
        flops = 0
        H, W, D = self.input_resolution
        # self.upop(inpt_glb)
        flops += self.upop.flops(H* W* D)
        # local and global attension
        flops += self.num_ders* 2* self.in_chans* H* W* D
        flops += self.num_ders* 2* (4* H*W* pow(self.in_chans,2)+ pow(H*W,2)*self.in_chans) + (4* D* pow(self.in_chans,2)+ pow(D,2)*self.in_chans)
        flops += self.num_ders* 2* 2* H* W* D* self.in_chans* self.in_chans* self.mlp_ratio
        flops += self.num_ders* 2* self.in_chans* H* W* D

        return flops


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )
    def forward(self, x):
        # print('x',x.shape)
        B, L, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)    
        return x_mamba


class MLP(nn.Module):
    """The MLP block
    The feed forward layer of the network
    Input: B, L, C
    Output: B, L, C
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, inputs):
        out_fc1 = self.fc1(inputs)
        out_act = self.act(out_fc1)
        out_drop1 = self.drop(out_act)
        out_fc2 = self.fc2(out_drop1)
        out = self.drop(out_fc2)
        return out



class GlobalAttention(nn.Module):
    r"""The GroupAttention block => can be replaced by the nn.MultiHeadAttention()
    It is just a multi-head attention block computed without relative position bias.
    Input:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: False
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    Output: attention output
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, inputs):
        """
        Input:
            inputs: input features with shape of (B, H*W, C). The dependency is computed independently among <B> but within channel <C>. Note that we should reshape the input tensor.
            H, W (int): height and width of the input tensor.
        Output: out
        """
        B, N, C = inputs.shape
        qkv = self.qkv(inputs).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B, n_head, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, n_head, N, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, n_head, N, N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)   # B, N, C
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def extra_repr(self):
        return f"dim={self.dim}, num_head={self.num_heads}"

    def flops(self, N):
        """The function for calculating flops for attention
        Input: N: the length of the token, that is: B, <N>, C
        output: calculating flops
        """
        flops = 0
        # qkv = self.qkv(inputs)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.dim * N * N
        #  x = (attn @ v)
        flops += self.dim * N * N
        # x = self.proj(x)
        flops += N * self.dim * self.dim

        return flops


class GlobalEncoderSep(nn.Module):
    r"""The GlobalEncoderSep Block
    It is the transformer encoder with GlobalAttention, which doesn't contain the patch embedding and positional encoding.
    The dependency is computed among H and W first and D finally.
    Formation: Norm-> W-MSA_HW-> W-MSA_T-> ADD-> Norm-> MLP-> ADD.
    Input:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int, int]): Input resulotion of H, W, and D.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value, better for training. Default: False.
        qk_scale (float | None, optional): If set, override default qk scale as head_dim ** -0.5.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic drop rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
    Output: Encoded output with internal dependency, (B, C, D, H, W).
    """

    def __init__(self, dim, input_resolution, num_heads=8, mlp_ratio=4, 
                qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., 
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop = drop

        self.norm1 = norm_layer(self.dim)
        self.attn_hw = GlobalAttention(self.dim, self.num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=self.drop)
        self.attn_t = GlobalAttention(self.dim, self.num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=self.drop)
        # self.attn_hw = MambaLayer(self.dim)
        # self.attn_t = MambaLayer(self.dim)
        
        self.drop_path1 = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(self.dim, mlp_hidden_dim, act_layer=act_layer, drop=self.drop)
        self.drop_path2 = DropPath(drop_path) if drop_path>0. else nn.Identity()
        attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        print("\033[0;33;40m=> WARN: GlobalEncoderSep block in/out channel sizes, num_head and resolution {} / {} / {} / {},{},{}, \033[0m".format(self.dim, self.dim, self.num_heads, self.input_resolution[0],self.input_resolution[1],self.input_resolution[2]))

    def forward(self, inputs):
        r"""
        Input:
            inputs: input features with shape of (B, C, D, H, W). The dependency is computed independently among <B> and window patches but partly within channel <C>. In this seperate version, we first compute dependency among H and W and then the D.
            workflow: B,C,DHW-> nB,C,D,h,w-> nBD,hw,C-> nBhw,D,C-> B,H,W,D,C-> B,C,D,H,W
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None, defult: None.
        Output: out: encoded output with full internal dependency, (B, C, D, H, W)
        MIND THE SHAPE CHANGES!!!
        """

        H, W, D = self.input_resolution
        B, C, _, _, _ = inputs.shape
        inputs = inputs.contiguous().view(B, C, -1)
        _, _, L = inputs.shape
        assert L == H* W* D, f"{L} should be equal to {H}*{W}*{D}."

        shortcut = inputs.permute(0, 2, 1)  # B, L, C
        out_norm1 = self.norm1(inputs.permute(0,2,1)).reshape(B, D, H* W, C)   # B, D, HW, C

        # MSA on spatial dimension
        out_norm1 = out_norm1.contiguous().view(-1, H* W, C)
        out_attn_hw = self.attn_hw(out_norm1)   # BD, HW, C
        out_attn_hw = out_attn_hw.contiguous().view(-1, D, H* W, C)

        # MSA on temporal dimension
        out_attn_hw = out_attn_hw.permute(0,2,1,3).contiguous().view(-1, D, C)  # BHW, D, C
        out_attn_t = self.attn_t(out_attn_hw)   # BHW, D, C
        out_attn = out_attn_t.reshape(B, H* W, D, C).contiguous().view(B, -1, C)  # B, L, C

        # FFN
        out_res = shortcut + self.drop_path1(out_attn)
        out_ffn = out_res + self.drop_path2(self.mlp(self.norm2(out_res)))   # B, L, C
        out = out_ffn.reshape(B, H, W, D, C).permute(0,4,3,1,2) # B,C, D, H, W

        return out

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_head={self.num_heads}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W, D = self.input_resolution
        # norm1
        flops += self.dim * H* W* D
        # MSA_HW and MSA_T
        flops += self.attn_hw.flops(H* W) + self.attn_t.flops(D)
        # mlp
        flops += 2* H* W* D* self.dim* self.dim* self.mlp_ratio
        # norm2
        flops += self.dim* H* W* D
        
        return flops
    
    
def modelClone(module, N):
    r"""The modelClone function
    It clones the given model with N copies.
    """

    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    

class PosiEncCNN(nn.Module):
    r"""The PosiEncCNN block
    It generates the positional encodings for the input tensor, which is a simple CNN on the input.
    Input:
        in_chans (int): channel size of input tokens
        out_chans (int): channel size of output tokens
    Output: out= inputs+ position
    """

    def __init__(self, in_chans):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = in_chans
        self.proj = nn.Conv3d(self.in_chans,self.out_chans,3,1,1,bias=True)
        fan_out = self.proj.kernel_size[0]* self.proj.kernel_size[1]* self.proj.kernel_size[2]* self.out_chans
        init.normal_(self.proj.weight, 0, sqrt(2.0 / fan_out)); init.constant_(self.proj.bias, 0.0)
        # init.kaiming_normal_(self.proj.weight, 0, 'fan_in', 'relu'); init.constant_(self.proj.bias, 0.0)
        
        print("\033[0;33;40m=> WARN: PosiEncCNN block in/out channel sizes {} / {} \033[0m".format(self.in_chans, self.out_chans))

    def forward(self, inputs):
        r"""
        Input: inputs has size of (B, Cin, D, H, W)
        Output: encoded output has size of (B, Cout, D, H, W)
        """

        pos_code = self.proj(inputs)
        out = pos_code + inputs

        return out

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

    def flops(self, N):
        flops = 0
        # self.proj(inputs)
        flops += self.proj.kernel_size[0]* self.proj.kernel_size[1]* self.proj.kernel_size[2]* self.out_chans* self.in_chans* N

        return flops

class mamba_up(nn.Module):
    def __init__(self, in_chans=1,temp_chans=8,sp_ds_scale=4,targrt_size=128,inplace=False):
        super(mamba_up, self).__init__()
        self.in_chans = in_chans
        self.temp_chans = temp_chans
        self.sp_ds_scale = sp_ds_scale
        self.targrt_size =targrt_size
        self.inplace = inplace
        
        self.denoise = MsFeat_2(in_channels= self.in_chans, out_channels=self.temp_chans)
        self.pos_code = PosiEncCNN(self.temp_chans * 4)   
        self.mamba = modelClone(GlobalEncoderSep(dim=self.temp_chans * 4,input_resolution=[self.targrt_size//self.sp_ds_scale,self.targrt_size//self.sp_ds_scale,256],num_heads=4), 6)
        self.spup = SPUP(self.temp_chans*4,self.sp_ds_scale, self.in_chans)
        # self.temup = TEMUP(self.temp_chans*4,2,self.in_chans)
        # self.ref2 = nn.Sequential(
        #     nn.Conv3d(self.temp_chans*2, self.temp_chans, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
        #     ResConv3D(self.temp_chans,self.inplace),
        #     nn.Conv3d(self.temp_chans, self.temp_chans//2, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
        #     ResConv3D(self.temp_chans//2,self.inplace),
        #     nn.Conv3d(self.temp_chans//2,1, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
        # )
        
    def forward(self, ds_meas):
        b,c,t,h,w = ds_meas.shape        # b,  1, t ,h , w  
        denoise = self.denoise(ds_meas)  # b, 32, t/2 ,h , w
        enhanced_fea = self.pos_code(denoise)
        for vit_up in self.mamba:
            enhanced_fea = vit_up(enhanced_fea)  
            
        # print(denoise.shape,enhanced_fea.shape)
        spup = self.spup(enhanced_fea)      # b, 32, t/2 ,h*self.sp_ds_scale, w*self.sp_ds_scale
        # temup = self.temup(spup)       # b, 32, t   ,h*self.sp_ds_scale, w*self.sp_ds_scale
        simple_up = F.interpolate(ds_meas.reshape(-1,t,h,w), size=(self.targrt_size,) * 2)
        simple_up = simple_up.reshape(b,c,t,self.targrt_size,self.targrt_size)
        simple_up = simple_up[:,:,::2,:,:] + simple_up[:,:,1::2,:,:]
        # print(simple_up.shape,spup.shape)
        final = spup + simple_up
        return spup
    
    

class mamba_up2(nn.Module):
    def __init__(self, in_chans=1,temp_chans=8,sp_ds_scale=4,targrt_size=128,inplace=False):
        super(mamba_up2, self).__init__()
        self.in_chans = in_chans
        self.temp_chans = temp_chans
        self.sp_ds_scale = sp_ds_scale
        self.targrt_size =targrt_size
        self.inplace = inplace
        
        self.denoise = MsFeat_2(in_channels= self.in_chans, out_channels=self.temp_chans)
        self.pos_code = PosiEncCNN(self.temp_chans * 4)   
        self.mamba = modelClone(GlobalEncoderSep(dim=self.temp_chans * 4,input_resolution=[self.targrt_size//self.sp_ds_scale,self.targrt_size//self.sp_ds_scale,256],num_heads=4), 12)
        # self.spup = SPUP(self.temp_chans*4,self.sp_ds_scale, self.in_chans)
        # self.temup = TEMUP(self.temp_chans*4,2,self.in_chans)
        self.ref = nn.Sequential(
            nn.Conv3d(self.temp_chans*4, self.temp_chans*2, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
            ResConv3D(self.temp_chans*2,self.inplace),
            nn.Conv3d(self.temp_chans*2, self.temp_chans, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
            ResConv3D(self.temp_chans,self.inplace),
            nn.Conv3d(self.temp_chans,1, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
        )
        
    def forward(self, ds_meas):
        b,c,t,h,w = ds_meas.shape        # b,  1, t ,h , w  
        denoise = self.denoise(ds_meas)  # b, 32, t/2 ,h , w
        enhanced_fea = self.pos_code(denoise)
        for vit_up in self.mamba:
            enhanced_fea = vit_up(enhanced_fea)  
            
        # print(denoise.shape,enhanced_fea.shape)
        # spup = self.spup(enhanced_fea)      # b, 32, t/2 ,h*self.sp_ds_scale, w*self.sp_ds_scale
        enhanced_fea_up = F.interpolate(enhanced_fea, [t//2,self.targrt_size,self.targrt_size])
        # print(enhanced_fea.shape,enhanced_fea_up.shape)
        enhanced_fea_up = enhanced_fea_up.reshape(b,self.temp_chans * 4,t//2,self.targrt_size,self.targrt_size)
        spup = self.ref(enhanced_fea_up)      # b, 32, t/2 ,h*self.sp_ds_scale, w*self.sp_ds_scale
        
        # temup = self.temup(spup)       # b, 32, t   ,h*self.sp_ds_scale, w*self.sp_ds_scale
        simple_up = F.interpolate(ds_meas.reshape(-1,t,h,w), size=(self.targrt_size,) * 2)
        simple_up = simple_up.reshape(b,c,t,self.targrt_size,self.targrt_size)
        simple_up = simple_up[:,:,::2,:,:] + simple_up[:,:,1::2,:,:]
        # print(simple_up.shape,spup.shape)
        final = spup + simple_up
        return final
    




class mamba_upf3(nn.Module):
    def __init__(self, in_chans=1,temp_chans=8,sp_ds_scale=4,targrt_size=128,inplace=False):
        super(mamba_upf3, self).__init__()
        self.in_chans = in_chans
        self.temp_chans = temp_chans
        self.sp_ds_scale = sp_ds_scale
        self.targrt_size =targrt_size
        self.inplace = inplace
        
        self.denoise = MsFeat_2(in_channels= self.in_chans, out_channels=self.temp_chans)
        self.pos_code = PosiEncCNN(self.temp_chans * 4)   
        self.mamba = modelClone(GlobalEncoderSep(dim=self.temp_chans * 4,input_resolution=[self.targrt_size//self.sp_ds_scale,self.targrt_size//self.sp_ds_scale,256],num_heads=4), 6)
        self.fusion1 = transient_fusion(self.temp_chans * 4, [self.targrt_size//self.sp_ds_scale,self.targrt_size//self.sp_ds_scale,256],8, 6)
        self.fusion2 = transient_fusion(self.temp_chans * 4, [self.targrt_size//self.sp_ds_scale,self.targrt_size//self.sp_ds_scale,256],8, 6)
        
        self.spup = SPUP(self.temp_chans*4,self.sp_ds_scale, self.in_chans)
        # self.temup = TEMUP(self.temp_chans*4,2,self.in_chans)
        # self.ref = nn.Sequential(
        #     nn.Conv3d(self.temp_chans*4, self.temp_chans*2, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
        #     ResConv3D(self.temp_chans*2,self.inplace),
        #     nn.Conv3d(self.temp_chans*2, self.temp_chans, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
        #     ResConv3D(self.temp_chans,self.inplace),
        #     nn.Conv3d(self.temp_chans,1, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
        # )
        
    def forward(self, ds_meas):
        b,f,c,t,h,w = ds_meas.shape        # b,  1, t ,h , w  
        ds_meas_re = ds_meas.reshape(-1,c,t,h,w)
        denoise = self.denoise(ds_meas_re)  # bf, 32, t/2 ,h , w
        enhanced_fea = self.pos_code(denoise)
        for vit_up in self.mamba:
            enhanced_fea = vit_up(enhanced_fea)  
            
        # print(denoise.shape,enhanced_fea.shape)
        enhanced_fea = enhanced_fea.reshape(b,f,*enhanced_fea.shape[1:])
        # print(enhanced_fea.shape)
        fu1_t,fu1_n = self.fusion1(enhanced_fea[:,1,...],enhanced_fea[:,0,...])
        fu2_t,fu2_n = self.fusion2(enhanced_fea[:,1,...],enhanced_fea[:,2,...])
        enhanced_fea = enhanced_fea[:,1,...] + fu1_t + fu2_t + fu1_n + fu2_n
        spup = self.spup(enhanced_fea)      # bf, 32, t/2 ,h*self.sp_ds_scale, w*self.sp_ds_scale

        # temup = self.temup(spup)       # b, 32, t   ,h*self.sp_ds_scale, w*self.sp_ds_scale
        simple_up = F.interpolate(ds_meas[:,1,...].reshape(-1,t,h,w), size=(self.targrt_size,) * 2)
        simple_up = simple_up.reshape(b,c,t,self.targrt_size,self.targrt_size)
        simple_up = simple_up[:,:,::2,:,:] + simple_up[:,:,1::2,:,:]
        # print(simple_up.shape,spup.shape)
        final = spup + simple_up
        return final
