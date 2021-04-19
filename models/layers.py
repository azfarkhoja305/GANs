import math

import torch
import torch.nn as nn

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_feat, hidden_feat=None, out_feat=None, act_layer=gelu, drop=0):
        super().__init__()
        out_feat = out_feat or in_feat
        hidden_feat = hidden_feat or in_feat
        self.fc1 = nn.Linear(in_feat, hidden_feat)
        self.act_layer = act_layer
        self.fc2 = nn.Linear(hidden_feat, out_feat)
        self.dropout = nn.Dropout(drop)
    
    def forward(self,x):
        x = self.dropout(self.act_layer(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x


def get_attn_mask(N, w):
    mask = torch.zeros(1, 1, N, N)
    for i in range(N):
        if i <= w:
            mask[:, :, i, 0:i+w+1] = 1
        elif N - i <= w:
            mask[:, :, i, i-w:N] = 1
        else:
            mask[:, :, i, i:i+w+1] = 1
            mask[:, :, i, i-w:i] = 1
    return mask


class Attention(nn.Module):
    def __init__(self, dims, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0,
                 proj_drop=0, att_mask=0):
        super().__init__()
        self.num_heads = num_heads
        assert dims % num_heads == 0, (f'Dims for Attention: {dims}, not divisible '
            f'by num_heads: {num_heads}')
        head_dims = dims // num_heads
        self.scale = qk_scale or head_dims ** -0.5
        
        self.qkv = nn.Linear(dims, dims*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dims, dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.att_mask = att_mask
        if att_mask:
            self.mask_1 = get_attn_mask(att_mask, 6)
            self.mask_2 = get_attn_mask(att_mask, 8) 
            self.mask_3 = get_attn_mask(att_mask, 10) 
            self.mask_4 = get_attn_mask(att_mask, 12) 
    
    def forward(self, x, epoch):
        B,N,C = x.shape
        # q,k,v shape: (B, num_heads, N, head_dims)
        q,k,v = self.qkv(x).view(B,N,self.num_heads,-1).permute(0,2,1,3).tensor_split(3, dim=-1)
        # score: (B, num_heads, N, N)
        score = (q @ k.transpose(-2,-1)) * self.scale

        if self.att_mask and epoch is not None:
            if epoch < 50:
                if epoch < 20:  mask = self.mask_1
                elif epoch < 30:  mask = self.mask_2
                elif epoch < 40:  mask = self.mask_3
                else:  mask = self.mask_4
                score = score.masked_fill(mask.to(score.device)==0, -1e9)

        score = self.attn_drop(score.softmax(dim=-1))
        # context: (B, N, C)
        context = (score @ v).permute(0,2,1,3).reshape(B,N,C)
        context = self.proj_drop(self.proj(context))
        return context


class Block(nn.Module):
    def __init__(self, dims, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, mlp_drop=0, attn_drop=False,
                 drop_path=0, att_mask=0, act_layer=gelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_1 = norm_layer(dims)
        self.attention = Attention(dims, num_heads, qkv_bias, qk_scale, attn_drop, 
                                   mlp_drop, att_mask)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(dims)
        mlp_hidden_dim = int(dims*mlp_ratio)
        self.mlp = MLP(in_feat=dims, hidden_feat=mlp_hidden_dim, act_layer=act_layer, 
                       drop=mlp_drop)

    def forward(self, x, epoch):
        x = x + self.drop_path(self.attention(self.norm_1(x), epoch))
        x = x + self.drop_path(self.mlp(self.norm_2(x)))
        return x  


class PixelUpsample(nn.Module):
    def __init__(self, start_width):
        super().__init__()
        self.width = start_width
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)
    def forward(self, x):
        B,N,C = x.size()
        assert N == self.width**2
        x = x.permute(0,2,1).view(B, C, self.width, self.width)
        x = self.pixelshuffle(x)
        B, C, H, W = x.size()
        x = x.view(B,C,H*W).permute(0,2,1)
        return x


class To_RGB(nn.Module):
    def __init__(self, ch_dims, img_size):
        super().__init__()
        self.img_size = img_size
        # TODO: Keep or remove bias ?
        self.linear = nn.Linear(ch_dims, 3, bias=True)
        self.act_fn = nn.Tanh()
    def forward(self, x):
        # From (B,N,C) to (B,3,img_size,img_size)
        x = self.act_fn(self.linear(x))
        x = x.permute(0,2,1).view(-1,3,self.img_size,self.img_size)
        return x


class LinearReshape(nn.Module):
    def __init__(self, in_dims, out_dims, width, embed_chs):
        super().__init__()
        self.width = width 
        self.embed_chs = embed_chs
        self.linear = nn.Linear(in_dims, out_dims, bias=False)
    def forward(self, x):
        x = self.linear(x)
        return x.view(-1, self.width**2, self.embed_chs)

