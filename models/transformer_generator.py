import functools

import torch
import torch.nn as nn

from utils.layers import Block, LinearReshape, PixelUpsample, To_RGB, gelu
from utils.trunc_normal import trunc_normal_

class TGenerator(nn.Module):
    def __init__(self, latent_dims=1024, img_size=32, bottom_width=8, embed_chs=384, depth=[5,2,2],
                 drop_path_rate=0, num_heads=4, mlp_ratio=4, qkv_bias=False, qk_scale=None,  
                 mlp_drop=0, attn_drop=0, att_mask=False, act_layer=gelu, norm_layer=nn.LayerNorm):
        
        super().__init__()
        # fix depth = 3 for now since it requires changes in upsample 
        assert isinstance(depth, list) and len(depth) == 3

        self.bottom_width, self.embed_chs = bottom_width, embed_chs
        self.in_layer = LinearReshape(in_dims=latent_dims, out_dims=embed_chs*bottom_width**2, 
                                      width=bottom_width, embed_chs=embed_chs)
        self.pos_embed = nn.ParameterList([
                          nn.Parameter(torch.zeros(1, bottom_width**2, embed_chs)),
                          nn.Parameter(torch.zeros(1, (bottom_width*2)**2, embed_chs//4)),
                          nn.Parameter(torch.zeros(1, (bottom_width*4)**2, embed_chs//16))
        ])
        for emb in self.pos_embed:
            trunc_normal_(emb, std=.02)

        Partial_Block = functools.partial(Block, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, qk_scale=qk_scale, mlp_drop=mlp_drop, attn_drop=attn_drop, 
                            att_mask=att_mask, act_layer=act_layer, norm_layer=norm_layer)
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth[0])]  
        self.bottom_block = nn.ModuleList([ 
                    Partial_Block(dims=embed_chs, drop_path=dpr[i]) for i in range(depth[0]) 
        ])
        self.upsample_block = nn.ModuleList([
                    nn.ModuleList([Partial_Block(dims=embed_chs//4, drop_path=0)] * depth[1]),
                    nn.ModuleList([Partial_Block(dims=embed_chs//16, drop_path=0)] * depth[2])
        ])

        self.pixel_upsample = nn.ModuleList([
                                PixelUpsample(start_width=bottom_width),
                                PixelUpsample(start_width=bottom_width*2)
        ])
        
        self.to_rgb = To_RGB(ch_dims=embed_chs//16, img_size=img_size)

    def forward(self, x, epoch=None):
        x = self.in_layer(x) + self.pos_embed[0]
        for blk in self.bottom_block:
            x = blk(x,epoch)
        for i, blocks in enumerate(self.upsample_block):
            x = self.pixel_upsample[i](x) + self.pos_embed[i+1]
            for blk in blocks:
                x = blk(x,epoch)
        x = self.to_rgb(x)
        return x.contiguous()