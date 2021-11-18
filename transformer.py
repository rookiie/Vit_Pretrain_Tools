import os
import torch
import torch.nn as nn
from .tools import *

# -*- encoding: utf-8 -*-
'''
Filename         :transformer.py
Description      :
Time             :2021/11/18 16:54:56
Author           :senxu
Version          :1.0
'''




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        x = self.proj(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

#        self.attend = nn.Softmax(dim = -1)
        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, model_name, embed_dim, image_size, patch_size, depth, heads, mlp_ratio, qkv_bias=True, 
                qk_scale=None, drop_rate = 0.1,attn_drop_rate=0.,drop_path_rate=0.,pretrained=None):
        super(VisionTransformer,self).__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = heads
        self.mlp_ratio = mlp_ratio
        self.pretrained = pretrained
        self.qkv_bias = qkv_bias
        self.image_size = image_size
        image_height, image_width = image_size
        assert image_height % patch_size == 0 and image_width % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patches = (image_height // patch_size) * (image_width // patch_size)
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = nn.LayerNorm
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        self.out_indices = list(range(self.depth // 4 - 1, self.depth, self.depth // 4))


        self.patch_embed = PatchEmbed(
                img_size=self.img_size, patch_size=self.patch_size, in_chans=3, embed_dim=self.embed_dim)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer)
            for i in range(self.depth)])
        
        # 缺全联接分类


    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2) # (b,n,h,w) -> b(h*w)n
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
              
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        
        # ======== TODO =========
        # 
        # out = self.fc(x)
        # return out 
        return x

    def init_weights(self,):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if self.pretrained is not None:
            self.pretrained = os.path.join(self.pretrained, self.model_name+'.pth')
            self.load_pretrained()
            return
        print("==> Random initialize the parameter of vit backbone")
        return

    def load_pretrained(self, keep_fc=True, strict=True, pos_embed_interp=True, align_corners=False):

        if os.path.exists(self.pretrained):
            state_dict = torch.load(self.pretrained)
            print('==> load pre-trained weight from ' + self.pretrained)
        else:
            print('Please check pretrained model, use random initialization')
            return
        
        # 'path_embed is not used in this work'
 #       del state_dict['patch_embed.proj.weight']
 #       del state_dict['patch_embed.proj.bias']
        
        classifier_name = 'head'
        if not keep_fc:
            if 'norm.weight' in state_dict.keys():
                del state_dict['norm.weight']
                del state_dict['norm.bias']
            # completely discard fully connected for all other differences between pretrained and created model
            if classifier_name + '.weight' in state_dict.keys():
                del state_dict[classifier_name + '.weight']
                del state_dict[classifier_name + '.bias']

        if pos_embed_interp:
            n, c, hw = state_dict['pos_embed'].transpose(1, 2).shape
            h = w = int(math.sqrt(hw))
            pos_embed_weight = state_dict['pos_embed'][:, (-h * w):]
            pos_embed_weight = pos_embed_weight.transpose(1, 2)
            n, c, hw = pos_embed_weight.shape
            h = w = int(math.sqrt(hw))
            pos_embed_weight = pos_embed_weight.view(n, c, h, w)

            pos_embed_weight = F.interpolate(pos_embed_weight, size=int(
                math.sqrt(self.num_patches)), mode='bilinear', align_corners=align_corners)
            pos_embed_weight = pos_embed_weight.view(n, c, -1).transpose(1, 2)

            cls_token_weight = state_dict['pos_embed'][:, 0].unsqueeze(1)

            state_dict['pos_embed'] = torch.cat(
                (cls_token_weight, pos_embed_weight), dim=1)
            print("==> change the pos_embedding to meet the image_size of {}".format(self.image_size))

        self.load_state_dict(state_dict, strict=strict)




def build_vit_from_cfg(cfg):

    args = cfg.copy()
    assert 'image_size' in args.keys(), 'Lack of para to build Vit backbone!'
    return VisionTransformer(**args)


# vit_config_factory = dict(
#     S_16_224_config = {
#         'depth': 12,
#         'model_name': 'vit_S_16_224_image21K',
#         'heads': 6,
#         'embed_dim': 384,
#         'mlp_ratio': 4,
#         'patch_size':16,
# #        'input': 224,
#         'drop_rate': 0.1,
#     },
# )




    