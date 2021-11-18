import os
import numpy as np
import torch
import collections
import argparse

# -*- encoding: utf-8 -*-
'''
Filename         :npz2pth.py
Description      :npz2pth for vision transformer pretrained model
Time             :2021/11/15 20:00:13
Author           :Sen Xu
Version          :1.0
'''
cfg_factory = dict(
    S_16_224_config = {
        'depth': 12,
        'num_head': 6,
        'hidden_dim': 384,
        'mlp_dim': 1536,
        'patch_size':16,
        'input': 224,
    },
    
)

# https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py
# gsutil ls -lh gs://vit_models/augreg/*
# https://github.com/google-research/vision_transformer

def parse_args():
    parser = argparse.ArgumentParser(description='npz2npy for vision transformer')
    parser.add_argument('--config', default='S_16_224_config', choices=['S_16_224_config','B_16_config','L_16_config'], type=str, help='config selection')
    parser.add_argument('--npz_path',default='./npz/',type=str,help='pretrained model path')
    parser.add_argument('--save_path',default='./',type=str, help='where to save the pth model')
    parser.add_argument('--keep_fc', default=True, type=bool, help='whether to keep classification head')
    return parser.parse_args()

args = parse_args()


def trans_cls_pos(npz_para, pth_dict, cfg):
    assert('cls' in npz_para.files and 'Transformer/posembed_input/pos_embedding' in npz_para.files), 'Please check cls_token in npz files'
    pth_dict['pos_embed']= torch.from_numpy(npz_para['Transformer/posembed_input/pos_embedding'])
    pth_dict['cls_token']= torch.from_numpy(npz_para['cls'])

    if (int(pth_dict['pos_embed'].shape[1]) != int((cfg['input'] / cfg['patch_size'])**2 + 1)):
        print('The input_size or path_size does not satisfy the settings you wish!')
        exit(1)
    elif (int(pth_dict['pos_embed'].shape[2]) != cfg['hidden_dim']):
        print('The hidden_dim does not satisfy the settings you wish!')
        exit(1)
    else: 
        return pth_dict

def trans_patch_embed(npz_para, pth_dict):
    assert('embedding/kernel' in npz_para.files and 'embedding/bias' in npz_para.files), 'Please check path_embedding in npz files'
    pth_dict['patch_embed.proj.weight']= torch.from_numpy(npz_para['embedding/kernel']).permute(3,2,0,1)
    pth_dict['patch_embed.proj.bias']= torch.from_numpy(npz_para['embedding/bias'])
    return pth_dict

def get_block_name(num_block, type=0, weight=True, layer=0):
    """
    Arguments
    ---------
    num_block -> "block_index"
    type -> "0: norm_layer1, 1: multi_head_attn, 2:norm_layer2, 3:mlp_block"
    weight -> "True for weight and False for bias"
    layer -> "0: qkv for attn and fc1 for mlp" "1: proj for attn and ..." ...  
    Returns
    -------
    npz_para_name, pth_para_name
    """
    pth_base = "blocks.{}.".format(num_block)
    npz_base = "Transformer/encoderblock_{}/".format(num_block)
    if type == 0:
        if weight:
            return npz_base+'LayerNorm_0/scale', pth_base+'norm1.weight'
        else:
            return npz_base+'LayerNorm_0/bias', pth_base+'norm1.bias'
    elif type == 1:
        npz_base = npz_base + "MultiHeadDotProductAttention_1/"
        if layer == 0:
            qkv_type = ['query','key','value']
            if weight:
                return [npz_base + "{}/kernel".format(i) for i in qkv_type], pth_base + "attn.qkv.weight"
            else:
                return [npz_base + "{}/bias".format(i) for i in qkv_type], pth_base + "attn.qkv.bias"
        elif layer == 1:
            if weight:
                return npz_base + "out/kernel", pth_base + "attn.proj.weight"
            else:
                return npz_base + "out/bias", pth_base + "attn.proj.bias"
    elif type == 2:
        if weight:
            return npz_base+'LayerNorm_2/scale', pth_base+'norm2.weight'
        else:
            return npz_base+'LayerNorm_2/bias', pth_base+'norm2.bias'
    elif type == 3:
        if weight:
            return npz_base + "MlpBlock_3/Dense_{}/kernel".format(layer), pth_base + "mlp.fc{}.weight".format(layer+1)
        else:
            return npz_base + "MlpBlock_3/Dense_{}/bias".format(layer), pth_base + "mlp.fc{}.bias".format(layer+1)



def copy2pth(npz_name, pth_name, npz_para, pth_dict, type, weight, cfg):

    if not isinstance(npz_name, list):
        assert(npz_name in npz_para.files), "Please check item {} in npz para files".format(npz_name)
        if type is 'norm':
            pth_dict[pth_name] = torch.from_numpy(npz_para[npz_name])
        elif type is 'mlp':
            if weight:
                assert cfg['mlp_dim'] in npz_para[npz_name].shape, "Mlp hidden dim in npz file does not satisfy the config!"
                pth_dict[pth_name] = torch.from_numpy(npz_para[npz_name]).permute(1,0)
            else:
                pth_dict[pth_name] = torch.from_numpy(npz_para[npz_name])
        elif type is 'attn':
            if weight:
                a,b,c = npz_para[npz_name].shape
                if not cfg['num_head'] == a:
                    print("Attention Layer head number in npz file does not satisfy the config!")
                    raise AssertionError
                pth_dict[pth_name] = torch.from_numpy(npz_para[npz_name]).view(-1,c).permute(1,0)
            else:
                pth_dict[pth_name] = torch.from_numpy(npz_para[npz_name])
    else:
        assert type is 'attn' and len(npz_name) == 3
        if weight:
            a,b,c = npz_para[npz_name[0]].shape
            if not cfg['num_head'] == b:
                print("Attention Layer head number in npz file does not satisfy the config!")
                raise AssertionError
            pth_dict[pth_name] = torch.cat([torch.from_numpy(npz_para[i]).view(a,-1) for i in npz_name],dim=1).permute(1,0)
        else:
            pth_dict[pth_name] = torch.cat([torch.from_numpy(npz_para[i]).view(-1) for i in npz_name],dim=0)

    return pth_dict


def trans_block(block_index, npz_para, pth_dict, cfg):

    for i in range(4):    
        if i % 2 == 0:
            for weight in [True, False]:
                npz_name, pth_name = get_block_name(block_index, i, weight)
                print(npz_name)
                print(pth_name)
                pth_dict = copy2pth(npz_name, pth_name, npz_para, pth_dict, 'norm', weight, cfg)
        elif i == 1:
            for j in range(2):
                for weight in [True, False]:
                    npz_name, pth_name = get_block_name(block_index, i, weight, j)
                    pth_dict = copy2pth(npz_name, pth_name, npz_para, pth_dict,'attn',weight,cfg)
        elif i == 3:
            for j in range(2):
                for weight in [True, False]:
                    npz_name, pth_name = get_block_name(block_index, i, weight, j)
                    pth_dict = copy2pth(npz_name, pth_name, npz_para, pth_dict,'mlp',weight,cfg)

    return pth_dict


def trans_out_fc(npz_para, pth_dict):
    pth_dict['norm.weight'] = torch.from_numpy(npz_para['Transformer/encoder_norm/scale'])
    pth_dict['norm.bias'] = torch.from_numpy(npz_para['Transformer/encoder_norm/bias'])
    pth_dict['head.weight'] = torch.from_numpy(npz_para['head/kernel']).permute(1,0)
    pth_dict['head.bias'] = torch.from_numpy(npz_para['head/bias'])
    return pth_dict    

def main():
    cfg = cfg_factory[args.config]
 #   print(cfg)
 #   print(type(cfg['input']))
    
    if not os.path.exists(args.npz_path):
        print("Sorry, npz file does not exist!")
        exit(1)
    npz_para = np.load(args.npz_path)
    pth_dict = collections.OrderedDict()
    pth_dict = trans_cls_pos(npz_para, pth_dict, cfg)
    pth_dict = trans_patch_embed(npz_para, pth_dict)

    for i in range(cfg['depth']):
        pth_dict = trans_block(i,npz_para, pth_dict,cfg)
    
    if args.keep_fc:
        pth_dict = trans_out_fc(npz_para, pth_dict)

    torch.save(pth_dict,args.save_path)

    return
    


if __name__ == '__main__':
    main()