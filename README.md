# Vit_Pretrain_Tools
PyTorch implementation tools for using vision transformer with Google pretrained model. (npz2pth)


## Details

* Transformer.py provides a generic pytorch implementation of [Vision Transformer](https://github.com/google-research/vision_transformer). And the basic module draws on [SETR](https://github.com/fudan-zvg/SETR). 

* weights/npz2pth.py help us to convert the [pre-trained model](https://github.com/google-research/vision_transformer) provided by Google in NPZ format into the applicable pth format.


## Usage

* Edit the config in Transformer.py to build the vit type you want. [More detail setting](https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py) 

* Edit the config in npz2pth.py. Note that we do not check the output categories number of the classification head. Download the suitable npz file according to your config from [Vision Transformer](https://github.com/google-research/vision_transformer). And this config dict has slight differences compared to config mentioned above.

```
python npz2pth.py --config=[$CONFIG_DICT] --npz_path=[$NPZ_FILE] --save_path=[$SAVE]
```

* Initialize the parameter with the pretrained model. When key 'pretrained' is not setting in the cfg dict, function init_weight() will initialize network randomly.
```
S_16_224_config = {
        'depth': 12,
        'model_name': 'vit_S_16_224_image21K', # pth name
        'image_size':(512,512), # size of the images fed into network
        'heads': 6,
        'embed_dim': 384,
        'mlp_ratio': 4,
        'patch_size':16,
        'drop_rate': 0.1,
    }
cfg = S_16_224_config
cfg['pretrained'] = './weight/' # root path of pth file

model = build_vit_from_cfg(cfg)
model.init_weight()
```
