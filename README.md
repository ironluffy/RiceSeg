# Rice growth abnormality recognition - Semantic Segmentation

## Preparation

### Code - git clone
After cloning this repo, please use the following command to initialize ```mmsegmentation``` submodule
```bash
git submodule update --init --recursive
```


### model config
- knet: knet_s3_upernet_r50-d8_8x2_512x512_adamw_80k_ade20k_rice.py

### model setting
1. ade20k_rice.py
- datset_type
- crop_size, rice_img_scale
- data_root

2. mmseg/datsets/__init__.py
- added our objects 'RiceDataset'
