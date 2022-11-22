# Rice growth abnormality recognition - Semantic Segmentation

## Preparation

### Docker image
```bash
docker pull ironluffy/rice:initial
```
### Code - git clone
```bash
git clone https://github.com/ironluffy/RiceSeg.git
```

### Project directory structure

#### Case 1: Before running the code, please make sure the directory structure is as follows:

```bash
.
├── rice_raw_data
│   ├── a01_2.zip
│   ├── a10_2.zip
│   └── ...
├── RiceSeg
│   ├── README.md
│   ├── src
│   ├── mmsegmentation
│   ├── pretrained_ckpt
│   └── .gitignore
└── index.html
```

#### Case 2: In case of downloading full dataset (not zipped), please unzip the dataset and put it in the `rice_unzipped` folder.

```bash
.
├── rice_unzipped
│   ├── org (origianl images)
│   └── png (PNG)
├── RiceSeg
│   └── ...
...
```

## Data processing

### Case 1 (in Project directory structure)
```bash
python3 src/data_preprocess.py
```

### Case 2 (in Project directory structure)
```bash
python3 src/data_preprocess.py  --skip_unzip
```

## Training

### Check-list
- `data_root` should be the path of the data directory.  
    In the `mmsegmentation/configs/_base_/datasets/rice.py`
    ```python
    dataset_type = 'RiceDataset'
    data_root= '/rice/data'
    ...
    ```
- download pre-trained checkpoints in `pretrained_ckpt`
  - [segmenter](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth)
  - [segformer](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20220617_162207-c00b9603.pth)

### Training Demo

#### KNet
```bash
python3 mmsegmentation/tools/train.py mmsegmentation/configs/rice/knet_s3_deeplabv3.py
```

#### Segmenter
```
 python3 ./mmsegmentation/tools/train.py ./mmsegmentation/configs/rice/segmenter_vit-b_mask.py
```     
#### Segformer
```bash
python3 ./mmsegmentation/tools/train.py ./mmsegmentation/configs/rice/_mit-b0.py
```