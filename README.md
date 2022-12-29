# Rice growth abnormality recognition - Semantic Segmentation

## Preparation (Pre-requisites)

### Docker image
Note: nvidia-docker2 is required. (maybe already installed)
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
├── label
│   ├── R2022720A18B0723.json
│   ├── R2022720A18B0724.json
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
│   └── org (origianl images)
│       ├── R20220720A18B
│       ├── R20220720A18E
│       ├── R20220720A18G
│       ├── R20220720A18N
│       ├── R20220720A18R
│       └── ...
├── label
│   ├── R2022720A18B0723.json
│   ├── R2022720A18B0724.json
│   └── ...
├── RiceSeg
│   └── ...
...
```
---
## Data processing

### Case 1 (in Project directory structure)
```bash
python3 src/data_preprocess.py
```

### Case 2 (in Project directory structure)
```bash
python3 src/data_preprocess.py  --skip_unzip
```
---
## Train

### Check-list
- `data_root` should be the path of the data directory.  
    In the `mmsegmentation/configs/_base_/datasets/rice.py`
    ```python
    dataset_type = 'RiceDataset'
    data_root= '../data'
    ...
    ```
### Training Demo

#### KNet
```bash
python3 mmsegmentation/tools/train.py mmsegmentation/configs/rice/knet_s3_upernet_swin-l_lovasz_gne_chw.py
```

#### Segmenter
```bash
python3 ./mmsegmentation/tools/train.py ./mmsegmentation/configs/rice/segmenter_vit-b_mask.py
```     
#### Segformer
```bash
python3 ./mmsegmentation/tools/train.py ./mmsegmentation/configs/rice/segformer_mit-b4_lovasz_gne_chw.py
```

### Customize training pipeline

#### Training configurations
Please refer to the files in `mmsegmentation/configs/rice` and add a new configuration file you want to use.
You can use several pre-defined models in `mmsegmentation/configs/_base_/models` and you can add a new model in there.
Also, you can change the configuration of dataset as well, in `mmsegmentation/configs/_base_/datasets`. The default dataset is `rice_gne_chw.py` in `mmsegmentation/configs/_base_/datasets`.

#### If you want to increase/decrease the nubmer of iterations.
Please refer to the fiels in `mmsegmentation/configs/_base_/rice_runtime.py` which is the the basic run-time configuration for tranining.

#### Tips.
- If you want to your multiple GPUs, your `./mmsegmentation/tools/dist_train.sh` instead of `./mmsegmentation/tools/train.py`. (Try it if the performance is lower than you think)


---
## Test
```bash
python3 ./mmsegmentation/tools/test.py .{config file path} {checkpoint_path} --eval mIoU 
```
Optionally, if you want to save inference results, please add `--show-dir {output path}` to the command.


### Examples (normal vs. dobok vs. doyeol vs. gyeolju vs. bujin)
#### KNet
After training step finished,
```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/rice/knet_s3_upernet_swin-l_lovasz_gne_chw.py ./work_dirs/knet_s3_upernet_swin-l_lovasz_gne_chw/latest.pth --eval mIoU
```

or using provided best checkpoint (tentative)
```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/rice/knet_s3_upernet_swin-l_lovasz_gne_chw.py ./best_ckpt/knet.pth --eval mIoU
```

#### Segmenter
After training step finished,
```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/rice/segmenter_vit-b_lovasz_gne_chw.py ./work_dirs/segmenter_vit-b_mask/latest.pth --eval mIoU
```

or using provided best checkpoint (tentative)
```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/rice/segmenter_vit-b_lovasz_gne_chw.py ./best_ckpt/segmenter.pth --eval mIoU
```

#### Segformer
After training step finished,
```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/rice/segformer_mit-b4_lovasz_gne_chw.py ./work_dirs/segformer_mit-b4_lovasz_gne_chw/latest.pth --eval mIoU
```

or using provided best checkpoint (tentative)
```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/rice/segformer_mit-b4_lovasz_gne_chw.py ./best_ckpt/segformer.pth --eval mIoU
```

### Specific configurations (e.g., normal vs. bujin)
Please refer to the directory `mmsegmentation/configs/{class_name}`.
For example, if you want to test the model with the configuration of `normal vs. gyeolju`, please use the following command. (you can choose specific model checkpoint)
And you should report mean accuracy (mAcc.) for the model performance
```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/gyeolju/segformer_mit-b4.py ./work_dirs/segformer_mit-b4_lovasz_gne_chw/latest.pth --eval mIoU
```
