# 2022 AI 학습용 데이터 구축사업 (1차) 
![그림1](https://user-images.githubusercontent.com/85090866/209904316-59a3c914-b647-49be-a233-55c6840f126b.png)

## 벼 생육이상 인식 데이터 의미론적 분할(Semantic Segmentation)

## Preparation (Pre-requisites)

### 도커 이미지 복제 (Docker image)
Note: nvidia-docker2 is required. (maybe already installed)
```bash
docker pull ironluffy/rice:initial
```
### 깃 허브 복제 (Code - git clone)
```bash
git clone https://github.com/ironluffy/RiceSeg.git
```

### MMSegmentation 설치 (Install MMSegmentation)
```bash
cd RiceSeg/mmsegmentation
pip install -v -e .
```

### 파일 구조 (Project directory structure)

#### Case 1(압축 된 상태): 코드를 실행하기 전에 파일 구조가 다음과 같은지 확인하십시오:
##### * Before running the code, please make sure the directory structure is as follows:

```bash
.
├── Dataset
│   ├── 1.원천데이터
│   │   ├── R20220726A25B1036.tif
│   │   ├── R20220726A25B1037.tif
│   │   └── ...
│   ├── 2.라벨링데이터
│   │   ├── R20220726A25B1036.json
│   │   ├── R20220726A25B1037.json
│   │   └── ...
├── RiceSeg
│   ├── README.md
│   ├── src
│   ├── mmsegmentation
│   ├── pretrained_ckpt
│   └── .gitignore
└── index.html
```

#### ![image](https://user-images.githubusercontent.com/85090866/210024180-c1415002-19c9-4373-8e68-b2917a402372.png) 하위 모든 과정 RiceSeg 폴더에서 진행 
## 데이터 전처리 (Data processing)

### Case 1 : 위의 Case 1번은 아래의 전처리 파일을 실행하세요.
##### * For Case 1 above, run the preprocessing file below.
```bash
python3 src/data_preprocess.py
```

## 훈련 (Train)

### 확인사항 (Check-list)
  
##### `mmsegmentation/configs/_base_/datasets/rice.py`의 `data_root`는 데이터 폴더의 경로여야 합니다. 
    ```python
    dataset_type = 'RiceDataset'
    data_root= '../data'
    ...
    ```
    
### 훈련 데모 (Training Demo)
##### 본 과제에서의 최종모델은 Segformer 입니다. Knet, Segmenter 모델은 성능비교를 위해 제공합니다.

#### Segformer
```bash
python3 ./mmsegmentation/tools/train.py ./mmsegmentation/configs/rice/segformer_mit-b4_lovasz_gne_chw.py
```

#### KNet 
```bash
python3 mmsegmentation/tools/train.py mmsegmentation/configs/rice/knet_s3_upernet_swin-l_lovasz_gne_chw.py
```

#### Segmenter
```bash
python3 ./mmsegmentation/tools/train.py ./mmsegmentation/configs/rice/segmenter_vit-b_mask.py
```     


## 맞춤형 훈련 파이프라인 (Customize training pipeline)

### 훈련 구성 (Training configurations)

##### 사용자는 `mmsegmentation/configs/rice` 폴더에 파일을 참조하고 구성파일을 추가할 수 있습니다. `mmsegmentation/configs/_base_/models` 폴더에 미리 정의된 여러개의 모델을 사용할 수 있으며 새로운 모델을 추가할 수 있습니다.`mmsegmentation/configs/_base_/datasets`에서 데이터 세트의 구성을 변경할 수 있으며 기본데이터 세트는`mmsegmentation/configs/_base_/datasets`의 `rice_gne_chw.py`입니다.

###### * Please refer to the files in `mmsegmentation/configs/rice` and add a new configuration file you want to use. You can use several pre-defined models in `mmsegmentation/configs/_base_/models` and you can add a new model in there. Also, you can change the configuration of dataset as well, in `mmsegmentation/configs/_base_/datasets`. The default dataset is `rice_gne_chw.py` in `mmsegmentation/configs/_base_/datasets`.
* * *
##### 사용자가 모델의 모델 학습 반복횟수(iterations)를 늘리거나 줄이려는 경우 훈련을 위한 기본 런타임 구성인 `mmsegmentation/configs/_base_/rice_runtime.py`를 참조하십시요.
###### * If you want to increase/decrease the nubmer of iterations. Please refer to the fiels in `mmsegmentation/configs/_base_/rice_runtime.py` which is the the basic run-time configuration for tranining.
* * *
##### Tips. 생각보다 성능이 나오지 않는다면 여러개의 GPU를 사용하여 `./mmsegmentation/tools/train.py` 대신 `./mmsegmentation/tools/dist_train.sh`를 사용하시길 권장드립니다.
###### * If you want to your multiple GPUs, your `./mmsegmentation/tools/dist_train.sh` instead of `./mmsegmentation/tools/train.py`. (Try it if the performance is lower than you think)


---
## 테스트(TEST)
```bash
python3 ./mmsegmentation/tools/test.py .{config file path} {checkpoint_path} --eval mIoU 
```
#### 테스트 결과를 저장하려면 명령어에 `--show-dir {출력 경로}`를 추가하십시오.
###### * Optionally, if you want to save inference results, please add `--show-dir {output path}` to the command.


### 예시(Examples) (정상 vs. 도복 vs. 도열 vs. 결주 vs. 생육부진)
#### KNet
###### 훈련이 종료된 후 (After training step finished)
```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/rice/knet_s3_upernet_swin-l_lovasz_gne_chw.py ./work_dirs/knet_s3_upernet_swin-l_lovasz_gne_chw/latest.pth --eval mIoU
```

###### 또는 모델의 체크포인트가 제공될 때 (using provided best checkpoint [tentative])
```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/rice/knet_s3_upernet_swin-l_lovasz_gne_chw.py ./best_ckpt/knet.pth --eval mIoU
```

#### Segmenter
###### 훈련이 종료된 후 (After training step finished)
```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/rice/segmenter_vit-b_lovasz_gne_chw.py ./work_dirs/segmenter_vit-b_mask/latest.pth --eval mIoU
```
###### 또는 모델의 체크포인트가 제공될 때 (using provided best checkpoint [tentative])
```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/rice/segmenter_vit-b_lovasz_gne_chw.py ./best_ckpt/segmenter.pth --eval mIoU
```

#### Segformer
###### 훈련이 종료된 후 (After training step finished)
```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/rice/segformer_mit-b4_lovasz_gne_chw.py ./work_dirs/segformer_mit-b4_lovasz_gne_chw/latest.pth --eval mIoU
```

###### 또는 모델의 체크포인트가 제공될 때 (using provided best checkpoint [tentative])
```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/rice/segformer_mit-b4_lovasz_gne_chw.py ./best_ckpt/segformer.pth --eval mIoU
```

### 단일 비교(Specific configurations) (e.g., 정상 vs. 생육부진)
`mmsegmentation/configs/{class_name}` 를 참조하시길 바랍니다.
예를 들어 `정상 vs. 결주`로 모델을 테스트하고 싶다면 다음 명령어를 사용하세요(특정모델의 체크포인트 선택가능)
모델의 성능에 대한 평균정확도를 확인하세요.
###### * Please refer to the directory `mmsegmentation/configs/{class_name}`. For example, if you want to test the model with the configuration of `normal vs. gyeolju`, please use the following command. (you can choose specific model checkpoint) And you should report mean accuracy (mAcc.) for the model performance

```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/gyeolju/segformer_mit-b4.py ./work_dirs/segformer_mit-b4_lovasz_gne_chw/latest.pth --eval mIoU
```
