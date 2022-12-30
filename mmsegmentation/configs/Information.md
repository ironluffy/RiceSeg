# 모델 정보 및 라이센스 가이드

## 인공지능 기반 벼 생육 이상 분할 모델

<!-- [ABSTRACT] -->

(모델 개발 목표) 벼의 생육 과정에서 일어나는 생육 이상을 분할하는 모델을 개발

(개발 내용) 벼 생육 시기 별로 촬영된 5개 채널(R, G, B, NIR, Red-Edge)의  분광이미지를 기반으로 벼의 생육 이상(도열병, 도복, 결주, 생육부진)을 분할하는 모델을 개발

오픈소스인 [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation) 도구를 활용하여 해당 Task에 적합한 모델을 선정하고, 실험을 거쳐 최종 후보를 선택하였습니다. 모델 후보는 K-Net, SegFormer, Segmenter 3종으로 선택되었고, 수집된 데이터로 성능 실험을 진행하여 SegFormer가 최종 모델로 선정되었습니다. 또한, 모델의 기본 설정인 Cross-entropy loss와 더불어 Multi-class data에 더 효과적인 Lovasz loss를 활용하였습니다. 

## 1. (메인모델) SegFormer

[SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

### Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/NVlabs/SegFormer">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/mit.py#L246">Code Snippet</a>

### Abstract

<!-- [ABSTRACT] -->

We present SegFormer, a simple, efficient yet powerful semantic segmentation framework which unifies Transformers with lightweight multilayer perception (MLP) decoders. SegFormer has two appealing features: 1) SegFormer comprises a novel hierarchically structured Transformer encoder which outputs multiscale features. It does not need positional encoding, thereby avoiding the interpolation of positional codes which leads to decreased performance when the testing resolution differs from training. 2) SegFormer avoids complex decoders. The proposed MLP decoder aggregates information from different layers, and thus combining both local attention and global attention to render powerful representations. We show that this simple and lightweight design is the key to efficient segmentation on Transformers. We scale our approach up to obtain a series of models from SegFormer-B0 to SegFormer-B5, reaching significantly better performance and efficiency than previous counterparts. For example, SegFormer-B4 achieves 50.3% mIoU on ADE20K with 64M parameters, being 5x smaller and 2.2% better than the previous best method. Our best model, SegFormer-B5, achieves 84.0% mIoU on Cityscapes validation set and shows excellent zero-shot robustness on Cityscapes-C. Code will be released at: [this http URL](https://github.com/NVlabs/SegFormer).

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142902600-e188073e-5744-4ba9-8dbf-9316e55c74aa.png" width="70%"/>
</div>

```bibtex
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.15203},
  year={2021}
}
```

### Results and models

#### 벼 생육이상 인식 데이터

| Method    | Backbone | Crop Size | Lr schd | Loss Funcion  | Mem (GB) | Inf time (fps) |  mAcc | config                                                                                              |
| --------- | -------- | --------- | ------: | ------------- | -------: | -------------- | ----: | --------------------------------------------------------------------------------------------------- |
| Segformer | MIT-B0   | 512x512   |  000000 | Cross-entropy |      2.1 | 00.00          | 00.00 | [config](https://github.com/RiceSeg/mmsegmentation/configs/rice/segformer_mit-b0_ce_gne_chw.py)     |
| Segformer | MIT-B4   | 512x512   |  000000 | Lovasz        |      6.1 | 00.00          | 00.00 | [config](https://github.com/RiceSeg/mmsegmentation/configs/rice/segformer_mit-b4_lovasz_gne_chw.py) |

Evaluation :

| Method    | Backbone | Crop Size | Lr schd |  mAcc |
| --------- | -------- | --------- | ------: | ----: |
| Segformer | MIT-B0   | 512x512   |  000000 | 00.00 |
| Segformer | MIT-B4   | 512x512   |  000000 | 00.00 |

## 2. K-Net

[K-Net: Towards Unified Image Segmentation](https://arxiv.org/abs/2106.14855)

### Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/ZwwWayne/K-Net/">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.23.0/mmseg/models/decode_heads/knet_head.py#L392">Code Snippet</a>

### Abstract

<!-- [ABSTRACT] -->

Semantic, instance, and panoptic segmentations have been addressed using different and specialized frameworks despite their underlying connections. This paper presents a unified, simple, and effective framework for these essentially similar tasks. The framework, named K-Net, segments both instances and semantic categories consistently by a group of learnable kernels, where each kernel is responsible for generating a mask for either a potential instance or a stuff class. To remedy the difficulties of distinguishing various instances, we propose a kernel update strategy that enables each kernel dynamic and conditional on its meaningful group in the input image. K-Net can be trained in an end-to-end manner with bipartite matching, and its training and inference are naturally NMS-free and box-free. Without bells and whistles, K-Net surpasses all previous published state-of-the-art single-model results of panoptic segmentation on MS COCO test-dev split and semantic segmentation on ADE20K val split with 55.2% PQ and 54.3% mIoU, respectively. Its instance segmentation performance is also on par with Cascade Mask R-CNN on MS COCO with 60%-90% faster inference speeds. Code and models will be released at [this https URL](https://github.com/ZwwWayne/K-Net/).

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/157008300-9f40905c-b8e8-4a2a-9593-c1177fa35b2c.png" width="90%"/>
</div>

```bibtex
@inproceedings{zhang2021knet,
    title={{K-Net: Towards} Unified Image Segmentation},
    author={Wenwei Zhang and Jiangmiao Pang and Kai Chen and Chen Change Loy},
    year={2021},
    booktitle={NeurIPS},
}
```

### Results and models

#### 벼 생육이상 인식 데이터

| Method           | Backbone | Crop Size | Lr schd | Loss Funcion  | Mem (GB) | Inf time (fps) | mAcc  | config                                                                                                     |
| ---------------- | -------- | --------- | ------- | ------------- | -------- | -------------- | ----- | ---------------------------------------------------------------------------------------------------------- |
| KNet + DeepLabV3 | R-50-D8  | 512x512   | 00000   | Cross-entropy | 7.42     | 00.00          | 00.00 | [config](https://github.com/RiceSeg/mmsegmentation/configs/rice/knet_s3_deeplabv3_ce_gne_chw.py)           |
| KNet + DeepLabV3 | R-50-D8  | 512x512   | 00000   | Lovasz        | 7.42     | 00.00          | 00.00 | [config](https://github.com/RiceSeg/mmsegmentation/configs/rice/knet_s3_deeplabv3_lovasz_gne_chw.py)       |
| KNet + UPerNet   | Swin-T   | 512x512   | 00000   | Lovasz        | 7.57     | 00.00          | 00.00 | [config](https://github.com/RiceSeg/mmsegmentation/configs/rice/knet_s3_upernet_swin-l_lovasz_gne_chw.py)  |
| KNet + UPerNet   | Swin-L   | 512x512   | 00000   | Lovasz        | 13.5     | 00.00          | 00.00 | [config](https://github.com/RiceSeg/mmsegmentation/configs/rice/knet_s3_upernet_swin-t_lovasz_gne_chw.py)  |

## 3. Segmenter

[Segmenter: Transformer for Semantic Segmentation](https://arxiv.org/abs/2105.05633)

### Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/rstrudel/segmenter">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.21.0/mmseg/models/decode_heads/segmenter_mask_head.py#L15">Code Snippet</a>

### Abstract

<!-- [ABSTRACT] -->

Image segmentation is often ambiguous at the level of individual image patches and requires contextual information to reach label consensus. In this paper we introduce Segmenter, a transformer model for semantic segmentation. In contrast to convolution-based methods, our approach allows to model global context already at the first layer and throughout the network. We build on the recent Vision Transformer (ViT) and extend it to semantic segmentation. To do so, we rely on the output embeddings corresponding to image patches and obtain class labels from these embeddings with a point-wise linear decoder or a mask transformer decoder. We leverage models pre-trained for image classification and show that we can fine-tune them on moderate sized datasets available for semantic segmentation. The linear decoder allows to obtain excellent results already, but the performance can be further improved by a mask transformer generating class masks. We conduct an extensive ablation study to show the impact of the different parameters, in particular the performance is better for large models and small patch sizes. Segmenter attains excellent results for semantic segmentation. It outperforms the state of the art on both ADE20K and Pascal Context datasets and is competitive on Cityscapes.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/148507554-87eb80bd-02c7-4c31-b102-c6141e231ec8.png" width="70%"/>
</div>

```bibtex
@inproceedings{strudel2021segmenter,
  title={Segmenter: Transformer for semantic segmentation},
  author={Strudel, Robin and Garcia, Ricardo and Laptev, Ivan and Schmid, Cordelia},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7262--7272},
  year={2021}
}
```
### Results and models

#### 벼 생육이상 인식 데이터

| Method           | Backbone | Crop Size | Lr schd | Loss Funcion | Mem (GB) | Inf time (fps) | mAcc  | config                                                                                               |
| ---------------- | -------- | --------- | ------- | -------------| -------- | -------------- | ----- | ---------------------------------------------------------------------------------------------------- |
| Segmenter Mask   | ViT-B_16 | 512x512   | 000000  | Cross-entropy| 4.20     | 00.00          | 00.00 | [config](https://github.com/RiceSeg/mmsegmentation/configs/rice/segmenter_vit-b_ce_gne_chw.py)       |
| Segmenter Mask   | ViT-B_16 | 512x512   | 000000  | Lovasz       | 4.20     | 00.00          | 00.00 | [config](https://github.com/RiceSeg/mmsegmentation/configs/rice/segmenter_vit-b_lovasz_gne_chw.py)   |

## 라이센스 

MMSegmentation is released under the Apache 2.0 license, while some specific features in this library are with other licenses.

#### Licenses for special features

In this file, we list the features with other licenses instead of Apache 2.0. Users should be careful about adopting these features in any commercial matters.

|  Feature  |                                                                        Files                                                                        |                            License                            |
| :-------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------: |
| SegFormer | [mmseg/models/decode_heads/segformer_head.py](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/segformer_head.py) | [NVIDIA License](https://github.com/NVlabs/SegFormer#license) |
