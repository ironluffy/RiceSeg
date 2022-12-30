# 모델 정보 및 라이센스 가이드

## 요약

<!-- [ABSTRACT] -->


We present SegFormer, a simple, efficient yet powerful semantic segmentation framework which unifies Transformers with lightweight multilayer perception (MLP) decoders. SegFormer has two appealing features: 1) SegFormer comprises a novel hierarchically structured Transformer encoder which outputs multiscale features. It does not need positional encoding, thereby avoiding the interpolation of positional codes which leads to decreased performance when the testing resolution differs from training. 2) SegFormer avoids complex decoders. The proposed MLP decoder aggregates information from different layers, and thus combining both local attention and global attention to render powerful representations. We show that this simple and lightweight design is the key to efficient segmentation on Transformers. We scale our approach up to obtain a series of models from SegFormer-B0 to SegFormer-B5, reaching significantly better performance and efficiency than previous counterparts. For example, SegFormer-B4 achieves 50.3% mIoU on ADE20K with 64M parameters, being 5x smaller and 2.2% better than the previous best method. Our best model, SegFormer-B5, achieves 84.0% mIoU on Cityscapes validation set and shows excellent zero-shot robustness on Cityscapes-C. Code will be released at: [this http URL](https://github.com/NVlabs/SegFormer).













## SegFormer

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

### Citation

```bibtex
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.15203},
  year={2021}
}
```

### Usage

We have provided pretrained models converted from [SegFormer](https://github.com/NVlabs/SegFormer).

If you want to convert keys on your own, we also provide a script [`mit2mmseg.py`](../../tools/model_converters/mit2mmseg.py) in the tools directory to convert the key of models from [the official repo](https://github.com/NVlabs/SegFormer) to MMSegmentation style.

```shell
python tools/model_converters/mit2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

### Results and models

#### 벼 생육이상 인식 데이터

| Method    | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | config                                                                                              | download                                                                                                                                                                                                                                                                                                                                               |
| --------- | -------- | --------- | ------: | -------: | -------------- | ----: | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Segformer | MIT-B0   | 512x512   |  160000 |      2.1 | 38.17          | 37.85 | [config](https://github.com/RiceSeg/mmsegmentation/configs/rice/segformer_mit-b0_lovasz_gne_chw.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20220617_162207-c00b9603.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20220617_162207.log.json) |
| Segformer | MIT-B4   | 512x512   |  160000 |      6.1 | 14.54          | 49.09 | [config](https://github.com/RiceSeg/mmsegmentation/configs/rice/segformer_mit-b4_lovasz_gne_chw.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_512x512_160k_ade20k/segformer_mit-b4_512x512_160k_ade20k_20220620_112216-4fa4f58f.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_512x512_160k_ade20k/segformer_mit-b4_512x512_160k_ade20k_20220620_112216.log.json) |

Evaluation with `AlignedResize`:

| Method    | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) |
| --------- | -------- | --------- | ------: | ----: | ------------- |
| Segformer | MIT-B0   | 512x512   |  160000 | 38.55 | 39.03         |
| Segformer | MIT-B4   | 512x512   |  160000 | 50.23 | 51.10         |



## K-Net

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

| Method           | Backbone | Crop Size | Loss Funcion  | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) | config                                                                                                                                   | download                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------- | -------- | --------- | ------------- | ------- | -------- | -------------- | ----- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| KNet + DeepLabV3 | R-50-D8  | 512x512   | Cross-entropy | 80000   | 7.42     | 12.10          | 45.06 | 46.11         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/knet/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_041642-00c8fbeb.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_041642.log.json) |
| KNet + DeepLabV3 | R-50-D8  | 512x512   | Lovasz        | 80000   | 7.42     | 12.10          | 45.06 | 46.11         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/knet/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_041642-00c8fbeb.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_041642.log.json) |
| KNet + UPerNet   | Swin-T   | 512x512   | Lovasz        | 80000   | 7.57     | 15.56          | 45.84 | 46.27         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/knet/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_ade20k.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_ade20k/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_ade20k_20220303_133059-7545e1dc.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_ade20k/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_ade20k_20220303_133059.log.json)         |
| KNet + UPerNet   | Swin-L   | 512x512   | Lovasz        | 80000   | 13.5     | 8.29           | 52.05 | 53.24         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/knet/knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k/knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k_20220303_154559-d8da9a90.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k/knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k_20220303_154559.log.json)         |


## Segmenter

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

### Usage

We have provided pretrained models converted from [ViT-AugReg](https://github.com/rwightman/pytorch-image-models/blob/f55c22bebf9d8afc449d317a723231ef72e0d662/timm/models/vision_transformer.py#L54-L106).

If you want to convert keys on your own to use the pre-trained ViT model from [Segmenter](https://github.com/rstrudel/segmenter), we also provide a script [`vitjax2mmseg.py`](../../tools/model_converters/vitjax2mmseg.py) in the tools directory to convert the key of models from [ViT-AugReg](https://github.com/rwightman/pytorch-image-models/blob/f55c22bebf9d8afc449d317a723231ef72e0d662/timm/models/vision_transformer.py#L54-L106) to MMSegmentation style.

```shell
python tools/model_converters/vitjax2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

E.g.

```shell
python tools/model_converters/vitjax2mmseg.py \
Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz \
pretrain/vit_tiny_p16_384.pth
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

In our default setting, pretrained models and their corresponding [ViT-AugReg](https://github.com/rwightman/pytorch-image-models/blob/f55c22bebf9d8afc449d317a723231ef72e0d662/timm/models/vision_transformer.py#L54-L106) models could be defined below:

| pretrained models     | original models                                                                                                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| vit_tiny_p16_384.pth  | ['vit_tiny_patch16_384'](https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz)   |
| vit_small_p16_384.pth | ['vit_small_patch16_384'](https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz) |
| vit_base_p16_384.pth  | ['vit_base_patch16_384'](https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz)  |
| vit_large_p16_384.pth | ['vit_large_patch16_384'](https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300epodels

#### 벼 생육이상 인식 데이터
           | Backbone | ss Funcion  | Mem (GB) | Inf time (fps) | mIoU  |                                                                                                  | download                                                                                                                                                                                                                                                                                                                                                                                       |
60000  | Cross-entropy | 4.20     | 13.20          | 49.60 | Cross-entropy | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706.log.json)         |
| Segmenter Mask   | ViT-  | 13.20          | 49.60 | Lovasz        | [config](https://github.com/open-mmlabntation/blob/master/configs/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706.log.json)         |
