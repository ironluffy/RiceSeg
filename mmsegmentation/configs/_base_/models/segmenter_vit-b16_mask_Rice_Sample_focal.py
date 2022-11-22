# /JH/jihye/mmsegmentation/configs/_base_/models/segmenter_vit-b16_mask_Rice_Sample_focal.py


checkpoint = '/JH/jihye/checkpoints/vit_base_p16_384_20220308-96dfe169.pth'
# checkpoint = '/JH/jihye/checkpoints/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth'
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth'  # noqa

# model settings
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=checkpoint,
    backbone=dict(
        type='VisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=True,
        interpolate_mode='bicubic',
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=768,
        channels=768,
        num_classes=4,
        num_layers=2,
        num_heads=12,
        embed_dims=768,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='FocalLoss', use_sigmoid=True, loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(480, 480)),
)
