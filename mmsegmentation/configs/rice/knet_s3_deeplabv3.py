_base_ = [
    '../_base_/datasets/rice.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_3k_knet.py'
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
num_stages = 3
conv_kernel_size = 1
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='IterativeDecodeHead',
        num_stages=num_stages,
        kernel_update_head=[
            dict(
                type='KernelUpdateHead',
                num_classes=6,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=512,
                out_channels=512,
                dropout=0.0,
                conv_kernel_size=conv_kernel_size,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN'))) for _ in range(num_stages)
        ],
        kernel_generate_head=dict(
            type='ASPPHead',
            in_channels=2048,
            in_index=3,
            channels=512,
            dilations=(1, 12, 24, 36),
            dropout_ratio=0.1,
            ignore_index = -100, 
            num_classes=6,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='LovaszLoss', loss_weight=1.0))),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        ignore_index = -100,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='LovaszLoss', loss_weight=0.4)),

    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[60000, 72000],
    by_epoch=False)
# In K-Net implementation we use batch size 2 per GPU as default
data = dict(samples_per_gpu=4, workers_per_gpu=4)