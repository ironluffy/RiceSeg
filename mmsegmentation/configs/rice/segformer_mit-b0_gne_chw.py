# /JH/jihye/mmsegmentation/configs/a_my_configs/segformer_mit-b0_512x512_3k_Rice_Sample_lovasz.py

_base_ = [
    '../_base_/models/segformer_mit-b0-_lovasz.py', '../_base_/datasets/rice_gne_chw.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_5k_segformer.py'
]

#checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'
checkpoint = 'pretrained_ckpt/segformer_mit-b0_512x512_160k_ade20k_20220617_162207-c00b9603.pth'# noqa

model = dict(pretrained=checkpoint, decode_head=dict(num_classes=4))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=2, workers_per_gpu=2)