_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/rice_gne_chw.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_5k_segmenter.py'
]

optimizer = dict(lr=0.001, weight_decay=0.0)

data = dict(samples_per_gpu=2, workers_per_gpu=2)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)