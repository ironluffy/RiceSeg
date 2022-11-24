_base_ = [
    '../_base_/models/knet_s3_deeplabv3.py',
    '../_base_/datasets/rice_gne_chw.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_3k_knet.py'
]
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