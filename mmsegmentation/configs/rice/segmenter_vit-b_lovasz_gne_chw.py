_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/rice_gne_chw.py', '../_base_/rice_runtime.py',
    '../_base_/schedules/schedule_segmenter.py'
]
model=dict(
    decode_head=dict(
        loss_decode=dict(
            type='LovaszLoss', loss_weight=1.0)
    )
)

data = dict(samples_per_gpu=2, workers_per_gpu=2)