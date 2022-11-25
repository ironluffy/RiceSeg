_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/rice_gne_chw.py',
    '../_base_/rice_runtime.py', '../_base_/schedules/schedule_segformer.py'
]
model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='LovaszLoss', loss_weight=1.0)
    )
)

batch_mul=4
data = dict(samples_per_gpu=2*batch_mul, workers_per_gpu=2*batch_mul)
optimizer = dict(lr=0.00006*batch_mul)