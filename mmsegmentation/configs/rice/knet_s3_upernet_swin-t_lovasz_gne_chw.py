_base_ = [
    '../_base_/models/knet_s3_upernet_swin-t.py',
    '../_base_/datasets/rice_gne_chw.py', '../_base_/rice_runtime.py',
    '../_base_/schedules/schedule_knet_upernet.py'
]

model = dict(
    decode_head=dict(
        kernel_generate_head=dict(loss_decode=dict(
            type='LovaszLoss', loss_weight=1.0))
    ),
    auxiliary_head=dict(loss_decode=dict(
        type='LovaszLoss', loss_weight=1.0)))



batch_mul=2
data = dict(samples_per_gpu=2*batch_mul, workers_per_gpu=2*batch_mul)
optimizer = dict(lr=0.00006*batch_mul)