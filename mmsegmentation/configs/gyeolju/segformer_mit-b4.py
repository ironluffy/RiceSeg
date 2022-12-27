_base_ = [
    '../_base_/models/segformer_mit-b4.py', '../_base_/datasets/rice_gne_chw.py',
    '../_base_/rice_runtime.py',
]

data = dict(test=dict(classes=('background', 'normal','gyulju')))
