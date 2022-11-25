_base_ = ['./rice_runtime.py']

label_map={2:0, 3:0, 5:0}
evaluation = dict(label_map=label_map)
