# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='policy', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(by_epoch=True, interval=1)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True, by_epoch=True)
