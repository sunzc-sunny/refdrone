_base_ = [
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py',
    '../_base_/datasets/visdrone_2019.py',
    '../_base_/models/faster-rcnn_r50_fpn.py'
]
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))

find_unused_parameters = True