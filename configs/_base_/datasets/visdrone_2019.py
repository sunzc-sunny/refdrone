# dataset settings
dataset_type = 'CocoDataset'
data_root = '/mnt/public/usr/sunzhichao/VisDrone2019/'
classes = (
 "pedestrian",
 "people",
 "bicycle",
 "car",
 "van",
 "truck",
 "tricycle",
 "awning-tricycle",
 "bus",
 "motor",
)
backend_args = None

metainfo = dict(classes=classes)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

evaluation = dict(interval=1, metric='bbox')


backend_args = None


train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_annotations/train.json',
        data_prefix=dict(img='all_image/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))


test_dataloader =  dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        # ann_file='test_annotations/test.json',
        ann_file='train_annotations/train.json',
        data_prefix=dict(img='all_image/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))


test_evaluator = dict(
    type='CocoMetric',
    # ann_file=data_root + 'test_annotations/test.json',
    ann_file=data_root + 'train_annotations/train.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

val_dataloader = test_dataloader
val_evaluator = test_evaluator

# test_dataloader = val_dataloader    
# test_evaluator = val_evaluator

