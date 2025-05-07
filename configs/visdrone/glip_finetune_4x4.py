_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


lang_model_name = '/mnt/public/usr/sunzhichao/hf_hub/models--google-bert--bert-base-uncased'
load_from = "/mnt/public/usr/sunzhichao/mmdetection/glip_tiny_a_mmdet-b3654169.pth"

model = dict(
    type='GLIP',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=False),
    neck=dict(
        type='FPN_DropBlock',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        relu_before_extra_convs=True,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSVLFusionHead',
        lang_model_name=lang_model_name,
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoderForGLIP',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    language_model=dict(type='BertModel', name=lang_model_name),
    train_cfg=dict(
        assigner=dict(
            type='ATSSAssigner',
            topk=9,
            iou_calculator=dict(type='BboxOverlaps2D_GLIP')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))



# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        imdecode_backend='pillow',
        backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='GTBoxSubOne_GLIP'),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 480), (1333, 560), (1333, 640), (1333, 720),
                (1333, 800)],
        keep_ratio=True,
        resize_type='FixScaleResize',
        backend='pillow'),
    dict(type='RandomFlip_GLIP', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='LLavaVGDataset',
        ann_file = '/mnt/public/usr/sunzhichao/RefDrone/finetune_RefDrone_train6_vg.json',
        data_prefix=dict(img='/mnt/public/usr/sunzhichao/VisDrone2019/all_image/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        return_classes=True,
        pipeline=train_pipeline))

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))

]


dataset_refdrone_test = dict(
    type='MDETRStyleRefCocoDataset',
    ann_file = '/mnt/public/usr/sunzhichao/RefDrone/finetune_RefDrone_test6.json',
    data_prefix=dict(img='/mnt/public/usr/sunzhichao/VisDrone2019/all_image/'),
    pipeline=test_pipeline,
)

val_evaluator_refdrone = dict(
    type='RefDroneMetric',
    ann_file = '/mnt/public/usr/sunzhichao/RefDrone/finetune_RefDrone_test6.json',
    metric='bbox',
    iou_thrs=0.5,
    thresh_score=0.7,
    thresh_f1=1.0)


# ----------Config---------- #
dataset_prefixes = ['refcocog']
datasets = [dataset_refdrone_test]
metrics = [val_evaluator_refdrone]

val_dataloader = dict(
    batch_size=4,
    num_workers=16,
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets)
    )
test_dataloader = val_dataloader


val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator




optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    clip_grad=None)

max_epochs = 5

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3],
        gamma=0.1)
]