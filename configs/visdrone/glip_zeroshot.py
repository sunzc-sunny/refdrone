_base_ = '../glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py'

lang_model_name = '/mnt/public/usr/sunzhichao/hf_hub/models--google-bert--bert-base-uncased'

model = dict(bbox_head=dict(early_fuse=True))

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
    type='RefDroneFlickr30kDataset',
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
