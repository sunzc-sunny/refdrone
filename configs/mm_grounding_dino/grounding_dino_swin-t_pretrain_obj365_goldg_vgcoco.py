_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'


o365v1_od_dataset = dict(
    type='ODVGDataset',
    data_root='data/objects365v1/',
    ann_file='o365v1_train_od.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None,
)

flickr30k_dataset = dict(
    type='ODVGDataset',
    data_root='data/flickr30k_entities/',
    ann_file='final_flickr_separateGT_train_vg.json',
    label_map_file=None,
    data_prefix=dict(img='flickr30k_images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None)

# --------------------------- coco2017 od dataset---------------------------


# --------------------------- coco2014 vg dataset---------------------------
coco2014_vg_dataset = dict(
    type='ODVGDataset',
    data_root='data/coco/',
    ann_file='mdetr_annotations/final_mixed_train_only_coco_vg.json',
    label_map_file=None,
    data_prefix=dict(img='train2014/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None)

# --------------------------- refcoco vg dataset---------------------------
refcoco_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='mdetr_annotations/finetune_refcoco_train_vg.json',
        label_map_file=None,
        data_prefix=dict(img='train2014'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- refcoco+ vg dataset---------------------------
refcoco_plus_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='mdetr_annotations/finetune_refcoco+_train_vg.json',
        label_map_file=None,
        data_prefix=dict(img='train2014'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- refcocog vg dataset---------------------------
refcocog_dataset = dict(
    type='RepeatDataset',
    times=3,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='mdetr_annotations/finetune_refcocog_train_vg.json',
        label_map_file=None,
        data_prefix=dict(img='train2014'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- grefcoco vg dataset---------------------------
grefcoco_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='ODVGDataset',
        data_root='data/coco/',
        ann_file='mdetr_annotations/finetune_grefcoco_train_vg.json',
        label_map_file=None,
        data_prefix=dict(img='train2014'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=_base_.train_pipeline,
        return_classes=True,
        backend_args=None))

# --------------------------- dataloader---------------------------
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(
        _delete_=True,
        type='CustomSampleSizeSampler',
        ratio_mode=True,
        dataset_size=[-1, -1, -1, -1, -1, -1, -1]),
    dataset=dict(datasets=[
        o365v1_od_dataset,  # 1.74M
        flickr30k_dataset,  # 0.15M
        coco2014_vg_dataset,  # 0.49M
        refcoco_dataset,  # 0.12M
        refcoco_plus_dataset,  # 0.12M
        refcocog_dataset,  # 0.08M
        grefcoco_dataset,  # 0.19M
    ]))

optim_wrapper = dict(optimizer=dict(lr=0.0001))

# learning policy
max_iter = 304680
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=10000)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[228510],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=20))
log_processor = dict(by_epoch=False)
