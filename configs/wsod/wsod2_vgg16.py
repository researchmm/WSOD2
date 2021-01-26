_base_ = './base.py'
# model settings
model = dict(
    type='WeakRCNN',
    pretrained=None,
    backbone=dict(type='VGG16'),
    neck=None,
    roi_head=dict(
        type='WSOD2RoIHead',
        steps=40000,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIPool', output_size=7),
            out_channels=512,
            featmap_strides=[8]),
        bbox_head=dict(
            type='OICRHead',
            in_channels=512,
            hidden_channels=4096,
            roi_feat_size=7,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            num_classes=20))
)
# dataset settings
dataset_type = 'VOCSSDataset'
data_root = '/datavoc/VOCdevkit/'
img_norm_cfg = dict(
    mean=[104., 117., 124.], std=[1., 1., 1.], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSuperPixelFromFile'),
    dict(type='LoadWeakAnnotations'),
    dict(type='LoadProposals'),
    dict(type='Resize', img_scale=[(488, 2000), (576, 2000), (688, 2000), (864, 2000), (1200, 2000)], keep_ratio=True, multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_labels', 'proposals', 'ss']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(688, 2000),
        #img_scale=[(500, 2000), (600, 2000), (700, 2000), (800, 2000), (900, 2000)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/trainval.txt',
        img_prefix=data_root + 'VOC2007/',
        proposal_file='/datavoc/selective_search_data/voc_2007_trainval.pkl',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        proposal_file='/datavoc/selective_search_data/voc_2007_test.pkl',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        proposal_file='/datavoc/selective_search_data/voc_2007_test.pkl',
        pipeline=test_pipeline))

work_dir = 'work_dirs/wsod2_vgg16/'
