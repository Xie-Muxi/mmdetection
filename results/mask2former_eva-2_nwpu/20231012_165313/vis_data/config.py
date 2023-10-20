backend_args = None
batch_size = 8
checkpoint = 'https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-tiny-p14_pre_in21k_20230505-d703e7b1.pth'
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmpretrain.models',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mask_pad_value=0,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_mask=True,
    pad_size_divisor=32,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='mmdet.DetDataPreprocessor')
data_root = '/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset'
dataset_type = 'NWPUInsSegDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=5, max_keep_ckpts=3, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
image_size = (
    1024,
    1024,
)
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 300
model = dict(
    backbone=dict(
        arch='tiny',
        final_norm=False,
        img_size=(
            1024,
            1024,
        ),
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-tiny-p14_pre_in21k_20230505-d703e7b1.pth',
            prefix='backbone.',
            type='Pretrained'),
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        out_type='featmap',
        patch_size=14,
        type='mmpretrain.ViTEVA02'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mask_pad_value=0,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmdet.DetDataPreprocessor'),
    init_cfg=None,
    panoptic_fusion_head=dict(
        init_cfg=None,
        loss_panoptic=None,
        num_stuff_classes=0,
        num_things_classes=10,
        type='MaskFormerFusionHead'),
    panoptic_head=dict(
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            192,
            192,
            192,
            192,
        ],
        loss_cls=dict(
            class_weight=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.1,
            ],
            loss_weight=2.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction='mean',
            type='DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        num_queries=60,
        num_stuff_classes=0,
        num_things_classes=10,
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=256,
                        feedforward_channels=1024,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4)),
                num_layers=6),
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128),
            type='MSDeformAttnPixelDecoder'),
        positional_encoding=dict(normalize=True, num_feats=128),
        strides=[
            4,
            8,
            16,
            32,
        ],
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8)),
            num_layers=9,
            return_intermediate=True),
        type='Mask2FormerHead'),
    test_cfg=dict(
        filter_low_score=True,
        instance_on=True,
        iou_thr=0.8,
        max_per_image=100,
        panoptic_on=True,
        semantic_on=False),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', use_sigmoid=True, weight=5.0),
                dict(eps=1.0, pred_act=True, type='DiceCost', weight=5.0),
            ],
            type='HungarianAssigner'),
        importance_sample_ratio=0.75,
        num_points=12544,
        oversample_ratio=3.0,
        sampler=dict(type='MaskPseudoSampler')),
    type='Mask2Former')
num_classes = 10
num_queries = 60
num_stuff_classes = 0
num_things_classes = 10
optim_wrapper = dict(
    optimizer=dict(lr=0.01, type='Adam', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1, start_factor=0.01, type='LinearLR'),
    dict(T_max=300, begin=1, by_epoch=True, end=300, type='CosineAnnealingLR'),
]
start_lr = 0.01
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    dataset=dict(
        ann_file=
        '/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset/nwpu-instances_val.json',
        backend_args=None,
        data_prefix=dict(img='positive image set'),
        data_root='/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        ],
        test_mode=True,
        type='NWPUInsSegDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset/nwpu-instances_val.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=300, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file=
        '/nfs/home/3002_hehui/xmx/RS-SA/RSPrompter/data/NWPU/annotations/NWPU_instances_train.json',
        backend_args=None,
        data_prefix=dict(img='positive image set'),
        data_root='/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='NWPUInsSegDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=
        '/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset/nwpu-instances_val.json',
        backend_args=None,
        data_prefix=dict(img='positive image set'),
        data_root='/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='NWPUInsSegDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset/nwpu-instances_val.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
val_interval = 5
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        init_kwargs=dict(
            group='mask2former',
            name='mask2former_eva-2-tiny_lr=0.01_nwpu_300e',
            project='pure-seg',
            resume=True),
        type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            init_kwargs=dict(
                group='mask2former',
                name='mask2former_eva-2-tiny_lr=0.01_nwpu_300e',
                project='pure-seg',
                resume=True),
            type='WandbVisBackend'),
    ])
work_dir = 'results/mask2former_eva-2_nwpu'
