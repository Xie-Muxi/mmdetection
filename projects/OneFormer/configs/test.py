
# runtime settings
max_epochs = 300
batch_size = 8
start_lr = 0.01
val_interval = 5

custom_imports = dict(imports=['mmpretrain.models','projects.OneFormer.oneformer'], allow_failed_imports=False)
# ------ model settings ------



batch_augments= None

image_size = (1024, 1024)

data_preprocessor = dict(
    type='mmdet.DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
)

num_things_classes = 10
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
num_queries = 60

checkpoint = 'https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-small-p14_dinov2-pre_3rdparty_20230426-5641ca5a.pth'  # noqa
# from projects.OneFormer_dev.oneformer import OneFormer
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255,
    batch_augments=batch_augments)

model = dict(
    type='OneFormer',
    data_preprocessor=data_preprocessor,
    #! mmpretrain.models.backbones.vision_transformer
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='dinov2-small',
        out_indices=[0, 1, 2, 3],
        out_type='featmap',
        init_cfg=dict(
            type='Pretrained', 
            checkpoint=checkpoint,
            prefix='backbone.')),

    
    panoptic_head=dict(
        type="OneFormerHead",
        # in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        in_channels=[384]*4,
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=150,
        task="instance",
        max_seq_len=77,
        task_seq_len=77,
        task_mlp=dict(input_dim=77, hidden_dim=256, output_dim=256, num_layers=2),
        text_encoder=dict(context_length=77, width=256, layers=6, vocab_size=49408),
        text_projector=dict(
            input_dim=256, hidden_dim=256, output_dim=256, num_layers=2
        ),
        prompt_ctx=dict(num_embeddings=16, embedding_dim=256),
        contrastive_multiScale_masked_transformer_decoder=dict(
            use_task_norm=True,
            class_transformer=dict(
                d_model=256,
                nhead=8,
                num_encoder_layers=0,
                num_decoder_layers=2,
                dim_feedforward=2048,
                dropout=0.1,
                normalize_before=False,
                return_intermediate_dec=False,
            ),
        ),
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type="MSDeformAttnPixelDecoder",
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.1,
                        batch_first=True,
                    ),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                ),
            ),
            positional_encoding=dict(num_feats=128, normalize=True),
        ),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256, num_heads=8, dropout=0.0, batch_first=True
                ),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256, num_heads=8, dropout=0.0, batch_first=True
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type="ReLU", inplace=True),
                ),
            ),
            init_cfg=None,
        ),
        use_task_norm=True,
        # loss_cls=dict(
        #     type="CrossEntropyLoss",
        #     use_sigmoid=False,
        #     loss_weight=2.0,
        #     reduction="mean",
        #     class_weight=[1.0] * num_classes + [0.1],
        # ),
        # loss_mask=dict(
        #     type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=5.0
        # ),
        # loss_dice=dict(
        #     type="DiceLoss",
        #     use_sigmoid=True,
        #     activate=True,
        #     reduction="mean",
        #     naive_dice=True,
        #     eps=1.0,
        #     loss_weight=5.0,
        # ),
        # loss_contrastive=dict(
        #     type='ContrastiveLoss',
        #     loss_weight=0.5,
        #     contrast_temperature=0.07)
    ),
    panoptic_fusion_head=dict(
        type='OneFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=150,
        iou_thr=0.8,

        filter_low_score=True),
    init_cfg=None)

# ----- coco_instance.py -----
# dataset settings
# data_root = '/nfs/home/3002_hehui/xmx/COCO2017/'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

from mmdet.datasets import NWPUInsSegDataset
dataset_type = NWPUInsSegDataset
data_root = '/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset'

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/nfs/home/3002_hehui/xmx/RS-SA/RSPrompter/data/NWPU/annotations/NWPU_instances_train.json',
        data_prefix=dict(img='positive image set'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset/nwpu-instances_val.json',
        data_prefix=dict(img='positive image set'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
)

test_dataloader = val_dataloader
from mmdet.evaluation.metrics import CocoMetric
val_evaluator = dict(
    type=CocoMetric,
    # ann_file=data_root + '/annotations/instances_val2017.json',
    ann_file='/nfs/home/3002_hehui/xmx/data/NWPU/NWPU VHR-10 dataset/nwpu-instances_val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# ----- schedule_1x.py -----
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=start_lr, by_epoch=False, begin=0, end=1),
    # dict(
    #     type='MultiStepLR',
    #     begin=0,
    #     end=max_epochs,
    #     by_epoch=True,
    #     milestones=[8, 11],
    #     gamma=0.1)
    # Cosine Anneal
    dict(
        type='CosineAnnealingLR', 
        by_epoch=True, 
        begin=1,
        T_max=max_epochs,
        end=max_epochs,
    )
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
    # optimizer=dict(type='Adam', lr=0.005, weight_decay=0.0001)
    optimizer=dict(
        type='Adam', 
        lr=start_lr, 
        weight_decay=1e-4,
    )
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)



# ----- default_runtime -----
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # checkpoint=dict(type='CheckpointHook', interval=4,max_keep_ckpts=3),
    checkpoint=dict(type='CheckpointHook',
                    interval=val_interval, 
                    max_keep_ckpts=3,
                    save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# vis_backends = [dict(type='LocalVisBackend')]
vis_backends = [dict(type='LocalVisBackend'), 
                dict(type='WandbVisBackend',
                     init_kwargs=dict(
                         project='dev',
                         name=\
    f'oneformer_dinov2-small_lr={start_lr}_nwpu_{max_epochs}e',
                         group='oneformer',
                         tags=['oneformer', 'dinov2', 'nwpu'],
                        #  resume=True
        )
    )
]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
# # load_from = '/nfs/home/3002_hehui/xmx/PureSeg/work_dirs/mask2former_dinov2_1x-wandb_nwpu/last_checkpoint'  # 从给定路径加载模型检查点作为预训练模型。这不会恢复训练。
# load_from = '/nfs/home/3002_hehui/xmx/PureSeg/work_dirs/mask2former_dinov2_nwpu_cosineannealinglr/best_coco_bbox_mAP_epoch_95.pth'
# resume = True  # 是否从 `load_from` 中定义的检查点恢复。 如果 `load_from` 为 None，它将恢复 `work_dir` 中的最新检查点。


