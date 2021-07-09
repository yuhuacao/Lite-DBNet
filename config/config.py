import os


file_dir = os.path.dirname(os.path.abspath(__file__))

Global = dict(
    use_gpu=True,
    epoch_num=300,
    log_smooth_window=20,
    print_batch_step=25,
    save_model_dir='./output/ch_db_mv3/',
    save_epoch_step=10,
    # evaluation is run every 5000 iterations after the 4000th iteration
    eval_batch_epoch=[1, 1],
    pretrained_model='',
    checkpoints=''
)

Architecture = dict(
    backbone=dict(
        name='mobilenet_v3_large',
        pretrained=True
        # scale=0.5,
        # model_name='large'
    ),
    neck=dict(
        name='DBFPN',
        out_channels=96
    ),
    head=dict(
        name='DBHead',
        k=50
    )
)

Loss = dict(
    name='DBLoss',
    balance_loss=True,
    main_loss_type='DiceLoss',
    alpha=5,
    beta=10,
    ohem_ratio=3
)

Optimizer = dict(
    name='Adam',
    beta=(0.9, 0.999),
    learning_rate=0.001,
    lr=dict(
        name='Cosine',
        warmup_epoch=5
    ),
    weight_decay=0
)

DBPostProcess = dict(
    name='DBPostProcess',
    thresh=0.3,
    box_thresh=0.6,
    max_candidates=1000,
    unclip_ratio=1.5
)

Metric = dict(
    name='DetMetric',
    main_indicator='hmean'
)

Train = dict(
    dataset=dict(
        name='ICDAR2015DataSet',
        data_dir=os.path.join(file_dir, '../dataset/'),
        label_file=os.path.join(file_dir, '../dataset/test_labels.txt'),
        transforms=[
            {'DecodeImage': dict(img_mode='BGR')},
            {'LabelEncode': None},  # Class handling label
            {"IaaAugment": dict(
                augmenter_args=[
                    {'type': 'Fliplr', 'args': {'p': 0.5}},
                    {'type': 'Affine', 'args': {'rotate': [-10, 10]}},
                    {'type': 'Resize', 'args': {'size': [0.5, 3]}}
                ]
            )},
            {'EastRandomCropData': dict(
                size=[960, 960],
                max_tries=50,
                keep_ratio=True
            )},
            {'MakeBorderMap': dict(
                shrink_ratio=0.4,
                thresh_min=0.3,
                thresh_max=0.7
            )},
            {'MakeShrinkMap': dict(
                shrink_ratio=0.4,
                min_text_size=8
            )},
            {'NormalizeImage': dict(
                scale=1./255.,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                order='hwc'
            )},
            {'ToCHWImage': None},
            {'KeepKeys': dict(
                keep_keys=['image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask']
            )}
        ]
    ),
    dataloader=dict(
        shuffle=True,
        drop_last=False,
        batch_size=4,
        num_workers=4
    )
)

Test = dict(
    dataset=dict(
        name='ICDAR2015DataSet',
        data_dir=os.path.join(file_dir, '../dataset/'),
        label_file=os.path.join(file_dir, '../dataset/test.txt'),
        transforms=[
            {'DecodeImage': dict(img_mode='BGR')},
            {'LabelEncode': None},
            {'DetResizeForTest': dict(image_shape=[736, 1280])},
            {'NormalizeImage': dict(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    order='hwc')},
            {'ToCHWImage': None},
            {'KeepKeys': dict(keep_keys=['image', 'shape', 'polys', 'ignore_tags'])}
        ]
    ),
    dataloader=dict(
        shuffle=False,
        drop_last=False,
        batch_size=1,
        num_workers=1
    )
)
