from backbones.mobilenet import MobileNetV1
from detectors.scrfd import SCRFD

config = dict(
    type='SCRFD',
    backbone=dict(
        type='MobileNetV1',
        block_cfg=dict(
            stage_blocks=(2, 3, 2, 6), stage_planes=[16, 16, 40, 72, 152,
                                                     288])),
    neck=dict(
        type='PAFPN',
        in_channels=[40, 72, 152, 288],
        out_channels=16,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=3),
    bbox_head=dict(
        type='SCRFDHead',
        num_classes=1,
        in_channels=16,
        stacked_convs=2,
        feat_channels=64,
        norm_cfg=dict(type='BN', requires_grad=True),
        # norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
        cls_reg_share=True,
        strides_share=False,
        dw_conv=True,
        scale_mode=0,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[1, 2],
            base_sizes=[16, 64, 256],
            strides=[8, 16, 32]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=False,
        reg_max=8,
        loss_bbox=dict(type='DIoULoss', loss_weight=2.0),
        use_kps=True,
        loss_kps=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=0.1),
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=-1,
            min_bbox_size=0,
            score_thr=0.02,
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=-1)))

backbone = MobileNetV1(block_cfg=config['backbone'])
neck = 
scrfd_detector = SCRFD(backbone=backbone,
                       neck=)
