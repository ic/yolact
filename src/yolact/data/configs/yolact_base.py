from yolact.data.configs import coco_base
from yolact.data.configs.backbones import resnet101_backbone
from yolact.data.configs.datasets import coco2017_dataset
from yolact.data.configs.helpers import mask_types, fpn_base

yolact_base_backbone = resnet101_backbone.copy()
yolact_base_backbone.update({
    'selected_layers': list(range(1, 4)),
    'use_pixel_scales': True,
    'preapply_sqrt': False,
    'use_square_anchors': True, # This is for backward compatability with a bug

    'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
    'pred_scales': [[24], [48], [96], [192], [384]],
})

yolact_base_fpn = fpn_base.copy()
yolact_base_fpn.update({
    'use_conv_downsample': True,
    'num_downsample': 2,
})

config = coco_base.config.copy()
config.update({
    'name': 'yolact_base',

    # Dataset stuff
    'dataset': coco2017_dataset,
    'num_classes': len(coco2017_dataset.class_names) + 1,

    # Image Size
    'max_size': 550,

    # Training params
    'lr_steps': (280000, 600000, 700000, 750000),
    'max_iter': 800000,

    # Backbone Settings
    'backbone': yolact_base_backbone,

    # FPN Settings
    'fpn': yolact_base_fpn,

    # Mask Settings
    'mask_type': mask_types['lincomb'],
    'mask_alpha': 6.125,
    'mask_proto_src': 0,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})],
    'mask_proto_normalize_emulate_roi_pooling': True,

    # Other stuff
    'share_prediction_module': True,
    'extra_head_net': [(256, 3, {'padding': 1})],

    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.4,

    'crowd_iou_threshold': 0.7,

    'use_semantic_segmentation_loss': True,
})
