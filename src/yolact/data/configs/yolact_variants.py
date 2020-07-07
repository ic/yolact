from yolact.data.configs.yolact_base import config as base
from yolact.data.configs.backbones import darknet53_backbone, resnet50_backbone.copy
from yolact.data.configs.datasets import pascal_sbd_dataset

yolact_im400_backbone = base['backbone'].copy()
yolact_im400_backbone.update({
    'pred_scales': [[int(x[0] / base['max_size'] * 400)] for x in base['backbone']['pred_scales']],
})

yolact_im400_config = base.copy()
yolact_im400_config.update({
    'name': 'yolact_im400',
    'max_size': 400,
    'backbone': yolact_im400_backbone,
})


yolact_im700_backbone = base['backbone'].copy()
yolact_im700_backbone.update({
    'pred_scales': [[int(x[0] / base['max_size'] * 700)] for x in base['backbone']['pred_scales']],
})

yolact_im700_config = base.copy({
    'name': 'yolact_im700',
    'masks_to_train': 300,
    'max_size': 700,
    'backbone': yolact_im700_backbone,
})


yolact_darknet53_backbone = darknet53_backbone.copy()
yolact_darknet53_backbone.update({
    'selected_layers': list(range(2, 5)),
    'pred_scales': base['backbone']['pred_scales'],
    'pred_aspect_ratios': base['backbone']['pred_aspect_ratios'],
    'use_pixel_scales': True,
    'preapply_sqrt': False,
    'use_square_anchors': True, # This is for backward compatability with a bug
})

yolact_darknet53_config = base.copy({
    'name': 'yolact_darknet53',
    'backbone': yolact_darknet53_backbone,
})


yolact_resnet50_backbone = resnet50_backbone.copy()
yolact_resnet50_backbone.update({
    'selected_layers': list(range(1, 4)),
    'pred_scales': base['backbone']['pred_scales'],
    'pred_aspect_ratios': base['backbone']['pred_aspect_ratios'],
    'use_pixel_scales': True,
    'preapply_sqrt': False,
    'use_square_anchors': True, # This is for backward compatability with a bug
})

yolact_resnet50_config = base.copy({
    'name': 'yolact_resnet50',
    'backbone': yolact_resnet50_backbone,
})


yolact_resnet50_pascal_backbone = yolact_resnet50_config['backbone'].copy()
yolact_resnet50_pascal_backbone.update({
    'pred_scales': [[32], [64], [128], [256], [512]],
    'use_square_anchors': False,
})

yolact_resnet50_pascal_config = yolact_resnet50_config.copy()
yolact_resnet50_pascal_config.update({
    'name': None, # Will default to yolact_resnet50_pascal
    'dataset': pascal_sbd_dataset,
    'num_classes': len(pascal_sbd_dataset.class_names) + 1,
    'max_iter': 120000,
    'lr_steps': (60000, 100000),
    'backbone': yolact_resnet50_pascal_backbone,
})
