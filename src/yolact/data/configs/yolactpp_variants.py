from yolact.data.configs.backbones import resnet101_dcn_inter3_backbone, resnet50_dcnv2_backbone
from yolact.data.configs import yolact_base

yolact_plus_base_backbone = resnet101_dcn_inter3_backbone.copy()
yolact_plus_base_backbone.update({
    'selected_layers': list(range(1, 4)),
    'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
    'pred_scales': [[i * 2 ** (j / 3.0) for j in range(3)] for i in [24, 48, 96, 192, 384]],
    'use_pixel_scales': True,
    'preapply_sqrt': False,
    'use_square_anchors': False,
})

yolact_plus_base_config = yolact_base.config.copy()
yolact_plus_base_config.udpate({
    'name': 'yolact_plus_base',

    'backbone': yolact_plus_base_backbone,

    'use_maskiou': True,
    'maskiou_net': [(8, 3, {'stride': 2}), (16, 3, {'stride': 2}), (32, 3, {'stride': 2}), (64, 3, {'stride': 2}), (128, 3, {'stride': 2})],
    'maskiou_alpha': 25,
    'rescore_bbox': False,
    'rescore_mask': True,

    'discard_mask_area': 5*5,
})


yolact_plus_resnet50_backbone = resnet50_dcnv2_backbone.copy()
yolact_plus_resnet50_backbone.update({
    'selected_layers': list(range(1, 4)),
    'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
    'pred_scales': [[i * 2 ** (j / 3.0) for j in range(3)] for i in [24, 48, 96, 192, 384]],
    'use_pixel_scales': True,
    'preapply_sqrt': False,
    'use_square_anchors': False,
})

yolact_plus_resnet50_config = yolact_plus_base_config.copy()
yolact_plus_resnet50_config.update({
    'name': 'yolact_plus_resnet50',
    'backbone': yolact_plus_resnet50_backbone,
})
