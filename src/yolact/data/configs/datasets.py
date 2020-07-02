from yolact.data.configs.dataset_base import dataset as base

coco2014_dataset = base.copy()
coco2014_dataset.update({
    'name': 'COCO 2014',
    'train_info': './src/yolact/data/coco/annotations/instances_train2014.json',
    'valid_info': './src/yolact/data/coco/annotations/instances_val2014.json',
})

coco2017_dataset = base.copy()
coco2017_dataset.update({
    'name': 'COCO 2017',
    'train_info': './src/yolact/data/coco/annotations/instances_train2017.json',
    'valid_info': './src/yolact/data/coco/annotations/instances_val2017.json',
})

coco2017_testdev_dataset = base.copy()
coco2017_testdev_dataset.update({
    'name': 'COCO 2017 Test-Dev',
    'valid_info': './src/yolact/data/coco/annotations/image_info_test-dev2017.json',
    'has_gt': False,
})

pascal_sbd_dataset = base.copy()
pascal_sbd_dataset.update({
    'name': 'Pascal SBD 2012',

    'train_images': './data/sbd/img',
    'valid_images': './data/sbd/img',

    'train_info': './data/sbd/pascal_sbd_train.json',
    'valid_info': './data/sbd/pascal_sbd_val.json',

    'class_names':  ("aeroplane", "bicycle", "bird", "boat", "bottle",
                  "bus", "car", "cat", "chair", "cow", "diningtable",
                  "dog", "horse", "motorbike", "person", "pottedplant",
                  "sheep", "sofa", "train", "tvmonitor"),
})
