resnet_transform = {
    'channel_order': 'RGB',
    'normalize': True,
    'subtract_means': False,
    'to_float': False,
}

vgg_transform = {
    # Note that though vgg is traditionally BGR,
    # the channel order of vgg_reducedfc.pth is RGB.
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': True,
    'to_float': False,
}

darknet_transform = {
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': False,
    'to_float': True,
}
