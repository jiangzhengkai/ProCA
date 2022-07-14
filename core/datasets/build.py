import logging
from . import transform
from .dataset_path_catalog import DatasetCatalog


def build_transform(cfg, mode, is_source):
    """
    data augmentation from FADA and combine SimCLR (GaussianBlur & Grayscale)
    Args:
        cfg:
        mode: train or test
        is_source: if True using GaussianBlur & Grayscale & ColorJitter for source_train_data
        train_transform = transforms.Compose([
           transforms.RandomScale(INPUT_SCALES_TRAIN),
           transforms.RandomCrop(h,w),
           transforms.RandomHorizontalFlip(HORIZONTAL_FLIP_PROB_TRAIN),
           transforms.RandomApply([transform.ColorJitter(BRIGHTNESS,CONTRAST,SATURATION,HUE)], RANDOMAPPLY),
           transform.RandomGrayscale(GRAYSCALE),
           transform.GaussianBlur(kernel_size=[3, 3]),
           transform.ToTensor(),
           transform.Normalize(PIXEL_MEAN, PIXEL_STD, TO_BGR255)
         ])
    Returns:

    """
    if mode == "train":
        w, h = cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN if is_source else cfg.INPUT.TARGET_INPUT_SIZE_TRAIN
        trans_list = [
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ]
        if is_source:
            if cfg.INPUT.GAUSSIANBLUR:
                trans_list = [
                                 transform.GaussianBlur(kernel_size=[3, 3])
                             ] + trans_list
            if cfg.INPUT.GRAYSCALE > 0:
                trans_list = [
                                 transform.RandomGrayscale(p=cfg.INPUT.GRAYSCALE)
                             ] + trans_list
            if cfg.INPUT.BRIGHTNESS > 0:
                trans_list = [
                                 transform.ColorJitter(
                                     brightness=cfg.INPUT.BRIGHTNESS,
                                     contrast=cfg.INPUT.CONTRAST,
                                     saturation=cfg.INPUT.SATURATION,
                                     hue=cfg.INPUT.HUE,
                                 )
                             ] + trans_list
        if cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN > 0:
            trans_list = [transform.RandomHorizontalFlip(p=cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN), ] + trans_list
        if cfg.INPUT.INPUT_SCALES_TRAIN[0] == cfg.INPUT.INPUT_SCALES_TRAIN[1] and cfg.INPUT.INPUT_SCALES_TRAIN[0] == 1:
            trans_list = [transform.Resize((h, w)), ] + trans_list
        else:
            trans_list = [
                             transform.RandomScale(scale=cfg.INPUT.INPUT_SCALES_TRAIN),
                             transform.RandomCrop(size=(h, w), pad_if_needed=True),
                         ] + trans_list
        trans = transform.Compose(trans_list)
    else:
        w, h = cfg.INPUT.INPUT_SIZE_TEST
        trans = transform.Compose([
            transform.Resize((h, w), resize_label=False),
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ])
    return trans



def build_dataset(cfg, mode='train', is_source=True, epochwise=False):
    assert mode in ['train', 'val', 'test']
    transform = build_transform(cfg, mode, is_source)
    logging.info(transform)
    iters = None
    if mode == 'train':
        if not epochwise:
            iters = cfg.SOLVER.MAX_ITER * cfg.SOLVER.BATCH_SIZE
        if is_source:
            dataset = DatasetCatalog.get(cfg.DATASETS.SOURCE_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES,
                                         max_iters=iters, transform=transform, cfg=cfg)
        else:
            dataset = DatasetCatalog.get(cfg.DATASETS.TARGET_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES,
                                         max_iters=iters, transform=transform, cfg=cfg)
    elif mode == 'val':
        dataset = DatasetCatalog.get(cfg.DATASETS.TEST, 'val', num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters,
                                     transform=transform, cfg=cfg)
    elif mode == 'test':
        dataset = DatasetCatalog.get(cfg.DATASETS.TEST, cfg.DATASETS.TEST.split('_')[-1],
                                     num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=transform, cfg=cfg)
    return dataset


def build_strong_weak_aug_dataset(cfg, mode='train', is_source=True, epochwise=False, logger=None):
    assert mode in ['train', 'val', 'test']
    logger.info("currently using strong weak augmentation!!!")
    iters = None
    if mode == 'train':
        if not epochwise:
            iters = cfg.SOLVER.MAX_ITER * cfg.SOLVER.BATCH_SIZE
        if is_source:
            dataset = DatasetCatalog.get(cfg.DATASETS.SOURCE_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES,
                                         max_iters=iters, transform=transform, cfg=cfg, logger=logger)
        else:
            dataset = DatasetCatalog.get(cfg.DATASETS.TARGET_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES,
                                         max_iters=iters, transform=transform, cfg=cfg, logger=logger)
    elif mode == 'val':
        dataset = DatasetCatalog.get(cfg.DATASETS.TEST, 'val', num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters,
                                     cfg=cfg)
    elif mode == 'test':
        dataset = DatasetCatalog.get(cfg.DATASETS.TEST, cfg.DATASETS.TEST.split('_')[-1],
                                     num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, cfg=cfg)
    return dataset
