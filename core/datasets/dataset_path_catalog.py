import os
from .cityscapes import cityscapesDataSet
from .cityscapes_strong_weak_aug import cityscapesDataSetStrongWeakAug
from .cityscapes_soft_label import cityscapesSoftLabelDataSet
from .synthia import synthiaDataSet
from .gtav import GTAVDataSet
from .gtav_strong_weak_aug import GTAVDataSetStrongWeakAug


class DatasetCatalog(object):
    DATASET_DIR = "datasets"
    DATASETS = {
        "gtav_train": {
            "data_dir": "gtav",
            "data_list": "gtav_train_list.txt"
        },
        "gtav_strong_weak_aug_train": {
            "data_dir": "gtav",
            "data_list": "gtav_train_list.txt"
        },
        "synthia_train": {
            "data_dir": "synthia",
            "data_list": "synthia_train_list.txt"
        },
        "cityscapes_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_strong_weak_aug_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_train_soft": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt",
            "label_dir": "soft_labels/cityscapes_train"
        },
        "cityscapes_val": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_val_list.txt"
        },
        "cityscapes_strong_weak_aug_val": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_val_list.txt"
        },
    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None, cfg=None, in_source=None, logger=None):
        if "gtav_strong_weak_aug" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return GTAVDataSetStrongWeakAug(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, cfg=cfg, logger=logger)
        elif "gtav" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return GTAVDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                               split=mode, transform=transform)
        elif "synthia" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return synthiaDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                  split=mode, transform=transform)

        elif "cityscapes_strong_weak_aug" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return cityscapesDataSetStrongWeakAug(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, cfg=cfg, logger=logger)
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if 'soft' in name:
                args['label_dir'] = os.path.join(cfg.PREPARE_DIR, attrs["label_dir"])
                print("Loading Cityscapes_train soft label from{}".format(args['label_dir']))
                return cityscapesSoftLabelDataSet(args["root"], args["data_list"], args['label_dir'],
                                                  max_iters=max_iters, num_classes=num_classes,
                                                  split=mode, transform=transform)
            return cityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        

        raise RuntimeError("Dataset not available: {}".format(name))
