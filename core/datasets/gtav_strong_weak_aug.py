import os
import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image, ImageFile
import pickle
import logging
from . import transform
ImageFile.LOAD_TRUNCATED_IMAGES = True



class GTAVDataSetStrongWeakAug(data.Dataset):
    """
        original resolution at 1914x1024
    """
    def __init__(self,
                 data_root,
                 data_list,
                 max_iters=None,
                 num_classes=19,
                 split="train",
                 ignore_label=255,
                 cfg=None,
                 debug=False,
                 logger=None,
        ):

        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.data_list = []
        with open(data_list, "r") as handle:
            content = handle.readlines()
        self.img_ids = [i_id.strip() for i_id in content]

        if max_iters is not None:
            self.label_to_file, self.file_to_label = pickle.load(open(osp.join(data_root, "gtav_label_info.p"), "rb"))
            self.img_ids = []
            SUB_EPOCH_SIZE = 3000
            tmp_list = []
            ind = dict()
            for i in range(self.NUM_CLASS):
                ind[i] = 0
            for e in range(int(max_iters / SUB_EPOCH_SIZE) + 1):
                cur_class_dist = np.zeros(self.NUM_CLASS)
                for i in range(SUB_EPOCH_SIZE):
                    if cur_class_dist.sum() == 0:
                        dist1 = cur_class_dist.copy()
                    else:
                        dist1 = cur_class_dist / cur_class_dist.sum()
                    w = 1 / np.log(1 + 1e-2 + dist1)
                    w = w / w.sum()
                    c = np.random.choice(self.NUM_CLASS, p=w)

                    if ind[c] > (len(self.label_to_file[c]) - 1):
                        np.random.shuffle(self.label_to_file[c])
                        ind[c] = ind[c] % (len(self.label_to_file[c]) - 1)

                    c_file = self.label_to_file[c][ind[c]]
                    tmp_list.append(c_file)
                    ind[c] = ind[c] + 1
                    cur_class_dist[self.file_to_label[c_file]] += 1

            self.img_ids = tmp_list

        for name in self.img_ids:
            self.data_list.append(
                {
                    "img": os.path.join(self.data_root, "images/%s" % name),
                    "label": os.path.join(self.data_root, "labels/%s" % name),
                    "name": name,
                }
            )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        self.id_to_trainid = {7: 0,
                              8: 1,
                              11: 2,
                              12: 3,
                              13: 4,
                              17: 5,
                              19: 6,
                              20: 7,
                              21: 8,
                              22: 9,
                              23: 10,
                              24: 11,
                              25: 12,
                              26: 13,
                              27: 14,
                              28: 15,
                              31: 16,
                              32: 17,
                              33: 18}
        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle"
        }
        
        if split == "train":
            w, h = cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN
            # build weak transform
            weak_trans_list = []
            if cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN > 0:
                weak_trans_list = [transform.RandomHorizontalFlip(p=cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN), ]
            if cfg.INPUT.INPUT_SCALES_TRAIN[0] == cfg.INPUT.INPUT_SCALES_TRAIN[1] and cfg.INPUT.INPUT_SCALES_TRAIN[0] == 1:
                weak_trans_list = [transform.Resize((h, w)), ] + weak_trans_list
            else:
                weak_trans_list = [
                             transform.RandomScale(scale=cfg.INPUT.INPUT_SCALES_TRAIN),
                             transform.RandomCrop(size=(h, w), pad_if_needed=True),
                         ] + weak_trans_list
            self.weak_trans = transform.Compose(weak_trans_list)
            logger.info(f"Source: weak aug {self.weak_trans}")
            # build strong transform 
            strong_trans_list = []
            if cfg.INPUT.GAUSSIANBLUR:
                strong_trans_list = [
                                 transform.GaussianBlur(kernel_size=[3, 3])
                             ]
            if cfg.INPUT.GRAYSCALE > 0:
                strong_trans_list = [
                                 transform.RandomGrayscale(p=cfg.INPUT.GRAYSCALE)
                             ] + strong_trans_list
            if cfg.INPUT.BRIGHTNESS > 0:
                strong_trans_list = [
                                 transform.ColorJitter(
                                     brightness=cfg.INPUT.BRIGHTNESS,
                                     contrast=cfg.INPUT.CONTRAST,
                                     saturation=cfg.INPUT.SATURATION,
                                     hue=cfg.INPUT.HUE,
                                 )
                             ] + strong_trans_list
            self.strong_trans = transform.Compose(strong_trans_list)
            logger.info(f"Source: strong aug {self.strong_trans}")
            # build normalize transform
            normalize_trans_list = [
                transform.ToTensor(),
                transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
            ]
            self.normalize_trans = transform.Compose(normalize_trans_list)
            logger.info(f"Source: normalize aug {self.normalize_trans}")
        else:
            # build eval transform
            w, h = cfg.INPUT.INPUT_SIZE_TEST
            self.val_trans = transform.Compose([
                transform.Resize((h, w), resize_label=False),
                transform.ToTensor(),
                transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
            ])
         
        self.ignore_label = ignore_label
        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = np.array(Image.open(datafiles["label"]), dtype=np.float32)
        name = datafiles["name"]

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = Image.fromarray(label_copy)

        if self.split == "train":
            weak_image, weak_label = self.weak_trans(image, label)
            strong_image, strong_label = self.strong_trans(weak_image, weak_label)
            # normalize
            weak_image, weak_label = self.normalize_trans(weak_image, weak_label)
            strong_image, strong_label = self.normalize_trans(strong_image, strong_label)
        else:
            if self.val_trans is not None:
                image, label = self.val_trans(image, label)
        if self.split == "train":
            return weak_image, weak_label, strong_image, strong_label, name
        else:
            return image, label, name
