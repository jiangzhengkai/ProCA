import torch
import torch.nn.functional as F
import numpy as np
import os


class PseudoLabel:
    def __init__(self, cfg):
        h, w = cfg.INPUT.TARGET_INPUT_SIZE_TRAIN
        self.prob_tar = np.zeros([1, h, w])
        self.label_tar = np.zeros([1, h, w])
        self.thres = []
        self.number_class = cfg.MODEL.NUM_CLASSES
        self.out_dir = cfg.OUTPUT_DIR
        self.iter = 0

    def save_results(self):
        np.save(os.path.join(self.out_dir, 'thres_const.npy'), self.thres)
        print("save done.")

    def update_pseudo_label(self, input):
        input = F.softmax(input.detach(), dim=1)
        prob, label = torch.max(input, dim=1)
        prob_np = prob.cpu().numpy()
        label_np = label.cpu().numpy()
        print(self.iter)
        if self.iter==0:
            self.prob_tar = prob_np
            self.label_tar = label_np
        else:
            self.prob_tar = np.append(self.prob_tar, prob_np, axis=0)
            self.label_tar = np.append(self.label_tar, label_np, axis=0)
        self.iter += 1

    def get_threshold_const(self, thred, percent=0.5):
        for i in range(self.number_class):
            x = self.prob_tar[self.label_tar == i]
            if len(x) == 0:
                self.thres.append(0)
                continue
            x = np.sort(x)
            self.thres.append(x[np.int(np.round(len(x) * percent))])
        self.thres = np.array(self.thres)
        self.thres[self.thres > thred] = thred
        return self.thres
