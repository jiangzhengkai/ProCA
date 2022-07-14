import argparse
import os
import datetime
import logging
import time
import numpy as np
from collections import OrderedDict

import torch
import torch.utils
import torch.distributed
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.models.classifier import ProjectionHead
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.lovasz_loss import lovasz_softmax
from core.utils.loss import PrototypeContrastiveLoss

import warnings
warnings.filterwarnings('ignore')


class MemoryBankContrastiveLoss(nn.Module):
    def __init__(self, cfg):
        super(MemoryBankContrastiveLoss, self).__init__()
        self.cfg = cfg
        self.temperature = cfg.MODEL.CONTRAST.TEMPERATURE 
        self.ignore_label = cfg.INPUT.IGNORE_LABEL
        self.max_samples = 1024
        self.min_samples = 10

    def hard_anchor_sampling(self, feat, label, predict):
        feat_dim = feat.shape[-1]
        total_classes = 0
        feat = feat.reshape(-1, feat_dim)
        label = label.reshape(-1)
        predict = predict.reshape(-1)
        classes = torch.unique(label)
        classes = [x for x in classes if x != self.ignore_label]
        classes = [x for x in classes if (label == x).nonzero().shape[0] > self.min_samples]  
        total_classes += len(classes)
        sample_each_class = self.max_samples // total_classes
        sample_each_class = min(sample_each_class, self.min_samples)

        select_feat = torch.zeros((total_classes, sample_each_class, feat_dim), dtype=torch.float).cuda()
        select_label = torch.zeros(total_classes, dtype=torch.float).cuda()
        x_ptr = 0
        for cls_id in classes:
            hard_indices = ((label == cls_id) & (predict != cls_id)).nonzero()
            easy_indices = ((label == cls_id) & (predict == cls_id)).nonzero()
            num_hard = hard_indices.shape[0]
            num_easy = easy_indices.shape[0]

            if num_hard >= sample_each_class // 2 and num_easy >= sample_each_class // 2:
                num_hard_keep = sample_each_class // 2
                num_easy_keep = sample_each_class - num_hard_keep
            elif num_hard >= sample_each_class // 2:
                num_easy_keep = num_easy
                num_hard_keep = sample_each_class - num_easy_keep
            elif num_easy >= sample_each_class // 2:
                num_hard_keep = num_hard
                num_easy_keep = sample_each_class - num_hard_keep

            perm = torch.randperm(num_hard)
            hard_indices = hard_indices[perm[:num_hard_keep]]
            perm = torch.randperm(num_easy)
            easy_indices = easy_indices[perm[:num_easy_keep]]
            indices = torch.cat((hard_indices, easy_indices), dim=0)
            select_feat[x_ptr, :, :] = feat[indices.squeeze(1), :].squeeze(1)
            select_label[x_ptr] = cls_id
            x_ptr += 1

        return select_feat, select_label


    def sample_negative(self, queue):
        num_class, queue_size, feat_size = queue.shape
        feat = torch.zeros((num_class * queue_size, feat_size), dtype=queue.dtype, device=queue.device)
        label = torch.zeros((num_class * queue_size, 1), dtype=queue.dtype, device=queue.device)
        feat_ptr = 0
        for i in range(num_class):
            queue_i = queue[i, :queue_size, :]
            feat[feat_ptr:feat_ptr + queue_size, ...] = queue_i
            label[feat_ptr:feat_ptr + queue_size, ...] = i
            feat_ptr += queue_size
        return feat, label

    def contrastive(self, feat, label, queue=None):
        anchor_count, num_view = feat.shape[0], feat.shape[1]
        label = label.reshape(-1, 1)
        anchor_feature = torch.cat(torch.unbind(feat, dim=1), dim=0)
        if queue is not None:
            queue_feat, queue_label = self.sample_negative(queue)
            queue_label = queue_label.reshape(-1, 1)

        mask = torch.eq(label, queue_label.T.float().cuda())
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, queue_feat.T),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(num_view, 1)
        neg_mask = ~mask
        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(num_view * anchor_count).view(-1, 1).cuda(),
                                                     0)

        mask = mask * logits_mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss


    def forward(self, feats, labels, predict, proto_queue, pixel_queue):
        """
        Args:
            C: NUM_CLASSES A: feat_dim B: batch_size H: feat_high W: feat_width N: number of pixels except IGNORE_LABEL
            feat: shape (B, HW, A)
            labels: shape (B, HW)
            predict: shape (B, HW)
            proto_queue: shape (C, proto_size, A)
            pixel_queue: shape (C, pixel_size, A)
        Returns:
        """
        assert not proto_queue.requires_grad
        assert not pixel_queue.requires_grad
        assert not labels.requires_grad
        assert feats.requires_grad
   
        queue = torch.cat([proto_queue, pixel_queue], dim=0)
        labels = labels.unsqueeze(1).float().clone()
        labels = F.interpolate(labels,
                              (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)
        batch_size = feats.shape[0]
        feat_dim = feats.shape[1]
        labels = labels.reshape(batch_size, -1)
        predict = predict.reshape(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1).reshape(batch_size, -1, feat_dim)

        feats_, labels_ = self.hard_anchor_sampling(feats, labels, predict)

        loss = self.contrastive(feats_, labels_, queue=queue)
        return loss / batch_size


class MemoryBank(nn.Module):
    def __init__(self, cfg):
        super(MemoryBank, self).__init__()
        self.cfg = cfg
        self.class_num = cfg.MODEL.NUM_CLASSES
        self.feature_num = cfg.MODEL.CONTRAST.PROJ_DIM

        self.memory_size = cfg.MODEL.CONTRAST.MEMORY_SIZE
        self.pixel_update_freq = cfg.MODEL.CONTRAST.PIXEL_UPDATE_FREQ
        device = cfg.MODEL.DEVICE
        # create the queue
        self.register_buffer("pixel_queue", torch.randn(self.class_num, self.memory_size, self.feature_num))
        self.pixel_queue = F.normalize(self.pixel_queue, dim=0).to(device)

        self.register_buffer("proto_queue", torch.randn(self.class_num, self.memory_size, self.feature_num))
        self.proto_queue = F.normalize(self.proto_queue, dim=0).to(device)

        self.register_buffer("pixel_queue_ptr", torch.zeros(self.class_num, dtype=torch.long, device=device))
        self.register_buffer("proto_queue_ptr", torch.zeros(self.class_num, dtype=torch.long, device=device))

    def update(self, features, labels):
        valid_mask = (labels != self.cfg.INPUT.IGNORE_LABEL)
        labels = labels[valid_mask]
        features = features[valid_mask]

        ids_unique = labels.unique()
        for i in ids_unique:
            i = i.item()
            mask_i = (labels == i)
            label = labels[mask_i]
            feature = features[mask_i]
            # proto queue
            proto = torch.mean(feature, dim=0)
            proto_ptr = self.proto_queue_ptr[i].item()

            self.proto_queue[i, proto_ptr, :] = proto
            self.proto_queue_ptr[i] = (self.proto_queue_ptr[i] + 1) % self.memory_size

            # pixel queue
            num_pixel = label.shape[0]
            perm = torch.randperm(num_pixel)
            K = min(num_pixel, self.pixel_update_freq)
            feat = feature[perm[:K], :]
            ptr = self.pixel_queue_ptr[i]

            if ptr + K > self.memory_size:
                self.pixel_queue[i, -K:, :] = feat
                self.pixel_queue_ptr[i] = 0
            else:
                self.pixel_queue[i, ptr:ptr + K, :] = feat
                self.pixel_queue_ptr[i] = (self.pixel_queue_ptr[i] + 1) % self.memory_size


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def train(cfg, local_rank, distributed, logger):
    # create network
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)

    # project head
    _, backbone_name = cfg.MODEL.NAME.split('_')
    feature_num = 2048 if backbone_name.startswith('resnet') else 1024
    project_head = ProjectionHead(feature_num, cfg.MODEL.CONTRAST.PROJ_DIM)
    project_head.to(device)

    # batch size: half for source and half for target
    batch_size = cfg.SOLVER.BATCH_SIZE // 2
    if distributed:
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size()) // 2
        if not cfg.MODEL.FREEZE_BN:
            feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
            project_head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(project_head)
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg2
        )
        pg3 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        project_head = torch.nn.parallel.DistributedDataParallel(
            project_head, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg3
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()

    if local_rank == 0:
        logger.info(classifier)
        logger.info(feature_extractor)
        logger.info(project_head)
    # init optimizer
    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()

    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    output_dir = cfg.OUTPUT_DIR

    # load checkpoint
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        feature_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(
            checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(
            checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    # init data loader
    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False)
    if distributed:
        src_train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_data)
        tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
    else:
        src_train_sampler = None
        tgt_train_sampler = None
    src_train_loader = torch.utils.data.DataLoader(src_train_data, batch_size=batch_size,
                                                   shuffle=(src_train_sampler is None), num_workers=4,
                                                   pin_memory=True, sampler=src_train_sampler, drop_last=True)
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size,
                                                   shuffle=(tgt_train_sampler is None), num_workers=4,
                                                   pin_memory=True, sampler=tgt_train_sampler, drop_last=True)

    # init loss
    ce_criterion = nn.CrossEntropyLoss(ignore_index=255)

    # init memory bank
    logger.info(">>>>>>>>>>>>>>>> Init Memory Bank >>>>>>>>>>>>>>>>")
    _, backbone_name = cfg.MODEL.NAME.split('_')
    memory_bank_estimator = MemoryBank(cfg=cfg)
    memory_bank_loss = MemoryBankContrastiveLoss(cfg=cfg)
    iteration = 0
    start_training_time = time.time()
    end = time.time()
    save_to_disk = local_rank == 0
    max_iters = cfg.SOLVER.MAX_ITER
    meters = MetricLogger(delimiter="  ")

    logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
    feature_extractor.train()
    classifier.train()
    best_mIoU = 0
    best_iteration = 0

    for i, ((src_input, src_label, src_name), (tgt_input, _, _)) in enumerate(zip(src_train_loader, tgt_train_loader)):
        data_time = time.time() - end
        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters,
                                          power=cfg.SOLVER.LR_POWER)
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr * 10

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()
        src_input = src_input.cuda(non_blocking=True)
        src_label = src_label.cuda(non_blocking=True).long()
        tgt_input = tgt_input.cuda(non_blocking=True)

        src_size = src_input.shape[-2:]
        src_feat = feature_extractor(src_input)
        src_out = classifier(src_feat)
        src_embedding = project_head(src_feat)

        tgt_feat = feature_extractor(tgt_input)
        tgt_out = classifier(tgt_feat)
        tgt_embedding = project_head(tgt_feat)

        # supervision loss
        src_pred = F.interpolate(src_out, size=src_size, mode='bilinear', align_corners=True)
        if cfg.SOLVER.LAMBDA_LOV > 0:
            pred_softmax = F.softmax(src_pred, dim=1)
            loss_lov = lovasz_softmax(pred_softmax, src_label, ignore=255)
            loss_sup = ce_criterion(src_pred, src_label) + cfg.SOLVER.LAMBDA_LOV * loss_lov
            meters.update(loss_lov=loss_lov.item())
        else:
            loss_sup = ce_criterion(src_pred, src_label)
        meters.update(loss_sup=loss_sup.item())

        # source mask: downsample the ground-truth label
        _, src_predict_mask = torch.max(src_out, dim=1)
        B, A, Hs, Ws = src_embedding.size()
        src_mask = F.interpolate(src_label.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
        src_mask_reshape = src_mask.contiguous().view(B * Hs * Ws, )
        assert not src_mask.requires_grad
        # target mask: constant threshold -- cfg.SOLVER.THRESHOLD
        _, _, Ht, Wt = tgt_embedding.size()
        tgt_out_maxvalue, tgt_mask = torch.max(tgt_out, dim=1)
        for i in range(cfg.MODEL.NUM_CLASSES):
            tgt_mask[(tgt_out_maxvalue < cfg.SOLVER.DELTA) * (tgt_mask == i)] = 255
        tgt_mask_reshape = tgt_mask.contiguous().view(B * Ht * Wt, )
        assert not tgt_mask.requires_grad
        src_embedding_reshape = src_embedding.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
        tgt_embedding_reshape = tgt_embedding.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)
        # update memory bank
        memory_bank_estimator.update(features=src_embedding_reshape.detach(), labels=src_mask_reshape)
        memory_bank_estimator.update(features=tgt_embedding_reshape.detach(), labels=tgt_mask_reshape)

        loss_src = memory_bank_loss(src_embedding,
                                    src_label,
                                    src_predict_mask,
                                    memory_bank_estimator.proto_queue,
                                    memory_bank_estimator.pixel_queue)
        loss_tgt = memory_bank_loss(tgt_embedding,
                                    tgt_mask,
                                    tgt_mask,
                                    memory_bank_estimator.proto_queue,
                                    memory_bank_estimator.pixel_queue)

        loss = loss_src + loss_tgt + loss_sup
        meters.update(loss_contrast_src=loss_src.item())
        meters.update(loss_contrast_tgt=loss_tgt.item())
        loss.backward()

        optimizer_fea.step()
        optimizer_cls.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        iteration += 1
        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.02f} GB"
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer_fea.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
                )
            )

        if (iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or iteration == cfg.SOLVER.STOP_ITER):
            current_mIoU, current_mAcc, current_allAcc = run_test(cfg, feature_extractor, classifier, local_rank, distributed, logger)
            feature_extractor.train()
            classifier.train()
            if save_to_disk:
                # update best model
                if current_mIoU > best_mIoU:
                    filename = os.path.join(output_dir, "model_best.pth")
                    torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(),
                        'classifier': classifier.state_dict(), 'optimizer_fea': optimizer_fea.state_dict(),
                        'optimizer_cls': optimizer_cls.state_dict()}, filename)
                    best_mIoU = current_mIoU
                    best_iteration = iteration
                else:
                    filename = os.path.join(output_dir, "model_current.pth")
                    torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(),
                        'classifier': classifier.state_dict(), 'optimizer_fea': optimizer_fea.state_dict(),
                        'optimizer_cls': optimizer_cls.state_dict()}, filename)
                logger.info(f"-------- Best mIoU {best_mIoU} at iteration {best_iteration} --------")
            torch.cuda.empty_cache()

        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / cfg.SOLVER.STOP_ITER
        )
    )

    return feature_extractor, classifier


def run_test(cfg, feature_extractor, classifier, local_rank, distributed, logger):
    if local_rank == 0:
        logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    if distributed:
        feature_extractor, classifier = feature_extractor.module, classifier.module
    torch.cuda.empty_cache()
    dataset_name = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)

    test_data = build_dataset(cfg, mode='test', is_source=False)
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
                                              pin_memory=True, sampler=test_sampler)
    feature_extractor.eval()
    classifier.eval()
    end = time.time()
    with torch.no_grad():
        for i, (x, y, _) in enumerate(test_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()

            size = y.shape[-2:]
            output = classifier(feature_extractor(x))
            output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
            output = output.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES,
                                                                  cfg.INPUT.IGNORE_LABEL)
            if distributed:
                torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(
                    union), torch.distributed.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if local_rank == 0:
        logger.info("Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(mIoU, mAcc, allAcc))
        for i in range(cfg.MODEL.NUM_CLASSES):
            logger.info(
                "Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.".format(i, test_data.trainid2name[i],
                                                                         iou_class[i], accuracy_class[i])
            )
    return mIoU, mAcc, allAcc


def main():
    parser = argparse.ArgumentParser(description="Pytorch Domain Adaptive Semantic Segmentation Training")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("ProCAMemoryBank", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, args.local_rank, args.distributed, logger)



if __name__ == '__main__':
    main()
