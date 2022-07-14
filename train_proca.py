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
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.lovasz_loss import lovasz_softmax
from core.utils.prototype_dist_estimator import prototype_dist_estimator
from core.utils.loss import PrototypeContrastiveLoss

import warnings
warnings.filterwarnings('ignore')


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

    # batch size: half for source and half for target
    batch_size = cfg.SOLVER.BATCH_SIZE // 2
    if distributed:
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size()) // 2
        if not cfg.MODEL.FREEZE_BN:
            feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg2
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()

    logger.info(feature_extractor)
    logger.info(classifier)
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
    pcl_criterion = PrototypeContrastiveLoss(cfg)

    # load init source prototypes
    logger.info(">>>>>>>>>>>>>>>> Load init prototypes >>>>>>>>>>>>>>>>")
    _, backbone_name = cfg.MODEL.NAME.split('_')
    feature_num = 2048 if backbone_name.startswith('resnet') else 1024
    feat_estimator = prototype_dist_estimator(feature_num=feature_num, cfg=cfg)
    if cfg.SOLVER.MULTI_LEVEL:
        out_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, cfg=cfg)

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
        tgt_feat = feature_extractor(tgt_input)
        tgt_out = classifier(tgt_feat)

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
        B, A, Hs, Ws = src_feat.size()
        src_mask = F.interpolate(src_label.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
        src_mask = src_mask.contiguous().view(B * Hs * Ws, )
        assert not src_mask.requires_grad
        # target mask: constant threshold -- cfg.SOLVER.THRESHOLD
        _, _, Ht, Wt = tgt_feat.size()
        tgt_out_maxvalue, tgt_mask = torch.max(tgt_out, dim=1)
        for i in range(cfg.MODEL.NUM_CLASSES):
            tgt_mask[(tgt_out_maxvalue < cfg.SOLVER.DELTA) * (tgt_mask == i)] = 255
        tgt_mask = tgt_mask.contiguous().view(B * Ht * Wt, )
        assert not tgt_mask.requires_grad

        src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
        tgt_feat = tgt_feat.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)
        # update feature-level statistics
        feat_estimator.update(features=tgt_feat.detach(), labels=tgt_mask)
        feat_estimator.update(features=src_feat.detach(), labels=src_mask)

        # contrastive loss on both domains
        loss_feat = pcl_criterion(Proto=feat_estimator.Proto.detach(),
                                  feat=src_feat,
                                  labels=src_mask) \
                    + pcl_criterion(Proto=feat_estimator.Proto.detach(),
                                  feat=tgt_feat,
                                  labels=tgt_mask)
        meters.update(loss_feat=loss_feat.item())

        if cfg.SOLVER.MULTI_LEVEL:
            src_out = src_out.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, cfg.MODEL.NUM_CLASSES)
            tgt_out = tgt_out.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, cfg.MODEL.NUM_CLASSES)

            # update output-level statistics
            out_estimator.update(features=tgt_out.detach(), labels=tgt_mask)
            out_estimator.update(features=src_out.detach(), labels=src_mask)

            # the proposed contrastive loss on prediction map
            loss_out = pcl_criterion(Proto=out_estimator.Proto.detach(),
                                     feat=src_out,
                                     labels=src_mask) \
                       + pcl_criterion(Proto=out_estimator.Proto.detach(),
                                       feat=tgt_out,
                                       labels=tgt_mask)
            meters.update(loss_out=loss_out.item())

            loss = loss_sup \
                   + cfg.SOLVER.LAMBDA_FEAT * loss_feat \
                   + cfg.SOLVER.LAMBDA_OUT * loss_out
        else:
            loss = loss_sup + cfg.SOLVER.LAMBDA_FEAT * loss_feat

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
            current_mIoU, _, _ = run_test(cfg, feature_extractor, classifier, local_rank, distributed, logger)
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

    logger = setup_logger("ProCA", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, args.local_rank, args.distributed, logger)



if __name__ == '__main__':
    main()
