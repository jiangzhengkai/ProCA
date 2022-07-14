import argparse
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from core.configs import cfg
from core.datasets import build_dataset
from core.utils.logger import setup_logger
from core.utils.misc import AverageMeter, intersection_and_union, mkdir

warnings.filterwarnings('ignore')


def strip_prefix_if_present(state_dict, prefix):
    from collections import OrderedDict
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix + 'layer5'):
            continue
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def run_test(cfg, feature_extractor, classifier, local_rank, distributed,
             logger):
    if local_rank == 0:
        logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
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
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_data)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=cfg.TEST.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              sampler=test_sampler)
    feature_extractor.eval()
    classifier.eval()
    end = time.time()
    with torch.no_grad():
        for i, (x, y, _) in enumerate(test_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()

            size = y.shape[-2:]
            output = classifier(feature_extractor(x))
            output = F.interpolate(output,
                                   size=size,
                                   mode='bilinear',
                                   align_corners=True)
            output = output.max(1)[1]
            intersection, union, target = intersection_and_union(
                output, y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
            if distributed:
                torch.distributed.all_reduce(
                    intersection), torch.distributed.all_reduce(
                        union), torch.distributed.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(
            ), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(
                union), target_meter.update(target)
            batch_time.update(time.time() - end)
            end = time.time()
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    m_iou = np.mean(iou_class)
    m_acc = np.mean(accuracy_class)
    all_acc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if local_rank == 0:
        logger.info("Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
            m_iou, m_acc, all_acc))
        for i in range(cfg.MODEL.NUM_CLASSES):
            logger.info(
                "Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.".format(
                    i, test_data.trainid2name[i], iou_class[i],
                    accuracy_class[i]))
    return m_iou, m_acc, all_acc


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training")
    parser.add_argument(
        "-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("SelfSupervised", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, args.local_rank, args.distributed, logger)


if __name__ == "__main__":
    main()
