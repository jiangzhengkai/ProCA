import argparse
import logging
from collections import OrderedDict
import  time
import datetime

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.distributed
import torch.backends.cudnn

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.utils.misc import mkdir, AverageMeter
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger

from core.utils.prototype_dist_estimator import prototype_dist_estimator

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


def prototype_dist_init(cfg, logger):
    _, backbone_name = cfg.MODEL.NAME.split('_')
    feature_num = 2048 if backbone_name.startswith('resnet') else 1024
    feat_estimator = prototype_dist_estimator(feature_num=feature_num, cfg=cfg)
    out_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, cfg=cfg)

    feature_extractor = build_feature_extractor(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)

    torch.cuda.empty_cache()

    # load checkpoint
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_extractor_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    src_train_data = build_dataset(cfg, mode='train', is_source=True, epochwise=True)
    src_train_loader = torch.utils.data.DataLoader(src_train_data,
                                                   batch_size=cfg.SOLVER.BATCH_SIZE_VAL,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   pin_memory=True,
                                                   drop_last=False)
    iteration = 0
    feature_extractor.eval()
    classifier.eval()
    end = time.time()
    start_time = time.time()
    max_iters = len(src_train_loader)
    meters = MetricLogger(delimiter="  ")
    logger.info(">>>>>>>>>>>>>>>> Initialize prototype >>>>>>>>>>>>>>>>")
    logger.info(max_iters)
    with torch.no_grad():
        for i, (src_input, src_label, _) in enumerate(src_train_loader):
            data_time = time.time() - end

            src_input = src_input.cuda(non_blocking=True)
            src_label = src_label.cuda(non_blocking=True).long()

            src_feat = feature_extractor(src_input)
            src_out = classifier(src_feat)
            B, N, Hs, Ws = src_feat.size()
            _, C, _, _ = src_out.size()

            # source mask: downsample the ground-truth label
            src_mask = F.interpolate(src_label.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
            src_mask = src_mask.contiguous().view(B * Hs * Ws, )

            # feature level
            src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, N)
            feat_estimator.update(features=src_feat.detach().clone(), labels=src_mask)

            # output level
            src_out = src_out.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, C)
            out_estimator.update(features=src_out.detach().clone(), labels=src_mask)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            iteration = iteration + 1
            eta_seconds = meters.time.global_avg * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iters:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "max mem: {memory:.02f}"
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
                    )
                )

            if iteration == max_iters:
                feat_estimator.save(name='prototype_feat_dist.pth')
                out_estimator.save(name='prototype_out_dist.pth')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_time / max_iters
        )
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("prototype_dist_init", output_dir, 0)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    prototype_dist_init(cfg, logger)


if __name__ == "__main__":
    main()
