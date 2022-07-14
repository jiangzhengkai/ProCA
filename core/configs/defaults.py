import os

from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = "deeplab_resnet101"
_C.MODEL.NUM_CLASSES = 19
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHTS = ""
_C.MODEL.FREEZE_BN = False
_C.MODEL.EVAL_BN = False
_C.MODEL.MOMENTUM_ITER = 100
_C.MODEL.THRESHOLD_PERCENT = 0.5

_C.MODEL.CONTRAST = CN()
_C.MODEL.CONTRAST.PROJ_DIM = 256
_C.MODEL.CONTRAST.MEMORY_SIZE = 1000
_C.MODEL.CONTRAST.PIXEL_UPDATE_FREQ = 10
_C.MODEL.CONTRAST.TAU = 50.0
_C.MODEL.CONTRAST.USE_MOMENTUM = False
_C.MODEL.CONTRAST.MOMENTUM = 0.9
# update
_C.INPUT = CN()
_C.INPUT.SOURCE_INPUT_SIZE_TRAIN = (1280, 720)
_C.INPUT.TARGET_INPUT_SIZE_TRAIN = (1024, 512)
_C.INPUT.INPUT_SIZE_TEST = (2048, 1024)
_C.INPUT.INPUT_SCALES_TRAIN = (1.0, 1.0)
_C.INPUT.IGNORE_LABEL = 255
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = False

# GaussianBlur
_C.INPUT.GAUSSIANBLUR = False

# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0

# RandomApply Transforms
_C.INPUT.RANDOMAPPLY = 0.0

# RandomGrayscale
_C.INPUT.GRAYSCALE = 0.0

# Flips
_C.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.0

_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.SOURCE_TRAIN = ""
_C.DATASETS.TARGET_TRAIN = ""
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ""

_C.SOLVER = CN()
_C.SOLVER.NUM_WORKERS = 4
_C.SOLVER.MAX_ITER = 16000
_C.SOLVER.STOP_ITER = 10000
_C.SOLVER.CHECKPOINT_PERIOD = 1000

_C.SOLVER.LR_METHOD = 'poly'
_C.SOLVER.BASE_LR = 0.02
_C.SOLVER.BASE_LR_D = 0.008
_C.SOLVER.LR_POWER = 0.9
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.EMA_DECAY = 0.99
_C.SOLVER.KD_WEIGHT = 1.0
_C.SOLVER.WEIGHT_DECAY = 0.0005

# Number of images per batch, if we have 4 GPUs and BATCH_SIZE = 8, each GPU will
# see 2 images per batch, one for source and one for target
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.BATCH_SIZE_VAL = 1


# Hyper-parameter
_C.SOLVER.MULTI_LEVEL = True
# lovasz_softmax loss
_C.SOLVER.LAMBDA_LOV = 0.0
# constant threshold for target mask
_C.SOLVER.DELTA = 0.9
# weight of feature level contrastive loss
_C.SOLVER.LAMBDA_FEAT = 1.0
# weight of output level contrastive loss
_C.SOLVER.LAMBDA_OUT = 1.0


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1

_C.OUTPUT_DIR = ""
_C.resume = ""
_C.PREPARE_DIR = ""
_C.CV_DIR = ""
