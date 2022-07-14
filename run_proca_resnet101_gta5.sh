########### ResNet-101 GTA5 -> cityscapes
NGPUS=4
# train on source data
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train_src.py -cfg configs/gta5/deeplabv2_r101_src.yaml SOLVER.LAMBDA_LOV 0.75 OUTPUT_DIR results/gta5/r101_g2c_src SOLVER.BATCH_SIZE 8
python3 prototype_dist_init.py -cfg configs/gta5/deeplabv2_r101_src.yaml resume  results/gta5/r101_g2c_src/model_current.pth OUTPUT_DIR results/gta5/r101_g2c_src/ SOLVER.BATCH_SIZE_VAL 1
# train proca
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train_proca.py -cfg configs/gta5/deeplabv2_r101_proca.yaml resume results/gta5/r101_g2c_src/model_current.pth CV_DIR results/gta5/r101_g2c_src/ OUTPUT_DIR results/gta5/r101_g2c_ours_proca/ SOLVER.BATCH_SIZE 8
# generate pseudo label
python3 pseudo_label.py -cfg configs/gta5/deeplabv2_r101_proca.yaml DATASETS.TEST cityscapes_train resume results/gta5/r101_g2c_ours_proca/model_best.pth OUTPUT_DIR results/gta5/r101_g2c_ours_proca SOLVER.BATCH_SIZE_VAL 1
# train with generated pseudo label
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train_ssl.py -cfg configs/gta5/deeplabv2_r101_ssl.yaml PREPARE_DIR results/gta5/r101_g2c_ours_proca/ OUTPUT_DIR results/gta5/r101_g2c_ours_proca_ssl SOLVER.BATCH_SIZE 8

