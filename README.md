# Prototypical Contrast Adaptation for Domain Adaptive Semantic Segmentation (ECCV 2022)

This repository is for **ProCA** introduced in the following paper

[Zhengkai Jiang](https://jiangzhengkai.github.io/), [Yuxi Li](https://scholar.google.com/citations?user=-24oYQoAAAAJ&hl=en), [Ceyuan Yang](https://scholar.google.com/citations?user=Rfj4jWoAAAAJ&hl=en), [Peng Gao](https://openreview.net/profile?id=~Peng_Gao3), [Yabiao Wang](https://scholar.google.com/citations?user=xiK4nFUAAAAJ&hl=zh-CN), [Ying Tai](https://tyshiwo.github.io/), [Chengjie Wang](https://scholar.google.com/citations?user=fqte5H4AAAAJ&hl=zh-CN), "*Prototypical Contrast Adaptation for Domain Adaptive Semantic Segmentation*", ECCV 2022 [[arxiv]]().

### Prerequisites

- Python 3.6
- Pytorch 1.7.1
- torchvision from master
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 10.1

**Step-by-step installation**

```bash
conda create --name ProCA -y python=3.6
conda activate ProCA

# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

pip install ninja yacs cython matplotlib tqdm opencv-python imageio mmcv tqdm torchvision==0.8.2 torch==1.7.1
```

### Data Preparation

- Download [The Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- Download [The SYNTHIA-RAND-CITYSCAPES Dataset](http://synthia-dataset.net/download/808/)
- Download [The GTAV Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/)

**The data folder should be structured as follows:**

```
├── datasets/
│   ├── cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── synthia/
|   |   ├── RAND_CITYSCAPES/
|   |   ├── synthia_label_info.p
│   ├── gtav/
|   |   ├── images/
|   |   ├── labels/
|   |   ├── gtav_label_info.p
```

**Symlink the required dataset**

```bash
ln -s /path_to_cityscapes_dataset datasets/cityscapes
ln -s /path_to_synthia_dataset datasets/synthia
ln -s /path_to_gtav_dataset datasets/gtav
```

**Generate the label statics file for SYNTHIA and GTAV Datasets by running** 

```
python3 datasets/generate_synthia_label_info.py -d datasets/synthia -o datasets/synthia/
python3 datasets/generate_gtav_label_info.py -d datasets/gtav -o datasets/gtav/
```

### Inference Using Pretrained Model

<details>
  <summary>
    <b>(1) SYNTHIA -> Cityscapes</b>
  </summary>

Download the [pretrained model (ResNet-101)](https://pan.baidu.com/s/1o63hJ6cv0w0H3i4WgdRERQ) (52.0 mIoU of single scale, extraction code:3e9h ) and save it in `results/`. Then run the command 
```bash
python test.py -cfg configs/deeplabv2_r101_ssl_synthia.yaml resume results/model_proca_ssl.pth
```
multi-scale testing results should preduce [result](./docs/synthia2cityscape.png)


</details>


<details>
  <summary>
    <b>(2) GTAV -> Cityscapes</b>
  </summary>

Download the [pretrained model (ResNet-101)](https://pan.baidu.com/s/1FMnPF9Tc-ubecXl0r7-5eQ) (55.1 mIoU of sinle scale, extraction code: f2ve) and save it in `results/`. Then run the command 
```bash
python test.py -cfg configs/deeplabv2_r101_ssl.yaml resume results/model_proca_ssl.pth
```
multi-scale testing results should preduce [result](./docs/gtav2cityscape.png)
</details>


### Train

**We provide the training script using 4 Tesla V100 GPUs.**

```
bash run_proca_resnet101_gta5.sh
```
We also provide Memory-Bank implementation which can be seen in [memory_bank](train_memory_bank.py)

## Acknowledgements
This code is partly based on the open-source implementations from [FADA](https://github.com/JDAI-CV/FADA) and [SDCA](https://github.com/BIT-DA/SDCA).


## Citation
If you find this code or idea useful, please cite our work:
```bib
@inproceedings{jiang2022prototypical,
  title={Prototypical Contrast Adaptation for Domain Adaptive Segmentation},
  author={Jiang, Zhengkai and Li, Yuxi and Yang, Ceyuan and Gao, Peng and Wang, Yabiao and Tai, Ying and Wang, Chengjie},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
