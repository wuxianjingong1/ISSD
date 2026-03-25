# ISSD
A Shadow Detection Framework Based on Illumination-Sensitive Modeling.

## Overview
![Model Architecture](figs/model.png)

## Datasets

In this work, we utilize five benchmark datasets for evaluation:

**(1) SBU**
The SBU dataset is available at [https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html).

**(2) UCF**
The UCF dataset is available at [https://drive.google.com/file/d/12DOmMVmE-oNuJVXmkBJrkfBvuDd0O70N/view](https://drive.google.com/file/d/12DOmMVmE-oNuJVXmkBJrkfBvuDd0O70N/view).

**(3) ISTD**
The ISTD dataset is available at [https://github.com/DeepInsight-PCALab/ST-CGAN](https://github.com/DeepInsight-PCALab/ST-CGAN).

**(4) SBUTestNew**
The SBUTestNew dataset is available at [https://github.com/hanyangclarence/SILT](https://github.com/hanyangclarence/SILT).

**(5) CUHK**
The CUHK dataset is not publicly available. Please contact the authors of [CUHK](https://github.com/xw-hu/CUHK-Shadow) to request access.


## Results & Weights
We provide the predicted results and pretrained weights of our model on the benchmark datasets.

| Resource | Google Drive | Baidu Disk |
|----------|-------------|------------|
| Predicted Results | [Google Drive](https://drive.google.com/drive/folders/1_1V_7gbTgc6ahzV1Yx1L-FcYBsJEGRIB?usp=sharing) | [Baidu Disk](https://pan.baidu.com/s/1wxv2v8LJtKT4HS8hfQOATg?pwd=1210) |
| Pretrained Weights | [Google Drive](https://drive.google.com/drive/folders/1nG3mrUkb_MNTb1N1FwgcTjCXdboys8j9?usp=sharing) | [Baidu Disk](https://pan.baidu.com/s/1nUa4NSeYFUX97Sdr-ay-MQ?pwd=1210) |


## Training
Before training, download the DINOv3 pretrained weights:

> ⚠️ **Note:** DINOv3 pretrained weights are required for training.
> Please download from [https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m)  and place them in `models/`.

Then run:
```bash
python train.py
```


## Evaluation
Before evaluating, make sure the pretrained weights are placed in `weights/`.

Then run:
```bash
python test.py
```


## Citation

If you find this work useful for your research, please consider citing our paper:
```bibtex
@article{lin2025issd,
  title={Illumination Matters: A Shadow Detection Framework via Illumination-Sensitive Modeling},
  author={Lin,Xiaofan},
  journal={xxx},
  year={2026}
}
```
