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
| Predicted Results | [Google Drive](https://drive.google.com/drive/folders/1xm8E8tGWO9J5nQT4yNdd9LlfExjctiDj?usp=sharing) | [Baidu Disk](https://pan.baidu.com/s/1qo8pDniGJfJFZ_o0XoYQaw?pwd=1210) |
| Pretrained Weights | [Google Drive](...) | [Baidu Disk](...) |


## Evaluation

To evaluate the model, run the following command:
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
