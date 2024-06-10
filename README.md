<p align="center">
  <a href="https://github.com/ESAOpenSR/opensr-test"><img src="docs/images/logo.png" alt="header" width="45%"></a>
</p>

<p align="center">
    <em>
    A comprehensive benchmark for real-world Sentinel-2 imagery super-resolution
    </em>
</p>

<p align="center">
<a href='https://pypi.python.org/pypi/opensr-test'>
<img src='https://img.shields.io/pypi/v/opensr-test.svg' alt='PyPI' />
</a>
<a href='https://colab.research.google.com/drive/1wDD_d0RUnAZkJTf63_t9qULjPjtCmbuv?usp=sharing'>
<img src='https://colab.research.google.com/assets/colab-badge.svg' alt='COLAB' />
</a>
<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href="https://github.com/psf/black" target="_blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://pycqa.github.io/isort/" target="_blank">
    <img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" alt="isort">
</a>
</p>

---

**GitHub**: [https://github.com/ESAOpenSR/opensr-test](https://github.com/ESAOpenSR/opensr-test)

**Documentation**: [https://esaopensr.github.io/opensr-test](https://esaopensr.github.io/opensr-test)

**PyPI**: [https://pypi.org/project/opensr-test/](https://pypi.org/project/opensr-test/)

**Paper**: [https://ieeexplore.ieee.org/document/10530998](https://ieeexplore.ieee.org/document/10530998)

---

#

## **Overview**

Super-Resolution (SR) aims to improve satellite imagery ground sampling distance. However, two problems are common in the literature. First, most models are **tested on synthetic data**, raising doubts about their real-world applicability and performance. Second, traditional evaluation metrics such as PSNR, LPIPS, and SSIM are not designed to assess SR performance. These metrics fall short, especially in conditions involving changes in luminance or spatial misalignments - scenarios frequently encountered in real world.

To address these challenges, 'opensr-test' provides a fair approach for SR benchmark. We provide three datasets carefully crafted to minimize spatial and spectral misalignment. Besides, 'opensr-test' precisely assesses SR algorithm performance across three independent metrics groups that measure consistency, synthesis, and correctness.

<p align="center">
  <img src="docs/images/diagram.png" alt="header">
</p>

## **How to use**

The example below shows how to use `opensr-test` to benchmark your SR model.


```python
import torch
import opensr_test

lr = torch.rand(4, 64, 64)
hr = torch.rand(4, 256, 256)
sr = torch.rand(4, 256, 256)

metrics = opensr_test.Metrics()
metrics.compute(lr=lr, sr=sr, hr=hr)
>>> {'reflectance': 0.253, 'spectral': 26.967, 'spatial': 1.0, 'synthesis': 0.2870, 'ha_percent': 0.892, 'om_percent': 0.0613, 'im_percent': 0.04625}
```

This model returns:

- **reflectance**: How SR affects the reflectance norm of the LR image. By default, it uses the L1 distance. The lower the value, the better the reflectance consistency.

- **spectral**: How SR affects the spectral signature of the LR image. By default, it uses the spectral angle distance (SAD). The lower the value, the better the spectral consistency. The angles are in degrees.

- **spatial**: The spatial alignment between the SR and LR images. By default, it uses Phase Correlation Coefficient (PCC). Some SR models introduce spatial shift, which can be detected by this metric.

- **synthesis**: The high-frequency details introduced by the SR model. By default, it uses the L1 distance. The higher the value, the better the synthesis quality.

- **ha_metric**: The amount of hallucinations in the SR image. A hallucination is a detail (high-gradient) in the SR image that **is not present in the HR image.** The lower the value, the better the correctness of the SR image.

- **om_metric**: The amount of omissions in the SR image. An omission is a detail in the HR image that **is not present in the SR image.** The lower the value, the better the correctness of the SR image.

- **im_metric**: The amount of improvements in the SR image. An improvement is a detail in the SR image that **is present in the HR image and not in the LR image.** The higher the value, the better the correctness of the SR image.

## **Benchmark**

Benchmark comparison of SR models. Downward arrows (↓) denote metrics in which lower values are preferable, and upward arrows (↑) indicate that higher values reflect better performance. This table is an improved version of the one presented in the opensr-test paper, considering the latest version of the datasets and other distance metrics. To get the datasets used in the [original paper](https://ieeexplore.ieee.org/document/10530998), set the `dataset` parameter to `version=020` in the `opensr_test.load` function.

### **CLIP**

We use the [**RemoteCLIP**](https://github.com/ChenDelong1999/RemoteCLIP) model to measure the distance between the LR, SR, and HR images. The parameters of the experiments are: `{"device": "cuda", "agg_method": "patch", "patch_size": 16, "correctness_distance": "clip"}`. This distance metric allows us to quantify the amount of semantic information introduced by the SR model. By comparing the three categories, we can assess the accuracy of the semantic information introduced.


|    model     | reflectance ↓   | spectral ↓       | spatial ↓       | synthesis ↑     | ha_metric ↓     | om_metric ↓     | im_metric ↑     |
|:-------------|:----------------|:-----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| ldm_baseline | 0.1239 ± 0.0405 | 12.8441 ± 2.7508 | 0.0717 ± 0.0683 | 0.0409 ± 0.0290 | 0.5963 ± 0.3055 | 0.2327 ± 0.2238 | 0.1710 ± 0.1435 |
| opensrmodel  | 0.0076 ± 0.0046 | 1.9739 ± 1.0507  | 0.0118 ± 0.0108 | 0.0194 ± 0.0120 | 0.1052 ± 0.0590 | 0.6927 ± 0.1343 | 0.2021 ± 0.0854 |
| satlas       | 0.1197 ± 0.0233 | 15.1521 ± 2.9876 | 0.2766 ± 0.0741 | 0.0648 ± 0.0302 | 0.6996 ± 0.2058 | 0.0872 ± 0.0947 | 0.2132 ± 0.1393 |
| sr4rs        | 0.0979 ± 0.0509 | 22.4905 ± 2.1168 | 1.0099 ± 0.0439 | 0.0509 ± 0.0237 | 0.3099 ± 0.1704 | 0.3486 ± 0.1753 | 0.3415 ± 0.1042 |
| superimage   | 0.0068 ± 0.0016 | 1.8977 ± 1.1053  | 0.0004 ± 0.0032 | 0.0130 ± 0.0073 | 0.0610 ± 0.0305 | 0.8524 ± 0.0586 | 0.0866 ± 0.0395 |



### **LPIPS**

We use the [**LPIPS**](https://github.com/richzhang/PerceptualSimilarity) model to measure the distance between the LR, SR, and HR images. The parameters of the experiments are: `{"device": "cuda", "agg_method": "patch", "patch_size": 16, "correctness_distance": "lpips"}`. This distance metric allows us to quantify the amount of perceptual information introduced by the SR model. By comparing the three categories, we can assess the accuracy of the perceptual information introduced.


|    model     | reflectance ↓   | spectral ↓       | spatial ↓       | synthesis ↑     | ha_metric ↓     | om_metric ↓     | im_metric ↑     |
|:-------------|:----------------|:-----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| ldm_baseline | 0.1239 ± 0.0405 | 12.8441 ± 2.7508 | 0.0717 ± 0.0683 | 0.0409 ± 0.0290 | 0.4558 ± 0.2932 | 0.3558 ± 0.2518 | 0.1884 ± 0.1232 |
| opensrmodel  | 0.0076 ± 0.0046 | 1.9739 ± 1.0507  | 0.0118 ± 0.0108 | 0.0194 ± 0.0120 | 0.0642 ± 0.0271 | 0.6690 ± 0.1291 | 0.2668 ± 0.1071 |
| satlas       | 0.1197 ± 0.0233 | 15.1521 ± 2.9876 | 0.2766 ± 0.0741 | 0.0648 ± 0.0302 | 0.5999 ± 0.2182 | 0.0588 ± 0.0552 | 0.3413 ± 0.1858 |
| sr4rs        | 0.0979 ± 0.0509 | 22.4905 ± 2.1168 | 1.0099 ± 0.0439 | 0.0509 ± 0.0237 | 0.3417 ± 0.1833 | 0.1924 ± 0.1402 | 0.4659 ± 0.1448 |
| superimage   | 0.0068 ± 0.0016 | 1.8977 ± 1.1053  | 0.0004 ± 0.0032 | 0.0130 ± 0.0073 | 0.0357 ± 0.0200 | 0.8844 ± 0.0391 | 0.0800 ± 0.0301 |



### **Normalized Difference (ND)**

We use the normalized difference (ND) distance to measure the distance between the LR, SR, and HR images. In contrast to L1 distance, ND is less sensitive to the magnitude of the reflectance values, providing a more fair detection of the hallucinations and omissions when assessing at the pixel level. The parameters of the experiments are: `{"device": "cuda", "agg_method": "patch", "patch_size": 1, "correctness_distance": "nd"}`. This distance metric allows us to quantify high-frequency details introduced by the SR model at pixel level. However, at pixel level, small changes in the reflectance values could introduce noise in the hallucination, omission, and improvement estimation.


|    model     | reflectance ↓   | spectral ↓       | spatial ↓       | synthesis ↑     | ha_metric ↓     | om_metric ↓     | im_metric ↑     |
|:-------------|:----------------|:-----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| ldm_baseline | 0.0505 ± 0.0161 | 9.6923 ± 2.1742  | 0.0715 ± 0.0679 | 0.0285 ± 0.0307 | 0.6067 ± 0.2172 | 0.3088 ± 0.1786 | 0.0845 ± 0.0428 |
| opensrmodel  | 0.0031 ± 0.0018 | 1.2632 ± 0.5878  | 0.0114 ± 0.0111 | 0.0068 ± 0.0044 | 0.3431 ± 0.0738 | 0.4593 ± 0.0781 | 0.1976 ± 0.0328 |
| satlas       | 0.0489 ± 0.0086 | 12.1231 ± 3.1529 | 0.2742 ± 0.0748 | 0.0227 ± 0.0107 | 0.8004 ± 0.0641 | 0.1073 ± 0.0393 | 0.0923 ± 0.0266 |
| sr4rs        | 0.0396 ± 0.0198 | 3.4044 ± 1.6882  | 1.0037 ± 0.1520 | 0.0177 ± 0.0083 | 0.7274 ± 0.0840 | 0.1637 ± 0.0572 | 0.1089 ± 0.0292 |
| superimage   | 0.0029 ± 0.0009 | 1.5672 ± 1.0692  | 0.0132 ± 0.1131 | 0.0046 ± 0.0027 | 0.2026 ± 0.0692 | 0.6288 ± 0.0754 | 0.1686 ± 0.0302 |


To reproduce the results, check this [**Colab notebook**](https://colab.research.google.com/drive/1wDD_d0RUnAZkJTf63_t9qULjPjtCmbuv).

## **Installation**

Install the latest version from PyPI:

```
pip install opensr-test
pip install opensr-test[perceptual] # Install to test perceptual metrics
```

Upgrade `opensr-test` by running:

```
pip install -U opensr-test
```

Install the latest dev version from GitHub by running:

```
pip install git+https://github.com/ESAOpenSR/opensr-test
```

## **Datasets**

The `opensr-test` package provides five datasets for benchmarking SR models. These datasets are carefully crafted to minimize spatial and spectral misalignment. See our Hugging Face repository for more details about the datasets. [**https://huggingface.co/datasets/isp-uv-es/opensr-test**](https://huggingface.co/datasets/isp-uv-es/opensr-test)

| Dataset | Scale factor | Number of images | HR patch size |
|---------|--------------|-------------------|--------------|
| NAIP    | x4           | 62               | 484x484       |
| SPOT    | x4           | 9                | 512x512       |
| Venµs   | x2           | 59               | 256x256       |
| SPAIN CROPS | x4       | 28               | 512x512       |
| SPAIN URBAN | x4       | 20               | 512x512       |


### **NAIP (X4 scale factor)**

The National Agriculture Imagery Program (NAIP) dataset is a high-resolution aerial imagery dataset that covers the continental United States. The dataset consists of 2.5m NAIP imagery captured in the visible and near-infrared spectrum (RGBNIR) and all Sentinel-2 L1C and L2A bands. The dataset focus in **crop fields, forests, and bare soil areas**.

```python
import opensr_test

naip = opensr_test.load("naip")
```

<p align="center">
  <a href="https://github.com/ESAOpenSR/opensr-test"><img src="docs/images/NAIP.gif" alt="header" width="80%"></a>
</p>

### **SPOT (X4 scale factor)**

The SPOT imagery were obtained from the worldstat dataset. The dataset consists of 2.5m SPOT imagery captured in the visible and near-infrared spectrum (RGBNIR) and all Sentinel-2 L1C and L2A bands. The dataset focus in **urban areas, crop fields, and bare soil areas**.

```python
import opensr_test

spot = opensr_test.load("spot")
```

<p align="center">
  <a href="https://github.com/ESAOpenSR/opensr-test"><img src="docs/images/SPOT.gif" alt="header" width="80%"></a>
</p>


### **Venµs (X2 scale factor)**

The Venµs images were obtained from the [**Sen2Venµs dataset**](https://zenodo.org/records/6514159). The dataset consists of 5m Venµs imagery captured in the visible and near-infrared spectrum (RGBNIR) and all Sentinel-2 L1C and L2A bands. The dataset focus in **crop fields, forests, urban areas, and bare soil areas**.

```python
import opensr_test

venus = opensr_test.load("venus")
```

<p align="center">
  <a href="https://github.com/ESAOpenSR/opensr-test"><img src="docs/images/VENUS.gif" alt="header" width="80%"></a>
</p>


### **SPAIN CROPS (x4 scale factor)**

The SPAIN CROPS dataset consists of 2.5m aerial imagery captured in the visible and near-infrared spectrum (RGBNIR) by the Spanish National Geographic Institute (IGN). The dataset includes all Sentinel-2 L1C and L2A bands. The dataset focus in **crop fields and forests**.

```python
import opensr_test

spain_crops = opensr_test.load("spain_crops")
```

<p align="center">
  <a href="https://github.com/ESAOpenSR/opensr-test"><img src="docs/images/SPAIN_CROPS.gif" alt="header" width="80%"></a>
</p>


### **SPAIN URBAN (x4 scale factor)**

The SPAIN URBAN dataset consists of 2.5m aerial imagery captured in the visible and near-infrared spectrum (RGBNIR) by the Spanish National Geographic Institute (IGN). The dataset includes all Sentinel-2 L1C and L2A bands. The dataset focus in **urban areas**.

```python

spain_urban = opensr_test.load("spain_urban")
```

<p align="center">
  <a href="https://github.com/ESAOpenSR/opensr-test"><img src="docs/images/SPAIN_URBAN.gif" alt="header" width="80%"></a>
</p>


## **Examples**

The following examples show how to use `opensr-test` to benchmark SR models.

| Model | Framework | Link |
|-------|-----------|------|
| SR4RS | TensorFlow | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/isp-uv-es/superIX/blob/main/sr4rs/run.py) |
| SuperImage | PyTorch | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/isp-uv-es/superIX/blob/main/superimage/run.py) |
| LDMSuperResolutionPipeline | Diffuser | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/isp-uv-es/superIX/blob/main/ldm_baseline/run.py) |
| opensr-model | Pytorch | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/isp-uv-es/superIX/blob/main/opensrmodel/run.py) |
| EvoLand | Pytorch | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/isp-uv-es/superIX/blob/main/evoland/run.py) |
| SWIN2-MOSE | Pytorch | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/isp-uv-es/superIX/blob/main/swin2_mose/run.py) |

## **Visualizations**

The `opensr-test` package provides a set of visualizations to help you understand the performance of your SR model.

```python
import torch
import opensr_test
import matplotlib.pyplot as plt

from super_image import HanModel

# Define the SR model
srmodel = HanModel.from_pretrained('eugenesiow/han', scale=4)
srmodel.eval()

# Load the data
dataset = opensr_test.load("spot")
lr, hr = dataset["L2A"], dataset["HRharm"]

# Define the benchmark experiment
config = opensr_test.Config()
metrics = opensr_test.Metrics(config=config)

# Define the image to be tested
idx = 0
lr_img = torch.from_numpy(lr[idx, 0:3]) / 10000
hr_img = torch.from_numpy(hr[idx, 0:3]) / 10000
with torch.no_grad():
    sr_img = srmodel(lr_img[None]).squeeze()

# Compute the metrics
metrics.compute(
    lr=lr_img, sr=sr_img, hr=hr_img
)
```

Now, we can visualize the results using the `opensr_test.plot_*` module. To display the triplets LR, SR and HR images:

```python
metrics.plot_triplets()
```

<p align="center">
  <img src="docs/images/img_01.gif">
</p>

Display the summary of the metrics. The plot shows:
  - LR: Low Resolution image 
  - LRdown: Downsampled Low Resolution image using bilinear interpolation and triangular antialiasing filter.
  - SR: Super-Resolved image.
  - SRharm: Harmonized super-resolution image.
  - HR: High Resolution image.
  - Reflectance Consistency: Reflectance consistency between the LR and HR images.
  - Spectral Consistency: Spectral consistency between the LR and HR images.
  - Distance normalized to the Omissiom, Hallucination, and Improvement spaces.

```python
metrics.plot_summary()
```

<p align="center">
  <img src="docs/images/img_02.gif">
</p>


Display the correctness of the SR image. The blue color represents the pixels closer to the improvement space (HR), green pixels are closer to the omission space (LR), and red pixels are closer to the hallucination space (neither in HR nor LR). The distance SR-LR and SR-HR are normalized to the LR-HR distance that is independent of the SR model. Threfore a pixel or patch with a improvement distance ($d_{im}$) of 3 means that the SR is further to the HR in 3 LR-HR units. The same applies to the omission distance.

```python
metrics.plot_tc()
```

<p align="center">
  <img src="docs/images/img_03.gif">
</p>


Display the histogram of the distances before and after the normalization:

```python
metrics.plot_stats()
```

<p align="center">
  <img src="docs/images/img_04.gif">
</p>

Display a ternary plot of the metrics:

```python
metrics.plot_ternary()
```

<p align="center">
  <img src="docs/images/img_05.gif">
</p>

## **Deeper understanding**

Explore the [**API**](https://esaopensr.github.io/opensr-test/docs/API/config_pydantic.html) section for more details about personalizing your benchmark experiments.

## **Citation**

If you use `opensr-test` in your research, please cite our paper:

```
@ARTICLE{10530998,
  author={Aybar, Cesar and Montero, David and Donike, Simon and Kalaitzis, Freddie and Gómez-Chova, Luis},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={A Comprehensive Benchmark for Optical Remote Sensing Image Super-Resolution}, 
  year={2024},
  volume={21},
  number={},
  pages={1-5},
  keywords={Measurement;Remote sensing;Spatial resolution;Superresolution;Reflectivity;Protocols;Inspection;Benchmarking;datasets;deep learning;NAIP;Sentinel-2 (S2);SPOT;super-resolution (SR)},
  doi={10.1109/LGRS.2024.3401394}}
```

## **Acknowledgements**

This work was make with the support of the European Space Agency (ESA) under the project “Explainable AI: application to trustworthy super-resolution (OpenSR)”. Cesar Aybar acknowledges support by the National Council of Science, Technology, and Technological Innovation (CONCYTEC, Peru) through the “PROYECTOS DE INVESTIGACIÓN BÁSICA – 2023-01” program with contract number PE501083135-2023-PROCIENCIA. Luis Gómez-Chova acknowledges support from the Spanish Ministry of Science and Innovation (project PID2019-109026RB-I00 funded by MCIN/AEI/10.13039/501100011033).
