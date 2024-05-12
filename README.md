<p align="center">
  <a href="https://github.com/ESAOpenSR/opensr-test"><img src="docs/images/logo.png" alt="header" width="50%"></a>
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

**Paper**: [https://www.techrxiv.org/users/760184/articles/735467-a-comprehensive-benchmark-for-optical-remote-sensing-image-super-resolution](https://www.techrxiv.org/users/760184/articles/735467-a-comprehensive-benchmark-for-optical-remote-sensing-image-super-resolution)

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
>>> {'reflectance': 0.253, 'spectral': 26.967, 'spatial': 0.0, 'synthesis': 0.2870, 'ha_percent': 0.892, 'om_percent': 0.0613, 'im_percent': 0.04625}
```

This model returns:

- **reflectance**: How SR affects the reflectance values of the LR image. By default, it uses the L1 norm. The lower the value, the better the reflectance consistency.

- **spectral**: How SR affects the spectral signature of the LR image. By default, it uses the spectral angle distance (SAM). The lower the value, the better the spectral consistency. The angles are in degrees.

- **spatial**: The spatial alignment between the SR and LR images. By default, it uses Phase Correlation Coefficient (PCC). Some SR models introduce spatial shift, which can be detected by this metric.

- **synthesis**: The high-frequency details introduced by the SR model. By default, it uses the L1 norm. The lower the value, the better the synthesis quality.

- **ha_percent**: The percentage of pixels in the SR image that are classified as hallucinations. A hallucination is a detail in the SR image that **is not present in the HR image.**

- **om_percent**: The percentage of pixels in the SR image that are classified as omissions. An omission is a detail in the HR image that **is not present in the SR image.**

- **im_percent**: The percentage of pixels in the SR image that are classified as improvements. An improvement is a detail in the SR image that **is present in the HR image and not in the LR image.**

## **Benchmark**

Benchmark comparison of SR models. Downward arrows (↓) denote metrics in which lower values are preferable, and upward arrows (↑) indicate metrics in which higher values reflect better performance.


## **Installation**

Install the latest version from PyPI:

```
pip install opensr-test
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


## **Examples**

The following examples show how to use `opensr-test` to benchmark your SR model.

- Use `opensr-test` with TensorFlow model (SR4RS) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cAGDGlj5Kqt343inNni3ByLE1856z0gE#scrollTo=xaivkcD5Zfw1&uniqifier=1)

- Use `opensr-test` with PyTorch model (SuperImage) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Db8-JSMTF-hNZQv2UyBDclxkO5hgP9VR#scrollTo=jVL7o6yOrJkY)

- Use `opensr-test` with a diffuser model (LDMSuperResolutionPipeline) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1banDovG43c2OBh9MODPN4OXgaSCXu1Dc#scrollTo=zz4Aw7_52ulT)

- Use `opensr-test` with a diffuser model (opensr-model) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1banDovG43c2OBh9MODPN4OXgaSCXu1Dc#scrollTo=zz4Aw7_52ulT)


- Use `opensr-test` with Pytorch (EvoLand) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1banDovG43c2OBh9MODPN4OXgaSCXu1Dc#scrollTo=zz4Aw7_52ulT)


- Use `opensr-test` with Pytorch (SWIN2-MOSE) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1banDovG43c2OBh9MODPN4OXgaSCXu1Dc#scrollTo=zz4Aw7_52ulT)

- Use `opensr-test` with Pytorch (synthetic dataset) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1banDovG43c2OBh9MODPN4OXgaSCXu1Dc#scrollTo=zz4Aw7_52ulT)

## **Visualizations**

The `opensr-test` package provides a set of visualizations to help you understand the performance of your SR model.

```python
import torch
import opensr_test
import matplotlib.pyplot as plt

from super_image import HanModel

# Define the SR model
srmodel = HanModel.from_pretrained('eugenesiow/han', scale=4)

# Load the data
lr, hr, parameters = opensr_test.load("spot").values()

# Define the benchmark experiment
metrics = opensr_test.Metrics()

# Define the image to be tested
idx = 0
lr_img = torch.from_numpy(lr[idx, 0:3])
hr_img = torch.from_numpy(hr[idx, 0:3])
sr_img = srmodel(lr_img[None]).squeeze().detach()

# Compute the metrics
metrics.compute(
    lr=lr_img, sr=sr_img, hr=hr_img,
    gradient_threshold=parameters[idx]
)
```

Now, we can visualize the results using the `opensr_test.visualize` module.
fDisplay the triplets LR, SR and HR images:

```python
metrics.plot_triplets()
```

<p align="center">
  <img src="docs/images/example01.png">
</p>

Display a summary of all the metrics:

```python
metrics.plot_summary()
```

<p align="center">
  <img src="docs/images/example04.png">
</p>


Display the correctness of the SR image:

```python
metrics.plot_tc()
```

<p align="center">
  <img src="docs/images/example05.png">
</p>

## **Deeper understanding**

Explore the [API](https://esaopensr.github.io/opensr-test/docs/API/config_pydantic.html) section for more details about personalizing your benchmark experiments.

<p align="center">
    <a href="/docs/api.md"><img src="docs/images/image02.png" alt="opensr-test" width="30%"></a>
</p>

## **Citation**

If you use `opensr-test` in your research, please cite our paper:

```
@article{aybar2024comprehensive,
  title={A Comprehensive Benchmark for Optical Remote Sensing Image Super-Resolution},
  author={Aybar, Cesar and Montero, David and Donike, Simon and Kalaitzis, Freddie and G{\'o}mez-Chova, Luis},
  journal={Authorea Preprints},
  year={2024},
  publisher={Authorea}
}
```

## **Acknowledgements**

This work was make with the support of the European Space Agency (ESA) under the project “Explainable AI: application to trustworthy super-resolution (OpenSR)”. Cesar Aybar acknowledges support by the National Council of Science, Technology, and Technological Innovation (CONCYTEC, Peru) through the “PROYECTOS DE INVESTIGACIÓN BÁSICA – 2023-01” program with contract number PE501083135-2023-PROCIENCIA. Luis Gómez-Chova acknowledges support from the Spanish Ministry of Science and Innovation (project PID2019-109026RB-I00 funded by MCIN/AEI/10.13039/501100011033).
