<p align="center">
  <a href="https://github.com/ESAOpenSR/opensr-test"><img src="docs/images/logo.png" alt="header" width="55%"></a>
</p>

<p align="center">
    <em>A comprehensive benchmark for real-world Sentinel-2 imagery super-resolution</em>
</p>

<p align="center">
<a href='https://pypi.python.org/pypi/opensr-test'>
    <img src='https://img.shields.io/pypi/v/opensr-test.svg' alt='PyPI' />
</a>

<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href='https://opensr-test.readthedocs.io/en/latest/?badge=main'>
    <img src='https://readthedocs.org/projects/opensr-test/badge/?version=main' alt='Documentation Status' />
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

**Documentation**: [https://opensr-test.readthedocs.io/](https://opensr-test.readthedocs.io/)

**PyPI**: [https://pypi.org/project/opensr-test/](https://pypi.org/project/opensr-test/)

**Paper**: Coming soon!

---

#

## Overview

In remote sensing, Image Super-Resolution (ISR) goal is to improve the ground sampling distance. However, two problems are common in the literature. First, most models are **tested on synthetic data**, raising doubts about their real-world applicability and performance. Second, traditional evaluation metrics such as PSNR, LPIPS, and SSIM are not designed for assessing ISR performance. These metrics fall short, especially in conditions involving changes in luminance or spatial misalignments - scenarios that are frequently encountered in remote sensing imagery.

To address these challenges, 'opensr-test' provides a fair approach for ISR benchmark. We provide **three datasets** that were carefully crafted to minimize spatial and spectral misalignment. Besides, 'opensr-test' precisely assesses ISR algorithm performance across **three independent metrics groups** that measure *consistency*, *synthesis*, and *correctness*.

## How to use

The example below shows how to use `opensr-test` to benchmark your SR model.

```python
import torch
import opensr_test

lr = torch.rand(4, 64, 64)
hr = torch.rand(4, 256, 256)
sr = torch.rand(4, 256, 256)

metrics = opensr_test.Metrics()
metrics.compute(lr=lr, sr=sr, hr=hr)
```

## Installation

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

## Examples

The following examples show how to use `opensr-test` to benchmark your SR model.

- Use `opensr-test` with TensorFlow model [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cAGDGlj5Kqt343inNni3ByLE1856z0gE#scrollTo=xaivkcD5Zfw1&uniqifier=1)

- Use `opensr-test` with PyTorch model [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Db8-JSMTF-hNZQv2UyBDclxkO5hgP9VR#scrollTo=jVL7o6yOrJkY)

- Use `opensr-test` with a diffuser model [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1banDovG43c2OBh9MODPN4OXgaSCXu1Dc#scrollTo=zz4Aw7_52ulT)

- Use `opensr-test` to test a multi-image SR model (Satlas Super Resolution) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OlrYome8gcBH6Wu3SQhaw6mSlr2apWdV?usp=sharing#scrollTo=NOk0G3-BWonm)

## Visualizations

The `opensr-test` package provides a set of visualizations to help you understand the performance of your SR model.

```python
import torch
import opensr_test
import matplotlib.pyplot as plt

from super_image import HanModel

# Define the SR model
srmodel = HanModel.from_pretrained('eugenesiow/han', scale=4)

# Load the data
lr, hr, landuse, parameters = opensr_test.load("spot").values()

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
    stability_threshold = parameters.stability_threshold[idx],
    im_score = parameters.correctness_params[0],
    om_score = parameters.correctness_params[1],
    ha_score = parameters.correctness_params[2]
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

Display the quadruplets LR, SR, HR and landuse images:

```python
metrics.plot_quadruplets()
```

<p align="center">
  <img src="docs/images/example02.png">
</p>


Display the matching points between the LR and SR images:

```python
metrics.plot_spatial_matches()
```

<p align="center">
  <img src="docs/images/example03.png" width="70%">
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

## Deeper understanding

Explore the [API](/docs/API/model_parameters.md) section for more details about personalizing your benchmark experiments.

<p align="center">
    <a href="/docs/api.md"><img src="docs/images/image02.png" alt="opensr-test" width="30%"></a>
</p>

## Citation

If you use `opensr-test` in your research, please cite our paper:

```
Coming soon!
```

## Acknowledgements

This work was make with the support of the European Space Agency (ESA) under the project “Explainable AI: application to trustworthy super-resolution (OpenSR)”. Cesar Aybar acknowledges support by the National Council of Science, Technology, and Technological Innovation (CONCYTEC, Peru) through the “PROYECTOS DE INVESTIGACIÓN BÁSICA – 2023-01” program with contract number PE501083135-2023-PROCIENCIA. Luis Gómez-Chova acknowledges support from the Spanish Ministry of Science and Innovation (project PID2019-109026RB-I00 funded by MCIN/AEI/10.13039/501100011033).