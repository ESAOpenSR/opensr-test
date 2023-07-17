Welcome to srcheck!
==================

.. raw:: html

   <embed>
     <p align="center">
       <a href="https://github.com/csaybar/srcheck"><img src="https://github.com/ESAOpenSR/opensr-test/assets/16768318/15661226-f4c4-4d55-8dd1-d73a228e36da" height="350px"/></a>
       <br>
       <b>A comprehensive benchmark for real-world Sentinel-2 imagery super-resolution
       </a>
       </b>
     </p>
   </embed>

.. image:: https://img.shields.io/pypi/v/opensr-test.svg
        :target: https://pypi.python.org/pypi/opensr-test
        :alt: PyPI Version
       
.. image:: https://img.shields.io/badge/License-MIT-blue.svg
        :target: https://opensource.org/licenses/MIT
        :alt: License
        
.. image:: https://readthedocs.org/projects/opensr-test/badge/?version=main
        :target: https://opensr-test.readthedocs.io/en/main/
        :alt: Documentation Status
                      
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black
        :alt: Black

.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
        :target: https://pycqa.github.io/isort/
        :alt: isort


.. toctree::   
   :maxdepth: 2
   :caption: Extended Classes 
   :hidden:      
   
   classes/index
   
.. toctree::
   :maxdepth: 2
   :caption: User Guide      
   :hidden:
      
   guide/index
      
.. toctree::
   :maxdepth: 2
   :caption: What's new? 
   :hidden:
      
   changelog
   
.. toctree::
   :maxdepth: 2
   :caption: Tutorials 
   :hidden:
      
   tutorials
   
.. toctree::
   :maxdepth: 2
   :caption: Contributors 
   :hidden:
      
   contributors


Overview
=================

In the domain of remote sensing, image super-resolution (ISR) goal is augmenting the ground sampling distance, also 
known as spatial resolution. Over recent years, numerous research papers have been proposed addressing ISR; however, 
they invariably suffer from two main issues. Firstly, the majority of these proposed models are tested on synthetic 
data. As a result, their applicability and performance in real-world scenarios remain unverified. Secondly, the 
frequently utilized evaluation metrics for these models, such as LPIPS and SSIM, are inherently designed for perceptual 
image analysis, not specifically for super-resolution.

In response to these challenges, **'opensr-test'** has been introduced as a Python package, designed to provide users with 
three meticulously curated test datasets. These datasets are tailored to minimize spatial distortions between Low Resolution 
(LR) and High Resolution (HR) images, while calibrating differences in the spectral domain. Moreover, the **'opensr-test'** 
package offers five distinct types of metrics aimed at accurately evaluating the performance of ISR algorithms, thus addressing 
the aforementioned shortcomings of conventional evaluation techniques.


Datasets
=================

The utilization of *synthetic data* in benchmarking could potentially introduce biased conclusions, as there are 
no guarantees that a degradation method could not inadvertently incorporate some form of bias, i.e. the degradation 
does not match the real-world ground sampling distance of the LR image. Therefore, to avoid *synthetic data* potential 
errors, the datasets utilized in **opensr-test** are meticulously created following a *cross-sensor approach* that means 
that HR and LR comes from different sensor but they are aligned as closely as possible, i.e. harmonized. Due to the limited 
availability of open HR resolution data that correspond with Sentinel-2, we propose using three HR sensors: SPOT, VENµS, and 
NAIP.


+----------------+-------+----------+--------+--------------------------------------------------------------------------------------------+
| Dataset        | Scale | # Scenes | HRsize | HR Reference                                                                               |
+================+=======+==========+========+============================================================================================+
| NAIP-19        | x4    | 200      | 512    | USDA Farm Production and Conservation - Business Center, Geospatial Enterprise Operations. |
| SPOTPAN-10     | x4    | 200      | 512    | European Space Agency, 2017, SPOT 1-5 ESA                                                  |
| Mini-SEN2VENµS | x2    | 200      | 512    | Vegetation and Environment monitoring on a New Micro-Satellite (SEN2VENμS).                |
+----------------+-------+----------+--------+--------------------------------------------------------------------------------------------+


Metrics
=================

We propose that evaluating the ISR process with a single metric is not adequate, and therefore, we suggest the use of five metrics, which will be detailed below.


.. raw:: html
   
   <details>
      <summary><b>Spectral consistency</b></summary>
      Measured based on the comparison of LR and SR images. The reflectance values are then compared using spectral angle distance metrics. The LR and degraded SR values should not be identical but rather similar.
   </details>

   <details>
      <summary><b>Spatial consistency</b></summary>
      Measured again from the comparison of the LR and SR images, ground control points are obtained using LightGlue + DISK. The difference between the reference points and a first-order polynomial is then calculated.
   </details>

   <details>
      <summary><b>High-frequency</b></summary>
      Measured from the comparison of LR and SR images, we calculate the MTF between the LR-SR pairs to assess the improvement in ground sampling distance. The MTF curve represents the system's response in different spatial frequencies. A higher MTF value at a specific frequency indicates a better resolution and sharper image details. The reported metric is the comparison of the area under the MTF curve between the LR and SR images, from the Nyquist frequency of Sentinel2 to the super-resolved image.
   </details>

   <details>
      <summary><b>Omission</b></summary>
      Error related to the inability to represent the actual high-frequency information from the landscape. The systematic error must be first removed.
   </details>

   <details>
      <summary><b>Hallucinations</b></summary>
      Hallucinations refer to errors that are solely related to high-frequency information inducted by the super-resolution model, with no correlation to the real continuous space. 
   </details>


   <img src="https://github.com/ESAOpenSR/opensr-test/assets/16768318/473865e5-5661-4b6a-b941-ae170ffd6d0e" alt="isort"  width="55%">


Installation
=================

Install the latest version from PyPI:

.. code-block::
      
   pip install opensr-test

Upgrade `opensr-test` by running:

.. code-block::
      
   pip install -U opensr-test

Install the latest dev version from GitHub by running:

.. code-block::
   pip install git+https://github.com/ESAOpenSR/opensr-test


How does it work?
=================

`opensr-test` needs either a PyTorch (`torch.nn.Module`, `torch.jit.trace` or `torch.jit.script`) 
or TensorFlow model. The following example shows how to run the benchmarks:

.. raw:: html
   
   <embed>
     <p align="center">
       <img src="https://github.com/ESAOpenSR/opensr-test/assets/16768318/60a34d0a-f7ab-4c52-b68c-978e51898733"/>
     </p>
   </embed>

The following example shows how to run the benchmarks:

.. code-block::
   
   import torch
   import opensr_test

   # Load your model
   model = torch.jit.load('/content/quantSRmodel.pt', map_location='cpu')

   # Check if the model works
   dataset = opensr_test.naip

   opensr_test.check(model, dataset)
   # { 'Spectral': 0.08, 'Spatial': 2.34, 'High-frequency': 1.32, 'Hallucination': 0.90, 'Omission': 1.03}



Pre-trained Models
==================

Pre-trained models are available at various scales and hosted at the
awesome
```huggingface_hub`` <https://huggingface.co/models?filter=super-image>`__.
By default the models were pretrained on
`DIV2K <https://huggingface.co/datasets/eugenesiow/Div2k>`__, a dataset
of 800 high-quality (2K resolution) images for training, augmented to
4000 images and uses a dev set of 100 validation images (images numbered
801 to 900).

The leaderboard below shows the
`PSNR <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Quality_estimation_with_PSNR>`__
/
`SSIM <https://en.wikipedia.org/wiki/Structural_similarity#Algorithm>`__
metrics for each model at various scales on various test sets
(`Set5 <https://huggingface.co/datasets/eugenesiow/Set5>`__,
`Set14 <https://huggingface.co/datasets/eugenesiow/Set14>`__,
`BSD100 <https://huggingface.co/datasets/eugenesiow/BSD100>`__,
`Urban100 <https://huggingface.co/datasets/eugenesiow/Urban100>`__). The
**higher the better**. All training was to 1000 epochs (some
publications, like a2n, train to >1000 epochs in their experiments).



Scale x2
--------

+---------+---------+---------+---------+---------+---------+---------+
| Rank    | Model   | Params  | Set5    | Set14   | BSD100  | U       |
|         |         |         |         |         |         | rban100 |
+=========+=========+=========+=========+=========+=========+=========+
| 1       | `       | 34m     | **      | 33.95   | **      | **      |
|         | drln-ba |         | 38.23/0 | /0.9206 | 33.95/0 | 32.81/0 |
|         | m <http |         | .9614** |         | .9269** | .9339** |
|         | s://hug |         |         |         |         |         |
|         | gingfac |         |         |         |         |         |
|         | e.co/eu |         |         |         |         |         |
|         | genesio |         |         |         |         |         |
|         | w/drln- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 2       | `edsr < | 41m     | 38.19   | **      | 33.89   | 32.68   |
|         | https:/ |         | /0.9612 | 33.99/0 | /0.9266 | /0.9331 |
|         | /huggin |         |         | .9215** |         |         |
|         | gface.c |         |         |         |         |         |
|         | o/eugen |         |         |         |         |         |
|         | esiow/e |         |         |         |         |         |
|         | dsr>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 3       | `msrn < | 5.9m    | 38.08   | 33.75   | 33.82   | 32.14   |
|         | https:/ |         | /0.9609 | /0.9183 | /0.9258 | /0.9287 |
|         | /huggin |         |         |         |         |         |
|         | gface.c |         |         |         |         |         |
|         | o/eugen |         |         |         |         |         |
|         | esiow/m |         |         |         |         |         |
|         | srn>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 4       | `mdsr < | 2.7m    | 38.04   | 33.71   | 33.79   | 32.14   |
|         | https:/ |         | /0.9608 | /0.9184 | /0.9256 | /0.9283 |
|         | /huggin |         |         |         |         |         |
|         | gface.c |         |         |         |         |         |
|         | o/eugen |         |         |         |         |         |
|         | esiow/m |         |         |         |         |         |
|         | dsr>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 5       | `       | 5.9m    | 38.02   | 33.73   | 33.78   | 32.08   |
|         | msrn-ba |         | /0.9608 | /0.9186 | /0.9253 | /0.9276 |
|         | m <http |         |         |         |         |         |
|         | s://hug |         |         |         |         |         |
|         | gingfac |         |         |         |         |         |
|         | e.co/eu |         |         |         |         |         |
|         | genesio |         |         |         |         |         |
|         | w/msrn- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 6       | `ed     | 1.5m    | 38.02   | 33.66   | 33.77   | 32.04   |
|         | sr-base |         | /0.9607 | /0.9180 | /0.9254 | /0.9276 |
|         |  <https |         |         |         |         |         |
|         | ://hugg |         |         |         |         |         |
|         | ingface |         |         |         |         |         |
|         | .co/eug |         |         |         |         |         |
|         | enesiow |         |         |         |         |         |
|         | /edsr-b |         |         |         |         |         |
|         | ase>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 7       | `       | 2.7m    | 38      | 33.68   | 33.77   | 32.04   |
|         | mdsr-ba |         | /0.9607 | /0.9182 | /0.9253 | /0.9272 |
|         | m <http |         |         |         |         |         |
|         | s://hug |         |         |         |         |         |
|         | gingfac |         |         |         |         |         |
|         | e.co/eu |         |         |         |         |         |
|         | genesio |         |         |         |         |         |
|         | w/mdsr- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 8       | `aw     | 1.4m    | 37.99   | 33.6    | 33.76   | 31.95   |
|         | srn-bam |         | /0.9606 | 6/0.918 | /0.9253 | /0.9265 |
|         |  <https |         |         |         |         |         |
|         | ://hugg |         |         |         |         |         |
|         | ingface |         |         |         |         |         |
|         | .co/eug |         |         |         |         |         |
|         | enesiow |         |         |         |         |         |
|         | /awsrn- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 9       | `a2n    | 1.0m    | 37.87   | 33.54   | 33.67   | 31.71   |
|         | <https: |         | /0.9602 | /0.9171 | /0.9244 | /0.9240 |
|         | //huggi |         |         |         |         |         |
|         | ngface. |         |         |         |         |         |
|         | co/euge |         |         |         |         |         |
|         | nesiow/ |         |         |         |         |         |
|         | a2n>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 10      | `carn < | 1.6m    | 37.89   | 33.53   | 33.66   | 31.62   |
|         | https:/ |         | /0.9602 | /0.9173 | /0.9242 | /0.9229 |
|         | /huggin |         |         |         |         |         |
|         | gface.c |         |         |         |         |         |
|         | o/eugen |         |         |         |         |         |
|         | esiow/c |         |         |         |         |         |
|         | arn>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 11      | `       | 1.6m    | 37.     | 33.51   | 33.6    | 31.5    |
|         | carn-ba |         | 83/0.96 | /0.9166 | 4/0.924 | 3/0.922 |
|         | m <http |         |         |         |         |         |
|         | s://hug |         |         |         |         |         |
|         | gingfac |         |         |         |         |         |
|         | e.co/eu |         |         |         |         |         |
|         | genesio |         |         |         |         |         |
|         | w/carn- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 12      | `pan    | 260k    | 37.77   | 33.42   | 33.6    | 31.31   |
|         | <https: |         | /0.9599 | /0.9162 | /0.9235 | /0.9197 |
|         | //huggi |         |         |         |         |         |
|         | ngface. |         |         |         |         |         |
|         | co/euge |         |         |         |         |         |
|         | nesiow/ |         |         |         |         |         |
|         | pan>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 13      | `pan-b  | 260k    | 37.7    | 33.4    | 33.6    | 31.     |
|         | am <htt |         | /0.9596 | /0.9161 | /0.9234 | 35/0.92 |
|         | ps://hu |         |         |         |         |         |
|         | ggingfa |         |         |         |         |         |
|         | ce.co/e |         |         |         |         |         |
|         | ugenesi |         |         |         |         |         |
|         | ow/pan- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+

Scale x3
--------

+---------+---------+---------+---------+---------+---------+---------+
| Rank    | Model   | Params  | Set5    | Set14   | BSD100  | U       |
|         |         |         |         |         |         | rban100 |
+=========+=========+=========+=========+=========+=========+=========+
| 1       | `       | 34m     | 35.3    | **      | **      | **      |
|         | drln-ba |         | /0.9422 | 31.27/0 | 29.78/0 | 29.82/0 |
|         | m <http |         |         | .8624** | .8224** | .8828** |
|         | s://hug |         |         |         |         |         |
|         | gingfac |         |         |         |         |         |
|         | e.co/eu |         |         |         |         |         |
|         | genesio |         |         |         |         |         |
|         | w/drln- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 1       | `edsr < | 44m     | **      | 31.1    | 29.77   | 29.75   |
|         | https:/ |         | 35.31/0 | 8/0.862 | /0.8224 | /0.8825 |
|         | /huggin |         | .9421** |         |         |         |
|         | gface.c |         |         |         |         |         |
|         | o/eugen |         |         |         |         |         |
|         | esiow/e |         |         |         |         |         |
|         | dsr>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 1       | `msrn < | 6.1m    | 35.12   | 31.08   | 29.67   | 29.31   |
|         | https:/ |         | /0.9409 | /0.8593 | /0.8198 | /0.8743 |
|         | /huggin |         |         |         |         |         |
|         | gface.c |         |         |         |         |         |
|         | o/eugen |         |         |         |         |         |
|         | esiow/m |         |         |         |         |         |
|         | srn>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 2       | `mdsr < | 2.9m    | 35.11   | 31.06   | 29.66   | 29.29   |
|         | https:/ |         | /0.9406 | /0.8593 | /0.8196 | /0.8738 |
|         | /huggin |         |         |         |         |         |
|         | gface.c |         |         |         |         |         |
|         | o/eugen |         |         |         |         |         |
|         | esiow/m |         |         |         |         |         |
|         | dsr>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 3       | `       | 5.9m    | 35.13   | 31.06   | 29.65   | 29.26   |
|         | msrn-ba |         | /0.9408 | /0.8588 | /0.8196 | /0.8736 |
|         | m <http |         |         |         |         |         |
|         | s://hug |         |         |         |         |         |
|         | gingfac |         |         |         |         |         |
|         | e.co/eu |         |         |         |         |         |
|         | genesio |         |         |         |         |         |
|         | w/msrn- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 4       | `       | 2.9m    | 35.07   | 31.04   | 29.62   | 29.16   |
|         | mdsr-ba |         | /0.9402 | /0.8582 | /0.8188 | /0.8717 |
|         | m <http |         |         |         |         |         |
|         | s://hug |         |         |         |         |         |
|         | gingfac |         |         |         |         |         |
|         | e.co/eu |         |         |         |         |         |
|         | genesio |         |         |         |         |         |
|         | w/mdsr- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 5       | `ed     | 1.5m    | 35.01   | 31.01   | 29.63   | 29.19   |
|         | sr-base |         | /0.9402 | /0.8583 | /0.8190 | /0.8722 |
|         |  <https |         |         |         |         |         |
|         | ://hugg |         |         |         |         |         |
|         | ingface |         |         |         |         |         |
|         | .co/eug |         |         |         |         |         |
|         | enesiow |         |         |         |         |         |
|         | /edsr-b |         |         |         |         |         |
|         | ase>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 6       | `aw     | 1.5m    | 35.05   | 31.01   | 29.63   | 29.1    |
|         | srn-bam |         | /0.9403 | /0.8581 | /0.8188 | 4/0.871 |
|         |  <https |         |         |         |         |         |
|         | ://hugg |         |         |         |         |         |
|         | ingface |         |         |         |         |         |
|         | .co/eug |         |         |         |         |         |
|         | enesiow |         |         |         |         |         |
|         | /awsrn- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 7       | `carn < | 1.6m    | 34.88   | 30.93   | 29.56   | 28.9    |
|         | https:/ |         | /0.9391 | /0.8566 | /0.8173 | 5/0.867 |
|         | /huggin |         |         |         |         |         |
|         | gface.c |         |         |         |         |         |
|         | o/eugen |         |         |         |         |         |
|         | esiow/c |         |         |         |         |         |
|         | arn>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 8       | `a2n    | 1.0m    | 34.8    | 30.94   | 29.56   | 28.95   |
|         | <https: |         | /0.9387 | /0.8568 | /0.8173 | /0.8671 |
|         | //huggi |         |         |         |         |         |
|         | ngface. |         |         |         |         |         |
|         | co/euge |         |         |         |         |         |
|         | nesiow/ |         |         |         |         |         |
|         | a2n>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 9       | `       | 1.6m    | 34.82   | 30.9    | 29.54   | 28.84   |
|         | carn-ba |         | /0.9385 | /0.8558 | /0.8166 | /0.8648 |
|         | m <http |         |         |         |         |         |
|         | s://hug |         |         |         |         |         |
|         | gingfac |         |         |         |         |         |
|         | e.co/eu |         |         |         |         |         |
|         | genesio |         |         |         |         |         |
|         | w/carn- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 10      | `pan-b  | 260k    | 34.62   | 30.83   | 29.47   | 28.6    |
|         | am <htt |         | /0.9371 | /0.8545 | /0.8153 | 4/0.861 |
|         | ps://hu |         |         |         |         |         |
|         | ggingfa |         |         |         |         |         |
|         | ce.co/e |         |         |         |         |         |
|         | ugenesi |         |         |         |         |         |
|         | ow/pan- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 11      | `pan    | 260k    | 34.64   | 30.8    | 29.4    | 28.61   |
|         | <https: |         | /0.9376 | /0.8544 | 7/0.815 | /0.8603 |
|         | //huggi |         |         |         |         |         |
|         | ngface. |         |         |         |         |         |
|         | co/euge |         |         |         |         |         |
|         | nesiow/ |         |         |         |         |         |
|         | pan>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+

Scale x4
--------

+---------+---------+---------+---------+---------+---------+---------+
| Rank    | Model   | Params  | Set5    | Set14   | BSD100  | U       |
|         |         |         |         |         |         | rban100 |
+=========+=========+=========+=========+=========+=========+=========+
| 1       | `drln < | 35m     | *       | **      | **      | **      |
|         | https:/ |         | *32.55/ | 28.96/0 | 28.65/0 | 26.56/0 |
|         | /huggin |         | 0.899** | .7901** | .7692** | .7998** |
|         | gface.c |         |         |         |         |         |
|         | o/eugen |         |         |         |         |         |
|         | esiow/d |         |         |         |         |         |
|         | rln>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 2       | `       | 34m     | 32.49   | 28.94   | 28.63   | 26.53   |
|         | drln-ba |         | /0.8986 | /0.7899 | /0.7686 | /0.7991 |
|         | m <http |         |         |         |         |         |
|         | s://hug |         |         |         |         |         |
|         | gingfac |         |         |         |         |         |
|         | e.co/eu |         |         |         |         |         |
|         | genesio |         |         |         |         |         |
|         | w/drln- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 3       | `edsr < | 43m     | 32.5    | 28.92   | 28.62   | 26.53   |
|         | https:/ |         | /0.8986 | /0.7899 | /0.7689 | /0.7995 |
|         | /huggin |         |         |         |         |         |
|         | gface.c |         |         |         |         |         |
|         | o/eugen |         |         |         |         |         |
|         | esiow/e |         |         |         |         |         |
|         | dsr>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 4       | `msrn < | 6.1m    | 32.19   | 28.78   | 28.53   | 26.12   |
|         | https:/ |         | /0.8951 | /0.7862 | /0.7657 | /0.7866 |
|         | /huggin |         |         |         |         |         |
|         | gface.c |         |         |         |         |         |
|         | o/eugen |         |         |         |         |         |
|         | esiow/m |         |         |         |         |         |
|         | srn>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 5       | `       | 5.9m    | 32.26   | 28.78   | 28.51   | 26.10   |
|         | msrn-ba |         | /0.8955 | /0.7859 | /0.7651 | /0.7857 |
|         | m <http |         |         |         |         |         |
|         | s://hug |         |         |         |         |         |
|         | gingfac |         |         |         |         |         |
|         | e.co/eu |         |         |         |         |         |
|         | genesio |         |         |         |         |         |
|         | w/msrn- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 6       | `mdsr < | 2.8m    | 32.26   | 28.77   | 28.53   | 26.07   |
|         | https:/ |         | /0.8953 | /0.7856 | /0.7653 | /0.7851 |
|         | /huggin |         |         |         |         |         |
|         | gface.c |         |         |         |         |         |
|         | o/eugen |         |         |         |         |         |
|         | esiow/m |         |         |         |         |         |
|         | dsr>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 7       | `       | 2.9m    | 32.19   | 28.73   | 28.50   | 26.02   |
|         | mdsr-ba |         | /0.8949 | /0.7847 | /0.7645 | /0.7834 |
|         | m <http |         |         |         |         |         |
|         | s://hug |         |         |         |         |         |
|         | gingfac |         |         |         |         |         |
|         | e.co/eu |         |         |         |         |         |
|         | genesio |         |         |         |         |         |
|         | w/mdsr- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 8       | `aw     | 1.6m    | 32.13   | 28.75   | 28.51   | 26.03   |
|         | srn-bam |         | /0.8947 | /0.7851 | /0.7647 | /0.7838 |
|         |  <https |         |         |         |         |         |
|         | ://hugg |         |         |         |         |         |
|         | ingface |         |         |         |         |         |
|         | .co/eug |         |         |         |         |         |
|         | enesiow |         |         |         |         |         |
|         | /awsrn- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 9       | `ed     | 1.5m    | 32.12   | 28.72   | 28.50   | 26.02   |
|         | sr-base |         | /0.8947 | /0.7845 | /0.7644 | /0.7832 |
|         |  <https |         |         |         |         |         |
|         | ://hugg |         |         |         |         |         |
|         | ingface |         |         |         |         |         |
|         | .co/eug |         |         |         |         |         |
|         | enesiow |         |         |         |         |         |
|         | /edsr-b |         |         |         |         |         |
|         | ase>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 10      | `a2n    | 1.0m    | 32.07   | 28.68   | 28.44   | 25.89   |
|         | <https: |         | /0.8933 | /0.7830 | /0.7624 | /0.7787 |
|         | //huggi |         |         |         |         |         |
|         | ngface. |         |         |         |         |         |
|         | co/euge |         |         |         |         |         |
|         | nesiow/ |         |         |         |         |         |
|         | a2n>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 11      | `carn < | 1.6m    | 32.05   | 28.67   | 28.44   | 25.85   |
|         | https:/ |         | /0.8931 | /0.7828 | /0.7625 | /0.7768 |
|         | /huggin |         |         |         |         |         |
|         | gface.c |         |         |         |         |         |
|         | o/eugen |         |         |         |         |         |
|         | esiow/c |         |         |         |         |         |
|         | arn>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 12      | `       | 1.6m    | 32.0    | 28.62   | 28.41   | 25.77   |
|         | carn-ba |         | /0.8923 | /0.7822 | /0.7614 | /0.7741 |
|         | m <http |         |         |         |         |         |
|         | s://hug |         |         |         |         |         |
|         | gingfac |         |         |         |         |         |
|         | e.co/eu |         |         |         |         |         |
|         | genesio |         |         |         |         |         |
|         | w/carn- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 13      | `pan    | 270k    | 31.92   | 28.57   | 28.35   | 25.63   |
|         | <https: |         | /0.8915 | /0.7802 | /0.7595 | /0.7692 |
|         | //huggi |         |         |         |         |         |
|         | ngface. |         |         |         |         |         |
|         | co/euge |         |         |         |         |         |
|         | nesiow/ |         |         |         |         |         |
|         | pan>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 14      | `pan-b  | 270k    | 31.9    | 28.54   | 28.32   | 25.6    |
|         | am <htt |         | /0.8911 | /0.7795 | /0.7591 | /0.7691 |
|         | ps://hu |         |         |         |         |         |
|         | ggingfa |         |         |         |         |         |
|         | ce.co/e |         |         |         |         |         |
|         | ugenesi |         |         |         |         |         |
|         | ow/pan- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 15      | `han    | 16m     | 31.21   | 28.18   | 28.09   | 25.1    |
|         | <https: |         | /0.8778 | /0.7712 | /0.7533 | /0.7497 |
|         | //huggi |         |         |         |         |         |
|         | ngface. |         |         |         |         |         |
|         | co/euge |         |         |         |         |         |
|         | nesiow/ |         |         |         |         |         |
|         | han>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| 16      | `       | 15m     | 30.8    | 27.91   | 27.91   | 24.75   |
|         | rcan-ba |         | /0.8701 | /0.7648 | /0.7477 | /0.7346 |
|         | m <http |         |         |         |         |         |
|         | s://hug |         |         |         |         |         |
|         | gingfac |         |         |         |         |         |
|         | e.co/eu |         |         |         |         |         |
|         | genesio |         |         |         |         |         |
|         | w/rcan- |         |         |         |         |         |
|         | bam>`__ |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
