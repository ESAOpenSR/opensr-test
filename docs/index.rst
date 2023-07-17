Welcome to opensr-test!
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
----------------

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
----------------

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
| NAIP           | 4     | 200      | 512    | USDA Farm Production and Conservation - Business Center, Geospatial Enterprise Operations. |
| SPOTPAN        | 4     | 200      | 512    | European Space Agency, 2017, SPOT 1-5 ESA                                                  |
| Mini-SEN2VENµS | 2     | 200      | 512    | Vegetation and Environment monitoring on a New Micro-Satellite (SEN2VENμS).                |
+----------------+-------+----------+--------+--------------------------------------------------------------------------------------------+


Metrics
----------------

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
----------------

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
----------------

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
awesome huggingface_hub (https://huggingface.co/models?filter=super-image).


Scale x2
^^^^^^^^^

+------+-----------+--------+------------------+------------------+------------------+------------------+
| Rank | Model     | Params | Set5             | Set14            | BSD100           | Urban100         |
+======+===========+========+==================+==================+==================+==================+
| 1    | drln-bam  | 34m    | **38.23/0.9614** | 33.95/0.9206     | **33.95/0.9269** | **32.81/0.9339** |
| 2    | edsr      | 41m    | 38.19/0.9612     | **33.99/0.9215** | 33.89/0.9266     | 32.68/0.9331     |
| 3    | msrn      | 5.9m   | 38.08/0.9609     | 33.75/0.9183     | 33.82/0.9258     | 32.14/0.9287     |
| 4    | mdsr      | 2.7m   | 38.04/0.9608     | 33.71/0.9184     | 33.79/0.9256     | 32.14/0.9283     |
| 5    | msrn-bam  | 5.9m   | 38.02/0.9608     | 33.73/0.9186     | 33.78/0.9253     | 32.08/0.9276     |
| 6    | edsr-base | 1.5m   | 38.02/0.9607     | 33.66/0.9180     | 33.77/0.9254     | 32.04/0.9276     |
| 7    | mdsr-bam  | 2.7m   | 38/0.9607        | 33.68/0.9182     | 33.77/0.9253     | 32.04/0.9272     |
| 8    | awsrn-bam | 1.4m   | 37.99/0.9606     | 33.66/0.918      | 33.76/0.9253     | 31.95/0.9265     |
| 9    | a2n       | 1.0m   | 37.87/0.9602     | 33.54/0.9171     | 33.67/0.9244     | 31.71/0.9240     |
| 10   | carn      | 1.6m   | 37.89/0.9602     | 33.53/0.9173     | 33.66/0.9242     | 31.62/0.9229     |
| 11   | carn-bam  | 1.6m   | 37.83/0.96       | 33.51/0.9166     | 33.64/0.924      | 31.53/0.922      |
| 12   | pan       | 260k   | 37.77/0.9599     | 33.42/0.9162     | 33.6/0.9235      | 31.31/0.9197     |
| 13   | pan-bam   | 260k   | 37.7/0.9596      | 33.4/0.9161      | 33.6/0.9234      | 31.35/0.92       |
+------+-----------+--------+------------------+------------------+------------------+------------------+


Scale x3
--------

+------+-----------+--------+------------------+------------------+------------------+------------------+
| Rank | Model     | Params | Set5             | Set14            | BSD100           | Urban100         |
+======+===========+========+==================+==================+==================+==================+
| 1    | drln-bam  | 34m    | **38.23/0.9614** | 33.95/0.9206     | **33.95/0.9269** | **32.81/0.9339** |
| 2    | edsr      | 41m    | 38.19/0.9612     | **33.99/0.9215** | 33.89/0.9266     | 32.68/0.9331     |
| 3    | msrn      | 5.9m   | 38.08/0.9609     | 33.75/0.9183     | 33.82/0.9258     | 32.14/0.9287     |
| 4    | mdsr      | 2.7m   | 38.04/0.9608     | 33.71/0.9184     | 33.79/0.9256     | 32.14/0.9283     |
| 5    | msrn-bam  | 5.9m   | 38.02/0.9608     | 33.73/0.9186     | 33.78/0.9253     | 32.08/0.9276     |
| 6    | edsr-base | 1.5m   | 38.02/0.9607     | 33.66/0.9180     | 33.77/0.9254     | 32.04/0.9276     |
| 7    | mdsr-bam  | 2.7m   | 38/0.9607        | 33.68/0.9182     | 33.77/0.9253     | 32.04/0.9272     |
| 8    | awsrn-bam | 1.4m   | 37.99/0.9606     | 33.66/0.918      | 33.76/0.9253     | 31.95/0.9265     |
| 9    | a2n       | 1.0m   | 37.87/0.9602     | 33.54/0.9171     | 33.67/0.9244     | 31.71/0.9240     |
| 10   | carn      | 1.6m   | 37.89/0.9602     | 33.53/0.9173     | 33.66/0.9242     | 31.62/0.9229     |
| 11   | carn-bam  | 1.6m   | 37.83/0.96       | 33.51/0.9166     | 33.64/0.924      | 31.53/0.922      |
| 12   | pan       | 260k   | 37.77/0.9599     | 33.42/0.9162     | 33.6/0.9235      | 31.31/0.9197     |
| 13   | pan-bam   | 260k   | 37.7/0.9596      | 33.4/0.9161      | 33.6/0.9234      | 31.35/0.92       |
+------+-----------+--------+------------------+------------------+------------------+------------------+


Scale x4
--------

+------+-----------+--------+------------------+------------------+------------------+------------------+
| Rank | Model     | Params | Set5             | Set14            | BSD100           | Urban100         |
+======+===========+========+==================+==================+==================+==================+
| 1    | drln-bam  | 34m    | **38.23/0.9614** | 33.95/0.9206     | **33.95/0.9269** | **32.81/0.9339** |
| 2    | edsr      | 41m    | 38.19/0.9612     | **33.99/0.9215** | 33.89/0.9266     | 32.68/0.9331     |
| 3    | msrn      | 5.9m   | 38.08/0.9609     | 33.75/0.9183     | 33.82/0.9258     | 32.14/0.9287     |
| 4    | mdsr      | 2.7m   | 38.04/0.9608     | 33.71/0.9184     | 33.79/0.9256     | 32.14/0.9283     |
| 5    | msrn-bam  | 5.9m   | 38.02/0.9608     | 33.73/0.9186     | 33.78/0.9253     | 32.08/0.9276     |
| 6    | edsr-base | 1.5m   | 38.02/0.9607     | 33.66/0.9180     | 33.77/0.9254     | 32.04/0.9276     |
| 7    | mdsr-bam  | 2.7m   | 38/0.9607        | 33.68/0.9182     | 33.77/0.9253     | 32.04/0.9272     |
| 8    | awsrn-bam | 1.4m   | 37.99/0.9606     | 33.66/0.918      | 33.76/0.9253     | 31.95/0.9265     |
| 9    | a2n       | 1.0m   | 37.87/0.9602     | 33.54/0.9171     | 33.67/0.9244     | 31.71/0.9240     |
| 10   | carn      | 1.6m   | 37.89/0.9602     | 33.53/0.9173     | 33.66/0.9242     | 31.62/0.9229     |
| 11   | carn-bam  | 1.6m   | 37.83/0.96       | 33.51/0.9166     | 33.64/0.924      | 31.53/0.922      |
| 12   | pan       | 260k   | 37.77/0.9599     | 33.42/0.9162     | 33.6/0.9235      | 31.31/0.9197     |
| 13   | pan-bam   | 260k   | 37.7/0.9596      | 33.4/0.9161      | 33.6/0.9234      | 31.35/0.92       |
+------+-----------+--------+------------------+------------------+------------------+------------------+
