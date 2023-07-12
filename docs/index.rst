Welcome to srcheck!
==================

.. raw:: html

   <embed>
     <p align="center">
       <a href="https://github.com/csaybar/srcheck"><img src="https://user-images.githubusercontent.com/16768318/213960001-66bb7d18-13d8-41d4-9de3-1e8a77f73787.png" height="350px"/></a>
       <br>
       <b>A comprehensive benchmark for real-world Sentinel-2 imagery super-resolution
       </a>
       </b>
     </p>
   </embed>

.. image:: https://img.shields.io/pypi/v/srcheck.svg
        :target: https://pypi.python.org/pypi/srcheck
        :alt: PyPI Version
        
.. image:: https://img.shields.io/conda/vn/conda-forge/srcheck.svg
        :target: https://anaconda.org/conda-forge/srcheck
        :alt: conda-forge Version
        
.. image:: https://img.shields.io/badge/License-MIT-blue.svg
        :target: https://opensource.org/licenses/MIT
        :alt: License
        
.. image:: https://readthedocs.org/projects/srcheck/badge/?version=latest
        :target: https://srcheck.readthedocs.io/en/latest/?badge=latest
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
-------------------

In remote sensing, image super-resolution (ISR) is a technique used to create high-resolution
(HR) images from low-resolution (R) satellite images, giving a more detailed view of the 
Earth's surface. However, with the constant development and introduction of new ISR algorithms, 
it can be challenging to stay updated on the latest advancements and to evaluate their performance 
objectively. To address this issue, we introduce SRcheck, a Python package that provides an 
easy-to-use interface for comparing and benchmarking various ISR methods. SRcheck includes a 
range of datasets that consist of high-resolution and low-resolution image pairs, as well as a 
set of quantitative metrics for evaluating the performance of SISR algorithms.


Installation
-------------------

Install the latest srcheck version from PyPI by running:

.. code-block::
      
   pip install srcheck

Upgrade srcheck by running:

.. code-block::   
      
   pip install -U srcheck

Install the development version from GitHub by running:

.. code-block::   
      
   pip install git+https://github.com/csaybar/srcheck
   
Install the latest srcheck version from conda-forge by running:

.. code-block::   
      
   conda install -c conda-forge srcheck


How does it work?
----------------

srcheck needs either a `torch.nn.Module` class or a compiled 
model via `torch.jit.trace` or `torch.jit.script`. The following
example shows how to run the benchmarks:

.. raw:: html
   
   <embed>
     <p align="center">
       <img src="https://user-images.githubusercontent.com/16768318/213967913-3c665d59-5053-43a7-a450-859b7442b345.png"/>
     </p>
   </embed>

.. code-block::
   
   import torch
   import srcheck

   model = torch.jit.load('/content/quantSRmodel.pt', map_location='cpu')
   srcheck.benchmark(model, dataset='SPOT-50', metrics=['PSNR', 'SSIM'], type= "NoSRI")

**srcheck** supports two group types of metrics: (a) Surface Reflectance Integrity (SRI) and 
(b) No Surface Reflectance Integrity (NoSRI). This difference is due to the fact that 
depending on the application, developers will be interested in optimizing the "image quality"
or the "image fidelity". *Image fidelity* refers to how closely the LR image represents 
the real source distribution (HR). Optimizing fidelity is crucial for applications that 
require preserving surface reflectance as close as possible to the original values. On 
the other hand, *image quality* refers to how pleasant the image is for the human eye. 
Optimizing image quality is important for creating HR image satellite base maps. The 
image below shows the natural trade-off that exists between these two group types 
of metrics.

.. raw:: html
   
   <embed>
     <p align="center">
       <img src="https://user-images.githubusercontent.com/16768318/213970463-5c2a8012-4e76-48ce-bb13-4d51590d359c.png">
     </p>
   </embed>

But what happens if my ISR algorithm increases the image by a factor of 8, but the datasets 
available in srcheck do not support 8X? In that case, *srcheck* will automatically convert 
the results to the native resolution of the datasets. For example, if your algorithm increases
the image by 2X, and you want to test it on SPOT-50 whose images are 10m in LR and 6m in HR,
*srcheck* will upscale the results from 5 meters to 6m using the bilinear interpolation algorithm. 
Similarly, in the MUS2-50 dataset, *srcheck* will downscale the results from 5m to 2m. This is 
done in order the results can be compared with the datasets available.


.. raw:: html
   
   <embed>
     <p align="center">
       <img src="https://user-images.githubusercontent.com/16768318/213971771-04b193e7-83e8-436a-b4a1-0c317cc7b756.png">
     </p>
   </embed>


Datasets
----------------

https://zenodo.org/record/7562334

More datasets are coming soon!


Metrics
----------------

Metrics documentation is coming soon!