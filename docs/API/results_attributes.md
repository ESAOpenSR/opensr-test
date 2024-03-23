#

## The `results` Attribute

The `results` attribute encapsulate all the metrics calculated at the `agg_method` level, along with the intermediate results produced by the `opensr-test` test. It organizes the output into four principal categories: *consistency*, *distance*, *correctness*, and *auxiliary*. The following example illustrates how to retrieve the outpus from the `results` attribute:

```python
# Import necessary libraries
import torch
import opensr_test

# Generate sample LR, HR, and SR images
lr = torch.rand(4, 64, 64)  # Low Resolution image
hr = torch.rand(4, 256, 256)  # High Resolution image
sr = torch.rand(4, 256, 256)  # Super Resolution image

# Initialize the Metrics object
metrics = opensr_test.Metrics()

# Setup the evaluation environment
metrics.setup(lr=lr, sr=sr, hr=hr)

# Compute the metrics
metrics.compute()

# Accessing Consistency Metrics
metrics.results.consistency

# Accessing Distance Metrics post SR Harmonization
metrics.results.distance

## Accessing Correctness Metrics
metrics.results.correctness

# Accessing Intermediate Results
metrics.results.auxiliary
```

Except the auxiliary field, the spatial resolution of all the results is determined by the `agg_method` parameter. For instance, if `agg_method` is set to "patch" with a patch_size of 32, then each metric in the results attribute will contain values aggregated over patches of size H/32 x W/32.


<details>
  <summary><b>consistency</b></summary>
    The consistency metrics within `opensr-test` play a crucial role in evaluating the harmony between LR and SR images prior to SR harmonization (SRharm). These metrics are calculated after resampling the SR images to match the dimensions of the LR (SRdown). There are three key metrics in this category: *reflectance*, *spectral*, and *spatial*.  
  <summary><b>reflectance</b></summary>
    This metric evaluates how well the SR image reflects the norm values of the LR image. The calculation of reflectance consistency utilizes the method defined by the reflectance_method parameter.  
  <summary><b>spectral</b></summary>
    This metric assesses the similarity in spectral characteristics between the LR and SR images. The computation of spectral consistency leverages the angle distance specified in the spectral_method parameter. This allows for a detailed comparison of the spectral profiles of the images, ensuring that the SR image preserves the original spectral properties of the LR image.
  <summary><b>spatial</b></summary>
    The spatial consistency metric is computed by calculating the difference between the matching points identified in the LR and HR images. This process evaluates the spatial alignment and structural integrity of the SR image compared to the LR image. If the agg_method parameter is set to pixel or patch, the grid is calculated applying a simple kernel interpolation method.
</details>


<details>
    <summary><b>distance</b></summary>
    distance metrics in `opensr-test` permit measure how far the SR image is from the LR and HR images. These metrics are computed post-harmonization (SRharm) to reduce the potential spatial and spectral bias introduced by the SR model. There are three distance metrics: <i>sr_lr</i>, <i>sr_hr</i> and <i>hr_lr</i>.
    <summary><b>sr_to_lr (SRharm - LR Distance)</b></summary>
    This metric quantifies the distance between SRharm and the LR image. It serves as an indicator of how close the SR image is to the LR image. The LR is upsampled (LRup) to the dimensions of the HR image using the method defined by the upsample_method parameter. We strongly advise opting for a method that does not require parameter tuning, such as bilinear interpolation or other similar techniques. This is crucial to prevent the introduction of hallucination artifacts, which can significantly bias the experimental results.
    <summary><b>sr_to_hr (SRharm - HR Distance)</b></summary>
    This metric measures the distance between the harmonized SR image (SRharm) and the HR image. It is essential for measure if the high-frequency details introduced by the SR model are consistent with the HR image (improvement) or if they are artificial (hallucination).
    <summary><b>lr_to_hr (HR - LR Distance)</b></summary>
    This metric calculates the distance between the HR and LR images. Although it doesn't directly involve the SR image, it offers a baseline understanding of the initial discrepancies between the HR and LR images, which can be useful for context and comparison.
</details>

<details>
  <summary><b>correctness</b></summary>
  The correctness metrics in `opensr-test` are crucial assessments conducted after the SR harmonization process and the evaluation of the triple distance. These metrics encompass four categories: <i>improvement</i>, <i>omission</i>, <i>hallucination</i>, and <i>classification</i>. All the correctness metrics are designed such that values closer to 0 indicate that the pixel, patch, or image is nearer to its respective target space.

  <summary><b>improvement</b></summary>
  The improvement matrix quantifies the extent of improvement space. It is a matrix of dimensions HxW. The `im_score` parameter, defined in the `compute` method, acts as a modulator, allowing for fine-tuning of the space. 

  <summary><b>omission</b></summary>
  The omission matrix, on the other hand, evaluates the extent of the omission space. This matrix provides insights into areas where the SR process might have failed to replicate crucial details from the HR image. 

  <summary><b>hallucination</b></summary>
  The hallucination matrix measures the extent of artificial details or 'hallucinations' introduced in the SR image. The definition of the improvement and omission spaces conditions the determination of the hallucination space.

  <summary><b>classification</b></summary>
  The classification matrix is computed by applying a np.argmin function across the aforementioned correctness matrices. This matrix forms the basis for categorizing each pixel into one of the three classes: improvement, omission, and hallucination.
</details>


<details>
  <summary><b>auxiliary</b></summary>
  The auxiliary results in opensr-test comprise a set of intermediate outputs generated during the execution of the compute method. These results are instrumental in understanding the internal workings and transformations applied during the computation process. There are four key auxiliary results: <b>sr_harm</b>, <b>lr_to_hr</b>, <b>matching_points_lr</b>, and <b>matching_points_hr</b>.

  <summary><b>sr_harm</b></summary>
  This is the SR product post-harmonization. The harmonization pipeline is influenced by the harm_apply_spectral and harm_apply_spatial parameters. When both parameters are enabled (set to True), the SR image undergoes a two-step enhancement: first, the reflectance values are corrected via histogram matching with the HR image; subsequently, spatial alignment is performed, aligning the SR image with the HR image based on the settings defined in the spatial_features and spatial_matcher parameters.

  <summary><b>lr_to_hr</b></summary>
  This represents the LR image resampled to match the dimensions of the HR image. In the absence of a specific upsample_method set during the setup phase, the lr_to_hr result is achieved using a classic method - bilinear interpolation complemented by an anti-aliasing kernel filter.

  <summary><b>matching_points_lr</b></summary>  
  These are the points of correspondence identified between the LR and HR images. This points help in understanding the spatial relationship and alignment between these two different tensor resolutions.

  <summary><b>matching_points_hr</b></summary>  
  These are the points of correspondence identified between the SRharm and HR images. It provides insights into how well the SR image aligns with the HR image after the harmonization process.
</details>
