#

## The `results` Attribute

The `results` attribute encapsulates all the metrics calculated at the `agg_method` level and the intermediate results produced by the `opensr-test` test. It organizes the output into four principal categories: *consistency*, *distance*, *correctness*, and *auxiliary*. The following example illustrates how to retrieve the outputs from the `results` attribute:

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

# Compute the metrics
metrics.compute(lr=lr, sr=sr, hr=hr)

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


### consistency

The consistency metrics within `opensr-test` play a crucial role in evaluating the harmony between LR and SR images prior to SR harmonization (SRharm). These metrics are calculated after resampling the SR images to match the dimensions of the LR (SRdown). There are three key metrics in this category: *reflectance*, *spectral*, and *spatial*.

#### reflectance
  
This metric evaluates how well the SR image reflects the norm values of the LR image. The calculation of reflectance consistency utilizes the method defined by the reflectance_method parameter.  

#### spectral

This metric assesses the similarity in spectral characteristics between the LR and SR images. The computation of spectral consistency leverages the angle distance specified in the spectral_method parameter. This allows for a detailed comparison of the spectral profiles of the images, ensuring that the SR image preserves the original spectral properties of the LR image.

#### spatial

The spatial consistency metric is computed by calculating the difference between the matching points identified in the LR and HR images. This process evaluates the spatial alignment and structural integrity of the SR image compared to the LR image. If the agg_method parameter is set to pixel or patch, the grid is calculated applying a simple kernel interpolation method.


### distance

distance metrics in `opensr-test` permit measure how far the SR image is from the LR and HR images. These metrics are computed post-harmonization (SRharm) to reduce the potential spatial and spectral bias introduced by the SR model. There are three distance metrics: *sr_lr*, *sr_hr*, and *hr_lr*.

#### sr_to_lr (SRharm - LR Distance)

This metric quantifies the distance between SRharm and the LR image. It serves as an indicator of how close the SR image is to the LR image. The LR is upsampled (LRup) to the dimensions of the HR image using the method defined by the upsample_method parameter. We strongly advise opting for a method that does not require parameter tuning, such as bilinear interpolation or other similar techniques. This is crucial to prevent the introduction of hallucination artifacts, which can significantly bias the experimental results.


#### sr_to_hr (SRharm - HR Distance)

This metric measures the distance between the harmonized SR image (SRharm) and the HR image. It is essential for measure if the high-frequency details introduced by the SR model are consistent with the HR image (improvement) or if they are artificial (hallucination).

#### lr_to_hr (HR - LR Distance)

This metric calculates the distance between the HR and LR images. Although it doesn't directly involve the SR image, it offers a baseline understanding of the initial discrepancies between the HR and LR images, which can be useful for context and comparison.


### correctness

The correctness metrics in `opensr-test` are crucial assessments conducted after the SR harmonization process and the evaluation of the triple distance. These metrics encompass four categories: *improvement*, *omission*, *hallucination*, and *classification*. All the correctness metrics are designed such that values closer to 0 indicate that the pixel, patch, or image is nearer to its respective target space.

#### improvement

The improvement matrix quantifies the extent of improvement space. It is a matrix of dimensions HxW. The `im_score` parameter, defined in the `compute` method, acts as a modulator, allowing for fine-tuning of the space.

#### omission

The omission matrix, on the other hand, evaluates the extent of the omission space. This matrix provides insights into areas where the SR process might have failed to replicate crucial details from the HR image.

#### hallucination

The hallucination matrix measures the extent of artificial details or 'hallucinations' introduced in the SR image. The definition of the improvement and omission spaces conditions the determination of the hallucination space.

#### classification

The classification matrix is computed by applying a np.argmin function across the aforementioned correctness matrices. This matrix forms the basis for categorizing each pixel into one of the three classes: improvement, omission, and hallucination.


### auxiliary

The auxiliary results in opensr-test comprise a set of intermediate outputs generated during the execution of the compute method. These results are instrumental in understanding the internal workings and transformations applied during the computation process. There are four key auxiliary results: *sr_harm*, *lr_to_hr*, *matching_points_lr*, and *matching_points_hr*.

#### sr_harm

This is the SR product post-harmonization. The harmonization pipeline is influenced by the harm_apply_spectral and harm_apply_spatial parameters. When both parameters are enabled (set to True), the SR image undergoes a two-step enhancement: first, the reflectance values are corrected via histogram matching with the HR image; subsequently, spatial alignment is performed, aligning the SR image with the HR image based on the settings defined in the spatial_features and spatial_matcher parameters.

#### lr_to_hr

This represents the LR image resampled to match the dimensions of the HR image. In the absence of a specific upsample_method set during the setup phase, the lr_to_hr result is achieved using a classic method - bilinear interpolation complemented by an anti-aliasing kernel filter.

#### matching_points_lr

These are the points of correspondence identified between the LR and HR images. This points help in understanding the spatial relationship and alignment between these two different tensor resolutions.

#### matching_points_hr

These are the points of correspondence identified between the SRharm and HR images. It provides insights into how well the SR image aligns with the HR image after the harmonization process.
