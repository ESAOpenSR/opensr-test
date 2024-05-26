#

## The `results` Attribute

The `results` attribute encapsulates all the metrics at pixel level and the intermediate results produced by the `opensr-test` test. It organizes the output into four principal categories: *consistency*, *synthesis*, *correctness*, and *auxiliary*. The following example illustrates how to retrieve the outputs from the `results` attribute:

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

# Accessing Synthesis Metrics
metrics.results.synthesis

## Accessing Correctness Metrics
metrics.results.correctness

# Accessing Intermediate Results
metrics.results.auxiliary
```

### consistency

The consistency metrics within `opensr-test` play a crucial role in evaluating the harmony between LR and SR images prior to SR harmonization (SRharm). These metrics are calculated after resampling the SR images to match the dimensions of the LR (SRdown). There are three key metrics in this category: *reflectance*, *spectral*, and *spatial*.

#### reflectance
  
This metric evaluates how well the SR image reflects the norm values of the LR image. The calculation of reflectance consistency utilizes the method defined by the reflectance_method parameter.  

#### spectral

This metric assesses the similarity in spectral characteristics between the LR and SR images. The computation of spectral consistency leverages the angle distance specified in the spectral_method parameter. This allows for a detailed comparison of the spectral profiles of the images, ensuring that the SR image preserves the original spectral properties of the LR image.

#### spatial

The spatial consistency metric is computed by calculating, first, the warp affine between the SRdown and LR images. The translation parameters are then extracted from the affine matrix and used to compute the spatial error.


### synthesis

#### distance

The synthesis metrics in `opensr-test` evaluate the distance between the SRharm and the LR image. The distancec matrix help to understand the high-frequency details introduced by the SR model at local level.


### Correctness

The correctness metrics in `opensr-test` are crucial assessments conducted after the SR harmonization process and the evaluation of the triple distance. These metrics encompass four categories: *improvement*, *omission*, *hallucination*, and *classification*. All the correctness metrics are designed such that values closer to 0 indicate that the pixel, patch, or image is nearer to its respective target space.

#### improvement

The improvement matrix quantifies the distance to the improvement space. It is a matrix of dimensions HxW. The idel value of the improvement matrix is 0, which indicates that the SR image is identical to the HR image.

#### omission

The omission matrix quantifies the distance to the improvement space. It is a matrix of dimensions HxW. A high value in the omission matrix indicates that the SR image has omitted crucial details from the HR image.

#### hallucination

The hallucination matrix measures the extent of artificial details or 'hallucinations' introduced in the SR image. It is a matrix of dimensions HxW. A high value in the hallucination matrix indicates that the SR image has introduced artificial details that are not present in the HR image.

#### classification

The classification matrix (HxW) is computed by applying a np.argmin function across the aforementioned correctness matrices. This matrix forms the basis for categorizing each pixel into one of the three classes: improvement, omission, and hallucination.


### auxiliary

The auxiliary subset comprises a set of intermediate outputs generated during the execution of the compute method. These results are instrumental in understanding the internal workings and transformations applied during the computation process. There are two auxiliary products: *sr_harm*, *lr_to_hr*.

#### sr_harm

This is the SR product post-harmonization. The harmonization pipeline is influenced by the harm_apply_spectral and harm_apply_spatial parameters. When both parameters are enabled (set to True, by default), the SR image undergoes a two-step enhancement: first, the reflectance values are corrected via histogram matching with the HR image; subsequently, spatial alignment is performed, aligning the SR image with the HR image based on the method specified in the `spatial_method` parameter.

#### lr_to_hr

This represents the LR image resampled to match the dimensions of the HR image. In the absence of a specific upsample_method set during the setup phase, the `lr_to_hr` result is achieved using a classic method - bilinear interpolation complemented by an anti-aliasing kernel filter.
