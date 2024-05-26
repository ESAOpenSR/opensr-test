# 

## **Config**

The Config class is a [`pydantic.BaseModel`](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel) that defines model parameters. The parameters are categorized into four groups, each catering to different aspects of the super-resolution (SR) evaluation process:

### **Global Parameters:**

These parameters control the overall behavior of the SR assessment:

- **device:** Selects the computation device for inference, defaulting to `cpu`. `cuda` is also supported.

- **agg_method:** Determines the granularity for distance metric application — pixel-wise by default ("pixel"), with "patch" and "image" level aggregations options. If "patch" is selected, the metric is computed on patches of size `patch_size` and then interpolated to the original image size using bicubic interpolation with anti-aliasing. If "image" is selected, the metric is computed on the entire image and output as a single value. If "pixel" is selected, the metric is computed on each pixel and output as a tensor of the same size as the input image.

- **patch_size:** Relevant when `agg_method` is "patch", this defines the patch size for distance metric computation.

- **border_mask:** Excludes image borders in metric calculations, ignoring the outer 16 pixels by default.

- **rgb_bands:** Necessary for certain metrics like LPIPS, CLIP and plot generation, assuming a default order of Red-Green-Blue ([0, 1, 2]).

- **harm_apply_spectral:** Applies histogram matching before correctness metrics. This is enabled by default.

- **harm_apply_spatial:** Activates spatial alignment before correctness metrics. This is enabled by default.

### **Spatial Parameters:**

These parameters set the spatial alignment pre-processing:

- **spatial_method:** The default method for spatial alignment. By default, pcc (Phase Correlation Coefficient) is used. However, other methods like "ecc" (Enhanced Correlation Coefficient) and "lgm" (SuperPoint + LightGlue) are also available. ecc is more robust to noise and usually more precise than pcc, but it is slower. lgm is slower than ecc and pcc, but it is more robust to large translations.

- **spatial_threshold_distance**: The maximum permissible translation distance for spatial alignment, set to 5 pixels by default. If the translation distance is greater than this threshold, the spatial alignment is skipped.

- **spatial_max_num_keypoints:** Only relevant when the spatial method is "lgm". This parameter caps the number of keypoints for feature matching at 500 by default.


### **Spectral and reflectance Parameters:**

These parameters are specific to spectral analysis and alignment:

- **reflectance_distance:** The default method for reflectance distance estimation. Reflectance distance measure how the SR image affects the reflectance of the original image. By default, "l1" is used. See section [Metrics](docs/Metrics/consistency.md) for more details and other options.

- **spectral_distance:** The default method for spectral distance calculation.  The spectral distance measure how the SR image affects the spectral signature of the original image. By default, "sam" is used. See section [Metrics](docs/Metrics/consistency.md) for more details and other options.

### **Synthesis parameters**

- **synthesis_distance:** Specifies the distance metric between SR and LR after harmonization. By default, "l1" is used. See section [Metrics](docs/Metrics/consistency.md) for more details and other options.

### **Correctness parameters**

- **correctness_distance:** Specifies the distance metric between harmonized SR, HR and LR. By default, "l1" is used. See section [Metrics](docs/Metrics/consistency.md) for more details and other options.


Below is an illustrative example of how to instantiate `opensr-test` with user-defined parameters:

```python
import torch
import opensr_test

# Define the Low Resolution (LR), High Resolution (HR), and Super-Resolved (SR) images.
lr = torch.rand(4, 64, 64)
hr = torch.rand(4, 256, 256)
sr = torch.rand(4, 256, 256)

# Initialize the Metrics object with custom parameters.
config = opensr_test.Config(
    device="cuda",
    spatial_features="ecc",
    harm_apply_spatial=True,
    harm_apply_spectral=False
)
metrics = opensr_test.Metrics(config)

# Compute the metrics based on the provided images.
metrics.compute(lr=lr, sr=sr, hr=hr)
```