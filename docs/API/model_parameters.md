# 

## Model Parameters

`opensr-test` utilizes [pydantic](https://docs.pydantic.dev/latest/) for defining model parameters, ensuring type safety and ease of configuration. The parameters are categorized into three groups, each catering to different aspects of the super-resolution (SR) evaluation process:

### Global Parameters:

These parameters control the overall behavior of the SR assessment:

- **device:** Selects the computation device for inference, defaulting to `cpu`. `cuda` is also supported for GPU acceleration.
- **distance_method:** Specifies the distance metric between SR, LR, and HR images. Defaults to "L1", with support for other metrics like KL, L2, PBIAS, RMSE, 1/PSNR, SAD, CROSSMTF, CLIP and LPIPS. See section [Metrics](docs/Metrics/consistency.md) for more details.
- **agg_method:** Determines the granularity for distance metric application — pixel-wise by default ("pixel"), with "patch" and "image" level aggregations options.
- **patch_size:** Relevant when `agg_method` is "patch", this defines the patch size for distance metric computation.
- **border_mask:** Excludes image borders in metric calculations, ignoring the outer 16 pixels by default.
- **rgb_bands:** Necessary for certain metrics like LPIPS and plot generation, assuming a default order of Red-Green-Blue ([0, 1, 2]).
- **harm_apply_spectral:** Applies histogram matching before distance metric calculation, enabled by default.
- **harm_apply_spatial:** Activates spatial alignment before distance metric calculation, enabled by default.

### Spatial Parameters:

These parameters fine-tune the spatial feature extraction and matching process:

- **spatial_features:** The default feature extractor is "superpoint", with "disk" also available.
- **spatial_matcher:** Sets the matcher for feature points, defaulting to "lightglue".
- **spatial_max_num_keypoints:** Caps the number of keypoints for feature matching at 1000 by default.
- **spatial_threshold_distance:** The maximum permissible distance between keypoints for a match, set to 5 pixels.
- **spatial_threshold_npoints:** The minimum required keypoints for alignment, set to 5. Spatial alignment is skipped if keypoints fall below this threshold.

### Spectral Parameters:

These parameters are specific to spectral analysis and alignment:

- **reflectance_method:** The default method for reflectance estimation is "l1", with "kl", "l1", "l2", and "pbias" as alternatives.

- **spectral_method:** For spectral alignment, "sad" is the default method.

Below is an illustrative example of how to instantiate `opensr-test` with user-defined parameters:

```python
import torch
import opensr_test

# Define the Low Resolution (LR), High Resolution (HR), and Super-Resolved (SR) images.
lr = torch.rand(4, 32, 32)
hr = torch.rand(4, 256, 256)
sr = torch.rand(4, 256, 256)

# Initialize the Metrics object with custom parameters.
metrics = opensr_test.Metrics(
    device="cpu",
    spatial_features="disk",
    spatial_max_num_keypoints=400,
    harm_apply_spectral=False,
)

# Setup the evaluation environment with the LR, SR, and HR images.
metrics.setup(lr=lr, sr=sr, hr=hr)

# Compute the metrics based on the provided images.
metrics.compute()

# Generate a summary report of the evaluation.
metrics.summary()
```