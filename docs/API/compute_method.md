#

## The `compute` Method

The `compute` method calculate the suite of metrics in the `opensr-test` framework. This method includes four optional parameters, all of them related to the calculation of correctness metrics ([Metrics section](docs/Metrics/correctness.md) section). These parameters are:

- **stability_threshold** (default=0.01): This threshold help for discerning pixels with significant distance differences within the triple image space (LR, SR, and HR). Setting this parameter helps in isolating pixels where the model has introduced notable high-frequency details, thereby focusing the analysis on areas of potential enhancement. By default, this parameter is set to 0.01, which is optimized for balanced assessment. However, this parameter can be adjusted to each image. Based on the expert judgment of three remote sensing specialists, the optimal value of this parameter has been determined for each image within the datasets of NAIP, SPOT, and Venus, allowing for a more tailored and precise evaluation for each specific dataset.

- **im_score** (default=0.8): This parameter is critical to defining the 'improvement space'. It acts as a modulator determining whether a pixel/patch is considered 'improved'. The default value is set to 0.8. This value have been determined by the perceptual evaluation of three remote sensing experts.

- **om_score** (default=0.8): Similarly, the 'omission space' is determined by this parameter. The default of 0.8. This value have been determined by the perceptual evaluation of three remote sensing experts.

- **ha_score** (default=0.4): This parameter delineates the 'hallucination space', evaluating whether a pixel is regarded as a hallucination. The default setting of 0.4 have been determined by the perceptual evaluation of three remote sensing experts.

Below is an example that demonstrates the usage of the `compute` method with the aforementioned parameters:

```python
# Import necessary libraries
import torch
import opensr_test
import matplotlib.pyplot as plt

# Generate sample LR, HR, and SR images
lr = torch.rand(4, 32, 32)  # Low Resolution image
hr = torch.rand(4, 256, 256)  # High Resolution image
sr = torch.rand(4, 256, 256)  # Super Resolution image

# Initialize the Metrics object
metrics = opensr_test.Metrics()

# Compute the metrics with specified parameters
metrics.compute( 
    lr=lr, sr=sr, hr=hr,
    stability_threshold=0.01,
    im_score=0.8,
    om_score=0.8,
    ha_score=0.4,
)
```
