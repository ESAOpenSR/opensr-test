#

## The `summary` Method

The `summary` method returns a dictionary containing all the metrics computed, with each metric represented as a single scalar value. When the *agg_method* parameter is configured to "patch" or "pixels", the values returned by the `summary` method represent the absolute mean values across the entire tensor. It is important to note that the `summary` method should be invoked only after executing the `compute` method, as it relies on the computations performed therein. The following example demonstrates the correct sequence and usage of the `summary` method:

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

# Retrieve and display the summary of computed metrics
metrics.summary()
```