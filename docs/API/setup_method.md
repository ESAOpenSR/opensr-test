# 

## The `setup` Method

The `setup` method is a pivotal function within  [`opensr_test.Metrics`](docs/API/model_parameters.md) class, responsible for initializing the evaluation process by defining the LR, SR, and HR images. In addition, it provides an option to include a land use image, must be the [ESA landcover product](https://worldcover2020.esa.int/), which is utilized solely for the purpose of visualization. The method also facilitates the specification of upsampling and downscaling strategies.

Internally, the `setup` method generates the harmonized SR image `SRharm` by applying the spatial and spectral alignment as determined by the user's Metric class settings. Below is a demonstration of how to employ the `setup` method:

```python
# Import necessary libraries
import torch
import opensr_test
import matplotlib.pyplot as plt

# Randomly generate LR, HR, SR, and land use images for illustration.
lr = torch.rand(4, 32, 32)  # Low Resolution image
hr = torch.rand(4, 256, 256)  # High Resolution image
sr = torch.rand(4, 256, 256)  # Super Resolution image
landuse = torch.rand(4, 256, 256)  # Land use image for visualization

# Initialize the Metrics object.
metrics = opensr_test.Metrics()

# Setup the evaluation environment with the specified images and methods.
metrics.setup(
    lr=lr,
    sr=sr,
    hr=hr,
    landuse=landuse, # Optional
    downsample_method="classic",  # Optional, Default downsample method
    upsample_method="classic"  # Optional, Default upsample method
)

# Plot the comparison triplets of LR, SR, and HR images.
metrics.plot_quadruplets()
plt.show()  # Display the plots
```

By default, the `setup` method employs what are known as `classic` methods for both downsampling and upsampling operations. These `classic` methods is just a bilinear interpolation enhanced with an anti-aliasing kernel to smooth out the image and reduce the risk of aliasing artifacts. In addition, the `setup` method also support downsample and upsample methods with degradation kernels fine-tuned for the different datasets supported by `opensr-test`. See the [Datasets](docs/Datasets/NAIP.md) section for more information.


```python
import torch
import opensr_test
import matplotlib.pyplot as plt

# Generate sample LR, HR, SR, and land use images
lr = torch.rand(4, 32, 32)
hr = torch.rand(4, 256, 256)
sr = torch.rand(4, 256, 256)
landuse = torch.rand(4, 256, 256)

# Initialize the Metrics object
metrics = opensr_test.Metrics()

# Setup the evaluation environment with custom downsample and default upsample methods
metrics.setup(
    lr=lr,
    sr=sr,
    hr=hr,
    landuse=landuse, 
    downsample_method="naip",  # Custom downsample method (options: "naip", "venus", "spot", "classic")
    upsample_method="classic"  # Default upsample method
)

# Plot comparison triplets of LR, SR, and HR images
metrics.plot_triplets()
plt.show()  # Display the generated plots
```