from typing import List, Optional
from skimage.exposure import match_histograms

import torch
import pathlib
import random
import numpy as np

def apply_upsampling(x: torch.Tensor, scale: int) -> torch.Tensor:
    """ Upsampling a tensor (B, C, H, W) to a lower resolution 
    (B, C, H', W') using bilinear interpolation with antialiasing.

    Args:
        x (torch.Tensor): The tensor to upsample.
        scale (int, optional): The super-resolution scale. Defaults 
            to 4.

    Returns:
        torch.Tensor: The upsampled tensor (B, C, H', W').
    """

    x_ref = torch.nn.functional.interpolate(
        input=x[None], scale_factor=1 / scale, mode="bilinear", antialias=True
    ).squeeze()

    return x_ref


def apply_downsampling(x: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """ Downsampling a tensor (B, C, H, W) to a upper resolution 
    (B, C, H', W') using bilinear interpolation with antialiasing.

    Args:
        x (torch.Tensor): The tensor to downsampling.
        scale (int, optional): The super-resolution scale. Defaults 
            to 4.

    Returns:
        torch.Tensor: The downscaled tensor (B, C, H', W').
    """

    x_ref = torch.nn.functional.interpolate(
        input=x, scale_factor=scale, mode="bilinear", antialias=True
    )

    return x_ref


def apply_mask(x: torch.Tensor, mask: int) -> torch.Tensor:
    """ Apply a mask to the tensor. Some deep learning models
    underperform at the borders of the image. This function
    crops the borders of the image to avoid this issue.

    Args:
        x (torch.Tensor): The tensor to apply the mask.
        mask (int): The border mask. 

    Returns:
        torch.Tensor: The tensor with the mask applied.
    """

    if mask is None:
        return x        
    if mask == 0:
        return x        
    return x[:, mask: -mask, mask: -mask]


def hq_histogram_matching(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """ Lazy implementation of histogram matching 

    Args:
        image1 (torch.Tensor): The low-resolution image (C, H, W).
        image2 (torch.Tensor): The super-resolved image (C, H, W).

    Returns:
        torch.Tensor: The super-resolved image with the histogram of
            the target image.
    """

    # Go to numpy
    np_image1 = image1.detach().cpu().numpy()
    np_image2 = image2.detach().cpu().numpy()

    if np_image1.ndim == 3:
        np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=0)
    elif np_image1.ndim == 2:
        np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=None)
    else:
        raise ValueError("The input image must have 2 or 3 dimensions.")

    # Go back to torch
    image1_hat = torch.from_numpy(np_image1_hat).to(image1.device)

    return image1_hat


def check_lpips() -> None:
    """ Check if the LPIPS library is installed. """

    try:
        import lpips
    except ImportError:
        raise ImportError(
            "The LPIPS library is not installed. Please install it with: pip install lpips"
        )
    
def check_openclip():
    """ Check if the open_clip library is installed. """
    try:
        import open_clip
    except ImportError as e:
        raise ImportError(
            "The open_clip library is not installed. Please install it with: pip install open_clip"
        ) from e    

def check_huggingface_hub():
    """ Check if the huggingface library is installed. """    

    try:
        import huggingface_hub
    except ImportError:
        raise ImportError(
            "The huggingface_hub library is not installed. Please "
            "install it with: pip install huggingface_hub"
        )


def seed_everything(seed: int):
    """ Seed everything for reproducibility.

    Args:
        seed (int): The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_data_path() -> str:
    """ Get the path of the opensr-test dataset

    Returns:
        str: _description_
    """
    cred_path = pathlib.Path.home() / ".config/opensr_test/"
    cred_path.mkdir(parents=True, exist_ok=True)
    return cred_path