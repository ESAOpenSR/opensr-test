from typing import Literal, Optional

import torch
from skimage.exposure import match_histograms


def spectral_reducer(
    X: torch.Tensor, method: Literal["mean", "median", "max", "min"] = "mean"
) -> torch.Tensor:
    """ Reduce the number of channels of a tensor from (C, H, W) to 
    (H, W) using a given method.
    
    Args:
        X (torch.Tensor): The tensor to reduce.
        
        method (str, optional): The method used to reduce the number of 
            channels. Must be one of "mean", "median", "max", "min", 
            "luminosity". Defaults to "mean".
            
    Raises:
        ValueError: If the method is not valid.
    
    Returns:
        torch.Tensor: The reduced tensor.
    """
    if method == "mean":
        return X.mean(dim=0)
    elif method == "median":
        return X.median(dim=0).values
    elif method == "max":
        return X.max(dim=0).values
    elif method == "min":
        return X.min(dim=0).values
    else:
        raise ValueError(
            "Invalid method. Must be one of 'mean', 'median', 'max', 'min', 'luminosity'."
        )


def spatial_reducer(
    x: torch.Tensor,
    reduction: Literal[
        "mean_abs", "mean", "median", "median_abs", "max", "max_abs", "min", "min_abs"
    ],
) -> torch.Tensor:
    """Reduces a given tensor by a given reduction method.
    
    Args:
        x (torch.Tensor): The tensor to reduce
        reduction (Literal["mean_abs", "mean", "median", 
            "median_abs", "max","max_abs", "min", "min_abs"]): 
            The reduction method to use.
                
    Return:
        torch.Tensor: The reduced tensor
        
    Raise:
        ValueError: If the reduction method is not supported.
    """
    if reduction == "mean":
        return torch.nanmean(x)
    if reduction == "mean_abs":
        return torch.nanmean(torch.abs(x))
    if reduction == "median":
        return torch.nanmedian(x)
    if reduction == "median_abs":
        return torch.nanmedian(torch.abs(x))
    if reduction == "max":
        return torch.max(x[x == x])
    if reduction == "max_abs":
        xabs = torch.abs(x)
        return torch.max(xabs[xabs == xabs])
    if reduction == "min":
        return torch.min(x[x == x])
    if reduction == "min_abs":
        xabs = torch.abs(x)
        return torch.min(xabs[xabs == xabs])
    if reduction == "sum":
        return torch.nansum(x)
    if reduction == "sum_abs":
        return torch.nansum(torch.abs(x))
    if reduction == "none" or reduction is None:
        raise ValueError("None is not a valid reduction method.")

    raise ValueError("Reduction parameter unknown.")


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


def estimate_medoid(tensor: torch.Tensor) -> torch.Tensor:
    """ Estimate the medoid of a tensor."""
    # Compute the pairwise distances between the elements of the tensor
    distances = torch.cdist(tensor[:, None], tensor[:, None])

    # Compute the sum of distances for each element
    sums = distances.sum(dim=1)

    # Select the element with the smallest sum of distances
    medoid_index = sums.argmin()
    medoid = tensor[medoid_index]

    return medoid


def do_square(
    tensor: torch.Tensor,
    patch_size: Optional[int] = 32,
    get_pad: bool = False,
    constant_value: float = torch.nan,
) -> torch.Tensor:
    """ Split a tensor into n_patches x n_patches patches and return
    the patches as a tensor.
    
    Args:
        tensor (torch.Tensor): The tensor to split.
        n_patches (int, optional): The number of patches to split the tensor into.
            If None, the tensor is split into the smallest number of patches.

    Returns:
        torch.Tensor: The patches as a tensor.
    """
    # tensor = torch.rand(3, 100, 100)
    # Check if it is a square tensor
    if tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError("The tensor must be square.")

    if patch_size is None:
        patch_size = 1

    # tensor (C, H, W)
    xdim = tensor.shape[1]
    ydim = tensor.shape[2]

    minimages_x = int(torch.ceil(torch.tensor(xdim / patch_size)))
    minimages_y = int(torch.ceil(torch.tensor(ydim / patch_size)))

    pad_x_01 = int((minimages_x * patch_size - xdim) // 2)
    pad_x_02 = int((minimages_x * patch_size - xdim) - pad_x_01)

    pad_y_01 = int((minimages_y * patch_size - ydim) // 2)
    pad_y_02 = int((minimages_y * patch_size - ydim) - pad_y_01)

    padded_tensor = torch.nn.functional.pad(
        input=tensor,
        pad=(pad_x_01, pad_x_02, pad_y_01, pad_y_02),
        mode="constant",
        value=constant_value,
    )

    # split the tensor (C, H, W) into (n_patches, n_patches, C, H, W)
    patches = padded_tensor.unfold(1, patch_size, patch_size).unfold(
        2, patch_size, patch_size
    )

    # move the axes (C, n_patches, n_patches, H, W) -> (n_patches, n_patches, C, H, W)
    patches = patches.permute(1, 2, 0, 3, 4)

    # compress dimensions (n_patches, n_patches, C, H, W) -> (n_patches*n_patches, C, H, W)
    patches = patches.reshape(-1, *patches.shape[2:])

    if get_pad:
        return patches, (pad_x_01, pad_x_02, pad_y_01, pad_y_02)

    return patches


def get_numbers(string: str) -> str:
    """ Get the numbers from a string. """
    n_patches = string.replace("patch", "")
    if n_patches == "":
        n_patches = 32
    else:
        try:
            n_patches = int(n_patches)
        except:
            raise ValueError("Invalid patch name.")
    return n_patches
