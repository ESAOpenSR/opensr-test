from typing import Literal, Dict, List, Optional, Tuple

import torch
from skimage.exposure import match_histograms
from sklearn.pipeline import Pipeline


def spectral_reducer(
    X: torch.Tensor,
    method: Literal["mean", "median", "max", "min", "luminosity"] = "luminosity",
    rgb_bands: Optional[List[int]] = [0, 1, 2],
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
    elif method == "luminosity":
        return (
            0.2126*X[rgb_bands[0]] +
            0.7152*X[rgb_bands[1]] +
            0.0722*X[rgb_bands[2]]
        )
    else:
        raise ValueError(
            "Invalid method. Must be one of 'mean', 'median', 'max', 'min', 'luminosity'."
        )


def spatial_reducer(
    x: torch.Tensor,
    reduction: Literal[
        "mean_abs", "mean", "median", "median_abs",
        "max","max_abs", "min", "min_abs"
    ]
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

class Value:
    def __init__(
        self,
        value: torch.Tensor,
        description: str,
        points: Optional[Tuple[List[int], List[int]]] = None,
        affine_model: Optional[Tuple[Pipeline, Pipeline]] = None,
        matching_points: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """ A class to store a value and its description.
        It is useful to create plots with titles that 
        highlight the method used for the computation of the 
        metric.
        
        Args:
            value (torch.Tensor): The metric value.
            description (str): The name of the metric used to compute 
                the value.
            points (Optional[Tuple[List[int], List[int]]], optional):
                The points used to fit the affine model. Useful for
                spatial error plots. Defaults to None.
            affine_model (Optional[Tuple[Pipeline, Pipeline]], optional):
                The affine model (X, y) used to compute the spatial error.
                It is a Pipeline object (see sklearn). Defaults to None.
            matching_points (Optional[Dict[str, torch.Tensor]], optional):
                Results of the matching algorithm. Defaults to None.
        """

        self.value = value
        self.description = description
        self.points = points
        self.affine_model = affine_model
        self.matching_points = matching_points

    def __repr__(self):
        return "%s: %s" % (self.description, self.value)


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
