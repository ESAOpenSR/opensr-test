from typing import List, Optional, Tuple

import numpy as np
import torch
from opensr_test.distance import get_distance


def sigm(x):
    return 1 / (1 + np.exp(-(x - 0) / 1)) 

def get_distances(
    lr_to_hr: torch.Tensor,
    sr_harm: torch.Tensor,
    hr: torch.Tensor,
    distance_method: str = "l1",
    agg_method: str = "pixel",
    scale: int = 4,
    patch_size: Optional[int] = 32,
    rgb_bands: Optional[List[int]] = [0, 1, 2],
    device: str = "cpu",
):
    """ Obtain the distances between the LR, SR and HR images.
        
    Args:
        lr_to_hr (torch.Tensor): The LR image upscaled to HR.
        sr_harm (torch.Tensor): The SR image harmonized to HR.
        hr (torch.Tensor): The HR image.
        distance_method (str, optional): The distance method to use. 
            Defaults to "psnr". Available methods are: "kl", "l1", 
            "l2", "pbias", "psnr", "sad", "mtf", "lpips", "clip".

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The distances
            between the LR and HR, SR and HR, and SR and LR images.
    """

    reference = get_distance(
        x=lr_to_hr,
        y=hr,
        method=distance_method,
        agg_method=agg_method,
        patch_size=patch_size,
        device=device,
        scale=scale,
        rgb_bands=rgb_bands,
    )

    dist_sr_to_hr = get_distance(
        x=sr_harm,
        y=hr,
        method=distance_method,
        agg_method=agg_method,
        patch_size=patch_size,
        device=device,
        scale=scale,
        rgb_bands=rgb_bands,
    )

    dist_sr_to_lr = get_distance(
        x=sr_harm,
        y=lr_to_hr,
        method=distance_method,
        agg_method=agg_method,
        patch_size=patch_size,
        device=device,
        scale=scale,
        rgb_bands=rgb_bands,
    )

    return reference, dist_sr_to_hr, dist_sr_to_lr


def tc_improvement(
    d_im: torch.Tensor,
    d_om: torch.Tensor,
    plambda: float = 0.85
) -> torch.Tensor:
    """ Obtain the relative distance to the center of the improvement space

    Args:
        d_im (torch.Tensor): The distance to the improvement space
        d_om (torch.Tensor): The distance to the omission space
        plambda (float): The parameter calibrated according to the
            human perception of the quality of the super-resolved
            image. Defaults to 0.85.

    Returns:
        torch.Tensor: The relative distance to the center 
        of the improvement space
    """
    H = d_im + d_om -1  
    return d_im + d_om*(1 - np.exp(-H*plambda))

def tc_omission(
    d_im: torch.Tensor,
    d_om: torch.Tensor,
    plambda: float = 0.85
) -> torch.Tensor:
    """ Obtain the relative distance to the center
    of the omission space

    Args:
        d_im (torch.Tensor): The distance to the improvement space
        d_om (torch.Tensor): The distance to the omission space
        plambda (float): The parameter calibrated according to the
            human perception of the quality of the super-resolved
            image. Defaults to 0.85.

    Returns:
        torch.Tensor: The relative distance to the center 
        of the improvement space
    """
    H = d_im + d_om -1  
    return d_om + d_im*(1 - np.exp(-H*plambda))


def tc_hallucination(
    d_im: torch.Tensor,
    d_om: torch.Tensor,
    plambda: float = 0.85
) -> torch.Tensor:
    """ Obtain the relative distance to the center
    of the hallucination space

    Args:
        d_im (torch.Tensor): The distance to the improvement space
        d_om (torch.Tensor): The distance to the omission space
        plambda (float): The parameter calibrated according to the
            human perception of the quality of the super-resolved
            image. Defaults to 0.85.

    Returns:
        torch.Tensor: The relative distance to the center 
        of the improvement space
    """
    Q = np.exp(-d_im * d_om * plambda)
    return Q