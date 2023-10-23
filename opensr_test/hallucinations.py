from typing import List, Optional

import torch
from opensr_test.distance import get_distance


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
    """ Obtain the distances between metrics in a 2D space.
        In the 2D space the y-axis is the distance between the SR and HR
        (error) and the x-axis is the distance between the LR and HR 
        (high-frequency).
        

    Args:
        lr_to_hr (torch.Tensor): The LR image upscaled to HR.
        sr_harm (torch.Tensor): The SR image harmonized to HR.
        hr (torch.Tensor): The HR image.
        distance_method (str, optional): The distance method to use. Defaults to 
            "psnr". Available methods are: psnr, kl, l1, l2, psnr, sad, lpips.

    Returns:
        torch.Tensor: The distances between the metrics.
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
    ).compute()

    dist_sr_to_hr = get_distance(
        x=sr_harm,
        y=hr,
        method=distance_method,
        agg_method=agg_method,
        patch_size=patch_size,
        device=device,
        scale=scale,
        rgb_bands=rgb_bands,
    ).compute()

    dist_sr_to_lr = get_distance(
        x=sr_harm,
        y=lr_to_hr,
        method=distance_method,
        agg_method=agg_method,
        patch_size=patch_size,
        device=device,
        scale=scale,
        rgb_bands=rgb_bands,
    ).compute()

    return reference.value, dist_sr_to_hr.value, dist_sr_to_lr.value


def tc_metric(d_im: torch.Tensor, d_om: torch.Tensor) -> torch.Tensor:
    """ Obtain the coordinates from both the center
    of the consystency line and the SR model

    Args:
        lr_to_hr (torch.Tensor): A tensor (B, C, H, W)
        sr_harm (torch.Tensor): A tensor (B, C, H, W)
        hr (torch.Tensor): A tensor (B, C, H, W)
    """
    # Make the distance relative to the center of the consistency line

    # dot product [1, 1] * [d_im, d_om]
    n = d_im.size(0)
    H = torch.matmul(
        torch.vstack([d_om.ravel(), d_im.ravel()]).T, torch.tensor([1, 1]).float()
    ).reshape(n, n)
    I = torch.matmul(
        torch.vstack([d_im.ravel(), d_om.ravel()]).T, torch.tensor([-1, 1]).float()
    ).reshape(n, n)
    H = H - 1
    TCscore = (I + torch.sign(I)) * torch.exp(-H) / 2
    return TCscore


def tc_metric_02(d_im: torch.Tensor, d_om: torch.Tensor) -> torch.Tensor:
    """ Obtain the coordinates from both the center
    of the consystency line and the SR model

    Args:
        lr_to_hr (torch.Tensor): A tensor (B, C, H, W)
        sr_harm (torch.Tensor): A tensor (B, C, H, W)
        hr (torch.Tensor): A tensor (B, C, H, W)
    """
    n = d_im.size(0)
    H = torch.matmul(
        torch.vstack([d_om.ravel(), d_im.ravel()]).T, torch.tensor([1, 1]).float()
    ).reshape(n, n)
    I = torch.matmul(
        torch.vstack([d_im.ravel(), d_om.ravel()]).T, torch.tensor([-1, 1]).float()
    ).reshape(n, n)
    H = H - 1
    TCscore = (I + torch.sign(I)) / (2 * torch.exp(H + 1))

    return TCscore
