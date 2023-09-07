from typing import Optional

import torch


def unsystematic_error(
    sr_norm: torch.Tensor,
    hr: torch.Tensor,
    lr_to_hr: torch.Tensor,
    hf: torch.Tensor,
    grid: Optional[bool] = None,
) -> torch.Tensor:
    """ Estimate the unsystematic error between the SR and HR image.

    Args:
        sr_norm (torch.Tensor): The SR with systematic errors 
            removed (C, H, W).
        hr (torch.Tensor): The HR image (C, H, W).
        lr_to_hr (torch.Tensor): The LR image in the HR space (C, H, W).
        hf (torch.Tensor): The HF raster (C, H, W).
        grid (Optional[bool], optional): Whether to return the metric as 
            a grid or not. Defaults to None.

    Returns:
        torch.Tensor: The metric value.
    """

    # Estimate unsystematic errors
    unsys_err = torch.median(torch.abs(sr_norm - hr), dim=0)[0]
    unsys_err_ref = torch.median(torch.abs(hr - lr_to_hr), dim=0)[0]

    # Estimate the Hallucinations
    hf_norm = (hf - hf.min()) / (hf.max() - hf.min())
    unsys_err_norm = (unsys_err - unsys_err.min()) / (unsys_err.max() - unsys_err.min())
    ha = hf_norm * unsys_err_norm

    # Estimate the improvements    
    imp_r_ratio_grid = 1 - unsys_err / unsys_err_ref
    imp_r_ratio = 1 - unsys_err.median() / unsys_err_ref.median()
    
    if not grid:
        ha = ha.median()

    return ha, imp_r_ratio_grid, imp_r_ratio
