from typing import Optional, Callable
from opensr_test.utils import Value
import torch

def metric_shift(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    space_search: int = 4,
    minimize: bool = True
) -> torch.Tensor:
    """ Given two tensors, x and y, compute the metric for all possible 
    2D shifts

    Args:
        x (torch.Tensor): The first tensor (H, W)
        y (torch.Tensor): The second tensor (H, W)
        metric (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The
            metric to use.         
        space_search (int, optional): This parameter is used to search 
            for the best shift that maximizes the PSNR. By default, it is 
            the same as the super-resolution scale.
        minimize (bool, optional): The metric is minimized if True, otherwise
            maximized. Defaults to True.
            
    Returns:
        torch.Tensor: The metric value and the shift that 
            minimizes/maximizes the metric.
    """
    s = space_search // 2
    x_padded = torch.nn.functional.pad(
        input=x,
        pad=(s, s, s, s),
        mode='constant',
        value=torch.nan
    )
    
    container = torch.zeros([space_search+1, space_search+1])
    for xshift in range(space_search+1):
        for yshift in range(space_search+1):
            x_shifted = x_padded[
                xshift:(xshift + x.shape[0]), yshift:(yshift + x.shape[1])
            ]
            m = metric(x_shifted, y).value
            container[xshift, yshift] = m
    
    # Get the position of the minimum/maximum
    if minimize:
        mask = (container == container.min())*1
    else:
        mask = (container == container.max())*1
    idx, idy = mask.nonzero(as_tuple=True)
    idx, idy = int(idx), int(idy)
    
    # Get the BEST shifted tensor
    x_best = x_padded[
        idx:(idx + x.shape[0]), idy:(idy + x.shape[1])
    ]
    
    return x_best

def simple_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the simple difference between two tensors
    
    Args:
        x (torch.Tensor): The first tensor (H, W)
        y (torch.Tensor): The second tensor (H, W)    
    """
    return torch.abs(x - y)

def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the PSNR between two tensors
    
    Args:
        x (torch.Tensor): The first tensor (H, W)
        y (torch.Tensor): The second tensor (H, W)    
    """
    return Value(
        value = 1 / (float(-10 * torch.log10(torch.nanmean((x - y) ** 2)))),
        description = "PSNR"
    )


def cpsnr(
    x: torch.Tensor,
    y: torch.Tensor,
    space_search: int
) -> torch.Tensor:
    """Compute the cPSNR between two tensors
    
    Args:
        x (torch.Tensor): The first tensor (H, W)
        y (torch.Tensor): The second tensor (H, W)
        space_search (int, optional): This parameter is used to search 
        for the best shift that maximizes the PSNR. By default, it is 
        the same as the super-resolution scale.
    """
    x_shifted = metric_shift(
        x, y, 
        metric=psnr,
        space_search=space_search,
        minimize=True
    )
    
    return Value(
        value = float(psnr(x_shifted, y).value),
        description = "cPSNR"
    )


def unsystematic_error(
    x: torch.Tensor,
    y: torch.Tensor,
    method: str,
    grid: bool,
    space_search: Optional[int] = None
):
    """ Estimate the unsystematic error between the SR and HR image.

    Args:
        x (torch.Tensor): The SR model (H, W).
        hr (torch.Tensor): The HR image (H, W).
        method (str): The method to use. Either "psnr" or "cpsnr".
        space_search (int, optional): This parameter is used to search 
            for the best shift that maximizes the PSNR. By default, it is 
            the same as the super-resolution scale.
        
    Returns:
        torch.Tensor: The metric value.
    """
    if not grid:
        if method == "psnr":
            unsys_err = psnr(x, y)
        elif method == "cpsnr":
            unsys_err = cpsnr(x, y, space_search)
        else:
            raise ValueError("method must be either psnr or cpsnr")
    else:
        unsys_err = Value(
            value = y - x,
            description = "HR - SRharm"
        )
        
    return unsys_err


def ha_im_ratio(
    sr: torch.Tensor,
    hr: torch.Tensor,
    lr_ref: torch.Tensor,
    hf: torch.Tensor,
    grid: bool
):  
    """Compute the hallucination/improvement ratio
    
    Args:
        sr (torch.Tensor): The SR model (H, W).
        hr (torch.Tensor): The HR image (H, W).
        lr_ref (torch.Tensor): The LR reference image (H, W).
        hf (torch.Tensor): The high-frequency image (H, W).
        grid (bool): If True, the difference between SR-HR
            on a grid. Otherwise, it is computed on the whole image.
    """
    # Obtain the SR unsystematic error: E_SR
    unsys_err_sr = simple_diff(x=sr, y=hr)

    # Obtain the SR unsystematic error: E_LR
    unsys_err_lr = simple_diff(x=lr_ref, y=hr)
    
    # Obtain the error space: E_LR - E_SR
    err_space = unsys_err_lr - unsys_err_sr
    
    if grid:
        return err_space*hf
    else:
        # Obtain the hallucinations: HF * (E_LR - E_SR) > 0        
        ha = torch.sum((hf * err_space)[err_space < 0] * -1)
        im = torch.sum((hf * err_space)[err_space > 0])
        return float(ha/im)