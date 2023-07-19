import torch
from typing import Optional

def spectral_angle_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Spectral angle distance between two tensors.
    Args:
        x: Tensor of shape (N, C, H, W).
        y: Tensor of shape (N, C, H, W).
    Returns:
        Tensor of shape (N, C, H, W).
    """
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
    return torch.acos(torch.clamp(torch.sum(x_norm*y_norm, dim=1), -1, 1))


def spectral_information_divergence(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Spectral information divergence between two tensors.
    Args:
        x: Tensor of shape (N, C, H, W).
        y: Tensor of shape (N, C, H, W).
    Returns:
        Tensor of shape (N, C, H, W).
    """
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
    return torch.sum(x_norm*torch.log(x_norm/y_norm), dim=1)

def spectral_pbias(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Percent bias between two tensors.
    Args:
        x: Tensor of shape (N, C, H, W).
        y: Tensor of shape (N, C, H, W).
    Returns:
        Tensor of shape (N, C, H, W).
    """
    return torch.max(torch.abs(x - y), axis=0).values


def spectral_metric(
    image1: torch.Tensor,
    image2: torch.Tensor,
    metric: str = 'pbias',
    grid: Optional[bool]=True
):
    """ Calculate spectral metrics between two images.

    Args:
        image1 (torch.Tensor): A tensor of shape (B, H, W).
        image2 (torch.Tensor): A tensor of shape (B, H, W).
        metric (str, optional): _description_. Defaults to 'sad'.
        grid (Optional[bool], optional): _description_. Defaults to True.
    """        

    if metric == 'sad':
        metric_value = spectral_angle_distance(image1, image2)
    elif metric == 'sid':
        metric_value = spectral_information_divergence(image1, image2)
    elif metric == 'pbias':
        metric_value = spectral_pbias(image1, image2)
    else:
        raise NotImplementedError(f"Metric {metric} not implemented.")
    
    if not grid:
        return metric_value.mean()
    
    return metric_value


if __name__ == "__main__":
    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    x = torch.rand([3, 4, 4], generator=torch.manual_seed(42))
    y = torch.rand([3, 4, 4], generator=torch.manual_seed(123))
    error = spectral_metric(x, y, metric='pbias', grid=False)