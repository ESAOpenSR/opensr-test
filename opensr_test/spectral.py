import torch
from opensr_test.utils import Value


def spectral_angle_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Spectral angle distance between two tensors.
    Args:
        x: Tensor of shape (C, H, W).
        y: Tensor of shape (C, H, W).
    Returns:
        Tensor of shape (C, H, W).
    """
    x_norm = torch.nn.functional.normalize(x, p=2, dim=0)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=0)
    return Value(
        torch.acos(torch.clamp(torch.sum(x_norm * y_norm, dim=0), -1.0, 1.0)),
        "Spectral Angle Distance",
    )


def spectral_information_divergence(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Spectral information divergence between two tensors.
    Args:
        x: Tensor of shape (C, H, W).
        y: Tensor of shape (C, H, W).
    Returns:
        Tensor of shape (C, H, W).
    """
    x_norm = torch.nn.functional.normalize(x, p=1, dim=0)
    y_norm = torch.nn.functional.normalize(y, p=1, dim=0)

    return Value(
        torch.sum(x_norm * torch.log(x_norm / y_norm), dim=0),
        "Spectral Information Divergence",
    )


def spectral_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Spectal difference between two tensors.
    Args:
        x: Tensor of shape (C, H, W).
        y: Tensor of shape (C, H, W).
    Returns:
        Tensor of shape (C, H, W).
    """
    return Value(
        torch.median(torch.abs(x - y), axis=0).values, "Spectral difference [X-Y]"
    )


def spectral_pbias(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Spectral bias between two tensors.

    Args:
        x (torch.Tensor): The LR image
        y (torch.Tensor): The SR image

    Returns:
        torch.Tensor: The bias
    """
    LARGE_NUMBER = 10e2

    # Compute the median of the absolute values of x/y,
    # replacing large values with LARGE_NUMBER
    x_over_y = torch.abs(x / y)
    x_over_y[x_over_y > LARGE_NUMBER] = LARGE_NUMBER
    median = torch.median(x_over_y, axis=0).values

    return Value(median, "Spectral pbias [X/Y]")


def spectral_metric(lr: torch.Tensor, sr_to_lr: torch.Tensor, metric: str, grid: bool):
    """ Calculate spectral metrics between two images.

    Args:
        lr (torch.Tensor): The LR image.
        sr_to_lr (torch.Tensor): The SR image converted to LR space.
        metric (str, optional): The metric to use. The following metrics 
            are supported: "sad", "sid", "sd", "pbias".
        grid (Optional[bool], optional): Whether to return the metric 
            as a grid or not
    """

    if metric == "sad":
        metric_value = spectral_angle_distance(lr, sr_to_lr)
    elif metric == "information_divergence":
        metric_value = spectral_information_divergence(lr, sr_to_lr)
    elif metric == "difference":
        metric_value = spectral_diff(lr, sr_to_lr)
    elif metric == "simple_ratio":
        metric_value = spectral_pbias(lr, sr_to_lr)
    else:
        raise NotImplementedError(f"Metric {metric} not implemented.")

    if grid:
        return metric_value

    return Value(float(metric_value.value.median()), metric_value.description)
