import torch
from opensr_test.distance import get_distance

def reflectance_metric(
    lr: torch.Tensor,
    sr_to_lr: torch.Tensor,
    metric: str
):
    """ Calculate spectral metrics between two images.

    Args:
        lr (torch.Tensor): The LR image.
        sr_to_lr (torch.Tensor): The SR image converted to LR space.
        metric (str, optional): The metric to use. The following metrics 
            are supported: "sid", "sd", "pbias".
        grid (Optional[bool], optional): Whether to return the metric 
            as a grid or not
    """
    if not metric in ["kl", "l1", "l2", "pbias"]:
        raise ValueError(f"Invalid metric. Must be one of 'kl', 'l1', 'l2', 'pbias'")
        
    metric_value = get_distance(lr, sr_to_lr, method=metric, agg_method="pixel").compute()
    
    return metric_value