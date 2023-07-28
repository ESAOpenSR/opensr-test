import torch
from opensr_test.utils import hq_histogram_matching
from opensr_test.spatial_check import (
    spatial_setup_model,
    spatial_get_matching_points,
    spatial_model_fit,
    spatial_model_transform
)
from opensr_test.hf_check import hf_metric


from typing import Optional, Dict, Any


def hallucination_unsystematic_error(
    lr_image: torch.Tensor,
    sr_image: torch.Tensor,
    hr_image: torch.Tensor,
    spatial_fix: bool = True,
    spatial_params:  Optional[Dict[str, Any]] = dict(),
    spectral_fix: bool = True,
    device: str = "cuda"
) -> torch.Tensor:
    """ Estimates the unsystematic error.
    Args:

        image1 (torch.Tensor):  The LR image as a tensor (C, H, W).
        image2 (torch.Tensor):  The SR image as a tensor (C, H, W).
        image3 (torch.Tensor):  The HR image as a tensor (C, H, W).

    Returns:
        torch.Tensor: The SR image with unsystematic error compensation (C, H, W).
    """
    # Remove the systematic error in the SR image

    # Set default spatial parameters
    base_spatial_params = {
            "features": "disk",
            "matcher": "lightglue",
            "max_num_keypoints": 2048,
            "threshold_distance": 5,
            "threshold_npoints": 5,
            "degree": 1,
            "precision": 4,
            "interpolation_mode": "nearest"            
    }

    # Overwrite the default spatial parameters
    if spatial_params is None:
        spatial_params = base_spatial_params
    else:
        spatial_params = {**base_spatial_params, **spatial_params}


    # Fix the HR image considering the LR image (systematic error - I)
    if spectral_fix:
        new_sr_image = hq_histogram_matching(sr_image, lr_image)
    else:
        new_sr_image = sr_image
    
    # Fix the HR image considering the LR image (systematic error - II)
    if spatial_fix:
        
        # Set spatial models
        models = spatial_setup_model(
            device=device,
            features=spatial_params["features"],
            matcher=spatial_params["matcher"],
            max_num_keypoints=spatial_params["max_num_keypoints"]
        )
        
        # From SR to LR
        lr_hat = torch.nn.functional.interpolate(
            hr_image[None],
            size=lr_image.shape[-2:],
            mode="bilinear",
            antialias=True
        ).squeeze(0)

        # Get the points and matches
        matching_points = spatial_get_matching_points(
            img01=lr_image,
            img02=lr_hat,
            model=models,
            device=device
        )

        # Fix a image according to the matching points
        spatial_models, spatial_rmse = spatial_model_fit(
            matching_points=matching_points,
            threshold_distance=spatial_params["threshold_distance"],
            degree=spatial_params["degree"],
            return_rmse=True
        ).values()
        
        if not (spatial_rmse[1] > spatial_rmse[0]):
            new_sr_image = spatial_model_transform(
                image1=new_sr_image,
                spatial_models=spatial_models,
                precision=spatial_params["precision"],
                interpolation_mode=spatial_params["interpolation_mode"],
                device=device
            )
    else:
        new_sr_image = sr_image


    # Estimate the unsystematic error
    unsystematic_error = torch.abs(hr_image - new_sr_image)


    # Obtain the high-frequency component
    high_frecuency = hf_metric(lr_image, new_sr_image, metric="convolve")
    
    # Obtain the hallucinations and omissions
    hallucinations = unsystematic_error * high_frecuency
    omissions = unsystematic_error * (1 - high_frecuency)

    return hallucinations, omissions

if __name__ == "__main__":

    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    import pathlib
    from opensr_test.lightglue.utils import load_image

    pathdir = pathlib.Path("demo/ROI_05021/")
    image1 = load_image(pathdir / "m_4111801_sw_11_060_20190831.tif")
    image2 = load_image(pathdir / "m_4111908_se_11_1_20130720.tif")
    
    image1 = image1[:, 900:1400, 900:1400]
    image2 = image2[:, 900:1400, 900:1400]
    
    lr_image = torch.nn.functional.interpolate(
        image1[None],
        scale_factor=0.5,
        mode="bilinear",
        antialias=True

    ).squeeze(0)
    sr_image = image1
    hr_image = image2