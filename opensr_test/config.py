from typing import Any, List, Optional, Union

import torch
from pydantic import BaseModel, field_validator, model_validator


class DatasetParams(BaseModel):
    blur_gaussian_sigma: Optional[List[float]]
    stability_threshold: Union[List[float], float]
    correctness_params: List[float]
    downsample_method: str
    upsample_method: str


class Metric(BaseModel):
    value: Any
    description: str

    @classmethod
    @field_validator("value")
    def check_value(cls, value) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise ValueError("value must be a torch.Tensor.")
        return value


class Consistency(BaseModel):
    reflectance: Metric
    spectral: Metric
    spatial: Metric


class Distance(BaseModel):
    lr_to_hr: Any
    sr_to_hr: Any
    sr_to_lr: Any

    @classmethod
    @field_validator("lr_to_hr", "sr_to_hr", "sr_to_lr")
    def check_value(cls, value) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise ValueError("value must be a torch.Tensor.")
        return value


class Correctness(BaseModel):
    omission: Any
    improvement: Any
    hallucination: Any
    classification: Any

    @classmethod
    @field_validator("omission", "improvement", "hallucination", "classification")
    def check_value(cls, value) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise ValueError("value must be a torch.Tensor.")
        return value


class Auxiliar(BaseModel):
    sr_harm: Any
    lr_to_hr: Any
    matching_points_lr: Union[dict, bool]
    matching_points_hr: Union[dict, bool]

    @classmethod
    @field_validator("sr_harm", "lr_to_hr")
    def check_value(cls, value) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise ValueError("value must be a torch.Tensor.")
        return value


class Results(BaseModel):
    consistency: Consistency
    distance: Distance
    correctness: Correctness
    auxiliar: Auxiliar


class Config(BaseModel):
    # General parameters
    agg_method: str = "pixel"  # pixel, image
    mask: Optional[int] = 32
    patch_size: Optional[int] = None        
    rgb_bands: Optional[List[int]] = [0, 1, 2]

    # downsample/upsample method
    downsample_method: Optional[str] = "classic"
    upsample_method: Optional[str] = "classic"

    # Spatial parameters
    spatial_features: str = "disk"
    spatial_matcher: str = "lightglue"
    spatial_max_num_keypoints: int = 1000
    spatial_threshold_distance: int = 3
    spatial_threshold_npoints: int = 5

    # Spectral parameters
    reflectance_method: str = "l1"
    spectral_method: str = "sad"

    # Create SRharm
    harm_apply_spectral: bool = True
    harm_apply_spatial: bool = True

    # Unsystematic error parameters
    distance_method: str = "l1"

    # General parameters - validator ----------------------------
    @field_validator("agg_method")
    @classmethod
    def check_agg_method(cls, value) -> str:
        valid_methods = ["pixel", "image", "patch"]
        if value not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of {valid_methods}")
        return value

    @field_validator("rgb_bands")
    @classmethod
    def check_rgb_bands(cls, value) -> List[int]:
        if len(value) != 3:
            raise ValueError("rgb_bands must have 3 elements.")
        return value


    # Spatial parameters - validator ----------------------------

    @field_validator("spatial_features")
    @classmethod
    def check_spatial_features(cls, value) -> str:
        valid_methods = ["disk", "superpoint", "sift", "aliked", "doghardnet"]
        if value not in valid_methods:
            raise ValueError(
                f"Invalid spatial_features. Must be one of {valid_methods}"
            )
        return value

    @field_validator("spatial_matcher")
    @classmethod
    def check_spatial_matcher(cls, value) -> str:
        valid_methods = ["lightglue", "superglue"]
        if value not in valid_methods:
            raise ValueError(f"Invalid spatial_matcher. Must be one of {valid_methods}")
        return value

    @field_validator("spatial_max_num_keypoints")
    @classmethod
    def check_spatial_max_num_keypoints(cls, value) -> int:
        if value < 0:
            raise ValueError("spatial_max_num_keypoints must be positive.")
        return value

    @field_validator("spatial_threshold_distance")
    @classmethod
    def check_spatial_threshold_distance(cls, value) -> int:
        if value < 0:
            raise ValueError("spatial_threshold_distance must be positive.")
        return value

    @field_validator("spatial_threshold_npoints")
    @classmethod
    def check_spatial_threshold_npoints(cls, value) -> int:
        if value < 0:
            raise ValueError("spatial_threshold_npoints must be positive.")
        return value

    # Spectral parameters - validator ----------------------------

    # Global ---------------------------------
    @field_validator("reflectance_method")
    @classmethod
    def check_reflectance_method(cls, value) -> str:
        valid_methods = ["kl", "l1", "l2", "pbias"]
        if value not in valid_methods:
            raise ValueError(
                f"Invalid reflectance_method. Must be one of {valid_methods}"
            )
        return value

    # Local ---------------------------------
    @field_validator("spectral_method")
    @classmethod
    def check_spectral_method(cls, value) -> str:
        valid_methods = ["sad"]
        if value not in valid_methods:
            raise ValueError(f"Invalid spectral_method. Must be one of {valid_methods}")
        return value

    # Create SRharm - validator ------------------------------------------
    @field_validator("harm_apply_spectral")
    @classmethod
    def check_harm_apply_spectral(cls, value) -> bool:
        if not isinstance(value, bool):
            raise ValueError("harm_apply_spectral must be boolean.")
        return value

    @field_validator("harm_apply_spatial")
    @classmethod
    def check_harm_apply_spatial(cls, value) -> bool:
        if not isinstance(value, bool):
            raise ValueError("harm_apply_spatial must be boolean.")
        return value

    # Create Unsystematic error parameters - validator -------------------------

    @field_validator("distance_method")
    @classmethod
    def check_unsys_method(cls, value) -> str:
        valid_methods = ["ipsnr", "lpips", "clip", "sam", "l1", "l2", "kl", "pbias"]
        if value not in valid_methods:
            raise ValueError(f"Invalid unsys_method. Must be one of {valid_methods}")
        return value

    @model_validator(mode='after')
    def check_perceptual_metrics(cls, values):
        if values.distance_method in ["lpips", "clip"]:
            if values.agg_method == "pixel":
                raise ValueError("agg_method must be image or patch for {values.distance_method}")
        return values

def create_param_config(dataset = "naip") -> DatasetParams:
    if dataset == "naip":
        params = DatasetParams(
            blur_gaussian_sigma = [2.28, 2.16, 2.10, 2.42],
            stability_threshold = [
                0.02, 0.016, 0.025, 0.015, 0.023, 0.019,
                0.021, 0.019, 0.012, 0.027, 0.016, 0.02,
                0.018, 0.02, 0.018, 0.019, 0.019, 0.019,
                0.013, 0.013, 0.013, 0.015, 0.015, 0.015,
                0.018, 0.017, 0.016, 0.015, 0.022, 0.016
            ],
            correctness_params=[0.80, 0.80, 0.40],
            downsample_method = "naip",
            upsample_method = "classic"
        )
    elif dataset == "spot":
        params = DatasetParams(
            blur_gaussian_sigma = [2.20, 2.18, 2.09, 2.76],
            stability_threshold = [
                0.014, 0.016, 0.013, 0.010, 0.032,
                0.040, 0.014, 0.037, 0.030, 0.036,
                0.026, 0.030
            ],
            correctness_params=[0.80, 0.80, 0.40],
            downsample_method = "spot",
            upsample_method = "classic"
        )

    elif dataset == "venus":
        params = DatasetParams(
            blur_gaussian_sigma = [0.5180, 0.5305, 0.5468, 0.5645],
            stability_threshold = [
                0.015, 0.012, 0.012, 0.012, 0.013, 0.011,
                0.01, 0.012, 0.011, 0.013, 0.01, 0.013,
                0.013, 0.01, 0.017, 0.01, 0.012, 0.013,
                0.011, 0.013, 0.017, 0.011, 0.01, 0.011,
                0.012, 0.015, 0.012, 0.011, 0.01, 0.017,
                0.016, 0.01, 0.011, 0.014, 0.012, 0.016,
                0.01, 0.012, 0.013, 0.01, 0.016, 0.016, 
                0.012, 0.011, 0.016, 0.01, 0.015, 0.016, 
                0.016, 0.01, 0.015, 0.011, 0.016, 0.016, 
                0.013, 0.013, 0.014, 0.01, 0.011
            ],
            correctness_params=[0.80, 0.80, 0.40],
            downsample_method = "venus",
            upsample_method = "classic"
        )
    elif dataset == "general":
        params = DatasetParams(
            blur_gaussian_sigma = None,
            stability_threshold = 0.01,
            correctness_params=[0.80, 0.80, 0.40],
            downsample_method = "classic",
            upsample_method = "classic"
        )

    return params
