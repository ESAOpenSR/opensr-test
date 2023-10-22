import torch
import math

from pydantic import BaseModel, field_validator
from typing import Optional, Union, Any, List


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


class TCscore(BaseModel):
    tc_score: Any
    ha_percent: float
    om_percent: float
    im_percent: float
    
    @classmethod
    @field_validator("tc_score")
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
    score: TCscore
    auxiliar: Auxiliar


class Config(BaseModel):
    # General parameters
    agg_method: str = "pixel" # pixel, image, patch[2...image]
    patch_size: Optional[int] = 32
    rgb_bands: Optional[List[int]] = [0, 1, 2]
    
    # Downsampling/Upsampling parameters
    downsample_method: str = "classic" #
    upsample_method: str = "classic"
            
    # Spatial parameters
    spatial_features: str = "superpoint"
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
    
    
    # Downsampling/Upsampling parameters - validator --------------    
    @field_validator("downsample_method")
    @classmethod
    def check_downsample_method(cls, value) -> str:
        valid_methods = ["classic", "naip", "spot", "venus"]
        if value not in valid_methods:
            raise ValueError(f"Invalid downsample_method. Must be one of {valid_methods}")
        return value
    
    @field_validator("upsample_method")
    @classmethod
    def check_upsample_method(cls, value) -> str:
        valid_methods = ["classic"]
        if value not in valid_methods:
            raise ValueError(f"Invalid upsample_method. Must be one of {valid_methods}")
        return value
    
    # Spatial parameters - validator ----------------------------
    
    @field_validator("spatial_features")
    @classmethod
    def check_spatial_features(cls, value) -> str:
        valid_methods = ["disk", "superpoint"]
        if value not in valid_methods:
            raise ValueError(f"Invalid spatial_features. Must be one of {valid_methods}")
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
            raise ValueError(f"Invalid reflectance_method. Must be one of {valid_methods}")
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
        valid_methods = ["psnr", "lpips", "sam", "l1", "l2", "kl", "pbias"]
        if value not in valid_methods:
            raise ValueError(f"Invalid unsys_method. Must be one of {valid_methods}")
        return value