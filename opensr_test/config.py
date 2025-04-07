from typing import Any, List, Literal, Optional, Union

import torch
from pydantic import BaseModel, field_validator, model_validator

DistanceMetrics = Literal[
    "kl", "l1", "l2", "pbias", "psnr", "sad", "mtf", "lpips", "clip", "fd", "nd"
]


class Config(BaseModel):
    # General parameters
    device: Union[str, Any] = "cpu"
    agg_method: str = "pixel"  # pixel, image, patch
    patch_size: Optional[int] = None
    border_mask: Optional[int] = 16
    rgb_bands: Optional[List[int]] = [0, 1, 2]
    harm_apply_spectral: bool = True
    harm_apply_spatial: bool = True
    clip_model_path: Optional[str] = None

    # Spatial parameters
    spatial_method: Literal["ecc", "pcc", "lgm"] = "pcc"
    spatial_threshold_distance: int = 5
    spatial_max_num_keypoints: int = 500

    # Spectral parameters
    reflectance_distance: DistanceMetrics = "l1"
    spectral_distance: DistanceMetrics = "sad"

    # Synthesis parameters
    synthesis_distance: DistanceMetrics = "l1"

    # Correctness parameters
    correctness_distance: DistanceMetrics = "nd"
    correctness_norm: Literal["softmin", "percent"] = "softmin"
    im_score: Optional[float] = 0.05
    om_score: Optional[float] = 0.05
    ha_score: Optional[float] = 0.05  # be quiet conservative about hallucinations
    correctness_temperature: Optional[float] = (
        0.25  # be quiet conservative about hallucinations
    )

    # General parameters - validator ----------------------------
    @field_validator("device")
    def check_device(cls, value) -> str:
        return torch.device(value)

    @field_validator("agg_method")
    def check_agg_method(cls, value) -> str:
        valid_methods = ["pixel", "image", "patch"]
        if value not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of {valid_methods}")
        return value

    @field_validator("rgb_bands")
    def check_rgb_bands(cls, value) -> List[int]:
        if len(value) != 3:
            raise ValueError("rgb_bands must have 3 elements.")
        return value

    @field_validator("spatial_max_num_keypoints")
    def check_spatial_max_num_keypoints(cls, value) -> int:
        if value < 0:
            raise ValueError("spatial_max_num_keypoints must be positive.")
        return value

    @field_validator("spatial_threshold_distance")
    def check_spatial_threshold_distance(cls, value) -> int:
        if value < 0:
            raise ValueError("spatial_threshold_distance must be positive.")
        return value

    # Create SRharm - validator ------------------------------------------
    @field_validator("harm_apply_spectral")
    def check_harm_apply_spectral(cls, value) -> bool:
        if not isinstance(value, bool):
            raise ValueError("harm_apply_spectral must be boolean.")
        return value

    @field_validator("harm_apply_spatial")
    def check_harm_apply_spatial(cls, value) -> bool:
        if not isinstance(value, bool):
            raise ValueError("harm_apply_spatial must be boolean.")
        return value


class Consistency(BaseModel):
    reflectance: Any
    spectral: Any
    spatial: Any


class Synthesis(BaseModel):
    distance: Any


class Correctness(BaseModel):
    omission: Any
    improvement: Any
    hallucination: Any
    classification: Any


class Auxiliar(BaseModel):
    sr_harm: Any
    lr_to_hr: Any
    d_ref: Any
    d_im: Any
    d_om: Any


class Results(BaseModel):
    consistency: Consistency
    synthesis: Synthesis
    correctness: Correctness
    auxiliar: Auxiliar
