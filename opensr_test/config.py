from pydantic import BaseModel, field_validator
from typing import Optional, List

class Config(BaseModel):
    # General parameters
    spectral_reducer: str = "mean"
    rgb_bands: List[int] = [0, 1, 2]    
    
    # Downsampling/Upsampling parameters
    downsample_method: str = "classic" # tailored
    upsample_method: str = "classic"
            
    # Spatial parameters
    spatial_features: str = "disk"
    spatial_matcher: str = "lightglue"
    spatial_max_num_keypoints: int = 1000
    spatial_threshold_distance: int = 3
    spatial_threshold_npoints: int = 5    
    
    # Spectral parameters
    spectral_global_method: str = "difference"
    spectral_global_reducer: str = "mean_abs"
    
    spectral_local_method: str = "sad"
    spectral_local_reducer: str = "mean_abs"

    # High.frequency parameters
    hf_method: str = "spatial_domain"
    hf_reducer: str = "mean_abs"
    
    # Create SRharm
    harm_apply_spectral: bool = True
    harm_apply_spatial: bool = True

    # Unsystematic error parameters
    unsys_method: str = "cpsnr"

    # Perceptual error parameters
    # LPIPS (only alexnet is supported)
    perceptual_method: str = "LPIPS"
    apply_stretch: Optional[str] = "no_stretch"
    
    # General parameters - validator ----------------------------
    
    @field_validator("spectral_reducer")
    @classmethod
    def check_spectral_reducer(cls, value) -> str:
        valid_methods = ["mean", "median", "max", "min", "luminosity"]
        if value not in valid_methods:
            raise ValueError(f"Invalid spectral_reducer method. Must be one of {valid_methods}")
        return value        
    
    
    @field_validator("rgb_bands")
    @classmethod
    def check_rgb_bands(cls, value) -> List[int]:
        if len(value) != 3:
            raise ValueError("rgb_bands must have 3 values.")
        
        if any([x > 3 for x in value]):
            return ValueError("rgb_bands must be between 0 and 3.")
        
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
    
    @field_validator("spectral_global_method")
    @classmethod
    def check_spectral_global_method(cls, value) -> str:
        valid_methods = ["difference", "simple_ratio", "information_divergence"]
        if value not in valid_methods:
            raise ValueError(f"Invalid spectral_global_method. Must be one of {valid_methods}")
        return value

    @field_validator("spectral_global_reducer")
    @classmethod
    def check_spectral_global_reducer(cls, value) -> str:
        valid_methods = [
            "mean_abs", "mean", "median", "median_abs", 
            "max","max_abs", "min", "min_abs"
        ]
        if value not in valid_methods:
            raise ValueError(f"Invalid spectral_global_reducer. Must be one of {valid_methods}")
        return value

    # Local ---------------------------------
    
    @field_validator("spectral_local_method")
    @classmethod
    def check_spectral_local_method(cls, value) -> str:
        valid_methods = ["sad"]
        if value not in valid_methods:
            raise ValueError(f"Invalid spectral_local_method. Must be one of {valid_methods}")
        return value


    @field_validator("spectral_local_reducer")
    @classmethod
    def check_spectral_local_reducer(cls, value) -> str:
        valid_methods = [
            "mean_abs", "mean", "median", "median_abs", 
            "max","max_abs", "min", "min_abs"
        ]
        if value not in valid_methods:
            raise ValueError(f"Invalid spectral_local_reducer. Must be one of {valid_methods}")
        return value

    
    # High-frequency parameters - validator ----------------------------
    
    @field_validator("hf_method")
    @classmethod
    def check_hf_method(cls, value) -> str:
        valid_methods = ["spatial_domain", "frequency_domain", "simple"]
        if value not in valid_methods:
            raise ValueError(f"Invalid hf_method. Must be one of {valid_methods}")
        return value
    
    @field_validator("hf_reducer")
    @classmethod
    def check_hf_reducer(cls, value) -> str:
        valid_methods = [
            "mean_abs", "mean", "median", "median_abs", 
            "max","max_abs", "min", "min_abs"
        ]
        if value not in valid_methods:
            raise ValueError(f"Invalid hf_reducer. Must be one of {valid_methods}")
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
    
    @field_validator("unsys_method")
    @classmethod
    def check_unsys_method(cls, value) -> str:
        valid_methods = ["cpsnr", "psnr"]
        if value not in valid_methods:
            raise ValueError(f"Invalid unsys_method. Must be one of {valid_methods}")
        return value

    # Create Perceptual parameters - validator -------------------------
    
    @field_validator("perceptual_method")
    @classmethod
    def check_perceptual_method(cls, value) -> str:
        valid_methods = ["LPIPS"]
        if value not in valid_methods:
            raise ValueError(f"Invalid perceptual_method. Must be one of {valid_methods}")
        return value
    
    @field_validator("apply_stretch")
    @classmethod
    def check_apply_stretch(cls, value) -> str:
        valid_methods = [None, "linear", "histogram"]        
        if value not in valid_methods:
            raise ValueError(f"Invalid apply_stretch. Must be one of {valid_methods}")
        return value

    def full_check(self, quiet: bool = False) -> None:
        """Check all parameters."""
        self.check_spectral_reducer(self.spectral_reducer)
        self.check_rgb_bands(self.rgb_bands)
        
        self.check_downsample_method(self.downsample_method)
        self.check_upsample_method(self.upsample_method)
        
        self.check_spatial_features(self.spatial_features)
        self.check_spatial_matcher(self.spatial_matcher)
        self.check_spatial_max_num_keypoints(self.spatial_max_num_keypoints)
        self.check_spatial_threshold_distance(self.spatial_threshold_distance)
        self.check_spatial_threshold_npoints(self.spatial_threshold_npoints)
        
        self.check_spectral_global_method(self.spectral_global_method)
        self.check_spectral_global_reducer(self.spectral_global_reducer)
        
        self.check_spectral_local_method(self.spectral_local_method)
        self.check_spectral_local_reducer(self.spectral_local_reducer)
        
        self.check_hf_method(self.hf_method)
        self.check_hf_reducer(self.hf_reducer)
        
        self.check_harm_apply_spectral(self.harm_apply_spectral)
        self.check_harm_apply_spatial(self.harm_apply_spatial)
        
        self.check_unsys_method(self.unsys_method)        
        
        self.check_perceptual_method(self.perceptual_method)
        self.check_apply_stretch(self.apply_stretch)
        
        if not quiet:
            print("Great job! All parameters are valid.")