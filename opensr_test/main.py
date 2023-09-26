import opensr_test.plot
import numpy as np
import lpips
import torch
import math
import warnings

from opensr_test.config import Config
from typing import Any, Optional, Union, Tuple
from opensr_test.kernels import apply_downsampling, apply_upsampling

from opensr_test.hallucinations import unsystematic_error, ha_im_ratio
from opensr_test.highfrequency import highfrequency
from opensr_test.spatial import (
    spatial_get_matching_points,
    spatial_metric, spatial_model_fit,
    spatial_model_transform,
    spatial_setup_model
)
from opensr_test.spectral import spectral_metric
from opensr_test.utils import Value, hq_histogram_matching, spectral_reducer

class Metrics:
    def __init__(
        self,
        params: Optional[Config] = None,
        grid: Optional[bool] = False,
        device: Union[str, torch.device, None] = "cpu",
    ) -> None:
        """ A class to evaluate the performance of a image 
        enhancement algorithm considering the triplets: LR[input], 
        SR[enhanced], HR[ground truth].

        Args:
            params (Optional[Config], optional): The parameters to
                setup the opensr-test experiment. Defaults to None.
                If None, the default parameters are used. See 
                config.py for more information.
            grid (Optional[bool], optional): Whether to return the
                metrics as grids or not. Defaults to False.
            device (Union[str, torch.device, None], optional): The
                device to use. Defaults to "cpu".
        """

        # Get the parameters
        if params is None:
            self.params = Config()
        else:
            self.params = params
            self.params.full_check(quiet=True)
        
        
        # Setup the spatial model
        self.spatial_model = spatial_setup_model(
            features=self.params.spatial_features,
            matcher=self.params.spatial_matcher,
            max_num_keypoints=self.params.spatial_max_num_keypoints,
            device=device
        )

        # Setup perceptual model
        self.perceptual_model = lpips.LPIPS(
            net="alex",
            verbose=False
        ).to(device)
                
        # Initial triplets: LR[input], SR[enhanced], HR[ground truth]
        self.lr = None
        self.sr = None
        
        self.hr = None
        self.hr_reduced = None

        # The sr without systematic error (spectral & spatial removed)
        self.sr_harm = None
        self.sr_harm_reduced = None

        # The high-frequency information, hf_raster is the
        # high-frequency map and hf_metric is the final metric
        # after applying the spatial reducer.
        self.hf_raster = None
        self.hf_metric = None
        
        # The LR in the HR space: (C, H, W) -> (H*scale, W*scale)
        # using a bilinear interpolation (zero-parameter model)
        # also applied the spectral_reducer since we don't need
        # to evaluate the spectral information in the HR space.
        self.lr_to_hr = None
        self.lr_to_hr_reduced = None

        # The SR in the LR space: (C, H*scale, W*scale) -> (C, H, W)
        # using a bilinear interpolation (zero-parameter model).
        self.sr_to_lr = None

        # The landuse map (H, W)
        self.landuse = None
        
        # The scale value of the experiment. Obtained from the
        # ratio between the HR and LR image sizes.
        self.scale_factor = None

        # RGB pairs (used for plotting and some metrics [spatial 
        # & perceptual])
        self.lr_RGB = None
        self.hr_RGB = None
        self.sr_RGB = None
        
        self.lr_to_hr_RGB = None
        self.sr_to_lr_RGB = None
        self.sr_harm_RGB = None
        
        # Metrics to be computed
        self.spectral_global_error = None
        self.spectral_local_error = None
        self.spatial_error = None
        self.highfrequency = None
        self.unsystematic_error = None
        self.ha_im_ratio = None
        self.perceptual_ratio = None
        
        # Global parameters
        self.grid = grid
        self.device = device
        
        
    def setup(
        self,
        lr: torch.Tensor,
        sr: torch.Tensor,
        hr: torch.Tensor,
        landuse: Optional[torch.Tensor] = None,
    ) -> None:
        """ Obtain the performance metrics of the SR image.

        Args:
            lr (torch.Tensor): The LR image as a tensor (C, H, W).
            sr (torch.Tensor): The SR image as a tensor (C, H, W).
            hr (torch.Tensor): The HR image as a tensor (C, H, W).
        """
        #lr = torch.rand(4, 32, 32)
        #hr = torch.rand(4, 128, 128)
        #sr = torch.rand(4, 128, 128)
        
        # Move all the images to the same device
        self.lr = lr.to(self.device)
        self.sr = sr.to(self.device)
        self.hr = hr.to(self.device)
        self.landuse = landuse.to(self.device) if landuse is not None else None

        # Obtain the scale factor
        scale_factor = self.hr.shape[-1] / self.lr.shape[-1]
        if not scale_factor.is_integer():
            raise ValueError("The scale factor must be an integer.")
        self.scale_factor = int(scale_factor)

        # Obtain the LR in the HR space
        self.lr_to_hr = apply_downsampling(
            X=self.lr[None],
            scale=self.scale_factor,
            method=self.params.upsample_method
        ).squeeze(0)
        
        # Apply the spectral reducer to SR and HR
        self.lr_to_hr_reduced = spectral_reducer(
            X=self.lr_to_hr,
            method=self.params.spectral_reducer,
            rgb_bands=self.params.rgb_bands            
        )
        self.hr_reduced = spectral_reducer(
            X=self.hr,
            method=self.params.spectral_reducer,
            rgb_bands=self.params.rgb_bands
        )
                
        # Obtain the SR in the LR space
        self.sr_to_lr = apply_upsampling(
            X=self.sr[None],
            scale = self.scale_factor,
            method = self.params.downsample_method
        ).squeeze(0)        
        
        # Obtain the RGB images
        rgb_idx = self.params.rgb_bands
        self.lr_RGB = self.lr[rgb_idx]
        self.hr_RGB = self.hr[rgb_idx]
        self.sr_RGB = self.sr[rgb_idx]
        self.lr_to_hr_RGB = self.lr_to_hr[rgb_idx]
        self.sr_to_lr_RGB = self.sr_to_lr[rgb_idx]
        
        return None

    def sr_harm_setup(self) -> None:
        """ Remove the systematic error from the SR image.
                        
        Returns:
            torch.Tensor: The SR image without systematic error.        
        """

        # Remove systematic reflectance error
        if self.params.harm_apply_spectral:
            sr_harm = hq_histogram_matching(self.sr, self.lr)
        else:
            sr_harm = self.sr
            
        if self.params.harm_apply_spatial:
            # Remove systematic spatial error
            init_error = self.spatial_error.affine_model["rmse"][0]

            if init_error > 1:
                # Get the points and matches
                matching_points = spatial_get_matching_points(
                    img01=sr_harm[self.params.rgb_bands],
                    img02=self.lr_to_hr_RGB,
                    model=self.spatial_model,
                    device=self.device,
                )

                if not isinstance(matching_points, float):
                    td = self.params.spatial_threshold_distance
                    new_td = td * self.scale_factor

                    if not isinstance(matching_points, float):
                        # Build the affine model
                        spatial_models = spatial_model_fit(
                            matching_points=matching_points,
                            threshold_distance=new_td,
                            n_points=self.params.spatial_threshold_npoints,
                            verbose=False,
                            degree=1
                        )

                        # Apply the affine transformation
                        sr_harm = spatial_model_transform(
                            image1=sr_harm,
                            spatial_models=spatial_models,
                            precision=4,
                            interpolation_mode="nearest",
                            device=self.device
                        )
        else:
            sr_harm = sr_harm
        
        # Save the SR harmonized image
        self.sr_harm = sr_harm
        self.sr_harm_reduced = spectral_reducer(
            X=self.sr_harm,
            method=self.params.spectral_reducer,
            rgb_bands=self.params.rgb_bands
        )
        self.sr_harm_RGB = self.sr_harm[self.params.rgb_bands]


    def _spectral_global_error(
        self,
        grid: Optional[bool] = None,
        metric: Optional[str] = None
    ) -> None:
        """ Estimate the spectral global error by comparing the
        reflectance of the LR and SR images.
        
        Returns:
            Value: The spectral global error.
        """

        if grid is None:
            grid = self.grid
            
        if metric is None:
            metric = self.params.spectral_global_method

        self.spectral_global_error = spectral_metric(
            lr=self.lr,
            sr_to_lr=self.sr_to_lr,
            metric=metric,
            grid=grid
        )

    def _spectral_local_error(
        self,
        grid: Optional[bool] = None,
        metric: Optional[str] = None
    ) -> None:
        """ Estimate the spectral local error by comparing the
        reflectance of the LR and SR images.

        Returns:
            float: The spectral local error.
        """

        if grid is None:
            grid = self.grid

        if metric is None:
            metric = self.params.spectral_local_method

        self.spectral_local_error = spectral_metric(
            lr=self.lr,
            sr_to_lr=self.sr_to_lr,
            metric=metric,
            grid=grid
        )

    def _spatial_error(self, grid: Optional[bool] = None) -> None:
        """ Estimate the spatial error by comparing the
        spatial information of the LR and SR images.
        
        Returns:
            Value: The spatial error.
        """
        if grid is None:
            grid = self.grid

        # Spatial parameters that control the set of
        # point used to fit the LINEAR spatial model.
        threshold_distance = self.params.spatial_threshold_distance
        threshold_npoints = self.params.spatial_threshold_npoints
        spatial_description = "%s & %s" % (
            self.params.spatial_features,
            self.params.spatial_matcher
        )
        
        self.spatial_error = spatial_metric(
            lr=self.lr_RGB,
            sr_to_lr=self.sr_to_lr_RGB,
            models=self.spatial_model,
            threshold_distance=threshold_distance,
            threshold_npoints=threshold_npoints,
            grid=grid,
            description=spatial_description,
            device=self.device,
        )

    def _highfrequency(
        self,
        method: Optional[str] = None,
        grid: Optional[bool] = None        
    ) -> None:
        """ Estimate the High-frecuency added in the
        image enhancement process by comparing the
        LR and SR (systematic error are removed) images.
        
        Returns:
            Value: The high-frequency information.
        """
        if method is None:
            method = self.params.hf_method

        if grid is None:
            grid = self.grid
                
        # Obtain the high-frequency information
        self.hf_raster, self.hf_metric = highfrequency(
            lr_to_hr=self.lr_to_hr_reduced,
            sr_harm=self.sr_harm_reduced,
            method=method,
            grid=grid,
            reducer=self.params.hf_reducer,
            scale=self.scale_factor
        )

    def _unsystematic_error(
        self,
        method: Optional[str] = None,
        grid: Optional[bool] = None       
    ) -> None:
        
        if method is None:
            method = self.params.unsys_method
            
        if grid is None:
            grid = self.grid
        
        # Estimate unsystematic errors
        self.unsystematic_error = unsystematic_error(
            x=self.sr_harm_reduced,
            y=self.hr_reduced,
            method=method,
            grid=grid,
            space_search=self.scale_factor
        )        
    
    def _ha_im_ratio(
        self,
        grid: Optional[bool] = None
    ) -> Tuple[float, torch.Tensor, float]:

        if grid is None:
            grid = self.grid

        # Estimate unsystematic errors
        self.ha_im_ratio = ha_im_ratio(
            sr=self.sr_harm_reduced,
            hr=self.hr_reduced,
            lr_ref=self.lr_to_hr_reduced,
            hf=self.hf_raster.value,
            grid=grid
        )

    def _perceptual_ratio(
        self,
        stretch: Optional[str] = None
    ) -> float:
        """ Estimate the perceptual ratio by comparing the
        perceptual information of the SRref and SRharm images.
        
        Returns:
            float: The perceptual error.
        """
        
        if stretch is None:
            stretch = self.params.apply_stretch
                
        if stretch == "linear":
            lr_to_hr = opensr_test.plot.linear_fix(self.lr_to_hr_RGB, permute=False)
            sr_norm = opensr_test.plot.linear_fix(self.sr_harm_RGB, permute=False)
            hr = opensr_test.plot.linear_fix(self.hr_RGB, permute=False)
        elif stretch == "histogram":
            lr_to_hr = opensr_test.plot.equalize_hist(self.lr_to_hr_RGB, permute=False)
            sr_norm = opensr_test.plot.equalize_hist(self.sr_harm_RGB, permute=False)
            hr = opensr_test.plot.equalize_hist(self.hr_RGB, permute=False)
        elif stretch == "no_stretch":
            lr_to_hr = self.lr_to_hr_RGB
            sr_norm = self.sr_harm_RGB
            hr = self.hr_RGB
                    
        # Normalize the images to [-1, 1] for lpips
        lr_to_hr = (lr_to_hr - lr_to_hr.min()) / (lr_to_hr.max() - lr_to_hr.min())
        lr_to_hr = lr_to_hr * 2 - 1
        
        sr_norm = (sr_norm - sr_norm.min()) / (sr_norm.max() - sr_norm.min())
        sr_norm = sr_norm * 2 - 1
        
        hr = (hr - hr.min()) / (hr.max() - hr.min())
        hr = hr * 2 - 1
        
        # Naive solution to avoid nan values in the LPIPS
        # estimation.
        mask = sr_norm == sr_norm
        lr_to_hr = lr_to_hr * mask
        hr = hr * mask
        sr_norm[~mask] = 0
        
        # Obtain the perceptual error
        with torch.no_grad():
            perceptual_error_ref = self.perceptual_model(hr, lr_to_hr).mean()
            perceptual_error_sr = self.perceptual_model(hr, sr_norm).mean()            
            
        self.perceptual_ratio = Value(
            value=float(perceptual_error_sr/perceptual_error_ref),
            description= "LPIPS ratio"
        )
        

    def compute(
        self,
        lr: torch.Tensor = None,
        sr: torch.Tensor = None,
        hr: torch.Tensor = None,
        grid: Optional[bool] = None,
        only_value: Optional[bool] = True,
    ) -> dict:

        """ Obtain the performance metrics of the SR image.
        
        Args:
            lr (torch.Tensor): The LR image as a tensor (C, H, W).
            sr (torch.Tensor): The SR image as a tensor (C, H, W).
            hr (torch.Tensor): The HR image as a tensor (C, H, W).
            grid (Optional[bool], optional): Whether to return the
                metrics as grids or not. Defaults to False.
            only_value (Optional[bool], optional): Whether to return
                only the values of the metrics or the full Value 
                object (with description). Defaults to True.
                
        Returns:
            dict: The performance metrics for the SR image.
        """
        # self = Metrics(grid=False)
        # lr = torch.rand(4, 32, 32)
        # hr = torch.rand(4, 128, 128)
        # sr = torch.rand(4, 128, 128)
        
        # Run forward is LR, SR and HR is they are not None
        if (lr is not None) and (sr is not None) and (hr is not None):
            self.setup(lr, sr, hr)

        if grid is None:
            grid = self.grid

        # Obtain the RS metrics
        self._spectral_global_error(grid=grid)
        self._spectral_local_error(grid=grid)
        self._spatial_error(grid=grid)
        
        # Create SR' without systematic error
        self.sr_harm_setup()
                
        # Estimate the high-frequency information
        self._highfrequency(grid=grid)
        
        # Estimate the unsystematic error
        self._unsystematic_error(grid=grid)
        
        # Estimate the ha/im ratio
        self._ha_im_ratio(grid=grid)

        # Estimate the perceptual ratio
        self._perceptual_ratio()
        
        if only_value:
            return {
                "spectral_local_error": self.spectral_local_error.value,
                "spectral_global_error": self.spectral_global_error.value,
                "spatial_error": self.spatial_error.value,
                "high_frequency": self.hf_metric.value,
                "unsystematic_error": self.unsystematic_error.value,
                "ha/im ratio": self.ha_im_ratio,
                "perceptual_improvement": self.perceptual_ratio.value
            }
        else:
            return {
                "spectral_local_error": self.spectral_local_error,
                "spectral_global_error": self.spectral_global_error,
                "spatial_error": self.spatial_error,
                "high_frequency": self.hf_metric,
                "unsystematic_error": self.unsystematic_error,
                "ha/im ratio": self.ha_im_ratio,
                "perceptual_improvement": self.perceptual_ratio
            }


    def plot_triplets(self, stretch: Optional[str] = "linear"):
        return opensr_test.plot.triplets(
            lr_img=self.lr_RGB,
            sr_img=self.sr_RGB,
            hr_img=self.hr_RGB,
            stretch=stretch
        )
    
    def plot_quadruplets(self, stretch: Optional[str] = "linear"):
        return opensr_test.plot.quadruplets(
            lr_img=self.lr_RGB,
            sr_img=self.sr_RGB,
            hr_img=self.hr_RGB,
            landuse_img=self.landuse,
            stretch=stretch
        )

    def plot_spatial_matches(self, stretch: Optional[str] = "linear"):

        # Retrieve the linear affine model and the matching points
        self._spatial_error(grid=True)
        sp_errors = self.spatial_error.affine_model["rmse"][0]
        
        if math.isnan(sp_errors):
            warnings.warn("Spatial model is nan. No spatial matches will be plotted.")
            return None
        
        
        spatial_models = self.spatial_error.affine_model
        matching_points = self.spatial_error.matching_points

        # Generate a text message informing about
        # the displacement of the SR image
        model_x = spatial_models["models"][0]
        sign = "+" if float(model_x[1].intercept_) > 0 else "-"
        message01 = "Best linear approximation: %.04f*X %s %.04f" % (
            float(model_x[1].coef_),
            sign,
            np.abs(float(model_x[1].intercept_)),
        )
        message02 = "RMSE: %.04f [in LR pixel units]" % spatial_models["rmse"][0]
        messages = [message01, message02]

        # plot it!
        return opensr_test.plot.spatial_matches(
            lr=self.lr_RGB,
            sr_to_lr=self.sr_to_lr_RGB,
            points0=matching_points["points0"],
            points1=matching_points["points1"],
            matches01=matching_points["matches01"],
            threshold_distance=self.params.spatial_threshold_distance,
            messages=messages,
            stretch=stretch
        )

    def plot_error_grids(self, stretch: Optional[str] = "linear"):

        # Return the error grids with the metrics names
        results_v = self.compute(grid=False, only_value=True)
        results_g = self.compute(grid=True, only_value=False)

        # Local error reflectance
        e1 = results_g["spectral_local_error"].value
        e1_title = results_g["spectral_local_error"].description
        e1_subtitle = "%.04f" % results_v["spectral_local_error"]
        
        # Spatial error
        #results["spatial_error"] = create_nan_value(self.lr, grid = True, description = "Spatial error")
        e2 = results_g["spatial_error"].value
        e2_p_np = results_g["spatial_error"].points
        e2_points = [list(x.flatten().astype(int)) for x in e2_p_np]
        e2_title = results_g["spatial_error"].description
        e2_subtitle = "%.04f" % results_v["spatial_error"]
        
        # High frequency
        e3 = results_g["high_frequency"].value
        e3_title = results_g["high_frequency"].description
        e3_subtitle = "%.04f" % results_v["high_frequency"]

        # Improvement
        e4 = results_g["unsystematic_error"].value
        e4_title = results_g["unsystematic_error"].description
        e4_subtitle = "%.04f" % results_v["unsystematic_error"]

        # Hallucinationdisplay_results
        e5 = results_g["ha/im ratio"]
        e5_title = "|HR - LRdown| - |HR - SRharm|"
        e5_subtitle = ""

        # Plot high frequency
        fig, axs = opensr_test.plot.display_results(
            self.lr_RGB,
            self.lr_to_hr_RGB,
            self.sr_RGB,
            self.sr_harm_RGB,
            self.hr_RGB,
            e1,
            e1_title,
            e1_subtitle,
            e2,
            e2_points,
            e2_title,
            e2_subtitle,
            e3,
            e3_title,
            e3_subtitle,
            e4,
            e4_title,
            e4_subtitle,
            e5,
            e5_title,
            e5_subtitle,
            stretch=stretch,
        )
            
        return fig, axs
    
    def reset(self):
        self.metrics = []
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.compute(*args, **kwds)