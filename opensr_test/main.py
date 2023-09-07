import opensr_test.plot
import numpy as np
import lpips
import torch

from typing import Any, Optional, Union, Tuple, List

from opensr_test.hallucinations import unsystematic_error
from opensr_test.highfrequency import highfrequency
from opensr_test.spatial import (spatial_get_matching_points,
                                       spatial_metric, spatial_model_fit,
                                       spatial_model_transform,
                                       spatial_setup_model)
from opensr_test.spectral import spectral_metric
from opensr_test.utils import Value, hq_histogram_matching


class Metrics:
    def __init__(
        self,
        RGBband: List[int] = [0, 1, 2],
        spatial_params: Optional[dict] = None,
        spectral_params: Optional[dict] = None,
        hf_params: Optional[dict] = None,
        perceptual_params: Optional[dict] = None,
        grid: Optional[bool] = False,
        device: Union[str, torch.device, None] = "cpu",
    ) -> None:
        """ A class to evaluate the performance of a image 
        enhancement algorithm considering the triplets: LR[input], 
        SR[enhanced], HR[ground truth].

        Args:
            RGBband (List[int], optional): The RGB bands to use.
                Defaults to [0, 1, 2].
            spatial_params (Optional[dict], optional): The parameters 
                to setup the spatial model. Defaults to None.
            spectral_params (Optional[dict], optional): The parameters
                to setup the spectral model. Defaults to None.
            hf_params (Optional[dict], optional): The parameters to
                setup the high-frequency model. Defaults to None.                                   
            grid (Optional[bool], optional): Whether to return the
                metrics as grids or not. Defaults to False.
            device (Union[str, torch.device, None], optional): The
                device to use. Defaults to "cpu".
        """

        # If the parameters are not provided, use the default ones
        if spatial_params is None:
            self.spatial_params = {
                "features": "superpoint",
                "matcher": "lightglue",
                "max_num_keypoints": 1000,
                "threshold_distance": 3,
                "threshold_npoints": 5,
            }

        if spectral_params is None:
            self.spectral_params = {"metric_global": "sd", "metric_local": "sad"}

        if hf_params is None:
            self.hf_params = {"metric": "simple"}
        
        if perceptual_params is None:
            self.perceptual_params = {
                "net": "alex",
                "stretch": "linear",
            }
            
        # Setup the spatial model
        self.spatial_model = spatial_setup_model(
            features=self.spatial_params["features"],
            matcher=self.spatial_params["matcher"],
            max_num_keypoints=self.spatial_params["max_num_keypoints"],
            device=device,
        )

        # Setup perceptual model
        self.perceptual_model = lpips.LPIPS(
            net=self.perceptual_params["net"],
            verbose=False
        )
        
        
        # Initial triplets: LR[input], SR[enhanced], HR[ground truth]
        self.lr = None
        self.sr = None
        self.hr = None

        # The sr without systematic error (spectral & spatial removed)
        self.sr_norm = None

        # The LR in the HR space: (C, H, W) -> (C, H*scale, W*scale)
        # using a bilinear interpolation (zero-parameter model)
        self.lr_to_hr = None

        # The SR in the LR space: (C, H*scale, W*scale) -> (C, H, W)
        # using a bilinear interpolation (zero-parameter model).
        self.sr_to_lr = None

        # The landuse map (C, H, W)
        self.landuse = None
        self.scale_factor = 1

        # RGB pairs (used for plotting and some metrics [spatial, & perceptual])
        self.lr_RGB = None
        self.hr_RGB = None
        self.lr_to_hr_RGB = None
        self.sr_to_lr_RGB = None
        self.sr_norm_RGB = None
        self.sr_RGB = None

        # The parameters
        self.grid = grid
        self.device = device
        self.RGBband = RGBband

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
        self.lr_to_hr = torch.nn.functional.interpolate(
            self.lr[None], scale_factor=self.scale_factor, mode="bilinear", antialias=True
        ).squeeze(0)
        

        # Obtain the SR in the LR space
        self.sr_to_lr = torch.nn.functional.interpolate(
            self.sr[None],
            scale_factor=1 / self.scale_factor,
            mode="bilinear",
            antialias=True,
        ).squeeze(0)


        # Obtain the RGB images
        self.lr_RGB = self.lr[self.RGBband]
        self.hr_RGB = self.hr[self.RGBband]
        self.sr_RGB = self.sr[self.RGBband]
        self.lr_to_hr_RGB = self.lr_to_hr[self.RGBband]
        self.sr_to_lr_RGB = self.sr_to_lr[self.RGBband]
        
        return None

    def sr_norm_setup(
        self,
        spatial_error: Value
    ) -> None:
        """ Remove the systematic error from the SR image.
        
        Args:
            spatial_error (Value): The spatial error result
                obtained from the spatial_error method.
                
        Returns:
            torch.Tensor: The SR image without systematic error.        
        """

        # Remove systematic reflectance error
        sr_norm = hq_histogram_matching(self.sr, self.lr)

        # Remove systematic spatial error
        init_error = spatial_error.affine_model["rmse"][0]

        if init_error > np.sqrt(2):

            # Get the points and matches
            matching_points = spatial_get_matching_points(
                img01=sr_norm,
                img02=self.lr_to_hr,
                model=self.spatial_model,
                device=self.device,
            )

            if not isinstance(matching_points, float):
                td = self.spatial_params["threshold_distance"]
                new_td = td * self.scale_factor

                if not isinstance(matching_points, float):
                    # Build the affine model
                    spatial_models = spatial_model_fit(
                        matching_points=matching_points,
                        threshold_distance=new_td,
                        n_points=self.spatial_params["threshold_npoints"],
                        degree=1,
                    )

                    # Apply the affine transformation
                    sr_norm = spatial_model_transform(
                        image1=sr_norm,
                        spatial_models=spatial_models,
                        precision=4,
                        interpolation_mode="nearest",
                        device=self.device,
                    )

        self.sr_norm = sr_norm
        self.sr_norm_RGB = self.sr_norm[self.RGBband]
        
        return None

    def spectral_global_error(
        self, grid: Optional[bool] = None, metric: Optional[str] = None
    ) -> Value:
        """ Estimate the spectral global error by comparing the
        reflectance of the LR and SR images.
        
        Returns:
            Value: The spectral global error.
        """

        if grid is None:
            grid = self.grid

        if metric is None:
            metric = self.spectral_params["metric_global"]

        return spectral_metric(
            lr=self.lr, sr_to_lr=self.sr_to_lr, metric=metric, grid=grid
        )

    def spectral_local_error(
        self, grid: Optional[bool] = None, metric: Optional[str] = None
    ) -> Value:
        """ Estimate the spectral local error by comparing the
        reflectance of the LR and SR images.

        Returns:
            float: The spectral local error.
        """

        if grid is None:
            grid = self.grid

        if metric is None:
            metric = self.spectral_params["metric_local"]

        return spectral_metric(
            lr=self.lr, sr_to_lr=self.sr_to_lr, metric=metric, grid=grid
        )

    def spatial_error(self, grid: Optional[bool] = None) -> Value:
        """ Estimate the spatial error by comparing the
        spatial information of the LR and SR images.
        
        Returns:
            Value: The spatial error.
        """
        if grid is None:
            grid = self.grid

        # Spatial parameters that control the set of
        # point used to fit the LINEAR spatial model.
        threshold_distance = self.spatial_params["threshold_distance"]
        threshold_npoints = self.spatial_params["threshold_npoints"]
        spatial_description = "%s & %s" % (
            self.spatial_params["features"],
            self.spatial_params["matcher"],
        )
        
        return spatial_metric(
            lr=self.lr_RGB,
            sr_to_lr=self.sr_to_lr_RGB,
            models=self.spatial_model,
            threshold_distance=threshold_distance,
            threshold_npoints=threshold_npoints,
            grid=grid,
            description=spatial_description,
            device=self.device,
        )

    def highfrequency(
        self,
        grid: Optional[bool] = None,
        metric: Optional[str] = None
    ) -> Value:
        """ Estimate the High-frecuency added in the
        image enhancement process by comparing the
        LR and SR (systematic error are removed) images.
        
        Returns:
            Value: The high-frequency information.
        """
        if metric is None:
            metric = self.hf_params["metric"]

        # Obtain the high-frequency information
        self.hf_raster = highfrequency(
            lr_to_hr=self.lr_to_hr,
            sr_norm=self.sr_norm,
            metric=metric,
            grid=True,
            scale=self.scale_factor,
        )

        if grid is None:
            grid = self.grid

        if grid:
            return self.hf_raster

        return Value(float(self.hf_raster.value.mean()), self.hf_raster.description)

    def rs_improvement(
        self,
        grid: Optional[bool] = None
    ) -> Tuple[float, torch.Tensor, float]:

        if grid is None:
            grid = self.grid

        # Estimate unsystematic errors
        ha, imp_r_ratio_grid, imp_r_ratio = unsystematic_error(
            sr_norm=self.sr_norm,
            hr=self.hr,
            lr_to_hr=self.lr_to_hr,
            hf=self.hf_raster.value,
            grid=grid,
        )

        if grid:
            return ha, imp_r_ratio_grid, imp_r_ratio

        return float(ha), imp_r_ratio_grid, float(imp_r_ratio)

    def perceptual_improvement(
        self,
        stretch: Optional[str] = None
    ) -> float:
        """ Estimate the perceptual improvement by comparing the
        perceptual information of the SRref and SRnorm images.
        
        Returns:
            float: The perceptual error.
        """
        
        if stretch is None:
            stretch = self.perceptual_params["stretch"]
                
        if stretch == "linear":
            lr_to_hr = opensr_test.plot.linear_fix(self.lr_to_hr_RGB, permute=False)
            sr_norm = opensr_test.plot.linear_fix(self.sr_norm_RGB, permute=False)
            hr = opensr_test.plot.linear_fix(self.hr_RGB, permute=False)
        elif stretch == "histogram":
            lr_to_hr = opensr_test.plot.equalize_hist(self.lr_to_hr_RGB, permute=False)
            sr_norm = opensr_test.plot.equalize_hist(self.sr_norm_RGB, permute=False)
            hr = opensr_test.plot.equalize_hist(self.hr_RGB, permute=False)
        elif stretch == "no_stretch":
            lr_to_hr = self.lr_to_hr_RGB
            sr_norm = self.sr_norm_RGB
            hr = self.hr_RGB
                    
        # Normalize the images to [-1, 1] for lpips
        lr_to_hr = (lr_to_hr - lr_to_hr.min()) / (lr_to_hr.max() - lr_to_hr.min())
        lr_to_hr = lr_to_hr * 2 - 1
        
        sr_norm = (sr_norm - sr_norm.min()) / (sr_norm.max() - sr_norm.min())
        sr_norm = sr_norm * 2 - 1
        
        hr = (hr - hr.min()) / (hr.max() - hr.min())
        hr = hr * 2 - 1
        
        # Obtain the perceptual error
        with torch.no_grad():
            perceptual_error_ref = self.perceptual_model(hr, lr_to_hr).mean()
            perceptual_error_sr = self.perceptual_model(hr, sr_norm).mean()
            lpips_ratio = float(perceptual_error_ref/perceptual_error_sr - 1)

        return lpips_ratio
        

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
        
        
        # Run forward is LR, SR and HR is they are not None
        if (lr is not None) and (sr is not None) and (hr is not None):
            self.setup(lr, sr, hr)

        if grid is None:
            grid = self.grid

        # Obtain the RS metrics
        spectral_global_error = self.spectral_global_error(grid=grid)
        spectral_local_error = self.spectral_local_error(grid=grid)
        spatial_error = self.spatial_error(grid=grid)
        
        # Create SR' without systematic error
        self.sr_norm_setup(spatial_error)
        
        # Estimate unsystematic errors (HA & SR improvement)
        hf_info = self.highfrequency(grid=grid)
        
        self.ha, self.imp_r_ratio_grid, self.imp_r_ratio = self.rs_improvement(
            grid=grid
        )
        
        if grid:
            imp_r = self.imp_r_ratio_grid
            perceptual_error = None
        else:
            imp_r = float(self.imp_r_ratio)            
            perceptual_error = self.perceptual_improvement()
            

        if only_value:
            return {
                "spectral_local_error": spectral_local_error.value,
                "spectral_global_error": spectral_global_error.value,
                "spatial_error": spatial_error.value,
                "high_frequency": hf_info.value,
                "hallucinations": self.ha,
                "rs_improvement": imp_r,
                "perceptual_improvement": perceptual_error
            }

        return {
            "spectral_local_error": spectral_local_error,
            "spectral_global_error": spectral_global_error,
            "spatial_error": spatial_error,
            "high_frequency": hf_info,
            "hallucinations": self.ha,
            "improvement": imp_r,
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
        spatial_error_value = self.spatial_error(grid=True)
        spatial_models = spatial_error_value.affine_model
        matching_points = spatial_error_value.matching_points

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
            threshold_distance=self.spatial_params["threshold_distance"],
            messages=messages,
            stretch=stretch,
        )

    def plot_error_grids(self, stretch: Optional[str] = "linear"):

        # Return the error grids with the metrics names
        results = self.compute(grid=True, only_value=False)

        # Local error reflectance
        e1 = results["spectral_local_error"].value
        e1_title = results["spectral_local_error"].description
        e1_subtitle = "%.04f" % torch.sqrt(torch.mean(e1 ** 2))

        # Spatial error
        #results["spatial_error"] = create_nan_value(self.lr, grid = True, description = "Spatial error")
        e2 = results["spatial_error"].value
        e2_p_np = results["spatial_error"].points
        e2_points = [list(x.flatten().astype(int)) for x in e2_p_np]
        e2_title = results["spatial_error"].description
        e2_subtitle = "%.04f" % torch.sqrt(torch.mean(e2 ** 2))            
        
        # High frequency
        e3 = results["high_frequency"].value
        e3_title = results["high_frequency"].description
        e3_subtitle = "%.04f" % torch.sqrt(torch.mean(e3 ** 2))

        # Improvement
        e4 = results["improvement"]
        e4_title = "1 - E_SR'/E_ref"
        e4_subtitle = "%.04f" % self.imp_r_ratio

        # Hallucinationdisplay_resultss
        e5 = results["hallucinations"]
        e5_title = "HF*|HR - SR'|"
        e5_subtitle = "%.04f" % torch.median(e5)

        # Plot high frequency
        fig, axs = opensr_test.plot.display_results(
            self.lr_RGB,
            self.sr_RGB,
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

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.compute(*args, **kwds)