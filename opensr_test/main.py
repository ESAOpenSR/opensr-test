import warnings
from typing import Any, Optional, Union

import opensr_test.plot
import torch
from opensr_test.config import (
    Auxiliar, Consistency, Distance,
    Results, Correctness, Config
)
from opensr_test.hallucinations import get_distances, tc_improvement, tc_omission, tc_hallucination
from opensr_test.kernels import apply_downsampling, apply_upsampling
from opensr_test.reflectance import reflectance_metric
from opensr_test.spatial import (SpatialMetric, spatial_aligment,
                                 spatial_setup_model,
                                 spatial_model_transform_pixel)
from opensr_test.spectral import spectral_metric
from opensr_test.utils import hq_histogram_matching, seed_everything, get_zeros_at_edges


class Metrics:
    def __init__(
        self,
        device: Union[str, torch.device, None] = "cpu",
        **kwargs: Any
    ) -> None:
        """ A class to evaluate the performance of a image 
        enhancement algorithm considering the triplets: LR[input], 
        SR[enhanced], HR[ground truth].

        Args:
            params (Optional[Config], optional): The parameters to
                setup the opensr-test experiment. Defaults to None.
                If None, the default parameters are used. See 
                config.py for more information.
            device (Union[str, torch.device, None], optional): The
                device to use. Defaults to "cpu".
        """
        

        # Set the parameters
        if kwargs is None:
            self.params = Config()
        else:
            self.params = Config(**kwargs)
        
        # If patch size is 1, then the aggregation method must be pixel
        if self.params.patch_size == 1:
            self.params.agg_method = "pixel"

        # Global parameters
        self.method = self.params.agg_method
        self.device = device

        # Set the spatial grid regulator
        self.apply_upsampling = apply_upsampling
        self.apply_downsampling = apply_downsampling

        # Setup the spatial model
        self.spatial_model = spatial_setup_model(
            features=self.params.spatial_features,
            matcher=self.params.spatial_matcher,
            max_num_keypoints=self.params.spatial_max_num_keypoints,
            device=device,
        )

        # Initial triplets: LR[input], SR[enhanced], HR[ground truth]
        self.lr = None
        self.sr = None
        self.hr = None

        # The sr without systematic error (spectral & spatial removed)
        self.sr_harm = None

        # The LR in the HR space: (C, H, W) -> (H*scale, W*scale)
        # using a bilinear interpolation (zero-parameter model)
        # also applied the spectral_reducer since we don't need
        # to evaluate the spectral information in the HR space.
        self.lr_to_hr = None

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

        ## consistency with low resolution
        self.reflectance_value = None
        self.spectral_value = None
        self.spatial_aligment_value = None

        ## distance to the omission or improvement space
        self.d_ref = None
        self.d_om = None
        self.d_hr = None

        ## Correctness metrics
        self.omission = None
        self.improvement = None
        self.hallucination = None
        self.classification = None

        ## Percentage of hallucination and improvement
        self.ha_percentage = None
        self.om_percentage = None
        self.im_percentage = None

    def setup(
        self,
        lr: torch.Tensor,
        sr: torch.Tensor,
        hr: torch.Tensor,
        landuse: Optional[torch.Tensor] = None,
        downsample_method: Optional[str] = "classic",
        upsample_method: Optional[str] = "classic",
    ) -> None:
        """ Obtain the performance metrics of the SR image.

        Args:
            lr (torch.Tensor): The LR image as a tensor (C, H, W).
            sr (torch.Tensor): The SR image as a tensor (C, H, W).
            hr (torch.Tensor): The HR image as a tensor (C, H, W).
        """
        # Check if SR has gradients
        if sr.requires_grad:
            raise ValueError("The SR image must not have gradients.")

        # If patch size is higher than the image size, then
        # return an error.
        if self.params.patch_size is not None:
            if (self.params.patch_size > lr.shape[1]) or (self.params.patch_size > lr.shape[2]):
                raise ValueError("The patch size must be lower than the image size.")

        # Obtain the scale factor
        scale_factor = hr.shape[-1] / lr.shape[-1]
        if not scale_factor.is_integer():
            raise ValueError("The scale factor must be an integer.")
        self.scale_factor = int(scale_factor)

        # Move all the images to the same device
        self.lr = self.apply_mask(lr.to(self.device), self.params.mask // self.scale_factor)
        self.sr = self.apply_mask(sr.to(self.device), self.params.mask)
        self.hr = self.apply_mask(hr.to(self.device), self.params.mask)
        self.landuse = self.apply_mask(landuse.to(self.device), self.params.mask // self.scale_factor) if landuse is not None else None


        # Obtain the LR in the HR space
        if self.scale_factor > 1:
            self.lr_to_hr = self.apply_downsampling(
                X=self.lr[None], 
                scale=self.scale_factor, 
                method=downsample_method
            ).squeeze(0)
        else:
            self.lr_to_hr = self.lr

        # Obtain the SR in the LR space
        self.sr_to_lr = self.apply_upsampling(
            X=self.sr[None],
            scale=self.scale_factor,
            method=upsample_method
        ).squeeze(0)

        # Obtain the RGB images
        if self.lr.shape[0] >= 3:
            self.lr_RGB = self.lr[self.params.rgb_bands]
            self.hr_RGB = self.hr[self.params.rgb_bands]
            self.sr_RGB = self.sr[self.params.rgb_bands]
            self.lr_to_hr_RGB = self.lr_to_hr[self.params.rgb_bands]
            self.sr_to_lr_RGB = self.sr_to_lr[self.params.rgb_bands]
        else:
            self.lr_RGB = self.lr[0][None]
            self.hr_RGB = self.hr[0][None]
            self.sr_RGB = self.sr[0][None]
            self.lr_to_hr_RGB = self.lr_to_hr[0][None]
            self.sr_to_lr_RGB = self.sr_to_lr[0][None]

        return None

    def sr_harm_setup(self) -> None:
        """ Remove the systematic error from the SR image.
                        
        Returns:
            torch.Tensor: The SR image without systematic error.        
        """

        # Remove systematic reflectance error
        if self.params.harm_apply_spectral:
            sr_harm = hq_histogram_matching(self.sr, self.hr)
        else:
            sr_harm = self.sr

        if self.params.harm_apply_spatial:
            sr_harm, matching_points, spatial_offset = spatial_aligment(
                sr=sr_harm,
                hr=self.hr,
                spatial_model=self.spatial_model,
                threshold_n_points=self.params.spatial_threshold_npoints,
                threshold_distance=self.params.spatial_threshold_distance,
                rgb_bands=self.params.rgb_bands,
            )
            self.matching_points_02 = matching_points
        else:
            self.sr_harm = sr_harm
            self.matching_points_02 = False

        # Remove the black edges
        xmin, xmax, ymin, ymax = get_zeros_at_edges(sr_harm, self.scale_factor)
        self.sr_harm = sr_harm[:, xmin: xmax, ymin: ymax]
        self.lr_to_hr = self.lr_to_hr[:, xmin: xmax, ymin: ymax]
        self.sr = self.sr[:, xmin: xmax, ymin: ymax]   
        self.hr = self.hr[:, xmin: xmax, ymin: ymax]
        self.hr_RGB = self.hr[self.params.rgb_bands]
        self.lr = self.lr[:, xmin//self.scale_factor: xmax//self.scale_factor, ymin//self.scale_factor: ymax//self.scale_factor]
        self.lr_RGB = self.lr[self.params.rgb_bands]    

        if self.lr.shape[0] >= 3:
            self.sr_harm_RGB = self.sr_harm[self.params.rgb_bands]
        else:
            self.sr_harm_RGB = self.sr_harm[0][None]

    def _reflectance_metric(self) -> None:
        """ Estimate the spectral global error by comparing the
        reflectance of the LR and SR images.
        
        Returns:
            Value: The spectral global error.
        """        
        self.reflectance_value = reflectance_metric(
            lr=self.lr,
            sr_to_lr=self.sr_to_lr,
            metric=self.params.reflectance_method,
            agg_method=self.params.agg_method,
            patch_size=self.params.patch_size,
        )

    def _spectral_metric(self) -> None:
        """ Estimate the spectral local error by comparing the
        reflectance of the LR and SR images.

        Returns:
            float: The spectral local error.
        """
        self.spectral_value = spectral_metric(
            lr=self.lr,
            sr_to_lr=self.sr_to_lr,
            metric=self.params.spectral_method,
            agg_method=self.params.agg_method,
            patch_size=self.params.patch_size,
        )

    def _spatial_metric(self) -> None:
        """ Estimate the spatial error by comparing the
        spatial information of the LR and SR images.
        
        Returns:
            Value: The spatial error.
        """
        # Spatial parameters that control the set of
        # point used to fit the LINEAR spatial model.
        threshold_distance = self.params.spatial_threshold_distance
        threshold_npoints = self.params.spatial_threshold_npoints
        spatial_description = "%s & %s" % (
            self.params.spatial_features,
            self.params.spatial_matcher,
        )

        spatial_metric_result = SpatialMetric(
            x=self.lr_RGB,
            y=self.sr_to_lr_RGB,
            spatial_model=self.spatial_model,
            name=spatial_description,
            threshold_n_points=threshold_npoints,
            threshold_distance=threshold_distance,
            method="pixel",
            patch_size=self.params.patch_size,
            device=self.device,
        )

        self.spatial_aligment_value = spatial_metric_result.compute()
        self.matching_points_01 = spatial_metric_result.matching_points

    def _create_mask(self, stability_threshold: float = 0.) -> None:
        d_ref, d_im, d_om = get_distances(
            lr_to_hr=self.lr_to_hr,
            sr_harm=self.sr_harm,
            hr=self.hr,
            distance_method="l1",
            agg_method="pixel"
        )

        # create mask
        mask1 = (d_ref > stability_threshold)*1
        mask2 = (d_im > stability_threshold)*1
        mask3 = (d_om > stability_threshold)*1
        mask = ((mask1 + mask2 + mask3) > 0) * 1.0
        mask[mask == 0] = torch.nan
        self.tc_mask = mask
        return mask

    def _distance_metric(self, stability_threshold: float = 0.) -> None:
        self.d_ref, self.d_im, self.d_om = get_distances(
            lr_to_hr=self.lr_to_hr,
            sr_harm=self.sr_harm,
            hr=self.hr,
            distance_method=self.params.distance_method,
            agg_method=self.params.agg_method,
            patch_size=self.params.patch_size,
            rgb_bands=self.params.rgb_bands,
            device=self.device,
        )

        # Apply mask
        mask = self._create_mask(stability_threshold)
        self.potential_pixels = torch.nansum(mask)
        self.d_ref_masked = self.d_ref * mask
        self.d_im_masked = self.d_im * mask
        self.d_om_masked = self.d_om * mask

        # Compute relative distance
        self.d_im_ref = self.d_im_masked / self.d_ref_masked
        self.d_om_ref = self.d_om_masked / self.d_ref_masked

        if sum(self.d_im_ref.shape) == 0:
            self.d_im_ref = self.d_im_ref.reshape(-1)
            self.d_om_ref = self.d_om_ref.reshape(-1)

    def _improvement(self, im_score: float = 0.80) -> None:
        self.improvement = tc_improvement(
            d_im=self.d_im_ref,
            d_om=self.d_om_ref,
            plambda=im_score
        )

    def _omission(self, om_score: float = 0.80) -> None:
        self.omission = tc_omission(
            d_im=self.d_im_ref,
            d_om=self.d_om_ref,
            plambda=om_score
        )

    def _hallucination(self, ha_score: float = 0.80) -> None:
        self.hallucination = tc_hallucination(
            d_im=self.d_im_ref,
            d_om=self.d_om_ref,
            plambda=ha_score
        )
        
        correctness_stack = torch.stack([
            self.improvement,
            self.omission,
            self.hallucination
        ], dim=0)
        
        self.classification = torch.argmin(correctness_stack, dim=0) * self.tc_mask
        
        self.im_percentage = torch.sum(self.classification==0) / self.potential_pixels
        self.om_percentage = torch.sum(self.classification==1) / self.potential_pixels
        self.ha_percentage = torch.sum(self.classification==2) / self.potential_pixels

    def _prepare(self) -> None:
        self.results = Results(
            consistency=Consistency(
                reflectance=dict(self.reflectance_value),
                spectral=dict(self.spectral_value),
                spatial=dict(self.spatial_aligment_value),
            ),
            distance=Distance(
                lr_to_hr=self.d_ref,
                sr_to_hr=self.d_im,
                sr_to_lr=self.d_om
            ),
            correctness=Correctness(
                omission=self.omission,
                improvement=self.improvement,
                hallucination=self.hallucination,
                classification=self.classification
            ),
            auxiliar=Auxiliar(
                sr_harm=self.sr_harm,
                lr_to_hr=self.lr_to_hr,
                matching_points_lr=self.matching_points_01,
                matching_points_hr=self.matching_points_02,
            ),
        )
    
    def summary(self) -> dict:
        return {
            "reflectance": float(self.reflectance_value.value.nanmean()),
            "spectral": float(self.spectral_value.value.nanmean()),
            "spatial": float(self.spatial_aligment_value.value.nanmean()),
            "synthesis": float(self.d_om.nanmean()),
            "ha_percent": float(self.ha_percentage),
            "om_percent": float(self.om_percentage),
            "im_percent": float(self.im_percentage),
        }

    def compute(
        self,
        stability_threshold: Optional[float] = 0.01,
        im_score: Optional[float] = 0.8,
        om_score: Optional[float] = 0.8,
        ha_score: Optional[float] = 0.8,
    ) -> None:

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
        seed_everything(42)

        # Obtain the RS metrics
        self._reflectance_metric()
        self._spectral_metric()
        self._spatial_metric()

        # Create SR' without systematic error
        self.sr_harm_setup()        

        # Obtain the distance metrics
        self._distance_metric(stability_threshold)

        # Obtain the improvement metrics
        self._improvement(im_score)

        # Obtain the omission metrics
        self._omission(om_score)

        # Obtain the hallucination metrics
        self._hallucination(ha_score)
        
        # Prepare the results
        self._prepare()

        return None

    def plot_triplets(self, stretch: Optional[str] = "linear"):
        return opensr_test.plot.triplets(
            lr_img=self.lr_RGB,
            sr_img=self.sr_harm_RGB,
            hr_img=self.hr_RGB,
            stretch=stretch,
        )

    def plot_quadruplets(
        self, apply_harm: bool = True, stretch: Optional[str] = "linear"
    ):
        return opensr_test.plot.quadruplets(
                lr_img=self.lr_RGB,
                sr_img=self.sr_RGB,
                hr_img=self.hr_RGB,
                landuse_img=self.landuse,
                stretch=stretch,
            )
    
    def plot_spatial_matches(self, stretch: Optional[str] = "linear"):
        
        # Retrieve the linear affine model and the matching points
        if self.results.auxiliar.matching_points_lr is False:
            warnings.warn("Spatial model is nan. No spatial matches will be plotted.")
            return None

        matching_points = self.results.auxiliar.matching_points_lr

        # plot it!
        return opensr_test.plot.spatial_matches(
            lr=self.lr_RGB,
            sr_to_lr=self.sr_to_lr_RGB,
            points0=matching_points["points0"],
            points1=matching_points["points1"],
            matches01=matching_points["matches01"],
            threshold_distance=self.params.spatial_threshold_distance,
            stretch=stretch,
        )        

    def plot_summary(self, stretch: Optional[str] = "linear"):
        contion1 = self.params.agg_method == "pixel"
        contion2 = self.method == "patch"
        if (contion1 and contion2):
            raise ValueError(
                "This method only works for pixel and patch evaluation."
            )

        # Reflectance metric
        e1 = self.results.consistency.reflectance.value
        e1_title = self.results.consistency.reflectance.description
        e1_subtitle = "%.04f" % float(e1.nanmean())

        # Spectral metric
        e2 = self.results.consistency.spectral.value
        e2_title = self.results.consistency.spectral.description
        e2_subtitle = "%.04f" % float(e2.nanmean())

        # Spatial error
        e3 = self.results.consistency.spatial.value
        e3_p_np = (
            self.results.auxiliar.matching_points_lr["points0"]
            .clone()
            .detach()
            .cpu()
            .numpy()
        )

        # if patch, then divide by the scale factor
        e3_points = [list(x.flatten().astype(int)) for x in e3_p_np]
        e3_title = self.results.consistency.spatial.description
        e3_subtitle = "%.04f" % float(e3.nanmean())

        # High frequency
        e4 = self.results.distance.sr_to_lr
        e4_title = self.params.distance_method.upper()
        e4_subtitle = "%.04f" % float(e4.nanmean())

        # Error grids
        e5 = self.results.distance.sr_to_hr
        e5_title = self.params.distance_method.upper()
        e5_subtitle = "%.04f" % float(e5.nanmean())

        # Plot high frequency
        fig, axs = opensr_test.plot.display_results(
            lr=self.lr_RGB,
            lrdown=self.lr_to_hr_RGB,
            sr=self.sr_RGB,
            srharm=self.sr_harm_RGB,
            hr=self.hr_RGB,
            e1=e1,
            e2=e2,
            e3=e3,
            e4=e4,
            e5=e5,
            e1_title=e1_title,
            e2_title=e2_title,
            e3_title=e3_title,
            e4_title=e4_title,
            e5_title=e5_title,
            e1_subtitle=e1_subtitle,
            e2_subtitle=e2_subtitle,
            e3_subtitle=e3_subtitle,
            e4_subtitle=e4_subtitle,
            e5_subtitle=e5_subtitle,
            e3_points=e3_points,
            stretch=stretch,
        )
        return fig, axs

    def plot_tc(
        self, 
        log_scale: Optional[bool] = True,
        stretch: Optional[str] = "linear"
    ):
        return opensr_test.plot.display_tc_score(
            sr_rgb=self.sr_harm_RGB,
            d_im_ref=self.d_im_ref,
            d_om_ref=self.d_om_ref,
            tc_score=self.classification,
            log_scale=log_scale,
            stretch=stretch
        )

    def __call__(self) -> Any:
        return self.compute()

    def apply_mask(self, X: torch.Tensor, mask: int) -> torch.Tensor:
        if mask is None:
            return X
        
        if mask == 0:
            return X
        
        # C, H, W
        if len(X.shape) == 3:
            return X[:, mask: -mask, mask: -mask]
        elif len(X.shape) == 2:
            return X[mask: -mask, mask: -mask]
        else:
            raise ValueError("The tensor must be 2D or 3D.")
