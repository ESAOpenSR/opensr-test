from typing import Any, Optional, Union

import warnings
import torch

from opensr_test.utils import (
    apply_downsampling, apply_upsampling,
    apply_mask, hq_histogram_matching,
    seed_everything
)
from opensr_test.spatial import SpatialMetricAlign
from opensr_test.config import Config, Results, Consistency, Synthesis, Correctness, Auxiliar
from opensr_test.distance import get_distance
from opensr_test.correctness import get_distances, tc_improvement, tc_omission, tc_hallucination, get_correctness_stats
from opensr_test import plot

class Metrics:
    def __init__(
        self,
        params: Optional[Config] = None,
        **kwargs
    ) -> None:
        """ A class to evaluate the performance of a super
        resolution algorithms considering the triplets: LR[input],
        SR[enhanced], HR[ground truth].

        Args:
            params (Optional[Config], optional): The parameters to
                setup the opensr-test experiment. Defaults to None.
                If None, the default parameters are used. See 
                config.py for more information.
        """
        
        # Set the parameters
        if params is None:
            if kwargs:
                self.params = Config(**kwargs)
            else:
                self.params = Config()
        else:
            self.params = params

        
            
        
        # If patch size is 1, then the aggregation method must be pixel
        if self.params.patch_size == 1:
            self.params.agg_method = "pixel"

        # Global parameters
        self.method = self.params.agg_method
        self.run_setup = True

        # Set the spatial grid regulator
        self.apply_upsampling = apply_upsampling
        self.apply_downsampling = apply_downsampling

        # Set the spatial aligner
        self.spatial_aligner = SpatialMetricAlign(
            method=self.params.spatial_method,
            max_translations=self.params.spatial_threshold_distance,
            max_num_keypoints=self.params.spatial_max_num_keypoints,
            device=self.params.device
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
        hr: torch.Tensor
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

        # If patch size is higher than the image size, then return an error.
        if self.params.patch_size is not None:
            if (self.params.patch_size > sr.shape[1]) or (self.params.patch_size > sr.shape[2]):
                raise ValueError("The patch size must be lower than the image size.")

        # Obtain the scale factor
        scale_factor = hr.shape[-1] / lr.shape[-1]
        if not scale_factor.is_integer():
            raise ValueError("The scale factor must be an integer.")
        self.scale_factor = int(scale_factor)

        # Move all the images to the same device
        self.lr = apply_mask(lr.to(self.params.device), self.params.border_mask // self.scale_factor)
        self.sr = apply_mask(sr.to(self.params.device), self.params.border_mask)
        self.hr = apply_mask(hr.to(self.params.device), self.params.border_mask)

        # Obtain the LR in the HR space
        if self.scale_factor > 1:
            self.lr_to_hr = self.apply_downsampling(
                x=self.lr[None], 
                scale=self.scale_factor
            ).squeeze(0)
        else:
            self.lr_to_hr = self.lr

        # Obtain the SR in the LR space
        self.sr_to_lr = self.apply_upsampling(
            x=self.sr,
            scale=self.scale_factor
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
        """ Remove the systematic error from the SR image. After
        super-resolving an image, the SR image may contain systematic
        errors. These errors are removed to only evaluate the high-frequency
        information added after the super-resolution process.
                        
        Returns:
            torch.Tensor: The SR image without systematic error.        
        """

        # Remove systematic reflectance error
        if self.params.harm_apply_spectral:
            sr_harm = hq_histogram_matching(self.sr, self.hr)
        else:            
            sr_harm = self.sr

        # Remove systematic spatial error
        if self.params.harm_apply_spatial:
            sr_harm = self.spatial_aligner.get_metric(sr_harm, self.hr)[0]
        else:
            sr_harm = sr_harm

        self.sr_harm = sr_harm

        # Obtain the RGB images
        if self.sr_harm.shape[0] >= 3:
            self.sr_harm_RGB = sr_harm[self.params.rgb_bands]
        else:
            self.sr_harm_RGB = sr_harm[0][None]

    def _reflectance_distance(self) -> None:
        """ Estimate the spectral global error by comparing the
        reflectance of the LR and SR images.
        
        Returns:
            Value: The reflectance global error.
        """        
        self.reflectance_error = get_distance(
            x=self.lr,
            y=self.sr_to_lr,
            method=self.params.reflectance_distance,
            agg_method=self.params.agg_method,
            patch_size=self.params.patch_size,
            scale=self.scale_factor
        )
    
    def _spectral_distance(self) -> None:
        """ Estimate the spectral global error by comparing the
        reflectance of the LR and SR images.
        
        Returns:
            Value: The spectral global error.
        """
        self.spectral_error = get_distance(
            x=self.lr,
            y=self.sr_to_lr,
            method=self.params.spectral_distance,
            agg_method=self.params.agg_method,
            patch_size=self.params.patch_size,
            scale=self.scale_factor
        )
    
    def _spatial_alignment(self) -> None:
        """ Estimate the spatial global error by comparing the
        spatial alignment of the SR and HR images.
        
        Returns:
            Value: The spatial global error.
        """
        self.spatial_error = self.spatial_aligner.get_metric(
            self.lr_RGB, self.sr_to_lr_RGB
        )[1]

    def _create_mask(
        self,
        d_ref: torch.Tensor,
        d_im: torch.Tensor,
        d_om: torch.Tensor,
        gradient_threshold: Union[float, str] = "auto"
    ) -> None:
        """ The goal of this mask is to avoid the pixels with
        gradients below a certain threshold. This mask is always
        created using l1 distance at pixel level.


        Args:
            stability_threshold (float, optional): The threshold
                to avoid the pixels with gradients below this value.
                Defaults to 0.05.

        Returns:
            torch.Tensor: The mask to avoid the pixels with gradients
                below a certain threshold.
        """

        # Get the optimal threshold based on quantiles
        if isinstance(gradient_threshold, str):
            if gradient_threshold == "auto":
                gradient_threshold = "auto75"
            t1 = int(gradient_threshold.replace("auto", ""))/100
            gradient_threshold = d_ref.flatten().kthvalue(
                int(t1 * d_ref.numel())
            ).values.item()

        # create mask
        mask1 = (d_ref > gradient_threshold)*1
        mask2 = (d_im > gradient_threshold)*1
        mask3 = (d_om > gradient_threshold)*1
        mask = ((mask1 + mask2 + mask3) > 0) * 1.0
        mask[mask == 0] = torch.nan
        return mask


    def consistency(
        self,
        lr: torch.Tensor,
        sr: torch.Tensor
    ) -> None:
        """ Obtain the consistency metrics trough comparing the 
        LR and SR images.

        Args:
            lr (torch.Tensor): The LR image as a tensor (C, H, W).
            sr (torch.Tensor): The SR image as a tensor (C, H, W).
            run_setup (bool, optional): Run the setup method. 
                Defaults to True.
        """
        # Make the experiment reproducible
        seed_everything(42)
        
        # Setup the LR and SR images
        self.setup(lr=lr, sr=sr, hr=sr)

        # Run the consistency metrics
        self._reflectance_distance()
        self._spectral_distance()
        self._spatial_alignment()

        return {
            "reflectance": self.reflectance_error.nanmean().item(),
            "spectral": self.spectral_error.nanmean().item(),
            "spatial": self.spatial_error.nanmean().item()
        }
        

    def synthesis(
        self,
        lr: torch.Tensor,
        sr: torch.Tensor,
        hr: Optional[torch.Tensor] = None
    ) -> None:
        """ Obtain the synthesis metrics trough comparing the
        LR, SR, and HR images.

        Args:
            lr (torch.Tensor): The LR image as a tensor (C, H, W).
            sr (torch.Tensor): The SR image as a tensor (C, H, W).
            hr (Optional[torch.Tensor], optional): The HR image as a 
                tensor (C, H, W). Defaults to None. If HR is set the
                SR is first harmonized using the HR image as a 
                reference. To deactivate harmonization, modify the
                Config object.
        """
        
        if hr is None:
                self.setup(lr=lr, sr=sr, hr=sr)
        else:
                self.setup(lr=lr, sr=sr, hr=hr)
            
        # Obtain the SR image without systematic error
        self.sr_harm_setup()
        
        # Obtain the distance between the SR and HR images
        self.synthesis_distance_value = get_distance(
            x=self.lr_to_hr,
            y=self.sr_harm,
            method=self.params.synthesis_distance,
            agg_method=self.params.agg_method,
            patch_size=self.params.patch_size,
            scale=self.scale_factor
        )

        return {
            "synthesis": self.synthesis_distance_value.nanmean().item()
        }
    
    def correctness(
        self,
        lr: torch.Tensor,
        sr: torch.Tensor,
        hr: torch.Tensor,
        gradient_threshold: float = 0.1
    ) -> None:
        """ Obtain the correctness metrics trough comparing the
        LR, SR, and HR images.

        Args:
            lr (torch.Tensor): The LR image as a tensor (C, H, W).
            sr (torch.Tensor): The SR image as a tensor (C, H, W).
            hr (torch.Tensor): The HR image as a tensor (C, H, W).
            gradient_threshold (float, optional): Ignore the pixels
                with gradients below this threshold. Defaults to 0.1.
        """

        # Make the experiment reproducible
        seed_everything(42)
        
        # Setup the LR, SR, and HR images
        self.setup(lr=lr, sr=sr, hr=hr)
        self.sr_harm_setup()

        # Obtain the distance between the LR, SR, and HR images
        d_ref, d_im, d_om = get_distances(
            lr_to_hr=self.lr_to_hr,
            sr_harm=self.sr_harm,
            hr=self.hr,
            distance_method=self.params.correctness_distance,
            agg_method=self.params.agg_method,
            scale=self.scale_factor,
            patch_size=self.params.patch_size,
            rgb_bands=self.params.rgb_bands,
            device=self.params.device
        )

        # Apply the mask to remove the pixels with low gradients
        mask = self._create_mask(
            d_ref=d_ref,
            d_im=d_im,
            d_om=d_om,
            gradient_threshold=gradient_threshold
        )
        self.d_ref = d_ref * mask
        self.d_im =  d_im * mask
        self.d_om =  d_om * mask
        
        # Compute relative distance
        self.d_im_ref = self.d_im / self.d_ref
        self.d_om_ref = self.d_om / self.d_ref

        # Estimate Hallucination
        self.hallucination = tc_hallucination(
            d_im=self.d_im_ref,
            d_om=self.d_om_ref,
            plambda=self.params.ha_score
        )

        # Estimate Omission
        self.omission = tc_omission(
            d_im=self.d_im_ref,
            d_om=self.d_om_ref,
            plambda=self.params.om_score
        )

        # Estimate Improvement
        self.improvement = tc_improvement(
            d_im=self.d_im_ref,
            d_om=self.d_om_ref,
            plambda=self.params.im_score
        )

        #  Get stats
        total_stats = get_correctness_stats(
            im_tensor=self.improvement,
            om_tensor=self.omission,
            ha_tensor=self.hallucination,
            mask=mask,
            correctness_norm=self.params.correctness_norm
        )
        
        self.classification = total_stats["classification"]
        self.improvement, self.omission, self.hallucination = total_stats["tensor_stack"]
        self.im_percentage = total_stats["stats"][0]
        self.om_percentage = total_stats["stats"][1]
        self.ha_percentage = total_stats["stats"][2]
        
        return {
            "ha_percent": self.ha_percentage.item(),
            "om_percent": self.om_percentage.item(),
            "im_percent": self.im_percentage.item()
        }

    def compute(
        self,
        lr: torch.Tensor,
        sr: torch.Tensor,
        hr: torch.Tensor,
        gradient_threshold: Optional[Union[float, str]] = "auto"
    ) -> None:
        """ Obtain the performance metrics of the SR image.

        Args:
            lr (torch.Tensor): The LR image as a tensor (C, H, W).
            sr (torch.Tensor): The SR image as a tensor (C, H, W).
            hr (torch.Tensor): The HR image as a tensor (C, H, W).
            gradient_threshold (float, optional): Ignore the pixels
                with gradients below this threshold. Defaults "auto",
                which means that the threshold is automatically
                generated based on LR-HR distance.
        """
        # Make the experiment reproducible
        seed_everything(42)
        
        # Obtain the correctness metrics
        correctness = self.correctness(
            lr=lr,
            sr=sr,
            hr=hr,
            gradient_threshold=gradient_threshold
        )

        # Obtain the consistency metrics
        consistency = self.consistency(lr=lr, sr=sr)

        # Obtain the synthesis metrics
        synthesis = self.synthesis(lr=lr, sr=sr, hr=hr)

        # Prepare the results
        self._prepare()

        # merge the results
        consistency.update(synthesis)
        consistency.update(correctness)
        
        return consistency
    
    def _prepare(self) -> None:
        self.results = Results(
            consistency=Consistency(
                reflectance=self.reflectance_error,
                spectral=self.spectral_error,
                spatial=self.spatial_error
            ),
            synthesis=Synthesis(
                distance=self.synthesis_distance_value
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
                d_ref=self.d_ref,
                d_im=self.d_im,
                d_om=self.d_om
            )
        )
    
    def plot_triplets(self, stretch: Optional[str] = "linear"):
        return plot.triplets(
            lr_img=self.lr_RGB.to("cpu"),
            sr_img=self.sr_harm_RGB.to("cpu"),
            hr_img=self.hr_RGB.to("cpu"),
            stretch=stretch,
        )
    
    def plot_summary(self, stretch: Optional[str] = "linear"):
        contion1 = self.params.agg_method == "pixel"
        contion2 = self.method == "patch"
        if not (contion1 or contion2):
            raise ValueError(
                "This method is only available for the "
                "pixel and patch aggregation methods."
            )
        
        # Reflectance metric
        e1 = self.results.consistency.reflectance
        e1_title = "Reflectance (%s)" % self.params.reflectance_distance
        e1_subtitle = "%.04f" % float(e1.nanmean())

        # Spectral metric
        e2 = self.results.consistency.spectral
        e2_title = "Spectral (%s)" % self.params.spectral_distance
        e2_subtitle = "%.04f" % float(e2.nanmean())
        
        # Distance to omission space
        e3 = self.results.correctness.omission
        e3_title = "Omission"
        e3_subtitle = "%.04f" % float(e3.nanmean())

        # Distance to improvement space
        e4 = self.results.correctness.improvement
        e4_title = "Improvement"
        e4_subtitle = "%.04f" % float(e4.nanmean())

        # Distance to hallucination space
        e5 = self.results.correctness.hallucination
        e5_title = "Hallucination"
        e5_subtitle = "%.04f" % float(e5.nanmean())

        # Plot high frequency
        fig, axs = plot.display_results(
            lr=self.lr_RGB.to("cpu"),
            lrdown=self.lr_to_hr_RGB.to("cpu"),
            sr=self.sr_RGB.to("cpu"),
            srharm=self.sr_harm_RGB.to("cpu"),
            hr=self.hr_RGB.to("cpu"),
            e1=e1.to("cpu"),
            e2=e2.to("cpu"),
            e3=e3.to("cpu"),
            e4=e4.to("cpu"),
            e5=e5.to("cpu"),
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
            stretch=stretch
        )        
        return fig, axs

    def plot_tc(
        self,
        log_scale: Optional[bool] = True,
        stretch: Optional[str] = "linear"
    ):
        return plot.display_tc_score(
            sr_rgb=self.sr_harm_RGB.to("cpu"),
            d_im_ref=self.d_im_ref.to("cpu"),
            d_om_ref=self.d_om_ref.to("cpu"),
            tc_score=self.classification.to("cpu"),
            log_scale=log_scale,
            stretch=stretch
        )

    def __call__(self) -> Any:
        return self.compute()