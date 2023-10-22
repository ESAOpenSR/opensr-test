import torch
import warnings

from opensr_test.config import Config, Results, Auxiliar, TCscore, Distance, Consistency
from typing import Any, Optional, Union
from opensr_test.kernels import apply_downsampling, apply_upsampling

from opensr_test.spatial import spatial_metric, spatial_aligment, spatial_setup_model
from opensr_test.spectral import spectral_metric
from opensr_test.reflectance import reflectance_metric
from opensr_test.hallucinations import get_distances, tc_metric
from opensr_test.utils import hq_histogram_matching
import opensr_test.plot

class Metrics:
    def __init__(
        self,
        params: Optional[Config] = None,
        method: Optional[str] = "pixel",
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
            method (Optional[str], optional): Whether to use a pixel,
                patch or grid-based evaluation. Defaults to "pixel".
                Multiple patch methods are supported: "patch2", "patch4",
                "patch8", etc. They refer to the size of the patch. For
                instance, "patch2" means that the metric is computed by
                averaging the metric of 2x2 pixel at LR space.                                
            device (Union[str, torch.device, None], optional): The
                device to use. Defaults to "cpu".
        """

        # Get the parameters
        if kwargs is None:
            if params is None:
                self.params = Config()
            else:
                self.params = params
        else:
            if params is None:
                self.params = Config(**kwargs)
            else:
                self.params = params

        # Global parameters
        self.method = method
        self.device = device
        
        # Set the spatial grid regulator
        self.apply_upsampling = apply_upsampling
        self.apply_downsampling = apply_downsampling
        
        # Setup the spatial model
        self.spatial_model = spatial_setup_model(
            features=self.params.spatial_features,
            matcher=self.params.spatial_matcher,
            max_num_keypoints=self.params.spatial_max_num_keypoints,
            device=device
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
        
        ## tc score
        self.tc_score = None
        
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
        self.lr_to_hr = self.apply_downsampling(
            X=self.lr[None],
            scale=self.scale_factor,
            method=self.params.upsample_method
        ).squeeze(0)

        # Obtain the SR in the LR space
        self.sr_to_lr = self.apply_upsampling(
            X=self.sr[None],
            scale = self.scale_factor,
            method = self.params.downsample_method
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
        
        # Apply harmozination
        self.sr_harm_setup()
        
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
            sr_harm, matching_points = spatial_aligment(
                sr=sr_harm,
                hr=self.hr,
                spatial_model=self.spatial_model,
                threshold_n_points=self.params.spatial_threshold_npoints,
                threshold_distance=self.params.spatial_threshold_distance,
                rgb_bands=self.params.rgb_bands
            )
        else:
            sr_harm = sr_harm

        # Save the SR harmonized image
        self.sr_harm = sr_harm
        self.matching_points_02 = matching_points
        
        # Estimate mask and apply to HR and SR
        mask = torch.sum(torch.abs(self.sr_harm), axis=0) != 0
        self.hr = self.hr * mask
        self.hr_RGB = self.hr_RGB * mask
        
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
            metric=self.params.reflectance_method
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
            metric=self.params.spectral_method
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
            self.params.spatial_matcher
        )
        
        self.spatial_aligment_value, self.matching_points_01 = spatial_metric(
            lr=self.lr_RGB,
            sr_to_lr=self.sr_to_lr_RGB,
            spatial_model=self.spatial_model,
            threshold_distance=threshold_distance,
            threshold_n_points=threshold_npoints,
            description=spatial_description,
            device=self.device,
        )
            
    def _distance_metric(self) -> None:        
        self.d_ref, self.d_im, self.d_om = get_distances(
            lr_to_hr=self.lr_to_hr,
            sr_harm=self.sr_harm,
            hr=self.hr,
            distance_method = self.params.distance_method,
            agg_method = self.params.agg_method,
            patch_size = self.params.patch_size,
            rgb_bands = self.params.rgb_bands,
            device = self.device
        )
        
    def _tc_score(self) -> None:
        self.d_im_ref = self.d_im / self.d_ref
        self.d_om_ref = self.d_om / self.d_ref
        self.tc_score = tc_metric(
            d_im=self.d_im_ref, d_om=self.d_om_ref
        )
        
        npixels = self.tc_score.numel()
        self.im_percentage = torch.sum((self.tc_score >= 0.5)) / npixels
        self.om_percentage = torch.sum((self.tc_score <= -0.5)) / npixels
        self.ha_percentage = torch.sum(((self.tc_score > -0.5) & (self.tc_score < 0.5))) / npixels
    
    def _prepare(self) -> None:        
        self.results = Results(
            consistency = Consistency(
                reflectance=dict(self.reflectance_value),
                spectral=dict(self.spectral_value),
                spatial=dict(self.spatial_aligment_value)
            ),
            distance = Distance(
                lr_to_hr=self.d_ref,
                sr_to_hr=self.d_im,
                sr_to_lr=self.d_om
            ),
            score = TCscore(
                tc_score=self.tc_score,
                ha_percent=float(self.ha_percentage),
                om_percent=float(self.om_percentage),
                im_percent=float(self.im_percentage)
            ),
            auxiliar = Auxiliar(
                sr_harm=self.sr_harm,
                lr_to_hr=self.lr_to_hr,
                matching_points_lr=self.matching_points_01,
                matching_points_hr=self.matching_points_02
            )
        )

        
    def compute(self) -> dict:

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
        # Obtain the RS metrics
        self._reflectance_metric()
        self._spectral_metric()
        self._spatial_metric()
        
        # Create SR' without systematic error
        self.sr_harm_setup()
                
        # Obtain the distance metrics
        self._distance_metric()
                
        # Obtain the tc score
        self._tc_score()
        
        self._prepare()
        
        return self.results

    def plot_triplets(self, apply_harm: bool = True, stretch: Optional[str] = "linear"):
        if apply_harm:
            tplot = opensr_test.plot.triplets(
                lr_img=self.lr_RGB,
                sr_img=self.sr_harm_RGB,
                hr_img=self.hr_RGB,
                stretch=stretch
            )
        else:
            tplot = opensr_test.plot.triplets(
                lr_img=self.lr_RGB,
                sr_img=self.sr_RGB,
                hr_img=self.hr_RGB,
                stretch=stretch
            )        
        return tplot
        
    def plot_quadruplets(self, apply_harm: bool = True, stretch: Optional[str] = "linear"):
        if apply_harm:
            qplot = opensr_test.plot.quadruplets(
                lr_img=self.lr_RGB,
                sr_img=self.sr_harm_RGB,
                hr_img=self.hr_RGB,
                landuse_img=self.landuse,
                stretch=stretch
            )
        else:
            qplot = opensr_test.plot.quadruplets(
                lr_img=self.lr_RGB,
                sr_img=self.sr_RGB,
                hr_img=self.hr_RGB,
                landuse_img=self.landuse,
                stretch=stretch
            )        
        return qplot

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
            stretch=stretch
        )

    def plot_pixel_summary(self, stretch: Optional[str] = "linear"):
        if not self.params.agg_method == "pixel":
            raise ValueError("This method only works with pixel-based metrics.")
        

        # Reflectance metric
        e1 = self.results.consistency.reflectance.value
        e1_title = self.results.consistency.reflectance.description
        e1_subtitle = "%.04f" % float(e1.mean())
        
        # Spectral metric
        e2 = self.results.consistency.spectral.value
        e2_title = self.results.consistency.spectral.description
        e2_subtitle = "%.04f" % float(e2.mean())
        
        # Spatial error
        e3 = self.results.consistency.spatial.value
        e3_p_np = self.results.auxiliar.matching_points_lr["points0"].clone().detach().cpu().numpy()
        e3_points = [list(x.flatten().astype(int)) for x in e3_p_np]
        e3_title = self.results.consistency.spatial.description
        e3_subtitle = "%.04f" % float(e3.mean())
        
        # High frequency
        e4 = self.results.distance.sr_to_lr
        e4_title = self.params.distance_method.upper()
        e4_subtitle = "%.04f" % float(e4.mean())
        
        # Error grids
        e5 = self.results.distance.sr_to_hr
        e5_title = self.params.distance_method.upper()
        e5_subtitle = "%.04f" % float(e5.mean())

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
        xylimits: Optional[list] = [0, 3]
    ):
        return opensr_test.plot.display_tc_score(
            d_im_ref=self.d_im_ref,
            d_om_ref=self.d_om_ref,
            tc_score=self.tc_score,
            log_scale=log_scale,
            xylimits=xylimits
        )
        
    def __call__(self) -> Any:
        return self.compute()