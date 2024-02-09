import itertools
import warnings
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from opensr_test.config import Metric
from opensr_test.distance import DistanceMetric
from opensr_test.lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from opensr_test.lightglue.utils import rbd
from scipy.spatial.distance import cdist
from skimage.registration import phase_cross_correlation
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures


# %-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# | Spatial transformation functions
# %-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def spatia_polynomial_fit(X: np.ndarray, y: np.ndarray, d: int) -> Pipeline:
    """Fit a polynomial regression using matched points

    Args:
        X (np.ndarray): Array with the x coordinates or y coordinates of the points (image 1)
        y (np.ndarray): Array with the x coordinates or y coordinates of the points (image 2)
        d (int): Degree of the polynomial

    Returns:
        Pipeline: The fitted model
    """

    pipe_model = make_pipeline(
        PolynomialFeatures(degree=d, include_bias=False), LinearRegression()
    ).fit(X, y)

    return pipe_model


def spatial_setup_model(
    features: str = "superpoint",
    matcher: str = "lightglue",
    max_num_keypoints: int = 2048,
    device: str = "cpu",
) -> tuple:
    """Setup the model for spatial check

    Args:
        features (str, optional): The feature extractor. Defaults to 'superpoint'.
        matcher (str, optional): The matcher. Defaults to 'lightglue'.
        max_num_keypoints (int, optional): The maximum number of keypoints. Defaults to 2048.
        device (str, optional): The device to use. Defaults to 'cpu'.

    Raises:
        ValueError: If the feature extractor or the matcher are not valid
        ValueError: If the device is not valid

    Returns:
        tuple: The feature extractor and the matcher models
    """

    # Local feature extractor
    if features == "superpoint":
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    elif features == "disk":
        extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(device)
    elif features == "sift":
        extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(device)
    elif features == "aliked":
        extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(device)
    elif features == "doghardnet":
        extractor = DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(device)
    else:
        raise ValueError(f"Unknown feature extractor {features}")

    # Local feature matcher
    if matcher == "lightglue":
        matcher = LightGlue(features=features).eval().to(device)
    else:
        raise ValueError(f"Unknown matcher {matcher}")

    return extractor, matcher


def spatial_get_matching_points(
    img01: torch.Tensor, img02: torch.Tensor, model: tuple, device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Predict the spatial error between two images

    Args:
        img01 (torch.Tensor): A torch.tensor with the input image (B, H, W)
        img02 (torch.Tensor): A torch.tensor with the reference image (B, H, W)
        model (tuple): A tuple with the feature extractor and the matcher.
        device (str, optional): The device to use. Defaults to 'cpu'.

    Returns:
        Dict[str, torch.Tensor]: A dictionary with the points0, 
            points1, matches01.
    """

    # unpack the model - send to device
    extractor, matcher = model
    extractor = extractor.to(device)
    matcher = matcher.to(device)

    # Send the data to the device
    img01 = img01.to(device)
    img02 = img02.to(device)

    # extract local features
    with torch.no_grad():
        # auto-resize the image, disable with resize=None
        feats0 = extractor.extract(img01, resize=None)
        if feats0["keypoints"].shape[1] == 0:
            warnings.warn("No keypoints found in image 1")
            return False

        feats1 = extractor.extract(img02, resize=None)
        if feats1["keypoints"].shape[1] == 0:
            warnings.warn("No keypoints found in image 2")
            return False

        # match the features
        matches01 = matcher({"image0": feats0, "image1": feats1})

    # remove batch dimension
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    matches = matches01["matches"]
    points0 = feats0["keypoints"][matches[..., 0]]
    points1 = feats1["keypoints"][matches[..., 1]]

    matching_points = {"points0": points0, "points1": points1, "matches01": matches01}

    return matching_points


def spatial_model_fit(
    matching_points: Dict[str, torch.Tensor],
    n_points: Optional[int] = 10,
    threshold_distance: Optional[int] = 5,
    verbose: Optional[bool] = False,
) -> Union[np.ndarray, dict]:
    """Get a model that minimizes the spatial error between two images

    Args:
        matching_points (Dict[str, torch.Tensor]): A dictionary with the points0, 
            points1 and image size.
        n_points (Optional[int], optional): The minimum number of points. Defaults
            to 10.
        threshold_distance (Optional[int], optional): The maximum distance between
            the points. Defaults to 5 pixels.
        verbose (Optional[bool], optional): If True, print the error. Defaults to
            False.
        scale (Optional[int], optional): The scale factor to use. Defaults to 1.
        
    Returns:
        np.ndarray: The spatial error between the two images
    """

    points0 = matching_points["points0"]
    points1 = matching_points["points1"]

    # if the distance between the points is higher than 5 pixels,
    # it is considered a bad match
    dist = torch.sqrt(torch.sum((points0 - points1) ** 2, dim=1))
    thres = dist < threshold_distance
    p0 = points0[thres]
    p1 = points1[thres]

    # if not enough points, return 0
    if p0.shape[0] < n_points:
        warnings.warn("Not enough points to fit the model")
        return False

    # from torch.Tensor to numpy array
    p0 = p0.detach().cpu().numpy()
    p1 = p1.detach().cpu().numpy()

    # Fit a polynomial of degree 2 to the points
    X_img0 = p0[:, 0].reshape(-1, 1)
    X_img1 = p1[:, 0].reshape(-1, 1)
    model_x = spatia_polynomial_fit(X_img0, X_img1, 1)

    y_img0 = p0[:, 1].reshape(-1, 1)
    y_img1 = p1[:, 1].reshape(-1, 1)
    model_y = spatia_polynomial_fit(y_img0, y_img1, 1)

    # display error
    xoffset = np.round(model_x.predict(np.array(0).reshape(-1, 1)))
    yoffset = np.round(model_y.predict(np.array(0).reshape(-1, 1)))

    xhat = X_img0 + xoffset
    yhat = y_img0 + yoffset

    # full error
    full_error1 = np.sqrt((xhat - X_img1) ** 2 + (yhat - y_img1) ** 2)
    full_error2 = np.sqrt((X_img0 - X_img1) ** 2 + (y_img0 - y_img1) ** 2)

    if verbose:
        print(f"Initial [RMSE]: %.04f" % np.mean(full_error2))
        print(f"Final [RMSE]: %.04f" % np.mean(full_error1))

    to_export = {
        "offset": (int(xoffset), int(yoffset)),
        "error": (np.mean(full_error2), np.mean(full_error1)),
    }

    return to_export


def spatial_model_transform_pixel(
    image1: torch.Tensor, spatial_offset: tuple
) -> torch.Tensor:
    """ Transform the image according to the spatial offset obtained by the
    spatial_model_fit function. This correction is done at pixel level.

    Args:
        image1 (torch.Tensor): The image 1 with shape (B, H, W)
        spatial_offset (tuple): The spatial offset estimated by the 
            spatial_model_fit function.
    Returns:
        torch.Tensor: The transformed image
    """
    x_offs, y_offs = spatial_offset["offset"]

    # get max offset
    moffs = np.max(np.abs([x_offs, y_offs]))

    # Add padding according to the offset
    image_pad = torch.nn.functional.pad(
        image1, (moffs, moffs, moffs, moffs), mode="constant", value=0
    )

    if x_offs < 0:
        image_pad = image_pad[:, :, (moffs + x_offs) :]
    elif x_offs > 0:
        image_pad = image_pad[:, :, (moffs - x_offs) :]

    if y_offs < 0:
        image_pad = image_pad[:, (moffs - y_offs) :, :]
    elif y_offs > 0:
        image_pad = image_pad[:, (moffs + y_offs) :, :]

    # remove padding
    final_image = image_pad[:, 0 : image1.shape[1], 0 : image1.shape[2]]

    return final_image


def spatial_model_transform(
    lr_to_hr: torch.Tensor, hr: torch.Tensor, spatial_offset: tuple
) -> torch.Tensor:
    """ Transform the image according to the spatial offset

    Args:
        lr_to_hr (torch.Tensor): The low resolution image
        hr (torch.Tensor): The high resolution image
        spatial_offset (tuple): The spatial offset estimated by the 
            spatial_model_fit function.
    Returns:
        torch.Tensor: The transformed image
    """

    # Fix the image according to the spatial offset
    offset_image1 = spatial_model_transform_pixel(
        image1=lr_to_hr, spatial_offset=spatial_offset
    )
    hr_masked = hr * (offset_image1 != 0)

    # Create a mask with the offset image
    offset_image1 = offset_image1.detach().cpu().numpy()
    hr_masked = hr_masked.detach().cpu().numpy()

    # Subpixel refinement
    shift, error, diffphase = phase_cross_correlation(
        offset_image1.mean(0), hr_masked.mean(0), upsample_factor=100
    )

    # Fix the offset_image according to the subpixel refinement
    offset_image2 = spatial_model_transform_pixel(
        image1=torch.from_numpy(offset_image1).float(),
        spatial_offset={"offset": list(np.int16(np.round(shift)))},
    )

    return offset_image2


def spatial_aligment(
    sr: torch.Tensor,
    hr: torch.Tensor,
    spatial_model: tuple,
    threshold_n_points: Optional[int] = 5,
    threshold_distance: Optional[int] = 2 ** 63 - 1,
    rgb_bands: Optional[List[int]] = [0, 1, 2],
) -> torch.Tensor:
    """ Transform the image according to the spatial offset

    Args:
        sr (torch.Tensor): The super resolved image (B, C, H, W)
        hr (torch.Tensor): The high resolution image (B, C, H, W)
        spatial_model (tuple): A tuple with the feature extractor and the matcher models.
        threshold_n_points (Optional[int], optional): The minimum number of points to fit
            the linear model. Defaults to 5.
        threshold_distance (Optional[int], optional): The maximum distance 
            possible between the points. Defaults to 2**63 - 1.

    Returns:
        torch.Tensor: The transformed image (B, C, H, W).
    """
    # replace nan values with 0
    # lightglue does not work with nan values
    sr = torch.nan_to_num(sr, nan=0.0)
    hr = torch.nan_to_num(hr, nan=0.0)

    # get imag01 and img02
    if sr.shape[0] >= 3:
        img01 = sr[rgb_bands, ...]
        img02 = hr[rgb_bands, ...]
    else:
        img01 = sr[0][None]
        img02 = hr[0][None]

    # Get the matching points
    matching_points = spatial_get_matching_points(
        img01=img01, img02=img02, model=spatial_model
    )

    if matching_points is False:
        warnings.warn("Not enough points to align the images")
        return sr, matching_points

    # Fix the image according to the spatial offset
    spatial_offset = spatial_model_fit(
        matching_points=matching_points,
        n_points=threshold_n_points,
        threshold_distance=threshold_distance,
        verbose=False,
    )

    if spatial_offset is False:
        warnings.warn("Not enough valid points to align the images")
        return sr, matching_points

    # Fix the image according to the spatial offset
    offset_image = spatial_model_transform(
        lr_to_hr=sr, hr=hr, spatial_offset=spatial_offset
    )

    return offset_image, matching_points


# %-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# | Spatial transformation functions
# %-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def distance_matrix(
    x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray
) -> np.ndarray:
    """Calculate the distance matrix between two sets of points

    Args:
        x0 (np.ndarray): Array with the x coordinates of the points (image 1)
        y0 (np.ndarray): Array with the y coordinates of the points (image 1)
        x1 (np.ndarray): Array with the x coordinates of the points (image 2)
        y1 (np.ndarray): Array with the y coordinates of the points (image 2)

    Returns:
        np.ndarray: Array with the distances between the points
    """
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T
    return cdist(obs, interp)


def linear_rbf(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, xi: np.ndarray, yi: np.ndarray
) -> np.ndarray:
    """Interpolate using radial basis functions

    Args:
        x (np.ndarray): Array with the x coordinates of the points (image 1)
        y (np.ndarray): Array with the y coordinates of the points (image 1)
        z (np.ndarray): Array with the z coordinates of the points (image 1)
        xi (np.ndarray): Array with the x coordinates of the points (target image)
        yi (np.ndarray): Array with the y coordinates of the points (target image)
    
    Returns:
        np.ndarray: Array with the interpolated values
    """
    dist = distance_matrix(x, y, xi, yi)

    # Mutual pariwise distances between observations
    internal_dist = distance_matrix(x, y, x, y)

    # Now solve for the weights such that mistfit at the observations is minimized
    weights = np.linalg.solve(internal_dist, z)

    # Multiply the weights for each interpolated point by the distances
    zi = np.dot(dist.T, weights)

    return zi


def spatial_error(
    matching_points: Dict[str, torch.Tensor],
    threshold_distance: Optional[int] = 5,
    threshold_npoints: Optional[int] = 5,
    img_size: Optional[Tuple[int, int]] = (1024, 1024),
) -> Union[Tuple[float, Tuple[list, list]], Tuple[np.ndarray, Tuple[list, list]]]:
    """Calculate the spatial error between two images

    Args:
        matching_points (Dict[str, torch.Tensor]): A dictionary with the points0,
            points1 and image size.
        threshold_distance (Optional[int], optional): The maximum distance
            between the points. Defaults to 5 pixels.
        threshold_npoints (Optional[int], optional): The minimum number of 
            points. Defaults to 5.            
        degree (Optional[int], optional): The degree of the polynomial. Defaults to 1.
        img_size (Optional[Tuple[int, int]], optional): The size of the image. Defaults
            to (1024, 1024).
    Returns:
        Union[Tuple[float, Tuple[list, list]], Tuple[np.ndarray, Tuple[list, list]]] : 
            The spatial error between the two images. If grid=True, return the 
            interpolated grid using RBF. Also, return the points0 and points1 as 
            a tuple.
            
    Exceptions:
        ValueError: If not enough points to calculate the spatial error.
    """

    points0 = matching_points["points0"]
    points1 = matching_points["points1"]

    # if the distance between the points is higher than 5 pixels,
    # it is considered a bad match
    dist = torch.sqrt(torch.sum((points0 - points1) ** 2, dim=1))
    thres = dist < threshold_distance
    p0 = points0[thres]
    p1 = points1[thres]

    # if not enough points, return 0
    if p0.shape[0] < threshold_npoints:
        warnings.warn("Not enough points to calculate the spatial error")
        return np.array(np.nan)

    # from torch.Tensor to numpy array
    p0 = p0.detach().cpu().numpy()
    p1 = p1.detach().cpu().numpy()

    # Fit a polynomial of degree 2 to the points
    X_img0 = p0[:, 0].reshape(-1, 1)
    X_img1 = p1[:, 0].reshape(-1, 1)

    X_error = np.abs(X_img1 - X_img0)

    y_img0 = p0[:, 1].reshape(-1, 1)
    y_img1 = p1[:, 1].reshape(-1, 1)

    y_error = np.abs(y_img1 - y_img0)

    # Calculate the error
    full_error = np.sqrt(X_error ** 2 + y_error ** 2)

    # Interpolate the error
    arr_grid = np.array(
        list(
            itertools.product(
                np.arange(0, img_size[0], 1), np.arange(0, img_size[1], 1)
            )
        )
    )

    spgrid_1D = linear_rbf(
        X_img0.flatten(),
        y_img0.flatten(),
        full_error.flatten(),
        arr_grid[:, 0],
        arr_grid[:, 1],
    )

    spgrid_2D = np.flip(
        np.transpose(np.flip(spgrid_1D.reshape(img_size), axis=0), axes=(1, 0)), axis=1
    )

    return spgrid_2D


def spatial_metric(
    lr: torch.Tensor,
    sr_to_lr: torch.Tensor,
    spatial_model: tuple,
    threshold_n_points: Optional[int] = 5,
    threshold_distance: Optional[int] = 2 ** 63 - 1,
    description: str = "DISK Lightglue",
    device: str = "cpu",
) -> Union[float, np.ndarray]:
    """Calculate the spatial error between two images

    Args:
        lr (torch.Tensor): The LR image with shape (C, H, W)
        sr_to_lr (torch.Tensor): The SR degradated to the spatial resolution of
            LR with shape (C, H, W).
        spatial_model (tuple): A tuple with the feature extractor and the matcher models.
        threshold_distance (Optional[int], optional): The maximum distance between the
            points. Defaults to 5 pixels.
        threshold_npoints (Optional[int], optional): The minimum number of points.
            Defaults to 5.
        degree (Optional[int], optional): The degree of the polynomial. Defaults to 1.
        grid (Optional[bool], optional): If True, return the grid with the error. Defaults 
            to True.
        description (str, optional): The description of the metric. Defaults to 
            'DISK & Lightglue'.
        device (str, optional): The device to use. Defaults to 'cpu'.

    Returns:
        Union[float, np.ndarray]: The spatial error between the two images. If grid=True,
            return the grid with the error, otherwise return the mean error (RMSEmean).
    """

    # replace nan values with 0
    # lightglue does not work with nan values
    lr = torch.nan_to_num(lr, nan=0.0)
    sr_to_lr = torch.nan_to_num(sr_to_lr, nan=0.0)

    # Apply histogram matching
    image1 = lr.mean(0)[None]
    image2 = sr_to_lr.mean(0)[None]

    # Get the points and matches
    matching_points = spatial_get_matching_points(
        img01=image1, img02=image2, model=spatial_model, device=device
    )

    # is matching_points is a dict:
    if matching_points is False:
        return torch.tensor(torch.nan), matching_points

    # Calculate the error
    sp_error = spatial_error(
        matching_points=matching_points,
        threshold_distance=threshold_distance,
        threshold_npoints=threshold_n_points,
        img_size=image1[0].shape,
    )

    return torch.from_numpy(sp_error).float(), matching_points


class SpatialMetric(DistanceMetric):
    """Spectral information divergence between two tensors"""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        name: str,
        threshold_n_points: int,
        threshold_distance: int,
        spatial_model: tuple,
        method: str = "image",
        patch_size: int = 32,
        device: str = "cpu",
    ):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size, name=name)
        self.spatial_model = spatial_model
        self.threshold_n_points = threshold_n_points
        self.threshold_distance = threshold_distance
        self.device = device

        # Estimate the spatial error at pixel level
        self.metric, self.matching_points = spatial_metric(
            lr=x,
            sr_to_lr=y,
            spatial_model=self.spatial_model,
            threshold_n_points=self.threshold_n_points,
            threshold_distance=self.threshold_distance,
            device="cpu",
        )

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.metric.mean()

    def compute_patch(self) -> torch.Tensor:
        metric_batched = self.do_square(self.metric[None], self.patch_size)
        metric_result = torch.zeros(metric_batched.shape[:2])
        xrange, yrange = metric_batched.shape[0:2]

        for x_index in range(xrange):
            for y_index in range(yrange):
                metric_batch = metric_batched[x_index, y_index]
                metric_result[x_index, y_index] = metric_batch.mean()

        return Metric(value=metric_result, description=self.name)

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.metric