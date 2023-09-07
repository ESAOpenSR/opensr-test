import itertools
import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from opensr_test.lightglue import DISK, LightGlue, SuperPoint
from opensr_test.lightglue.utils import rbd
from opensr_test.utils import Value, hq_histogram_matching
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures


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


def spatia_polynomial_fit(X: np.ndarray, y: np.ndarray, d: int) -> Pipeline:
    """Fit a polynomial of degree d to the points

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
        img01 (torch.Tensor): A torch.tensor with the image 1 (B, H, W)
        img02 (torch.Tensor): A torch.tensor with the ref image (B, H, W)
        model (tuple): A tuple with the feature extractor and the matcher
        device (str, optional): The device to use. Defaults to 'cpu'.

    Returns:
        Dict[str, torch.Tensor]: A dictionary with the points0, 
            points1, matches01 and image size.
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
            return torch.nan

        feats1 = extractor.extract(img02, resize=None)
        if feats1["keypoints"].shape[1] == 0:
            warnings.warn("No keypoints found in image 2")
            return torch.nan

        # match the features
        matches01 = matcher({"image0": feats0, "image1": feats1})

    # remove batch dimension
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    matches = matches01["matches"]  # indices with shape (K,2)
    points0 = feats0["keypoints"][
        matches[..., 0]
    ]  # coordinates in image #0, shape (K,2)
    points1 = feats1["keypoints"][
        matches[..., 1]
    ]  # coordinates in image #1, shape (K,2)

    matching_points = {
        "points0": points0,
        "points1": points1,
        "matches01": matches01,
        "img_size": tuple(img01.shape[-2:]),
    }

    return matching_points


def spatial_error(
    matching_points: Dict[str, torch.Tensor],
    threshold_distance: Optional[int] = 5,
    threshold_npoints: Optional[int] = 5,
    grid: Optional[bool] = None,
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
        grid (Optional[bool], optional): If True, return the grid with the error.
            Defaults to None.

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
    img_size = matching_points["img_size"]

    # if the distance between the points is higher than 5 pixels,
    # it is considered a bad match
    dist = torch.sqrt(torch.sum((points0 - points1) ** 2, dim=1))
    thres = dist < threshold_distance
    p0 = points0[thres]
    p1 = points1[thres]

    # if not enough points, return 0
    if p0.shape[0] < threshold_npoints:
        warnings.warn("Not enough points to calculate the spatial error")
        return torch.nan

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

    if grid:
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
            np.transpose(np.flip(spgrid_1D.reshape(img_size), axis=0), axes=(1, 0)),
            axis=1,
        )

        return spgrid_2D, (X_img0, y_img0)
    else:
        # rmse
        return np.mean(full_error), (X_img0, y_img0)


def spatial_model_fit(
    matching_points: Dict[str, torch.Tensor],
    n_points: Optional[int] = 10,
    threshold_distance: Optional[int] = 5,
    degree: Optional[int] = 1,
    verbose: Optional[bool] = True,
    scale: Optional[int] = 1,
    return_rmse: Optional[bool] = False,
) -> Union[np.ndarray, dict]:
    """Get a model that minimizes the spatial error between two images

    Args:
        matching_points (Dict[str, torch.Tensor]): A dictionary with the points0, 
            points1 and image size.
        n_points (Optional[int], optional): The minimum number of points. Defaults
            to 10.
        threshold_distance (Optional[int], optional): The maximum distance between
            the points. Defaults to 5 pixels.
        degree (Optional[int], optional): The degree of the polynomial. Defaults 
            to 1.
        verbose (Optional[bool], optional): If True, print the error. Defaults to
            False.
        scale (Optional[int], optional): The scale factor to use. Defaults to 1.
        return_rmse (Optional[bool], optional): If True, return the RMSE. Defaults
            to False.
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
        return torch.nan

    # from torch.Tensor to numpy array
    p0 = p0.detach().cpu().numpy() * scale
    p1 = p1.detach().cpu().numpy() * scale

    # Fit a polynomial of degree 2 to the points
    X_img0 = p0[:, 0].reshape(-1, 1)
    X_img1 = p1[:, 0].reshape(-1, 1)
    model_x = spatia_polynomial_fit(X_img0, X_img1, degree)

    y_img0 = p0[:, 1].reshape(-1, 1)
    y_img1 = p1[:, 1].reshape(-1, 1)
    model_y = spatia_polynomial_fit(y_img0, y_img1, degree)

    # display error
    xhat = model_x.predict(X_img0)
    yhat = model_y.predict(y_img0)

    # full error
    full_error1 = np.sqrt((xhat - X_img1) ** 2 + (yhat - y_img1) ** 2)
    full_error2 = np.sqrt((X_img0 - X_img1) ** 2 + (y_img0 - y_img1) ** 2)

    if verbose:
        if degree == 1:
            # get the params polynomial features
            print(
                "[X] Model: %.04f*X + %.04f"
                % (float(model_x[1].coef_), float(model_x[1].intercept_))
            )
            print(
                "[Y] Model: %.04f*X + %.04f"
                % (float(model_y[1].coef_), float(model_y[1].intercept_))
            )
        print(f"Initial [RMSE]: %.04f" % np.mean(full_error2))
        print(f"Final [RMSE]: %.04f" % np.mean(full_error1))

    if return_rmse:
        return {
            "models": (model_x, model_y),
            "rmse": (np.mean(full_error2), np.mean(full_error1)),
        }

    return model_x, model_y


def spatial_model_transform(
    image1: torch.Tensor,
    spatial_models: tuple,
    precision: Optional[int] = 2,
    interpolation_mode: Optional[str] = "bilinear",
    device: str = "cpu",
) -> torch.Tensor:
    """ Transform the image according to the spatial model

    Args:
        image1 (torch.Tensor): The image 1 with shape (B, H, W)
        spatial_models (tuple): A tuple with the models for x and y
        device (str, optional): The device to use. Defaults to 'cpu'.
        interpolation_mode (Optional[str], optional): The interpolation
            mode. Defaults to 'bilinear'.
    Returns:
        torch.Tensor: The transformed image
    """

    # Get the output device
    output_device = image1.device

    # Unpack the model - send to device
    model_y, model_x = spatial_models

    # Add padding to the image
    image1 = torch.nn.functional.pad(
        image1, pad=(8, 8, 8, 8), mode="constant", value=torch.nan
    )

    # Send the data to the device
    image1 = image1.to(device)

    # Create a super-grid
    image1 = torch.nn.functional.interpolate(
        image1.unsqueeze(0), scale_factor=precision, mode="nearest-exact"
    ).squeeze(0)

    # Get the coordinates
    x = torch.arange(0, image1.shape[-2] / precision, 1 / precision).to(device)
    y = torch.arange(0, image1.shape[-1] / precision, 1 / precision).to(device)

    xx, yy = torch.meshgrid(x, y)

    # Flatten the coordinates
    xx = xx.flatten().unsqueeze(1)
    yy = yy.flatten().unsqueeze(1)

    # Predict the new coordinates
    xx_new = model_x.predict(xx.cpu().numpy())
    yy_new = model_y.predict(yy.cpu().numpy())

    # Reshape the coordinates
    xx_new = xx_new.reshape(image1.shape[-2], image1.shape[-2]) * precision
    yy_new = yy_new.reshape(image1.shape[-1], image1.shape[-1]) * precision

    # Send the coordinates to torch and to the device
    xx_new = torch.Tensor(xx_new).to(device)
    yy_new = torch.Tensor(yy_new).to(device)

    # Grid must be normalized to [-1, 1]
    xx_new = (xx_new / (image1.shape[-2] / 2) - 1) * -1
    yy_new = (yy_new / (image1.shape[-1] / 2) - 1) * -1

    # Image mirror torch
    image1_1 = torch.flip(image1.transpose(2, 1), [2])

    # Interpolate the image
    new_image1 = torch.nn.functional.grid_sample(
        image1_1.unsqueeze(0),
        torch.stack([xx_new, yy_new], dim=2).unsqueeze(0),
        mode=interpolation_mode,
        padding_mode="border",
        align_corners=False,
    ).squeeze(0)

    # Remove the padding
    new_image1 = new_image1[
        :, (8 * precision) : -(8 * precision), (8 * precision) : -(8 * precision)
    ]

    # Go back to the original size
    new_image1 = torch.nn.functional.interpolate(
        new_image1.unsqueeze(0), scale_factor=1 / precision, mode="nearest-exact"
    ).squeeze(0)

    # Save the image
    final_image1 = new_image1.flip(2).to(output_device)

    return final_image1

def create_nan_value(
    image1:torch.Tensor,
    grid:bool,
    description:str
):
    """Create a empty value object when spatial models
    return nan values    
    """
    
    # Create fake points
    points = (
        np.array([[0], [1], [2]]),
        np.array([[0], [1], [2]])
    )

    
    # Create a affine model that does nothing
    affine_model = {
        "models": spatia_polynomial_fit(*points, 1),
        "rmse": (torch.nan, torch.nan)
    }
    
    # Create a fake model
    matching_points = {
        "points0": np.array([[0, 1], [0,1]]),
        "points1": np.array([[0, 1], [0,1]]),
        "matches01": np.array([[0, 1], [0,1]]),
        "img_size": image1.shape[1:]
    }
    
    # Create a fake value matrix
    value_matrix = torch.zeros_like(image1[0])
    value_matrix[value_matrix==0] = torch.nan
    
    if not grid:
        value_matrix = float(value_matrix.median())
        
    return Value(
        value=value_matrix,
        points=points,
        affine_model=affine_model,
        matching_points=matching_points,
        description=description,
    )

def spatial_metric(
    lr: torch.Tensor,
    sr_to_lr: torch.Tensor,
    models: tuple,
    threshold_distance: Optional[int] = 5,
    threshold_npoints: Optional[int] = 5,
    grid: Optional[bool] = True,
    description: str = "DISK Lightglue",
    device: str = "cpu",
) -> Union[float, np.ndarray]:
    """Calculate the spatial error between two images

    Args:
        lr (torch.Tensor): The LR image with shape (C, H, W)
        sr_to_lr (torch.Tensor): The SR degradated to the spatial resolution of
            LR with shape (C, H, W).
        model (tuple): A tuple with the feature extractor and the matcher models.
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
    image1 = torch.nan_to_num(lr, nan=0.0)

    # Apply histogram matching
    image2 = hq_histogram_matching(sr_to_lr, lr)

    # Get the points and matches    
    matching_points = spatial_get_matching_points(
        img01=image1, img02=image2, model=models, device=device
    )

    # is matching_points is a dict:
    if not isinstance(matching_points, dict):
        return create_nan_value(image1, description)
    
    # Fix a image according to the matching points
    affine_model = spatial_model_fit(
        matching_points=matching_points,
        threshold_distance=threshold_distance,
        n_points=threshold_npoints,
        degree=1,
        verbose=False,
        return_rmse=True,
    )

    # is matching_points is a dict:
    if not isinstance(affine_model, dict):
        return create_nan_value(image1, description)
        
    
    # Calculate the error
    sp_error = spatial_error(
        matching_points=matching_points,
        threshold_distance=threshold_distance,
        threshold_npoints=threshold_npoints,
        grid=grid
    )

    # is matching_points is a dict:
    if not isinstance(sp_error, tuple):
        return create_nan_value(image1, description)
     
    grid_error, points = sp_error


    # if grid_error is a grid, convert to torch.Tensor
    if isinstance(grid_error, np.ndarray):
        grid_error = torch.from_numpy(grid_error).to(image1.device).float()

    return Value(
        value=grid_error,
        points=points,
        affine_model=affine_model,
        matching_points=matching_points,
        description=description,
    )
