import itertools
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from opensr_test.lightglue import DISK, LightGlue, SuperPoint, viz2d
from opensr_test.lightglue.utils import rbd


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


def spatia_polynomial_fit(X: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """Fit a polynomial of degree d to the points

    Args:
        X (np.ndarray): Array with the x coordinates or y coordinates of the points (image 1)
        y (np.ndarray): Array with the x coordinates or y coordinates of the points (image 2)
        d (int): Degree of the polynomial

    Returns:
        np.ndarray: Array with the predicted values
    """

    pipe_model = make_pipeline(
        PolynomialFeatures(degree=d, include_bias=False), LinearRegression()
    ).fit(X, y)

    return pipe_model


def spatial_setup_model(
    features: str = "superpoint",
    matcher: str = "lightglue",
    max_num_keypoints: int = 2048,
    device: str = "cuda",
) -> tuple:
    """Setup the model for spatial check

    Args:
        features (str, optional): The feature extractor. Defaults to 'superpoint'.
        matcher (str, optional): The matcher. Defaults to 'lightglue'.
        max_num_keypoints (int, optional): The maximum number of keypoints. Defaults to 2048.
        device (str, optional): The device to use. Defaults to 'cuda'.

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
) -> tuple:
    """Predict the spatial error between two images

    Args:
        img01 (torch.Tensor): A torch.tensor with the image 1 (B, H, W)
        img02 (torch.Tensor): A torch.tensor with the ref image (B, H, W)
        model (tuple): A tuple with the feature extractor and the matcher
        device (str, optional): The device to use. Defaults to 'cpu'.

    Returns:
        tuple: A tuple with the points0, points1 and matches01
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
            raise ValueError("No keypoints found in image 1")

        feats1 = extractor.extract(img02, resize=None)
        if feats1["keypoints"].shape[1] == 0:
            raise ValueError("No keypoints found in image 2")

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
        "img_size": tuple(img01.shape[-2:]),
        "matches01": matches01,
    }

    return matching_points


def spatial_error(
    matching_points: Dict[str, torch.Tensor],
    threshold_distance: Optional[int] = 5,
    threshold_npoints: Optional[int] = 5,
    grid: Optional[bool] = None,
):
    """Calculate the spatial error between two images

    Args:
        matching_points (Dict[str, torch.Tensor]): A dictionary with the points0, points1 and image size.
        threshold_distance (Optional[int], optional): The maximum distance between the points.
            Defaults to 5 pixels.
        threshold_npoints (Optional[int], optional): The minimum number of points. Defaults to 5.            
        degree (Optional[int], optional): The degree of the polynomial. Defaults to 1.
        grid (Optional[bool], optional): If True, return the grid with the error. Defaults to None.

    Returns:
        np.ndarray: The spatial error between the two images

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
        raise ValueError("Not enough points to calculate the spatial error")

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
    full_error = np.sqrt(X_error**2 + y_error**2)

    if grid:
        # Interpolate the error
        arr_grid = np.array(
            list(
                itertools.product(
                    np.arange(0, img_size[0], 1), np.arange(0, img_size[1], 1)
                )
            )
        )

        spgrid = linear_rbf(
            X_img0.flatten(),
            y_img0.flatten(),
            full_error.flatten(),
            arr_grid[:, 0],
            arr_grid[:, 1],
        ).reshape(img_size[0], img_size[1])

        return spgrid
    else:
        return np.mean(full_error)


def spatial_metric(
    image1: torch.Tensor,
    image2: torch.Tensor,
    models: tuple,
    threshold_distance: Optional[int] = 5,
    threshold_npoints: Optional[int] = 5,
    grid: Optional[bool] = True,
    previous_matching: Optional[Dict[str, torch.Tensor]] = None,
    device: str = "cuda",
) -> Union[float, np.ndarray]:
    """Calculate the spatial error between two images

    Args:
        image1 (torch.Tensor): The image 1 with shape (B, H, W)
        image2 (torch.Tensor): The image 2 with shape (B, H, W)
        model (tuple): A tuple with the feature extractor and the matcher models
        threshold_distance (Optional[int], optional): The maximum distance between the points.
            Defaults to 5 pixels.
        threshold_npoints (Optional[int], optional): The minimum number of points. 
            Defaults to 5.
        degree (Optional[int], optional): The degree of the polynomial. Defaults to 1.
        grid (Optional[bool], optional): If True, return the grid with the error. Defaults to True.
        previous_matching (Optional[Dict[str, torch.Tensor]], optional): A dictionary with the points0,
            points1 and image size before the spatial transformation. Defaults to None.
        device (str, optional): The device to use. Defaults to 'cuda'.

    Returns:
        Union[float, np.ndarray]: The spatial error between the two images. If grid=True, return
            the grid with the error, otherwise return the mean error (RMSEmean).
    """

    # replace nan values with 0
    image1 = torch.nan_to_num(image1, nan=0.0)
    
    # Get the points and matches
    matching_points = spatial_get_matching_points(
        img01=image1, img02=image2, model=models, device=device
    )

    if previous_matching is not None:
        p0 = matching_points["points0"]
        p1 = matching_points["points1"]

        pp0 = previous_matching["points0"]

        # obtain the same elements in both arrays (B, H)
        index1 = torch.where(
            torch.all(torch.eq(p0.unsqueeze(1), pp0.unsqueeze(0)), dim=2)
        )[0]

        newp0 = p0[index1]
        newp1 = p1[index1]

        if (newp0.shape[0] < threshold_npoints) & (newp1.shape[0] < threshold_npoints):
            raise ValueError("Not enough points to calculate the spatial error")

        matching_points["points0"] = newp0
        matching_points["points1"] = newp1

    # Calculate the error
    grid_error = spatial_error(
        matching_points=matching_points,
        threshold_distance=threshold_distance,
        threshold_npoints=threshold_npoints,
        grid=grid,
    )

    # if grid_error is a grid, convert to torch.Tensor
    if isinstance(grid_error, np.ndarray):
        grid_error = torch.from_numpy(grid_error).to(image1.device).float()

    return grid_error


def spatial_model_fit(
    matching_points: Dict[str, torch.Tensor],
    n_points: Optional[int] = 10,
    threshold_distance: Optional[int] = 5,
    degree: Optional[int] = 1,
    verbose: Optional[bool] = True,
    scale: Optional[int] = 1,
    return_rmse: Optional[bool] = False,
):
    """Get a model that minimizes the spatial error between two images

    Args:
        matching_points (Dict[str, torch.Tensor]): A dictionary with the points0, points1 and image size.
        n_points (Optional[int], optional): The minimum number of points. Defaults to 10.
        threshold_distance (Optional[int], optional): The maximum distance between the points. Defaults to 5 pixels.
        degree (Optional[int], optional): The degree of the polynomial. Defaults to 1.
        verbose (Optional[bool], optional): If True, print the error. Defaults to False.
        scale (Optional[int], optional): The scale factor to use. Defaults to 1.
        return_rmse (Optional[bool], optional): If True, return the RMSE. Defaults to False.
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
        raise ValueError("Not enough points to fit the model")

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
            "rmse": (np.mean(full_error2), np.mean(full_error1))
        }

    return model_x, model_y


def spatial_model_transform(
    image1: torch.Tensor,
    spatial_models: tuple,
    precision: Optional[int] = 2,
    interpolation_mode: Optional[str] = "bilinear",
    device: str = "cuda",
) -> torch.Tensor:
    """ Transform the image according to the spatial model

    Args:
        image1 (torch.Tensor): The image 1 with shape (B, H, W)
        spatial_models (tuple): A tuple with the models for x and y
        device (str, optional): The device to use. Defaults to 'cuda'.
        interpolation_mode (Optional[str], optional): The interpolation mode. 
            Defaults to 'bilinear'.
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
    x = torch.arange(0, image1.shape[-2]/precision, 1/precision).to(device)
    y = torch.arange(0, image1.shape[-1]/precision, 1/precision).to(device)
    
    xx, yy = torch.meshgrid(x, y)

    # Flatten the coordinates
    xx = xx.flatten().unsqueeze(1)
    yy = yy.flatten().unsqueeze(1)

    # Predict the new coordinates
    xx_new = model_x.predict(xx.cpu().numpy())
    yy_new = model_y.predict(yy.cpu().numpy())

    # Reshape the coordinates
    xx_new = xx_new.reshape(image1.shape[-2], image1.shape[-2])*precision
    yy_new = yy_new.reshape(image1.shape[-1], image1.shape[-1])*precision

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


def spatial_plot_01(
    image0: torch.Tensor,
    image1: torch.Tensor,
    points0: torch.Tensor,
    points1: torch.Tensor,
    matches01: Dict[str, torch.Tensor],
    threshold_distance: Optional[int] = 5,
):
    
    # if the distance between the points is higher than threshold_distance pixels,
    # it is considered a bad match
    dist = torch.sqrt(torch.sum((points0 - points1) ** 2, dim=1))
    thres = dist < threshold_distance
    p0 = points0[thres]
    p1 = points1[thres]

    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(p0, p1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    plt.show()


if __name__ == "__main__":

    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    import pathlib
    from opensr_test.lightglue.utils import load_image

    pathdir = pathlib.Path("demo/ROI_05021/")
    image1 = load_image(pathdir / "m_4111801_sw_11_060_20190831.tif")
    image2 = load_image(pathdir / "m_4111908_se_11_1_20130720.tif")
    
    image1 = image1[:, 900:1400, 900:1400]
    image2 = image2[:, 900:1400, 900:1400]

    # Set spatial models
    models = spatial_setup_model(device="cuda", features="disk", max_num_keypoints=4096)
    
    # Get the points and matches
    matching_points = spatial_get_matching_points(img01=image1, img02=image2, model=models, device="cuda")

    # Fix a image according to the matching points
    spatial_models = spatial_model_fit(
        matching_points=matching_points, threshold_distance=3, degree=1
    )
    
    new_image1 = spatial_model_transform(
        image1=image1,
        spatial_models=spatial_models,
        precision=4,
        interpolation_mode="nearest",
        device="cuda"
    )

    # Spatial error
    e1 = spatial_metric(image1, image2, models, device="cuda", grid=True, threshold_distance=3)
    e2 = spatial_metric(new_image1, image2, models, device="cuda", grid=True, threshold_distance=3)
    

    # Plots ----------------------------------------------------

    ## Display the error
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(e1)
    ax[0].set_title("Initial error")
    ax[1].imshow(e2)
    ax[1].set_title("Transformed error")
    plt.show()


    ## Display the matching points 
    spatial_plot_01(
        image0=image1,
        image1=image2,
        points0=matching_points["points0"],
        points1=matching_points["points1"],
        matches01=matching_points["matches01"],
        threshold_distance=5
    )

    new_m_p = spatial_get_matching_points(img01=image1, img02=image2, model=models, device="cuda")
    spatial_plot_01(
        image0=new_image1,
        image1=image2,
        points0=new_m_p["points0"],
        points1=new_m_p["points1"],
        matches01=new_m_p["matches01"],
        threshold_distance=5
    )

    ## Save results as GEOTIFF
    import rasterio as rio
    save_pathdir = pathdir / "demo/"
    save_pathdir.mkdir(exist_ok=True)

    metadata = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": np.nan,
        "width": new_image1.shape[2],
        "height": new_image1.shape[1],
        "count": new_image1.shape[0]
    }

    # Save the image1    
    img1_path = save_pathdir / "image1.tif"
    with rio.open(img1_path, "w", **metadata) as dst:
        dst.write(image1.cpu().numpy())

    # Save the image2
    img2_path = save_pathdir / "image2.tif"
    with rio.open(img2_path, "w", **metadata) as dst:
        dst.write(image2.cpu().numpy())

    # Save the new image1
    new_img1_path = save_pathdir / "new_image2.tif"
    with rio.open(new_img1_path, "w", **metadata) as dst:
        dst.write(new_image1.cpu().numpy())
