import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch

from opensr_test.lightglue import LightGlue, SuperPoint, DISK
from opensr_test.lightglue.utils import load_image, rbd
from opensr_test.lightglue import viz2d

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from typing import Optional, Union

def distance_matrix(x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray) -> np.ndarray:
    """ Calculate the distance matrix between two sets of points

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
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])
    return np.hypot(d0, d1)


def linear_rbf(x: np.ndarray, y: np.ndarray, z: np.ndarray, xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    """ Interpolate using radial basis functions

    Args:
        x (np.ndarray): Array with the x coordinates of the points (image 1)
        y (np.ndarray): Array with the y coordinates of the points (image 1)
        z (np.ndarray): Array with the z coordinates of the points (image 1)
        xi (np.ndarray): Array with the x coordinates of the points (target image)
        yi (np.ndarray): Array with the y coordinates of the points (target image)
    Returns:
        np.ndarray: Array with the interpolated values
    """
    dist = distance_matrix(x,y, xi,yi)

    # Mutual pariwise distances between observations
    internal_dist = distance_matrix(x,y, x,y)

    # Now solve for the weights such that mistfit at the observations is minimized
    weights = np.linalg.solve(internal_dist, z)

    # Multiply the weights for each interpolated point by the distances
    zi =  np.dot(dist.T, weights)

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

    # Create polynomial features
    poly_reg = PolynomialFeatures(degree=d)
    X_poly = poly_reg.fit_transform(X)
    # Create linear regression object
    lin_reg = LinearRegression()
    
    # Fit polynomial features
    linear_model = lin_reg.fit(X_poly, y)

    # Predict using linear model
    y_pred = linear_model.predict(X_poly)

    return y_pred


def spatial_setup_model(
        features: str='superpoint',
        matcher: str='lightglue',
        max_num_keypoints: int=2048,
        device: str='cuda'
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
    if features == 'superpoint':
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    elif features == 'disk':
        extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(device)
    else:
        raise ValueError(f'Unknown feature extractor {features}')

    # Local feature matcher   
    if matcher == 'lightglue':
        matcher = LightGlue(features=features).eval().to(device)
    else:
        raise ValueError(f'Unknown matcher {matcher}')

    return extractor, matcher


def spatial_prediction(img01: torch.Tensor, img02: torch.Tensor, model: tuple) -> tuple:
    """Predict the spatial error between two images

    Args:
        img01 (torch.Tensor): A torch.tensor with the image 1 (B, H, W)
        img02 (torch.Tensor): A torch.tensor with the ref image (B, H, W)
        model (tuple): A tuple with the feature extractor and the matcher

    Returns:
        tuple: A tuple with the points0, points1 and matches01
    """
    
    # unpack the model
    extractor, matcher = model

    # extract local features
    with torch.no_grad():
        # auto-resize the image, disable with resize=None
        feats0 = extractor.extract(img01, resize=None)  
        feats1 = extractor.extract(img02, resize=None)

        # match the features
        matches01 = matcher({'image0': feats0, 'image1': feats1})
    
    # remove batch dimension
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
    
    # Save image size
    matches01["image_size"] = tuple(img01.shape[-2:])

    return points0, points1, matches01

def spatial_error(
        points0: torch.Tensor,
        points1: torch.Tensor,
        matches01: torch.Tensor,
        threshold_score: Optional[float]=0.2,
        threshold_distance: Optional[int]=5,
        degree: Optional[int]=1,
        grid: Optional[bool]=None
):
    """ Calculate the spatial error between two images

    Args:
        points0 (torch.Tensor): The keypoints of the image 1.
        points1 (torch.Tensor): The keypoints of the image 2.
        matches01 (torch.Tensor): The matches between the keypoints of the image 1 and the image 2.
        threshold_score (Optional[float], optional): A assigned score to the matches. Defaults to 0.2.
        threshold_distance (Optional[int], optional): The maximum distance between the points. Defaults to 5 pixels.
        degree (Optional[int], optional): The degree of the polynomial. Defaults to 1.
        grid (Optional[bool], optional): If True, return the grid with the error. Defaults to None.

    Returns:
        np.ndarray: The spatial error between the two images
    """    

    # filter out invalid matches
    thres = matches01["scores"] > threshold_score
    p0 = points0[thres]
    p1 = points1[thres]

    # if not enough points, return 0
    if p0.shape[0] < 10:
        return -1

    # if the distance between the points is higher than 5 pixels, 
    # it is considered a bad match
    dist = torch.sqrt(torch.sum((p0 - p1) ** 2, dim=1))
    thres = dist < threshold_distance
    p0 = p0[thres]
    p1 = p1[thres]

    # if not enough points, return 0
    if p0.shape[0] < 10:
        return -1
    
    # from torch.Tensor to numpy array
    p0 = p0.detach().cpu().numpy()
    p1 = p1.detach().cpu().numpy()

    # Fit a polynomial of degree 2 to the points
    X = p0[:,0].reshape(-1,1)
    y = p1[:,0].reshape(-1,1)
    error_1 = np.abs(y - spatia_polynomial_fit(X, y,  degree))
    
    X = p0[:,1].reshape(-1,1)
    y = p1[:,1].reshape(-1,1)
    error_2 = np.abs(y - spatia_polynomial_fit(X, y,  degree))
    
    # Calculate the error
    full_error = np.sqrt(error_1**2 + error_2**2)
    
    if grid:
        # Spatial interpolation using RBF
        x, y = p0[:, 0], p0[:, 1]
        z = full_error[:, 0]

        ## Create the grid
        grid_x, grid_y = matches01["image_size"]
        arr_grid = np.array(list(
            itertools.product(
                np.arange(0, grid_x),
                np.arange(0, grid_y)
            )
        ))
        spgrid = linear_rbf(x, y, z, arr_grid[:,0], arr_grid[:,1]).reshape(grid_x, grid_y)
        return spgrid
    else:
        return np.mean(full_error)

def spatial_metric(
    image1: torch.Tensor,
    image2: torch.Tensor,
    features: str='superpoint',
    matcher: str='lightglue',
    max_num_keypoints: int=2048,    
    threshold_score: Optional[float]=0.2,
    threshold_distance: Optional[int]=5,
    degree: Optional[int]=1,
    grid: Optional[bool]=True,
    device: str='cuda'
) -> Union[float, np.ndarray]:
    """ Calculate the spatial error between two images

    Args:
        image1 (torch.Tensor): The image 1 with shape (B, H, W)
        image2 (torch.Tensor): The image 2 with shape (B, H, W)
        features (str, optional): The feature extractor. Defaults to 'superpoint'.
        matcher (str, optional): The matcher. Defaults to 'lightglue'.
        max_num_keypoints (int, optional): The maximum number of keypoints. Defaults to 2048.
        threshold_score (Optional[float], optional): A assigned score to the matches. Defaults to 0.2.
        threshold_distance (Optional[int], optional): The maximum distance between the points. 
            Defaults to 5 pixels.
        degree (Optional[int], optional): The degree of the polynomial. Defaults to 1. 
        grid (Optional[bool], optional): If True, return the grid with the error. Defaults to True.
        device (str, optional): The device to use. Defaults to 'cuda'.

    Returns:
        Union[float, np.ndarray]: The spatial error between the two images. If grid=True, return
            the grid with the error, otherwise return the mean error (RMSEmean).

    """        


    # Create the model
    models = spatial_setup_model(
        features=features,
        matcher=matcher,
        max_num_keypoints=max_num_keypoints,
        device=device
    )

    # Get the points and matches
    points0, points1, matches01 = spatial_prediction(
        img01=image1, 
        img02=image2,
        model=models
    )
    
    # Calculate the error
    grid_error = spatial_error(
        points0=points0,
        points1=points1,
        matches01=matches01,
        threshold_score=threshold_score,
        threshold_distance=threshold_distance,
        degree=degree,
        grid=grid
    )

    return grid_error

def spatial_plot_01(image0, image1, points0, points1, matches01):
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(points0, points1, color='lime', lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    plt.show()


if __name__ == "__main__":
    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    image1 = load_image('/home/gonzalo/Desktop/S2NAIP/ROI_09075/m_3712141_nw_10_1_20100529.tif')
    image2 = load_image('/home/gonzalo/Desktop/S2NAIP/ROI_09075/m_3712141_nw_10_060_20200525.tif')
    image1 = image1[:, 0:1024, 0:1024]
    image2 = image2[:, 0:1024, 0:1024]

    # Calculate the spatial error 
    error = spatial_metric(image1, image2, device='cpu')
    
    # Plot the spatial error
    models = spatial_setup_model(device='cpu')

    # Get the points and matches
    points0, points1, matches01 = spatial_prediction(
        img01=image1, 
        img02=image2,
        model=models
    )

    spatial_plot_01(image1, image2, points0, points1, matches01)
