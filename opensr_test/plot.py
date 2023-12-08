from typing import Dict, Optional, Tuple

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from opensr_test.lightglue import viz2d
from skimage import exposure


def min_max_range(tensor):
    """ 
    Return the min and max range of a tensor
    """
    tensor[torch.isnan(tensor)] = 0
    rmax = torch.tensor([tensor.min(), tensor.max()]).abs().min()
    rmin = rmax * -1
    return rmin, rmax


def linear_fix(img: torch.Tensor, permute=True) -> torch.Tensor:
    """ Linearly stretch the values of the image to increase contrast.

    Args:
        img (torch.Tensor): The RGB image to stretch (3xHxW).
        permute (bool, optional): Permute the dimensions to HxWx3.
            Defaults to True.
    Returns:
        torch.Tensor: The stretched image (HxWx3)
    """
    img = img.detach().cpu().numpy()
    container = []
    for i in range(3):
        p2, p98 = np.percentile(img[i, ...], (2, 98))
        img_rescale = exposure.rescale_intensity(img[i, ...], in_range=(p2, p98))
        container.append(img_rescale)
    new_img = np.array(container)
    if permute:
        new_img = np.moveaxis(new_img, 0, -1)
    return torch.from_numpy(new_img).float()


def do_nothing(img: torch.Tensor, permute=True) -> torch.Tensor:
    """ Do nothing to the image.
    
    Args:
        img (torch.Tensor): The RGB image to stretch (3xHxW).
        permute (bool, optional): Permute the dimensions to HxWx3.
            Defaults to True.
    Returns:
        torch.Tensor: The stretched image (HxWx3)
    """
    img = img.detach().cpu().numpy()
    container = []
    for i in range(3):
        img_rescale = img[i, ...]
        container.append(img_rescale)
    new_img = np.array(container)
    if permute:
        new_img = np.moveaxis(new_img, 0, -1)
    return torch.from_numpy(new_img).float()


def equalize_hist(img: torch.Tensor, permute=True) -> torch.Tensor:
    """ Equalize the histogram of the image to increase contrast.

    Args:
        img (torch.Tensor): The RGB image to stretch (3xHxW).
        permute (bool, optional): Permute the dimensions to HxWx3.
    Returns:
        torch.Tensor: The stretched image (HxWx3)
    """
    img = img.detach().cpu().numpy()
    container = []
    for i in range(3):
        img_rescale = exposure.equalize_hist(img[i, ...])
        container.append(img_rescale)
    new_img = np.array(container)
    if permute:
        new_img = np.moveaxis(new_img, 0, -1)
    return torch.from_numpy(new_img).float()


def quadruplets(
    lr_img: torch.Tensor,
    sr_img: torch.Tensor,
    hr_img: torch.Tensor,
    landuse_img: Optional[torch.Tensor] = None,
    stretch: Optional[str] = "linear",
) -> Tuple[plt.figure, plt.Axes]:
    """ Display LR, SR, HR and Landuse images in a single figure.

    Args:
        lr_img (torch.Tensor): The LR RGB (3xHxW) image.
        sr_img (torch.Tensor): The SR RGB (3xHxW) image.
        hr_img (torch.Tensor): The HR RGB (3xHxW) image.
        landuse_img (torch.Tensor): The landuse grayscale (HxW) image.
            Optional, defaults to None.
        stretch (Optional[str], optional): Option to stretch the values 
            to increase contrast: "lin" (linear) or "hist" (histogram)

    Raises:
        ValueError: The LR image must be a RGB (3xHxW) image.
        ValueError: The SR image must be a RGB (3xHxW) image.
        ValueError: The HR image must be a RGB (3xHxW) image.
        ValueError: The landuse image must be a grayscale (HxW) image.

    Returns:
        Tuple[plt.figure, plt.Axes]: The figure and axes to plot.
    """
    if lr_img.shape[0] == 4:
        raise ValueError("The LR image must be a RGB (3xHxW) image.")
    if sr_img.shape[0] == 4:
        raise ValueError("The SR image must be a RGB (3xHxW) image.")
    if hr_img.shape[0] == 4:
        raise ValueError("The HR image must be a RGB (3xHxW) image.")

    if landuse_img is None:
        return triplets(lr_img, sr_img, hr_img, stretch=stretch)

    # Apply the stretch
    if stretch == "linear":
        lr_img = linear_fix(lr_img)
        sr_img = linear_fix(sr_img)
        hr_img = linear_fix(hr_img)
    elif stretch == "histogram":
        lr_img = equalize_hist(lr_img)
        sr_img = equalize_hist(sr_img)
        hr_img = equalize_hist(hr_img)

    # Define the categorical values and colors
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    colors_list = [
        "#006400",
        "#ffbb22",
        "#ffff4c",
        "#f096ff",
        "#fa0000",
        "#b4b4b4",
        "#f0f0f0",
        "#0064c8",
        "#0096a0",
        "#00cf75",
        "#fae6a0",
    ]
    labels = [
        "Tree cover (10)",
        "Shrubland (20)",
        "Grassland (30)",
        "Cropland (40)",
        "Built-up (50)",
        "Bare / sparse vegetation (60)",
        "Snow and ice (70)",
        "Permanent water bodies (80)",
        "Herbaceous wetland (90)",
        "Mangroves (95)",
        "Moss and lichen (100)",
    ]

    # Set bounds and ticks
    bounds = [values[i] - 2.5 for i in range(len(values))]
    bounds.append(values[-1] + 2.5)
    ticks = [values[i] for i in range(len(values))]

    # Create a iterable colormap
    cmap = colors.ListedColormap(colors_list)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Plot the images
    scale_factor = hr_img.shape[0] / lr_img.shape[0]
    fig, axs = plt.subplots(1, 4, figsize=(25, 5))
    axs[0].imshow(lr_img)
    axs[0].set_title("LR")
    axs[1].imshow(sr_img)
    axs[1].set_title("SR")
    axs[2].imshow(hr_img)
    axs[2].set_title("HR")
    axs[3].imshow(landuse_img, cmap=cmap, interpolation="nearest", norm=norm)
    axs[3].set_title("LandUse")

    # Add the colorbar
    cbar_ax = axs[0].inset_axes([1.05, 0, 0.05, 1], transform=axs[3].transAxes)
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=ticks, boundaries=bounds)
    cbar.ax.set_yticklabels(labels)

    # Add the suptitle
    fig.suptitle("Scale factor: %.2f" % scale_factor, fontsize=16)

    # return the figure to plot
    return fig, axs


def triplets(
    lr_img: torch.Tensor,
    sr_img: torch.Tensor,
    hr_img: torch.Tensor,
    stretch: Optional[str] = "linear",
) -> Tuple[plt.figure, plt.Axes]:
    """ Display LR, SR, HR and Landuse images in a single figure.

    Args:
        lr_img (torch.Tensor): The LR RGB (3xHxW) image.
        sr_img (torch.Tensor): The SR RGB (3xHxW) image.
        hr_img (torch.Tensor): The HR RGB (3xHxW) image.
        stretch (Optional[str], optional): Option to stretch the values
            to increase contrast: "lin" (linear) or "hist" (histogram)

    Raises:
        ValueError: The LR image must be a RGB (3xHxW) image.
        ValueError: The SR image must be a RGB (3xHxW) image.
        ValueError: The HR image must be a RGB (3xHxW) image.
        ValueError: The landuse image must be a grayscale (HxW) image.

    Returns:
        Tuple[plt.figure, plt.Axes]: The figure and axes to plot.
    """
    if lr_img.shape[0] == 4:
        raise ValueError("The LR image must be a RGB (3xHxW) image.")
    if sr_img.shape[0] == 4:
        raise ValueError("The SR image must be a RGB (3xHxW) image.")
    if hr_img.shape[0] == 4:
        raise ValueError("The HR image must be a RGB (3xHxW) image.")

    # Apply the stretch
    if stretch == "linear":
        lr_img = linear_fix(lr_img)
        sr_img = linear_fix(sr_img)
        hr_img = linear_fix(hr_img)
    elif stretch == "histogram":
        lr_img = equalize_hist(lr_img)
        sr_img = equalize_hist(sr_img)
        hr_img = equalize_hist(hr_img)
    else:
        lr_img = do_nothing(lr_img)
        sr_img = do_nothing(sr_img)
        hr_img = do_nothing(hr_img)

    # Plot the images
    scale_factor = hr_img.shape[0] / lr_img.shape[0]
    fig, axs = plt.subplots(1, 3, figsize=(25, 5))
    axs[0].imshow(lr_img)
    axs[0].set_title("LR")
    axs[1].imshow(sr_img)
    axs[1].set_title("SR")
    axs[2].imshow(hr_img)
    axs[2].set_title("HR")

    # Add the suptitle
    fig.suptitle("Scale factor: %.2f" % scale_factor, fontsize=16)
    
    # return the figure to plot
    return fig, axs


def spatial_matches(
    lr: torch.Tensor,
    sr_to_lr: torch.Tensor,
    points0: torch.Tensor,
    points1: torch.Tensor,
    matches01: Dict[str, torch.Tensor],
    threshold_distance: Optional[int] = 5,
    stretch: Optional[str] = "linear",
):

    if stretch == "linear":
        sr_to_lr = linear_fix(sr_to_lr, permute=False)
        lr = linear_fix(lr, permute=False)
    elif stretch == "histogram":
        sr_to_lr = equalize_hist(sr_to_lr, permute=False)
        lr = equalize_hist(lr, permute=False)

    # if the distance between the points is higher than
    # threshold_distance pixels, it is considered a bad match
    dist = torch.sqrt(torch.sum((points0 - points1) ** 2, dim=1))
    thres = dist < threshold_distance
    p0 = points0[thres]
    p1 = points1[thres]

    axes = viz2d.plot_images([lr, sr_to_lr], titles=["LR", "SR to LR"])
    viz2d.plot_matches(p0, p1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    return axes


def display_results(
    lr: torch.Tensor,
    lrdown: torch.Tensor,
    sr: torch.Tensor,
    srharm: torch.Tensor,
    hr: torch.Tensor,
    e1: torch.Tensor,
    e1_title: str,
    e1_subtitle: str,
    e2: torch.Tensor,
    e2_title: str,
    e2_subtitle: str,
    e3: torch.Tensor,
    e3_points: Tuple[torch.Tensor, torch.Tensor],
    e3_title: str,
    e3_subtitle: str,
    e4: torch.Tensor,
    e4_title: str,
    e4_subtitle: str,
    e5: torch.Tensor,
    e5_title: str,
    e5_subtitle: str,
    stretch: Optional[str] = "linear",
):
    """ Display the results of the SR algorithm

    Args:
        lr (torch.Tensor): The LR image (3, H, W).
        sr (torch.Tensor): The SR image (3, H, W).
        hr (torch.Tensor): The HR image (3, H, W).
        e1 (torch.Tensor): The local reflectance map error (H, W).
        e1_title (str): The local reflectance error method.
        e1_subtitle (str): The median value of the local reflectance error.
        e2 (torch.Tensor): The spatial local errors (H, W).
        e2_points (Tuple[torch.Tensor, torch.Tensor]): The points to plot on 
            the spatial local errors.
        e2_title (str): The spatial local error method.
        e2_subtitle (str): The median value of the spatial local error.
        e3 (torch.Tensor): The HF map error (H, W).
        e3_title (str): The HF map error method.
        e3_subtitle (str): The median value of the HF map error.
        e4 (torch.Tensor): The improvement ratio (H, W).
        e4_title (str): The improvement ratio method.
        e4_subtitle (str): The median value of the improvement ratio.
        e5 (torch.Tensor): The hallucination error (H, W).
        e5_title (str): The hallucination error method.
        e5_subtitle (str): The median value of the hallucination error.

    Returns:
        fig, axs: The figure and axes of the plot.
    """

    # Apply the stretch
    if stretch == "linear":
        lr = linear_fix(lr)
        lrdown = linear_fix(lrdown)
        sr = linear_fix(sr)
        srharm = linear_fix(srharm)
        hr = linear_fix(hr)

    elif stretch == "histogram":
        lr = equalize_hist(lr)
        lrdown = equalize_hist(lrdown)
        sr = equalize_hist(sr)
        srharm = equalize_hist(srharm)
        hr = equalize_hist(hr)
    else:
        lr = do_nothing(lr)
        lrdown = do_nothing(lrdown)
        sr = do_nothing(sr)
        srharm = do_nothing(srharm)
        hr = do_nothing(hr)

    # Create the figure and axes (remove white space around the images)
    fig, axs = plt.subplots(2, 5, figsize=(20, 10), tight_layout=True)

    # Desactive all the axis
    for ax in axs.flatten():
        ax.axis("off")

    # Plot the first row (lr, sr, hr)
    axs[0, 0].imshow(lr)
    axs[0, 0].set_title("LR", fontsize=20, fontweight="bold")
    axs[0, 1].imshow(lrdown)
    axs[0, 1].set_title("LRdown", fontsize=20, fontweight="bold")
    axs[0, 2].imshow(sr)
    axs[0, 2].set_title("SR", fontsize=20, fontweight="bold")
    axs[0, 3].imshow(srharm)
    axs[0, 3].set_title("SRharm", fontsize=20, fontweight="bold")
    axs[0, 4].imshow(hr)
    axs[0, 4].set_title("HR", fontsize=20, fontweight="bold")

    # Display the local reflectance map error
    axs[1, 0].imshow(e1)
    axs[1, 0].set_title(
        "%s \n %s: %s" % (r"$\bf{Reflectance\ Consistency \downarrow}$", e1_title, e1_subtitle)
    )

    # Display the spectral map error
    axs[1, 1].imshow(e2)
    axs[1, 1].set_title(
        "%s \n %s: %s" % (r"$\bf{Spectral\ Consistency \downarrow}$", e2_title, e2_subtitle)
    )

    # Display the Spatial consistency
    if len(torch.tensor(e3).shape) == 0:
        axs[1, 2].imshow(e2 * torch.nan)
    else:
        axs[1, 2].imshow(e3, vmin=e3.min(), vmax=e3.max(), cmap="RdBu")
        axs[1, 2].plot(
            [x[0] for x in e3_points], [x[1] for x in e3_points], "r*", markersize=5
        )
    axs[1, 2].set_title(
        "%s \n %s: %s" % (r"$\bf{Spatial\ Consistency \downarrow}$", e3_title, e3_subtitle)
    )

    # Display the distance to the ommission space
    axs[1, 3].imshow(e4)
    axs[1, 3].set_title(
        "%s \n %s: %s"
        % (r"$\bf{Distance\ to\ Omission\ Space \uparrow}$", e4_title, e4_subtitle)
    )

    # Display the Hallucination error
    axs[1, 4].imshow(e5)
    axs[1, 4].set_title(
        "%s \n %s: %s"
        % (r"$\bf{Distance\ to\ Improvement\ Space \downarrow}$", e5_title, e5_subtitle)
    )

    return fig, axs


def display_tc_score(
    sr_rgb: torch.Tensor,
    d_im_ref: torch.Tensor,
    d_om_ref: torch.Tensor,
    tc_score: torch.Tensor,
    log_scale: bool = True,
    stretch: Optional[str] = "linear",
):    
    # Apply the stretch
    if stretch == "linear":
        sr_rgb = linear_fix(sr_rgb)
    elif stretch == "histogram":
        sr_rgb = equalize_hist(sr_rgb)
    else:
        sr_rgb = do_nothing(sr_rgb)

    # Custom categorical colormap - Blue[0], Green[1], Red[2]
    colorbar = colors.ListedColormap(["blue", "green", "red"])
    
    # Define triplets
    p1 = d_im_ref.ravel()
    p1 = p1[~torch.isnan(p1)]

    p2 = d_om_ref.ravel()
    p2 = p2[~torch.isnan(p2)]
    
    p3 = tc_score.ravel()
    p3 = p3[~torch.isnan(p3)]
    
    if log_scale:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(sr_rgb)
        ax[0].set_title("SR RGB", fontsize=20, fontweight="bold")
        ax[1].imshow(tc_score, cmap=colorbar)
        ax[1].set_title("TC score - GRID", fontsize=20, fontweight="bold")
        ax[2].scatter(p1, p2, c=p3, cmap=colorbar)
        ax[2].set_ylabel("$d_{im}$", fontsize=18)
        ax[2].set_xlabel("$d_{om}$", fontsize=18)
        ax[2].set_title("TC score - 2D", fontsize=20, fontweight="bold")
        ax[2].set_yscale("log")
        ax[2].set_xscale("log")    
        # make square and equal
        ax[2].set_aspect(1.0/ax[2].get_data_ratio(), adjustable='box')
    else:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(sr_rgb.permute(1, 2, 0)*3)
        ax[0].set_title("SR RGB", fontsize=20, fontweight="bold")
        ax[1].imshow(tc_score, cmap=colorbar)
        ax[1].set_title("TC score - GRID", fontsize=20, fontweight="bold")
        ax[2].scatter(p1, p2, c=p3, cmap=colorbar)        
        ax[2].set_ylabel("$d_{im}$", fontsize=18)
        ax[2].set_xlabel("$d_{om}$", fontsize=18)
        ax[2].set_title("TC score - 2D", fontsize=20, fontweight="bold")        
    return fig, ax
