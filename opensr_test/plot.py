from typing import Dict, Optional, Tuple

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mpltern
import numpy as np
import torch
from skimage import exposure
from matplotlib.patches import ArrowStyle, FancyArrowPatch


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
    axs[0].set_title("LR", fontsize=14, fontweight="bold")
    axs[1].imshow(sr_img)
    axs[1].set_title("SR", fontsize=14, fontweight="bold")
    axs[2].imshow(hr_img)
    axs[2].set_title("HR", fontsize=14, fontweight="bold")

    # Add the suptitle
    fig.suptitle("Scale factor: %.2f" % scale_factor, fontsize=16)
    
    # return the figure to plot
    return fig, axs


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
    axs[1, 0].imshow(e1.cpu().clone())
    axs[1, 0].set_title(
        "%s \n %s: %s" % (r"$\bf{Reflectance\ Consistency \downarrow}$", e1_title, e1_subtitle)
    )

    # Display the spectral map error
    axs[1, 1].imshow(e2.cpu().clone())
    axs[1, 1].set_title(
        "%s \n %s: %s" % (r"$\bf{Spectral\ Consistency \downarrow}$", e2_title, e2_subtitle)
    )

    # Display the distance to the omission space
    e3p = e3[~torch.isnan(e3)]
    p5, p95 = np.percentile(e3p.flatten().cpu().clone().numpy(), [2, 98])
    axs[1, 2].imshow(e3, vmin=p5, vmax=p95)
    axs[1, 2].set_title(
        "%s \n %s: %s"
        % (r"$\bf{Distance\ to\ Omission\ Space \uparrow}$", e3_title, e3_subtitle)
    )

    # Display the distance to the hallucination space
    e5p = e5[~torch.isnan(e5)]
    p5, p95 = np.percentile(e5p.flatten().cpu().clone().numpy(), [2, 98])
    axs[1, 3].imshow(e5, vmin=p5, vmax=p95)
    axs[1, 3].set_title(
        "%s \n %s: %s"
        % (r"$\bf{Distance\ to\ Hallucination\ Space \uparrow}$", e5_title, e5_subtitle)
    )


    # Display the distance to the improvement space
    e4p = e4[~torch.isnan(e4)]
    p5, p95 = np.percentile(e4p.flatten().cpu().clone().numpy(), [2, 98])
    axs[1, 4].imshow(e4, vmin=p5, vmax=p95)
    axs[1, 4].set_title(
        "%s \n %s: %s"
        % (r"$\bf{Distance\ to\ Improvement\ Space \downarrow}$", e4_title, e4_subtitle)
    )

    return fig, axs


def display_tc_score(
    sr_rgb: torch.Tensor,
    hr_rgb: torch.Tensor,
    d_im_ref: torch.Tensor,
    d_om_ref: torch.Tensor,
    tc_score: torch.Tensor,
    log_scale: bool = False,
    stretch: Optional[str] = "linear",
):    
    # Apply the stretch
    if stretch == "linear":
        sr_rgb = linear_fix(sr_rgb)
        hr_rgb = linear_fix(hr_rgb)
    elif stretch == "histogram":
        sr_rgb = equalize_hist(sr_rgb)
        hr_rgb = equalize_hist(hr_rgb)
    else:
        sr_rgb = do_nothing(sr_rgb)
        hr_rgb = do_nothing(hr_rgb)

    # Custom categorical colormap - Blue[0], Green[1], Red[2]
    categorical_map = colors.ListedColormap(["blue", "green", "red"])
    bounds = [0, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, categorical_map.N)

    # Define triplets
    p1 = d_im_ref.ravel().cpu().clone()
    p1 = p1[~torch.isnan(p1)]

    p2 = d_om_ref.ravel().cpu().clone()
    p2 = p2[~torch.isnan(p2)]
    
    p3 = tc_score.ravel().cpu().clone()
    p3 = p3[~torch.isnan(p3)]
    
    if log_scale:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        ax = axs.flatten()
        ax[0].imshow(sr_rgb)
        ax[0].set_title("SR RGB", fontsize=20, fontweight="bold")
        ax[1].imshow(hr_rgb)
        ax[1].set_title("HR RGB", fontsize=20, fontweight="bold")
        ax[2].imshow(tc_score, cmap=categorical_map, norm=norm)
        ax[2].set_title("TC score - GRID", fontsize=20, fontweight="bold")
        ax[3].scatter(p2, p1, c=p3, cmap=categorical_map, norm=norm)
        ax[3].set_ylabel("$d_{im}$", fontsize=18)
        ax[3].set_xlabel("$d_{om}$", fontsize=18)
        ax[3].set_title("TC score - 2D", fontsize=20, fontweight="bold")
        ax[3].set_yscale("log")
        ax[3].set_xscale("log")    
        # make square and equal
        ax[3].set_aspect(1.0/ax[3].get_data_ratio(), adjustable='box')
    else:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        ax = axs.flatten()
        ax[0].imshow(sr_rgb)
        ax[0].set_title("SR RGB", fontsize=20, fontweight="bold")
        ax[1].imshow(hr_rgb)
        ax[1].set_title("HR RGB", fontsize=20, fontweight="bold")
        ax[2].imshow(tc_score, cmap=categorical_map, norm=norm)
        ax[2].set_title("TC score - GRID", fontsize=20, fontweight="bold")
        ax[3].scatter(p2, p1, c=p3, cmap=categorical_map, norm=norm)
        ax[3].set_ylabel("$d_{im}$", fontsize=18)
        ax[3].set_xlabel("$d_{om}$", fontsize=18)
        ax[3].set_title("TC score - 2D", fontsize=20, fontweight="bold")
    return fig, ax


def display_ternary(
    ha: torch.Tensor,
    om: torch.Tensor,
    im: torch.Tensor
):
    """ Display the ternary plot of the HA, OM and IM values.

    Args:
        ha (torch.Tensor): The hallucination error (H, W).
        om (torch.Tensor): The omission error (H, W).
        im (torch.Tensor): The improvement error (H, W).
    
    Returns:
        fig, ax: The figure and axes of the plot.
    """

    # Set background color
    # matplotlib.rc('axes',edgecolor='black',linewidth=1.2)

    fig = plt.figure()
    ax = fig.add_subplot(projection='ternary')

    # Generate data
    pc = ax.hexbin(im, om, ha, edgecolors="none", gridsize=50, cmap='viridis', bins='log')
    cax = ax.inset_axes([1.1, 0.1, 0.05, 0.9], transform=ax.transAxes)
    colorbar = fig.colorbar(pc, cax=cax)
    colorbar.set_label("Count", rotation=270, va="baseline")

    # Set labels with formatting
    ax.set_tlabel('Im', fontsize=15, color='blue', weight='bold')
    ax.set_llabel('Om', fontsize=15, color='green', weight='bold')
    ax.set_rlabel('Ha', fontsize=15, color='red', weight='bold')

    # Set grid properties
    ax.grid(color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Configure axis tick and color properties
    ax.taxis.set_tick_params(grid_color='blue', labelcolor='blue', color='red', labelsize=12)
    ax.laxis.set_tick_params(grid_color='green', labelcolor='green', color='blue', labelsize=12)
    ax.raxis.set_tick_params(grid_color='red', labelcolor='red', color='blue', labelsize=12)

    # Remove tick positions
    ax.taxis.set_ticks_position('none')
    ax.laxis.set_ticks_position('none')
    ax.raxis.set_ticks_position('none')

    # Add arrows
    arrowstyle = ArrowStyle('simple', head_length=10, head_width=5)
    kwargs_arrow = {
        'transform': ax.transAxes,  # Used with ``ax.transAxesProjection``
        'arrowstyle': arrowstyle,
        'linewidth': 1,
        'clip_on': False,  # To plot arrows outside triangle
        'zorder': -10,  # Very low value not to hide e.g. tick labels.
    }

    # Start of arrows in barycentric coordinates.
    ta = np.array([ 0.0, -0.1,  1.1])
    la = np.array([ 1.1,  0.0, -0.1])
    ra = np.array([-0.1,  1.1,  0.0])

    # End of arrows in barycentric coordinates.
    tb = np.array([ 1.0, -0.1,  0.1])
    lb = np.array([ 0.1,  1.0, -0.1])
    rb = np.array([-0.1,  0.1,  1.0])

    # This transforms the above barycentric coordinates to the original Axes
    # coordinates. In combination with ``ax.transAxes``, we can plot arrows fixed
    # to the Axes coordinates.
    f = ax.transAxesProjection.transform

    tarrow = FancyArrowPatch(f(ta), f(tb), ec='blue', fc='blue', **kwargs_arrow)
    larrow = FancyArrowPatch(f(la), f(lb), ec='green', fc='green', **kwargs_arrow)
    rarrow = FancyArrowPatch(f(ra), f(rb), ec='red', fc='red', **kwargs_arrow)
    ax.add_patch(tarrow)
    ax.add_patch(larrow)
    ax.add_patch(rarrow)

    # To put the axis-labels at the positions consistent with the arrows above, it
    # may be better to put the axis-label-text directly as follows rather than
    # using e.g.  ax.set_tlabel.
    kwargs_label = {
        'transform': ax.transTernaryAxes,
        'backgroundcolor': 'w',
        'ha': 'center',
        'va': 'center',
        'rotation_mode': 'anchor',
        'fontsize': 12,
        'zorder': -9,  # A bit higher on arrows, but still lower than others.
    }

    # Put axis-labels on the midpoints of arrows.
    tpos = (ta + tb) * 0.5
    lpos = (la + lb) * 0.5
    rpos = (ra + rb) * 0.5

    ax.text(*tpos, 'Improvement'  , color='blue', rotation=-60, **kwargs_label)
    ax.text(*lpos, 'Omission' , color='green', rotation= 60, **kwargs_label)
    ax.text(*rpos, 'Hallucination', color='red', rotation=  0, **kwargs_label)
    
    return fig, ax


def display_stats(object):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()

    d1 = object.d_ref.cpu().numpy().flatten()
    axs[0].hist(d1, bins=25, alpha=0.5, label='LR-HR distance')
    axs[0].set_title("Distance between the LR and HR images", fontsize=12, fontweight="bold")
    axs[0].legend()

    d2 = object.d_im.cpu().numpy().flatten()
    axs[1].hist(d2, bins=25, alpha=0.5, label='SR-HR distance')
    axs[1].set_title("Distance between the SR and HR images", fontsize=12, fontweight="bold")
    axs[1].legend()

    d3 = object.d_om.cpu().numpy().flatten()
    axs[2].hist(d3, bins=25, alpha=0.5, label='SR-LR distance')
    axs[2].set_title("Distance between the SR and LR images", fontsize=12, fontweight="bold")
    axs[2].legend()

    omission = object.omission.cpu().numpy().flatten()
    axs[3].hist(omission, bins=25, alpha=0.5, label='lower is better')
    axs[3].set_title("Omission - reference distance [0-1]", fontsize=12, fontweight="bold")
    axs[3].legend()

    axs[4].hist(object.improvement.cpu().numpy().flatten(), bins=25, alpha=0.5, label='higher is better')
    axs[4].set_title("Improvement - reference distance [0-1]", fontsize=12, fontweight="bold")
    axs[4].legend()

    axs[5].hist(object.hallucination.cpu().numpy().flatten(), bins=25, alpha=0.5, label='lower is better')
    axs[5].set_title("Hallucination - reference distance [0-1]", fontsize=12, fontweight="bold")
    axs[5].legend()    
    
    return fig, axs
