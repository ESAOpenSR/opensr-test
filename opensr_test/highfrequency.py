from typing import Literal

import numpy as np
import torch
from opensr_test.utils import Value, hq_histogram_matching


def hf_energy(image: torch.Tensor, downsample_factor: int) -> torch.Tensor:
    """ Computes the high-frequency energy of an image.
    By downsampling and upsampling the image using a
    bilinear interpolation with antialiasing filter.

    Args:
        image (torch.Tensor): The image to compute the
            high-frequency energy of.
        downsample_factor (int, optional): The factor to
            use for the downsampling.

    Return:
        torch.Tensor: The high-frequency energy of the 
            image.
    """

    # Downsample the image
    downsampled_image = torch.nn.functional.interpolate(
        input=image, scale_factor=1 / downsample_factor, mode="bilinear", antialias=True
    )

    # Upsample the downsampled image to the same size as the original image
    upsampled_downsampled_image = torch.nn.functional.interpolate(
        input=downsampled_image, size=image.shape[-2:], mode="bilinear", antialias=True
    )

    # Compute the high-frequency introduced by the upsampling
    return image - upsampled_downsampled_image


def fill_nan_with_mean(arr: torch.Tensor) -> torch.Tensor:
    """ Fills NaN values with the mean of the closest neighbors.
    
    Args:
        arr (torch.Tensor): The array to fill.
    
    Return:
        torch.Tensor: The filled array.
    """

    # Convert the tensor to a numpy array
    device = arr.device
    arr = arr.detach().cpu().numpy()

    # Find the indices of the NaN values
    nan_indices = np.argwhere(np.isnan(arr))

    # Pad the array to handle edge cases
    padded_arr = np.pad(arr, ((1, 1), (1, 1)), mode="reflect")

    # Compute the mean of the closest neighbors
    mean = np.zeros_like(arr)
    for i, j in nan_indices:
        mean[i, j] = np.nanmean(padded_arr[i : i + 3, j : j + 3])

    # Replace the NaN values with the mean of the closest neighbors
    filled_arr = np.where(np.isnan(arr), mean, arr)

    # Convert the array back to a tensor
    filled_arr = torch.from_numpy(filled_arr)

    return filled_arr.to(device).float()


def hq_reduce(
    x: torch.Tensor, reduction: Literal["mean", "sum", "median"]
) -> torch.Tensor:
    """Reduces a given tensor by a given reduction method.
    
    Args:
        x (torch.Tensor): The tensor to reduce
        reduction (Literal["mean", "sum", "median"]): The reduction
            method to use. ('elementwise_mean', 'none', 'sum')
            
    Return:
        torch.Tensor: The reduced tensor
        
    Raise:
        ValueError: If the reduction method is not supported.
    """
    if reduction == "mean":
        return torch.nanmean(x)
    if reduction == "none" or reduction is None:
        raise ValueError("None is not a valid reduction method.")
    if reduction == "sum":
        return torch.nansum(x)
    if reduction == "median":
        return torch.nanmedian(x)

    raise ValueError("Reduction parameter unknown.")


def hq_mtf_each(
    image1: torch.Tensor, image2: torch.Tensor, reduction: str, scale: int
) -> torch.Tensor:
    """Calculates a cross-sensor MTF metric by comparing the
        LR image and SR image.

    Args:
        image1 (torch.Tensor): The LR to HR image (C, H, W).
        image2 (torch.Tensor): The SR or target image (C, H, W).
        reduction (str, optional): The reduction method to use for
            the MTF metric. Defaults to "mean". Other options 
            include "sum", and "median".
        scale (int, optional): The scale of the super-resolution.
            Defaults to 4.
    
    Returns:
        torch.Tensor: The MTF value.
    """

    # Compute mask
    freq = torch.fft.fftfreq(image2.shape[-1])
    freq = torch.fft.fftshift(freq)
    kfreq2D = torch.meshgrid(freq, freq, indexing="ij")
    knrm = torch.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    tensormask = (knrm > (0.5 * 1 / scale)) & (knrm < (0.5))

    fft_preds = torch.abs(torch.fft.fftshift(torch.fft.fft2(image1)))
    fft_target = torch.abs(torch.fft.fftshift(torch.fft.fft2(image2)))

    mtf = torch.masked_select(fft_preds / fft_target, tensormask)

    return hq_reduce(1 - torch.clamp(mtf, 0, 1), reduction)


def hq_mtf(lr_to_hr: torch.Tensor, sr: torch.Tensor, scale: int) -> Value:
    """ Calculates the MTF between two images.

    Args:
        lr_to_hr (torch.Tensor): The LR image in the
            HR space (C, H, W).
        sr (torch.Tensor): The SR image (C, H, W).
        scale (int, optional): The scale of the super-resolution.

    Returns:
        torch.Tensor: The MTF value.
    """
    ks = scale * 2 + 1

    # Pad the images to be divisible by the patch size
    lr_to_hr_pad = torch.nn.functional.pad(
        input=lr_to_hr,
        pad=(0, ks - lr_to_hr.shape[-1] % ks, 0, ks - lr_to_hr.shape[-2] % ks),
        mode="reflect",
    )

    sr_pad = torch.nn.functional.pad(
        input=sr,
        pad=(0, ks - sr.shape[-1] % ks, 0, ks - sr.shape[-2] % ks),
        mode="reflect",
    )

    # Dimensions of the new image (after chopping)
    high_dim = lr_to_hr_pad.shape[-2:]
    newdiv = (high_dim[0] // ks, high_dim[1] // ks)

    # chop image in patches of size ks x ks an
    # image of size (C, H, W) -> (C, newdiv[0], newdiv[1], ks, ks)
    # considering the stride of the patches as 4
    lr_to_hr_patch = lr_to_hr_pad.unfold(1, ks, ks).unfold(2, ks, ks)
    sr_patch = sr_pad.unfold(1, ks, ks).unfold(2, ks, ks)

    # Compute MTF for each patch.
    mtf_results = torch.zeros(newdiv)
    for i in range(newdiv[0]):
        for j in range(newdiv[1]):
            # Get the patch
            preds_p_iter = lr_to_hr_patch[:, i, j, :, :]
            target_p_iter = sr_patch[:, i, j, :, :]

            preds_p_iter = hq_histogram_matching(preds_p_iter, target_p_iter)

            ## Estimate the MTF
            mtf_pred = hq_mtf_each(
                image1=preds_p_iter, image2=target_p_iter, reduction="mean", scale=scale
            )

            # Save the results
            mtf_results[i, j] = mtf_pred

    # Fill NaN values with the mean of the closest neighbors
    mtf_results_filled = fill_nan_with_mean(mtf_results)
    mtf_results_up = torch.nn.functional.interpolate(
        input=mtf_results_filled[None, None], size=sr.shape[-2:], mode="bicubic"
    ).squeeze()

    # Remove the padding
    mtf_results_up = mtf_results_up[0 : lr_to_hr.shape[-2], 0 : lr_to_hr.shape[-1]]

    return Value(mtf_results_up, "Cross-sensor MTF")


def hq_simple(lr_to_hr: torch.Tensor, sr: torch.Tensor) -> Value:
    """ Calculates the difference between the 
    SR and LR to HR images.

    Args:
        lr_to_hr (torch.Tensor): The LR image in the
            HR space (C, H, W).
        sr (torch.Tensor): The SR image (C, H, W).

    Returns:
        Value: The difference between the SR and 
            LR' image.
    """
    return Value(
        value=torch.median(torch.abs(sr - lr_to_hr), axis=0)[0],
        description="|SR'-LR| diff",
    )


def hq_convolve(sr: torch.Tensor, factor: int) -> Value:
    """ Estimate high-frequency information by substract a 
    loss-pass filtered version of the image

    Args:
        sr (torch.Tensor): The SR image (C, H, W).
        factor (int, optional): The factor to use for the 
            low-pass filter.
    Returns:
        Value: The difference between the SR and SR' [Up/Down] 
            image.            
    """
    sr_energy = hf_energy(sr[None], downsample_factor=factor).squeeze()
    sr_energy_median = torch.median(torch.abs(sr_energy), axis=0)[0]

    return Value(value=sr_energy_median, description="|SR'| Up/Down")


def highfrequency(
    lr_to_hr: torch.Tensor, sr_norm: torch.Tensor, metric: str, scale: int, grid: bool
) -> Value:
    """ Calculate the high-frequencies difference between two images.

    Args:
        lr_to_hr (torch.Tensor): The LR image in the HR space (C, H, W).
        sr_norm (torch.Tensor): The SR with systematic errors removed (C, H, W).
        metric (str, optional): The metric to use. Defaults to 'mtf'.
        scale (int, optional): The scale of the super-resolution.
        grid (bool, optional): Whether to return the metric as a grid or not
        
    Returns:
        torch.Tensor: The metric value.
    """

    # if images are no square return error
    if lr_to_hr.shape[-1] != lr_to_hr.shape[-2]:
        raise ValueError("image1 must be square.")

    if sr_norm.shape[-1] != sr_norm.shape[-2]:
        raise ValueError("image2 must be square.")

    if metric == "mtf":
        metric_value = hq_mtf(lr_to_hr, sr_norm, scale=scale)
    elif metric == "convolve":
        metric_value = hq_convolve(sr_norm, factor=scale)
    elif metric == "simple":
        metric_value = hq_simple(lr_to_hr, sr_norm)
    else:
        raise ValueError("Metric not supported.")

    if grid:
        return metric_value

    return Value(float(metric_value.value.median()), metric_value.description)
