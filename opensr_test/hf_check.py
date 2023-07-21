import torch
from typing import Optional, Literal
from skimage.exposure import match_histograms

def hq_reduce(x: torch.Tensor, reduction: Literal["mean", "sum", "median"]) -> torch.Tensor:
    """Reduces a given tensor by a given reduction method.
    Args:
        x: the tensor, which shall be reduced
        reduction:  a string specifying the reduction method ('elementwise_mean', 'none', 'sum')
    Return:
        reduced Tensor
    Raise:
        ValueError if an invalid reduction parameter was given
    """
    if reduction == "mean":
        return torch.mean(x)
    if reduction == "none" or reduction is None:
        raise ValueError("None is not a valid reduction method.")
    if reduction == "sum":
        return torch.sum(x)
    if reduction == "median":
        return torch.median(x)

    raise ValueError("Reduction parameter unknown.")


def hq_histogram_matching(
    image1: torch.Tensor,
    image2: torch.Tensor,
) -> torch.Tensor:
    """ Lazy implementation of histogram matching 

    Args:
        image1 (torch.Tensor): The low-resolution image (C, H, W).
        image2 (torch.Tensor): The super-resolved image (C, H, W).

    Returns:
        torch.Tensor: The super-resolved image with the histogram of the target image.
    """

    # Go to numpy
    np_image1 = image1.detach().cpu().numpy()
    np_image2 = image2.detach().cpu().numpy()
    
    # Apply histogram matching
    np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=0)

    # Go back to torch
    image1_hat = torch.from_numpy(np_image1_hat).to(image1.device)

    return image1_hat


def hq_mtf(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """ Calculates the MTF between two images.

    Args:
        image1 (torch.Tensor): The low-resolution image (C, H, W).
        image2 (torch.Tensor): The super-resolved or target image (C, H, W).

    Returns:
        torch.Tensor: The MTF value.
    """

    # Obtain the scale
    scale = image2.shape[-1] // image1.shape[-1]

    # Convert the LR to SRimage scale
    image1 = torch.nn.functional.interpolate(
        input=image1[None],
        size=image2.shape[-2:],
        mode='bilinear',
        antialias=True
    ).squeeze(0)
        
    # Ratio of chopping
    ks = 8

    # Dimensions of the new image (after chopping)
    newdiv = (
        image1.shape[-2:][0] // ks,
        image1.shape[-2:][1] // ks
    )

    # Split (C, H, W) into (C, H/ks, W/ks, ks, ks) patches.
    preds_p = image1.unfold(-1, ks, ks).unfold(-3, ks, ks)
    target_p = image2.unfold(-1, ks, ks).unfold(-3, ks, ks)


    # Compute MTF for each patch.
    mtf_results = torch.zeros(newdiv)
    for i in range(newdiv[0]):
        for j in range(newdiv[1]):
            # Get the patch
            preds_p_iter = preds_p[:, i, j, :, :]
            target_p_iter = target_p[:, i, j, :, :]
                
            # Apply histogram matching to the predicted image
            preds_p_iter = hq_histogram_matching(preds_p_iter, target_p_iter)

            # Estimate the MTF
            mtf_pred = hq_mtf_each(preds_p_iter, target_p_iter, scale=scale)

            # Save the results
            mtf_results[i, j] = mtf_pred
        
    return mtf_results


def hq_mtf_each(
    image1: torch.Tensor,
    image2: torch.Tensor,
    reduction: str = "mean",
    scale: int = 4
) -> int:
    """Calculates a cross-sensor MTF metric by comparing the
        LR image and SR image.

    Args:
        image1 (torch.Tensor): The low-resolution image (C, H, W).
        image2 (torch.Tensor): The super-resolved or target image (C, H, W).
        reduction (str, optional): The reduction method to use for the MTF
            metric. Defaults to "mean". Other options include "sum", and "median".
        scale (int, optional): The scale of the super-resolution. Defaults to 4.
    
    Returns:
        int: The MTF value.
    """

    # Compute Fourier transform
    fft_preds = torch.fft.fft2(image1)
    fft_target = torch.fft.fft2(image2)
    
    # Compute MTF
    mtf = torch.mean(torch.abs(fft_preds/fft_target), axis=0)

    # clip values higher than 1
    mtf = torch.clamp(mtf, 0, 1)
    
    # Compute mask
    freq = torch.fft.fftfreq(image2.shape[-1]) # * 1.0 / self.pixel_size
    kfreq2D = torch.meshgrid(freq, freq, indexing='ij')
    knrm = torch.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    tensormask = (knrm > (0.5 * 1/scale)) & (knrm < (0.5))
    
    # Apply mask
    mtf_masked = torch.masked_select(mtf, tensormask)
    
    return hq_reduce(mtf_masked, reduction)


def hq_convolve(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """ Estimate high-frequency information by substract a loss-pass filtered
        version of the LR image

    Args:
        image1 (torch.Tensor): The LR image.
        image2 (torch.Tensor): The SR image.
        reduction (str, optional): The reduction method to use for the MTF
            metric. Defaults to "mean". Other options include "sum", and "median".
        scale (int, optional): The scale of the super-resolution. Defaults to 4.
    Returns:
        int: The metric value. 
    """

    # Obtain the scale
    scale = image2.shape[-1] // image1.shape[-1]

    # Convert the LR to SRimage scale
    image1 = torch.nn.functional.interpolate(
        input=image1[None],
        size=image2.shape[-2:],
        mode='bilinear',
        antialias=True
    ).squeeze(0)


    # Create a LR reference image (down and up)
    image1_ref = torch.nn.functional.interpolate(
        input=image1[None],
        scale_factor=1/(scale*2),
        mode='bilinear',
        antialias=True
    )
    image1_ref = torch.nn.functional.interpolate(
        input=image1_ref,
        size=image2.shape[-2:],
        mode='bilinear',
        antialias=True
    ).squeeze(0)


    # Compute the difference
    diff1 = torch.abs(image1 - image1_ref)
    diff2 = torch.abs(image2 - image1_ref)

    # Group get the mean of each image patch (8x8)
    diff1_r = torch.nn.functional.avg_pool2d(diff1, kernel_size=scale*2, stride=8)    
    diff2_r = torch.nn.functional.avg_pool2d(diff2, kernel_size=scale*2, stride=8)
    
    return torch.abs(diff2_r - diff1_r)

def hf_metric(
    image1: torch.Tensor,
    image2: torch.Tensor,
    metric: str = 'mtf'
) -> torch.Tensor:
    """ Calculate the high-frequencies difference between two images.

    Args:
        image1 (torch.Tensor): The low-resolution image (C, H, W).
        image2 (torch.Tensor): The super-resolved or target image (C, H, W).
        metric (str, optional): The metric to use. Defaults to 'mtf'.
    
    Returns:
        torch.Tensor: The high-frequency difference between the two images.
    """        
    
    # Only get channel
    image1 = image1.mean(axis=0, keepdims=True)
    image2 = image2.mean(axis=0, keepdims=True)

    # if images are no square return error
    if image1.shape[-1] != image1.shape[-2]:
        raise ValueError("image1 must be square.")

    if image2.shape[-1] != image2.shape[-2]:
        raise ValueError("image2 must be square.")

    
    if metric == 'mtf':
        results = hq_mtf(image1, image2)
    elif metric == "convolve":
        results = hq_convolve(image1, image2)        
    else:
        raise ValueError("Metric not supported.")
    
    return results

if __name__ == "__main__":

    image1 = torch.rand(3, 128, 128)
    image2 = torch.rand(3, 512, 512)
    hf_metric(image1, image2, metric="convolve")
