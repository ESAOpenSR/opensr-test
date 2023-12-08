import torch
from typing import List, Optional


def classic_upsampling(x: torch.Tensor, scale: int) -> torch.Tensor:
    """ Upsampling a tensor (B, C, H, W) to a lower resolution 
    (B, C, H', W') using bilinear interpolation with antialiasing.

    Args:
        x (torch.Tensor): The tensor to upsample.
        scale (int, optional): The super-resolution scale. Defaults 
            to 4.

    Returns:
        torch.Tensor: The upsampled tensor (B, C, H', W').
    """

    x_ref = torch.nn.functional.interpolate(
        input=x, scale_factor=1 / scale, mode="bilinear", antialias=True
    )

    return x_ref


def naip_upsampling(
        x: torch.Tensor,
        params: Optional[List[float]] = [0.5291, 0.4943, 0.5110, 0.4771],
        rgb_bands: Optional[List[int]] = [0, 1, 2]
) -> torch.Tensor:
    """ Upsampling a tensor (B, C, H, W) to a lower resolution 
    (B, C, H', W') using a gaussian blur and upsampling
    withouth antialiasing. The blur kernel is trained using
    the curated dataset opensr-test-spot.

    Args:
        x (torch.Tensor): The tensor to upsample.
        scale (int, optional): The super-resolution scale. Defaults 
            to 4.

    Returns:
        torch.Tensor: The upsampled tensor (B, C, H', W').
    """
    # get the nir band according to the rgb bands    
    if x.shape[0] == 4:
        nir_band_index = [i for i in range(4) if i not in rgb_bands][0]
        x = x[rgb_bands + [nir_band_index], ...]
    elif x.shape[0] == 3:
        x = x[rgb_bands, ...]
    else:
        raise ValueError("The input tensor must have 3 or 4 bands.")

    # Blur 5x5 kernel by band trained using the
    # curated dataset opensr-test-naip
    with torch.no_grad():
        blur_R = GaussianBlur(kernel_size=7, params=[params[0]])
        blur_G = GaussianBlur(kernel_size=7, params=[params[1]])
        blur_B = GaussianBlur(kernel_size=7, params=[params[2]])
        blur_N = GaussianBlur(kernel_size=7, params=[params[3]])

        # Apply the blur kernel to each band
        container = []
        for i in range(x.shape[0]):
            if i == 0:
                _x_blurred = blur_R(x[i][None, None, ...]).squeeze()
            elif i == 1:
                _x_blurred = blur_G(x[i][None, None, ...]).squeeze()
            elif i == 2:
                _x_blurred = blur_B(x[i][None, None, ...]).squeeze()
            elif i == 3:
                _x_blurred = blur_N(x[i][None, None, ...]).squeeze()
            container.append(_x_blurred)
        
    # Apply the blur kernel to each band
    x_blurred = torch.stack(container, dim=0)
    
    # Downsample using bilinear interpolation
    x_ref = torch.nn.functional.interpolate(
        input=x_blurred[None], scale_factor=0.25, mode="bilinear", antialias=False
    ).squeeze()

    return x_ref


def spot_upsampling(
    x: torch.Tensor,
    rgb_bands: Optional[List[int]] = [0, 1, 2],
    params: Optional[List[float]] = [0.5795, 0.6057, 0.6451, 0.6145]
) -> torch.Tensor:
    """ Upsampling a tensor (B, C, H, W) to a lower resolution 
    (B, C, H', W') using a gaussian blur and upsampling
    withouth antialiasing. The blur kernel is trained using
    the curated dataset opensr-test-spot.

    Args:
        x (torch.Tensor): The tensor to upsample.
        scale (int, optional): The super-resolution scale. Defaults 
            to 4.

    Returns:
        torch.Tensor: The upsampled tensor (B, C, H', W').
    """
    # get the nir band according to the rgb bands    
    if x.shape[0] == 4:
        nir_band_index = [i for i in range(4) if i not in rgb_bands][0]
        x = x[rgb_bands + [nir_band_index], ...]
    elif x.shape[0] == 3:
        x = x[rgb_bands, ...]
    else:
        raise ValueError("The input tensor must have 3 or 4 bands.")

    # Blur 5x5 kernel by band trained using the
    # curated dataset opensr-test-naip
    with torch.no_grad():
        blur_R = GaussianBlur(kernel_size=7, params=[params[0]])
        blur_G = GaussianBlur(kernel_size=7, params=[params[1]])
        blur_B = GaussianBlur(kernel_size=7, params=[params[2]])
        blur_N = GaussianBlur(kernel_size=7, params=[params[3]])

        # Apply the blur kernel to each band
        container = []
        for i in range(x.shape[0]):
            if i == 0:
                _x_blurred = blur_R(x[i][None, None, ...]).squeeze()
            elif i == 1:
                _x_blurred = blur_G(x[i][None, None, ...]).squeeze()
            elif i == 2:
                _x_blurred = blur_B(x[i][None, None, ...]).squeeze()
            elif i == 3:
                _x_blurred = blur_N(x[i][None, None, ...]).squeeze()
            container.append(_x_blurred)
        
    # Apply the blur kernel to each band
    x_blurred = torch.stack(container, dim=0)
    
    # Downsample using bilinear interpolation
    x_ref = torch.nn.functional.interpolate(
        input=x_blurred[None], scale_factor=0.25, mode="bilinear", antialias=False
    ).squeeze()

    return x_ref


def venus_upsampling(
    x: torch.Tensor,
    params: Optional[List[float]] = [0.5180, 0.5305, 0.5468, 0.5645],
    rgb_bands: Optional[List[int]] = [0, 1, 2]
) -> torch.Tensor:
    """ Upsampling a tensor (B, C, H, W) to a lower resolution
    (B, C, H', W') using a gaussian blur and upsampling
    withouth antialiasing. The blur kernel is trained using
    the curated dataset opensr-test-venus.

    Args:
        x (torch.Tensor): The tensor to upsample.
        scale (int, optional): The super-resolution scale. Defaults
            to 4.

    Returns:
        torch.Tensor: The upsampled tensor (B, C, H', W').
    """
    # get the nir band according to the rgb bands    
    if x.shape[0] == 4:
        nir_band_index = [i for i in range(4) if i not in rgb_bands][0]
        x = x[rgb_bands + [nir_band_index], ...]
    elif x.shape[0] == 3:
        x = x[rgb_bands, ...]
    else:
        raise ValueError("The input tensor must have 3 or 4 bands.")

    # Blur 5x5 kernel by band trained using the
    # curated dataset opensr-test-naip
    with torch.no_grad():
        blur_R = GaussianBlur(kernel_size=3, params=[params[0]])
        blur_G = GaussianBlur(kernel_size=3, params=[params[1]])
        blur_B = GaussianBlur(kernel_size=3, params=[params[2]])
        blur_N = GaussianBlur(kernel_size=3, params=[params[3]])

        # Apply the blur kernel to each band
        container = []
        for i in range(x.shape[0]):
            if i == 0:
                _x_blurred = blur_R(x[i][None, None, ...]).squeeze()
            elif i == 1:
                _x_blurred = blur_G(x[i][None, None, ...]).squeeze()
            elif i == 2:
                _x_blurred = blur_B(x[i][None, None, ...]).squeeze()
            elif i == 3:
                _x_blurred = blur_N(x[i][None, None, ...]).squeeze()
            container.append(_x_blurred)
        
    # Apply the blur kernel to each band
    x_blurred = torch.stack(container, dim=0)
    
    # Downsample using bilinear interpolation
    x_ref = torch.nn.functional.interpolate(
        input=x_blurred[None], scale_factor=0.5, mode="bilinear", antialias=False
    ).squeeze()

    return x_ref


def apply_upsampling(
    X: torch.Tensor, scale: int = 4, method: str = "classic"
) -> torch.Tensor:
    """ Apply a upsampling method to a tensor (B, C, H, W).

    Args:
        X (torch.Tensor): The tensor to downsample.
        scale (int, optional): The super-resolution scale. Defaults
            to 4.
        method (str, optional): The upsampling method. Must be one 
            of "classic", "naip", "spot", "venus". Defaults to "classic".

    Raises:
        ValueError: If the method is not valid.

    Returns:
        torch.Tensor: The upscaled tensor.
    """

    if method == "classic":
        return classic_upsampling(X, scale)
    elif method == "naip":
        return naip_upsampling(X)
    elif method == "spot":
        return spot_upsampling(X)
    elif method == "venus":
        return venus_upsampling(X)
    else:
        raise ValueError(
            "Invalid method. Must be one of 'classic', 'naip', 'spot', 'venus'."
        )


def classic_downsampling(x: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """ Downsampling a tensor (B, C, H, W) to a upper resolution 
    (B, C, H', W') using bilinear interpolation with antialiasing.

    Args:
        x (torch.Tensor): The tensor to downsampling.
        scale (int, optional): The super-resolution scale. Defaults 
            to 4.

    Returns:
        torch.Tensor: The downscaled tensor (B, C, H', W').
    """

    x_ref = torch.nn.functional.interpolate(
        input=x, scale_factor=scale, mode="bilinear", antialias=True
    )

    return x_ref


def apply_downsampling(
    X: torch.Tensor, scale: int = 4, method: str = "classic"
) -> torch.Tensor:
    """ Apply a downsampling method to a tensor (B, C, H, W).

    Args:
        X (torch.Tensor): The tensor to downsample.
        scale (int, optional): The super-resolution scale. Defaults
            to 4.
        method (str, optional): The downsampling method. Must be one 
            of ["classic"]. Defaults to "classic".

    Raises:
        ValueError: If the method is not valid.

    Returns:
        torch.Tensor: The downscaled tensor.
    """

    if method == "classic":
        return classic_downsampling(X, scale)
    else:
        raise ValueError("Invalid method. Must be one of ['classic']")



def gaussian(x, sigma: float= 2.5):
    """
    Returns a 1D Gaussian distribution tensor.

    Args:
        x (Tensor): input tensor.
        sigma (float, optional): standard deviation of the Gaussian distribution. Default is 2.5.

    Returns:
        Tensor: 1D Gaussian distribution tensor.
    """    
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return gauss / gauss.sum(-1, keepdim=True)


def gaussian2D(sigma: torch.Tensor, window_size: int = 33):
    """
    Returns a 2D Gaussian distribution tensor.

    Args:
        sigma (Tensor): standard deviation of the Gaussian distribution.
        window_size (int, optional): size of the window to apply the Gaussian
            distribution. Default is 33.
        batch_size (int, optional): size of the batch. Default is 1.

    Returns:
        Tensor: 2D Gaussian distribution tensor.
    """        
    # kernel 1D
    ky = gaussian(torch.linspace(-1, 1, window_size), sigma=sigma)
    kx = gaussian(torch.linspace(-1, 1, window_size), sigma=sigma)
    
    kernel = kx.unsqueeze(1) * ky.unsqueeze(0)
    
    return kernel

class GaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size=65, params=[2.5], device=torch.device("cpu")):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = torch.nn.Parameter(torch.tensor(params[0]))
        # if type is str
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

    def forward(self, x):
        kernel = gaussian2D(self.sigma, self.kernel_size).expand(x.shape[0], 1, self.kernel_size, self.kernel_size)
        kernel = kernel.to(self.device)
        return torch.nn.functional.conv2d(x, kernel, groups=x.shape[0], padding=self.kernel_size//2)
