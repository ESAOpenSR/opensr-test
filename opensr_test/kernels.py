import torch

def classic_upsampling(
    x: torch.Tensor,
    scale: int = 4
) -> torch.Tensor:
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
        input=x,
        scale_factor=1/scale,
        mode='bilinear',
        antialias=True
    )
        
    return x_ref
    

def naip_upsampling(
    x: torch.Tensor,
    scale: int = 4
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
    # Blur 5x5 kernel by band trained using the 
    # curated dataset opensr-test-naip
    blur_R = torch.tensor([
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0001, 0.0000, 0.0000],
        [0.0000, 0.0001, 0.0002, 0.0001, 0.0000],
        [0.0000, 0.0000, 0.0001, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    ])
    
    blur_G = torch.tensor([
        [0.0000, 0.0000, 0.0001, 0.0000, 0.0000],
        [0.0000, 0.0002, 0.0006, 0.0002, 0.0000],
        [0.0001, 0.0006, 0.0010, 0.0006, 0.0001],
        [0.0000, 0.0002, 0.0006, 0.0002, 0.0000],
        [0.0000, 0.0000, 0.0001, 0.0000, 0.0000]
    ])
    
    blur_B = torch.tensor([
        [0.0000, 0.0001, 0.0002, 0.0001, 0.0000],
        [0.0001, 0.0006, 0.0010, 0.0006, 0.0001],
        [0.0002, 0.0010, 0.0016, 0.0010, 0.0002],
        [0.0001, 0.0006, 0.0010, 0.0006, 0.0001],
        [0.0000, 0.0001, 0.0002, 0.0001, 0.0000]
    ])
    
    blur_N = torch.tensor([
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0002, 0.0006, 0.0002, 0.0000],
        [0.0000, 0.0006, 0.0010, 0.0006, 0.0000],
        [0.0000, 0.0002, 0.0006, 0.0002, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    ])
    
    blur_kernel = torch.stack([
        blur_R, blur_G, blur_B, blur_N
    ]).to(x.device)
    
        
    # Apply the blur kernel to each band
    x_blurred = torch.nn.functional.conv2d(
        input=x,
        weight=blur_kernel[:, None, ...],
        padding='same',
        groups=4
    )
    
    # Downsample using bilinear interpolation
    x_ref = torch.nn.functional.interpolate(
        input=x_blurred,
        scale_factor=1/scale,
        mode='bilinear',
        antialias=False
    )
        
    return x_ref



def spot_upsampling(
    x: torch.Tensor,
    scale: int = 4
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
    # Blur 5x5 kernel by band trained using the 
    # curated dataset opensr-test-naip
    blur_R = torch.tensor([
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0001, 0.0000, 0.0000],
        [0.0000, 0.0001, 0.0002, 0.0001, 0.0000],
        [0.0000, 0.0000, 0.0001, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    ])
    
    blur_G = torch.tensor([
        [0.0000, 0.0000, 0.0001, 0.0000, 0.0000],
        [0.0000, 0.0002, 0.0006, 0.0002, 0.0000],
        [0.0001, 0.0006, 0.0010, 0.0006, 0.0001],
        [0.0000, 0.0002, 0.0006, 0.0002, 0.0000],
        [0.0000, 0.0000, 0.0001, 0.0000, 0.0000]
    ])
    
    blur_B = torch.tensor([
        [0.0000, 0.0001, 0.0002, 0.0001, 0.0000],
        [0.0001, 0.0006, 0.0010, 0.0006, 0.0001],
        [0.0002, 0.0010, 0.0016, 0.0010, 0.0002],
        [0.0001, 0.0006, 0.0010, 0.0006, 0.0001],
        [0.0000, 0.0001, 0.0002, 0.0001, 0.0000]
    ])
    
    blur_N = torch.tensor([
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0002, 0.0006, 0.0002, 0.0000],
        [0.0000, 0.0006, 0.0010, 0.0006, 0.0000],
        [0.0000, 0.0002, 0.0006, 0.0002, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    ])
    
    blur_kernel = torch.stack([
        blur_R, blur_G, blur_B, blur_N
    ]).to(x.device)
    
        
    # Apply the blur kernel to each band
    x_blurred = torch.nn.functional.conv2d(
        input=x,
        weight=blur_kernel[:, None, ...],
        padding='same',
        groups=4
    )
    
    # Downsample using bilinear interpolation
    x_ref = torch.nn.functional.interpolate(
        input=x_blurred,
        scale_factor=1/scale,
        mode='bilinear',
        antialias=False
    )
        
    return x_ref


def venus_upsampling(
    x: torch.Tensor,
    scale: int = 4
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
    # Blur 5x5 kernel by band trained using the 
    # curated dataset opensr-test-naip
    blur_R = torch.tensor([
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0001, 0.0000, 0.0000],
        [0.0000, 0.0001, 0.0002, 0.0001, 0.0000],
        [0.0000, 0.0000, 0.0001, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    ])
    
    blur_G = torch.tensor([
        [0.0000, 0.0000, 0.0001, 0.0000, 0.0000],
        [0.0000, 0.0002, 0.0006, 0.0002, 0.0000],
        [0.0001, 0.0006, 0.0010, 0.0006, 0.0001],
        [0.0000, 0.0002, 0.0006, 0.0002, 0.0000],
        [0.0000, 0.0000, 0.0001, 0.0000, 0.0000]
    ])
    
    blur_B = torch.tensor([
        [0.0000, 0.0001, 0.0002, 0.0001, 0.0000],
        [0.0001, 0.0006, 0.0010, 0.0006, 0.0001],
        [0.0002, 0.0010, 0.0016, 0.0010, 0.0002],
        [0.0001, 0.0006, 0.0010, 0.0006, 0.0001],
        [0.0000, 0.0001, 0.0002, 0.0001, 0.0000]
    ])
    
    blur_N = torch.tensor([
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0002, 0.0006, 0.0002, 0.0000],
        [0.0000, 0.0006, 0.0010, 0.0006, 0.0000],
        [0.0000, 0.0002, 0.0006, 0.0002, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    ])
    
    blur_kernel = torch.stack([
        blur_R, blur_G, blur_B, blur_N
    ]).to(x.device)
    
        
    # Apply the blur kernel to each band
    x_blurred = torch.nn.functional.conv2d(
        input=x,
        weight=blur_kernel[:, None, ...],
        padding='same',
        groups=4
    )
    
    # Downsample using bilinear interpolation
    x_ref = torch.nn.functional.interpolate(
        input=x_blurred,
        scale_factor=1/scale,
        mode='bilinear',
        antialias=False
    )
    
    return x_ref


def apply_upsampling(
    X: torch.Tensor,
    scale: int = 4,
    method: str = "classic"
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
        return naip_upsampling(X, scale)
    elif method == "spot":
        return spot_upsampling(X, scale)
    elif method == "venus":
        return venus_upsampling(X, scale)
    else:
        raise ValueError(
            "Invalid method. Must be one of 'classic', 'naip', 'spot', 'venus'."
        )


def classic_downsampling(
    x: torch.Tensor,
    scale: int = 4
) -> torch.Tensor:
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
        input=x,
        scale_factor=scale,
        mode='bilinear',
        antialias=True
    )

    return x_ref


def apply_downsampling(
    X: torch.Tensor,
    scale: int = 4,
    method: str = "classic"
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
        raise ValueError(
            "Invalid method. Must be one of ['classic']"
        )