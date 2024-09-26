from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
from opensr_test.utils import (check_huggingface_hub, check_lpips,
                               check_openclip)


class DistanceMetric(ABC):
    """An abstract class to compute the distance between two tensors.

    Parameters:
        method (str): The method to use. Either "pixel", "patch", or "image".
        patch_size (int): The patch size to use if the patch method is used.
        x (torch.Tensor): The SR harmonized image (C, H, W).
        y (torch.Tensor): The HR image (C, H, W).
        **kwargs: The parameters to pass to the distance function.

    Abstract methods:
        compute_patch: Compute the distance metric at patch level.
        compute_image: Compute the distance metric at image level.
        compute_pixel: Compute the distance metric at image level.
        compute: Compute the distance metric.
    """

    def __init__(
        self,
        method: str,
        patch_size: int,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs
    ):
        self.method = method
        self.patch_size = patch_size
        self.kwargs = kwargs
        self.axis: int = 0
        self.x = x
        self.y = y

    @staticmethod
    def do_square(tensor: torch.Tensor, patch_size: Optional[int] = 32) -> torch.Tensor:
        """Split a tensor into n_patches x n_patches patches and return
        the patches as a tensor.

        Args:
            tensor (torch.Tensor): The tensor to split.
            n_patches (int, optional): The number of patches to split the tensor into.
                If None, the tensor is split into the smallest number of patches.

        Returns:
            torch.Tensor: The patches as a tensor.
        """
        
        # Check if it is a square tensor
        if tensor.shape[-1] != tensor.shape[-2]:
            raise ValueError("The tensor must be square.")

        # Get the image size
        xdim = tensor.shape[1]
        ydim = tensor.shape[2]

        # Get the patch size
        minimages_x = int(torch.ceil(torch.tensor(xdim / patch_size)))
        minimages_y = int(torch.ceil(torch.tensor(ydim / patch_size)))

        # pad the tensor to be divisible by the patch size            
        pad_x_01 = int((minimages_x * patch_size - xdim) // 2)
        pad_x_02 = int((minimages_x * patch_size - xdim) - pad_x_01)

        pad_y_01 = int((minimages_y * patch_size - ydim) // 2)
        pad_y_02 = int((minimages_y * patch_size - ydim) - pad_y_01)

        padded_tensor = torch.nn.functional.pad(
            tensor, (pad_x_01, pad_x_02, pad_y_01, pad_y_02)
        )

        # split the tensor (C, H, W) into (n_patches, n_patches, C, H, W)
        patches = padded_tensor.unfold(1, patch_size, patch_size).unfold(
            2, patch_size, patch_size
        )

        # move the axes (C, n_patches, n_patches, H, W) -> (n_patches, n_patches, C, H, W)
        patches = patches.permute(1, 2, 0, 3, 4)

        return patches

    @abstractmethod
    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def compute_image(self) -> torch.Tensor:
        return self._compute_image(self.x, self.y)

    def compute_patch(self) -> torch.Tensor:
        # Create the patches
        x_batched = self.do_square(self.x, self.patch_size)
        y_batched = self.do_square(self.y, self.patch_size)
        
        # Compute the metric for each patch
        metric_result = torch.zeros(x_batched.shape[:2])
        xrange, yrange = x_batched.shape[0:2]
        for x_index in range(xrange):
            for y_index in range(yrange):
                x_batch = x_batched[x_index, y_index]
                y_batch = y_batched[x_index, y_index]
                metric_result[x_index, y_index] = self._compute_image(
                    x_batch, y_batch
                )

        # Go back to the original size
        metric_result = torch.nn.functional.interpolate(
            metric_result[None, None],
            size=self.x.shape[-2:],
            mode="nearest"
        ).squeeze()

        return metric_result

    def compute_pixel(self) -> torch.Tensor:
        return self._compute_pixel(self.x, self.y)

    def compute(self) -> torch.Tensor:
        if self.method == "pixel":
            return self.compute_pixel()
        elif self.method == "image":
            return self.compute_image()
        elif self.method == "patch":
            return self.compute_patch()
        else:
            raise ValueError("Invalid method.")


class KL(DistanceMetric):
    """Spectral information divergence between two tensors"""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32,
    ):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

        self.large_number = 1e2
        self.epsilon = 1e-8

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pbias_d = torch.abs(x / (y + self.epsilon))
        pbias_d[pbias_d > self.large_number] = self.large_number
        pbias_d[pbias_d < 1 / self.large_number] = 1 / self.large_number
        return torch.mean(x) * torch.log(torch.mean(pbias_d))

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pbias_d = torch.abs(x / (y + self.epsilon))
        pbias_d[pbias_d > self.large_number] = self.large_number
        pbias_d[pbias_d < 1 / self.large_number] = 1 / self.large_number
        return torch.mean(x, axis=0) * torch.log(torch.mean(pbias_d, axis=0))


class L1(DistanceMetric):
    """Spectral information divergence between two tensors"""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32,
    ):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean(torch.abs(x - y))

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean(torch.abs(x - y), axis=0)


class L2(DistanceMetric):
    """Spectral information divergence between two tensors"""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32,
    ):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean((x - y) ** 2)

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean((x - y) ** 2, axis=0)


class PBIAS(DistanceMetric):
    """Spectral information divergence between two tensors"""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32,
    ):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)
        
        self.large_number = 1e2
        self.epsilon = 1e-8
        
    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ratio = torch.abs((x - y) / (y + self.epsilon))
        ratio[ratio > self.large_number] = self.large_number
        ratio[ratio < 1 / self.large_number] = 1 / self.large_number
        return torch.nanmean(ratio)

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ratio = torch.abs((x - y) / (y + self.epsilon))
        ratio[ratio > self.large_number] = self.large_number
        ratio[ratio < 1 / self.large_number] = 1 / self.large_number
        return torch.nanmean(ratio, axis=0)


class IPSNR(DistanceMetric):
    """Inverse Peak signal to noise ratio between two tensors"""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32,
    ):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

        self.data_range = torch.tensor(1)
        self.epsilon = torch.tensor(1e-10)

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        l2_distance = torch.nanmean((x - y) ** 2)
        l2_distance[l2_distance < self.epsilon] = self.epsilon
        return 1 / (10 * torch.log10(self.data_range**2 / l2_distance))

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        l2_distance = torch.nanmean((x - y) ** 2, axis=0)
        l2_distance[l2_distance < self.epsilon] = self.epsilon
        return 1 / (10 * torch.log10(self.data_range**2 / l2_distance))


class SAD(DistanceMetric):
    """Spectral Angle distance between two tensors"""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32,
    ):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dot_product = (x * y).squeeze().sum()
        preds_norm = x.squeeze().norm()
        target_norm = y.squeeze().norm()
        sam_score = torch.clamp(dot_product / (preds_norm * target_norm), -1, 1).acos()
        return torch.rad2deg(sam_score)

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dot_product = (x * y).sum(dim=0)
        preds_norm = x.norm(dim=0)
        target_norm = y.norm(dim=0)
        sam_score = torch.clamp(dot_product / (preds_norm * target_norm), -1, 1).acos()
        return torch.rad2deg(sam_score)



class LPIPS(DistanceMetric):
    """Learned Perceptual Image Patch Similarity between two tensors"""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 16,
        device: Union[str, torch.device] = "cpu",
    ):
        if method == "patch":
            if patch_size < 16:
                raise ValueError("The patch size must be at least 16.")

        check_lpips()
        import lpips

        # Set the model
        self.model = lpips.LPIPS(net="alex", verbose=False).to(device)
        self.model.eval()

        # Normalize the tensors to [-1, 1]
        y = y * 2 - 1
        x = x * 2 - 1

        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

    @torch.no_grad()
    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Scale image is lower than 64
        if (x.shape[1] < 64) or (x.shape[2] < 64):
            x = torch.nn.functional.interpolate(
                x[None],
                size=(64, 64),
                mode="bilinear",
                antialias=True
            ).squeeze()
            
        
        if (y.shape[1] < 64) or (y.shape[2] < 64):
            y = torch.nn.functional.interpolate(
                y[None],
                size=(64, 64),
                mode="bilinear",
                antialias=True
            ).squeeze()

        with torch.no_grad():
            result = self.model(x, y).mean()

        return result

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("LPIPS cannot be computed at pixel level.")


class CLIP(DistanceMetric):
    """Estimate the CLIP score between two tensors"""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32,
        device: Union[str, torch.device] = "cpu",
    ):
        if method == "patch":
            if patch_size < 16:
                raise ValueError("The patch size must be at least 16.")
                    
        # if extra_requires (setup.py) is not fulfilled, raise error
        check_openclip()
        check_huggingface_hub()
        import open_clip
        from huggingface_hub import hf_hub_download

        # Set the model
        # Copy the pretrained model in the current directory
        checkpoint_path = hf_hub_download(
            "chendelong/RemoteCLIP",
            f"RemoteCLIP-RN50.pt",
            cache_dir='checkpoints'
        )
        ckpt = torch.load(checkpoint_path, map_location=device)
        model, _, preprocess = open_clip.create_model_and_transforms("RN50")
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()

        # desactivate the gradients
        for param in model.parameters():
            param.requires_grad = False

        # Scale the tensors values to [0, 1]
        x_norm = x.clamp(0, 1)
        y_norm = y.clamp(0, 1)
        self.model = model
        super().__init__(
            x=x_norm, y=y_norm, method=method, patch_size=patch_size
        )

    @torch.no_grad()
    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Scale image to be always Bx3x224x224
        if x.shape != (3, 224, 224):
            x = torch.nn.functional.interpolate(
                x[None],
                size=224,
                mode="bilinear",
                antialias=True
            ).squeeze()
            
        
        if y.shape != (3, 224, 224):
            y = torch.nn.functional.interpolate(
                y[None],
                size=224,
                mode="bilinear",
                antialias=True
            ).squeeze()
        
        # normalize
        means = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(x.device)
        stds = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(x.device)

        x = (x - means) / stds
        y = (y - means) / stds

        # Run the CLIP model
        with torch.no_grad():
            x_emb = self.model.encode_image(x[None]).squeeze()
            y_emb = self.model.encode_image(y[None]).squeeze()

        return torch.nn.functional.l1_loss(x_emb, y_emb)

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("CLIP cannot be computed at pixel level.")


class MTF(DistanceMetric):
    """Estimate the cross-modulation transfer function between two tensors"""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32,
        scale: int = 4,
    ):
        super().__init__(
            x=x, y=y, method=method, patch_size=patch_size
        )

        if method == "patch":
            if patch_size < 16:
                raise ValueError("The patch size must be at least 16.")

        self.scale = scale

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # do computation in cpu to avoid problems with fft
        x = x.cpu()
        y = y.cpu()

        # Compute mask
        freq = torch.fft.fftfreq(x.shape[-1])
        freq = torch.fft.fftshift(freq)
        kfreq2D = torch.meshgrid(freq, freq, indexing="ij")
        knrm = torch.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
        tensormask = (knrm > (0.5 * 1 / self.scale)) & (knrm < (0.5))

        fft_preds = torch.abs(torch.fft.fftshift(torch.fft.fft2(x)))
        fft_target = torch.abs(torch.fft.fftshift(torch.fft.fft2(y)))

        mtf = torch.masked_select((fft_target - fft_preds) / fft_target, tensormask)
        mtf = torch.abs(mtf)

        return torch.mean(torch.clamp(mtf, 0, 1))

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("MTF cannot be computed at pixel level.")


class FractionalDifference(DistanceMetric):
    """Estimate the fractional difference between two tensors"""
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32
    ):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)
        self.epsilon = 1e-8

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean(torch.abs(x - y) / torch.abs(x + y + self.epsilon))

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean(torch.abs(x - y) / torch.abs(x + y + self.epsilon), axis=0)

def get_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    method: str,
    agg_method: str,
    patch_size: int = 32,
    scale: int = 4,
    device: Union[str, torch.device] = "cpu",
    rgb_bands: Optional[List[int]] = [0, 1, 2],
):
    """Estimate the distance between two tensors. All the distances
    are normalized to be between 0 and n,  where n is the maximum


    Args:
        x (torch.Tensor): The SR harmonized image (C, H, W).
        hr (torch.Tensor): The HR image (C, H, W).
        method (str): The method to use. Either "psnr" or "cpsnr".
        agg_method (str): The method to use to aggregate the distance.
            Either "pixel", "image", or "patch".
        patch_size (int, optional): The patch size to use if the patch
            method is used.
        scale (int, optional): The scale of the super-resolution.
        space_search (int, optional): This parameter is used to search
            for the best shift that maximizes the PSNR. By default, it is
            the same as the super-resolution scale.

    Returns:
        torch.Tensor: The metric value.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("The number of channels in x and y must be the same.")

    if x.shape[1] != y.shape[1]:
        raise ValueError("The height of x and y must be the same.")

    if method == "kl":
        distance_fn = KL(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method == "l1":
        distance_fn = L1(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method == "l2":
        distance_fn = L2(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method == "pbias":
        distance_fn = PBIAS(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method == "psnr":
        distance_fn = IPSNR(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method == "sad":
        distance_fn = SAD(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method == "mtf":
        distance_fn = MTF(
            x=x, y=y, method=agg_method, patch_size=patch_size, scale=scale
        )
    elif (method == "fd") or (method == "nd"):
        distance_fn = FractionalDifference(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method == "lpips":
        x_rgb = x[rgb_bands, :, :]
        y_rgb = y[rgb_bands, :, :]
        distance_fn = LPIPS(
            x_rgb, y_rgb, method=agg_method, patch_size=patch_size, device=device
        )
    elif method == "clip":
        x_rgb = x[rgb_bands, :, :]
        y_rgb = y[rgb_bands, :, :]
        distance_fn = CLIP(
            x_rgb, y_rgb, method=agg_method, patch_size=patch_size, device=device
        )
    else:
        raise ValueError("No valid distance method.")

    return distance_fn.compute()