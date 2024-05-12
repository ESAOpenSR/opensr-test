import opensr_test
import satalign
import pydantic
import torch
import numpy as np

from typing import List, Union, Any

class SpatialMetricAlign(pydantic.BaseModel):
    """ Get the spatial error between two images
    and the aligned image.

    Args:
        method (str): The alignment method to use. One of 
            "ecc", "pcc", or "light
        max_translations (int): The maximum number of translations
            to search for.
        max_num_keypoints (int): The maximum number of keypoints
            to use for the lightglue method.
        device (Union[str, Any]): The device to use for the lightglue
            method. One of "cpu", "cuda", or a torch device.
    """

    method: str
    max_translations: int
    max_num_keypoints: int
    device: Union[str, Any]

    def get_metric(self, x, y):
        # torch.Tensor to numpy
        x = x.detach().cpu().numpy()[None]
        y = y.detach().cpu().numpy()


        if self.method == "ecc":
            align_model = satalign.ECC(
                datacube=x,
                reference=y,
                max_translations=self.max_translations
            )
        elif self.method == "pcc":
            align_model = satalign.PCC(
                datacube=x,
                reference=y,
                max_translations=self.max_translations
            )
        elif self.method == "lgm":
            align_model = satalign.LGM(
                datacube=x,
                reference=y,
                max_translations=self.max_translations,
                max_num_keypoints=self.max_num_keypoints,
                device=self.device            
            )
        else:
            raise ValueError("Invalid method")
        
        # Run the alignment model
        image_fixed, warp = align_model.run()
        image_fixed_to_torch = torch.from_numpy(image_fixed).to(self.device)


        # Get the spatial error from the affine matrix
        spatial_error = np.sqrt(warp[0][0, 2]**2 + warp[0][1, 2]**2)

        return image_fixed_to_torch[0], torch.tensor(spatial_error).type(torch.float32)
    
    @pydantic.field_validator("device")
    def check_device(cls, value):
        return torch.device(value)
