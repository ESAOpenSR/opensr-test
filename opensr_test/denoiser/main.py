from argparse import Namespace

import pkg_resources
import rasterio as rio
import torch
from opensr_test.denoiser.utils import (NoisyEstimator, RealNoisyEstimator,
                                        denoiser)


def setup_denoiser_model(device=torch.device("cpu")):

    # get the package directory
    weights01 = pkg_resources.resource_filename(
        "opensr_test", "denoiser/weights/NoisyEstimator.pth"
    )
    weights02 = pkg_resources.resource_filename(
        "opensr_test", "denoiser/weights/RealNoisyEstimator.pth"
    )

    # Load Noisy Calibration Model
    model = NoisyEstimator(channels=3, num_of_layers=20, num_of_est=6)
    weights = torch.load(weights01)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)

    model_real = RealNoisyEstimator(input_channels=3, output_channels=6)
    weights = {k.replace("module.", ""): v for k, v in torch.load(weights02).items()}
    model_real.load_state_dict(weights)
    model_real.eval()
    model_real.to(device)

    return {"noisy_estimator": model, "real_noisy_estimator": model_real}


def remove_noise(img, models):
    # all the parameters
    parameters = {
        "color": 1,
        "cond": 1,
        "ext_test_noise_level": None,
        "k": 0,
        "keep_ind": [0, 1, 2],
        "mode": "MC",
        "num_of_layers": 20,
        "output_map": 0,
        "ps": 2,
        "ps_scale": 2,
        "real_n": 1,
        "refine": 0,
        "refine_opt": 1,
        "rescale": 1,
        "scale": 1,
        "spat_n": 0,
        "test_noise_level": None,
        "wbin": 64,
        "zeroout": 0,
    }

    # from dict to namespace
    opt = Namespace(**parameters)
    pss = opt.ps_scale
    model, model_est = models.values()

    return denoiser(img, 3, pss, model, model_est, opt) / 255
