from typing import Optional, List, Literal
from opensr_test.utils import get_data_path

import pathlib
import pickle
import numpy as np
import requests
import torch


def download(url: str, save_path: str) -> str:
    """ Download a file from a url.

    Args:
        url (str): The url of the file in HuggingFace Hub.
        save_path (str): The path to save the file.

    Returns:
        str: The path to the file.
    """
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return None

def load(
    dataset: str = "naip",
    model_dir: Optional[str] = None,
    version: Literal["v1", "v2"] = "v2",
    force: bool = False
) -> torch.Tensor:
    """ Load a dataset.

    Args:
        dataset (str, optional): The dataset to load. Defaults to "naip".
        force (bool, optional): If True, force the download. Defaults to False.
        
    Raises:
        NotImplementedError: If the dataset is not implemented.

    Returns:
        torch.Tensor: The dataset in a tensor of shape (N, C, H, W).
    """
    if version == "v1":
        version_key = "020"
    elif version == "v2":
        version_key = "100"

    if model_dir is None:
        ROOT_FOLDER = get_data_path()
    else:
        ROOT_FOLDER = pathlib.Path(model_dir)

    DATASETS = [
        "naip", "spot", "venus", "spain_urban",
        "spain_crops", "satellogic"
    ]
    if dataset not in DATASETS:        
        raise NotImplementedError("The dataset %s is not implemented." % dataset)

    URL = "https://huggingface.co/datasets/isp-uv-es/opensr-test/resolve/main"
    
    ## download the dataset
    dataset_file = "%s/%s/%s/%s.pkl" % (URL, version_key, dataset, dataset)
    hrfile_path = ROOT_FOLDER  / f"{dataset}.pkl"
    if not hrfile_path.exists() or force:
        print(f"Downloading {dataset} dataset  - {hrfile_path.stem} file.")
        download(dataset_file, hrfile_path)


    ## Load the dataset
    
    with open(hrfile_path, "rb") as f:
        dataset = pickle.load(f)
        
    return dataset