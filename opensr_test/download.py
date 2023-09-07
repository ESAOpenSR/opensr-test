import pathlib

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


def load(dataset: str = "naip", force: bool = False) -> torch.Tensor:
    """ Load a dataset.

    Args:
        dataset (str, optional): The dataset to load. Defaults to "naip".
        force (bool, optional): If True, force the download. Defaults to False.
        
    Raises:
        NotImplementedError: If the dataset is not implemented.

    Returns:
        torch.Tensor: The dataset in a tensor of shape (N, C, H, W).
    """

    ROOT_FOLDER = get_credentials_path()
    DATASETS = ["naip", "spot", "venus"]
    URL = "https://huggingface.co/csaybar/opensr-test/resolve/main"
    HRFILES = ["HR_NAIP.npy", "HR_SPOT67.npy", "HR_VENUS.npy"]

    # Create folder
    [(ROOT_FOLDER / x).mkdir(exist_ok=True) for x in DATASETS]

    # Download the files
    for d in DATASETS:

        ## hr file
        hrfile_url = "%s/%s/%s" % (URL, d, HRFILES[DATASETS.index(d)])
        hrfile_path = ROOT_FOLDER / d / HRFILES[DATASETS.index(d)]
        if not hrfile_path.exists() or force:
            print(f"Downloading {d} dataset  - {hrfile_path.stem} file.")
            download(hrfile_url, hrfile_path)

        ## lr file
        lrfile_url = "%s/%s/LR_S2.npy" % (URL, d)
        lrfile_path = ROOT_FOLDER / d / "LR_S2.npy"
        if not lrfile_path.exists() or force:
            print(f"Downloading {d} dataset  - {lrfile_path.stem} file.")
            download(lrfile_url, lrfile_path)

        ## landuse file
        landuse_url = "%s/%s/LANDUSE.npy" % (URL, d)
        landuse_path = ROOT_FOLDER / d / "LANDUSE.npy"
        if not landuse_path.exists() or force:
            print(f"Downloading {d} dataset  - {landuse_path.stem} file.")
            download(landuse_url, landuse_path)

        ## metadata file
        csvfile_url = "%s/%s/metadata.csv" % (URL, d)
        csvfile_path = ROOT_FOLDER / d / "metadata.csv"
        if not csvfile_path.exists() or force:
            print(f"Downloading {d} dataset  - {csvfile_path.stem} file.")
            download(csvfile_url, csvfile_path)

    # Load the dataset

    ## LR file
    lr_data = np.load(ROOT_FOLDER / dataset / "LR_S2.npy")
    lr_data_torch = torch.from_numpy(lr_data).float()

    ## HR file
    hr_data = np.load(ROOT_FOLDER / dataset / HRFILES[DATASETS.index(dataset)])
    hr_data_torch = torch.from_numpy(hr_data).float()

    ## LandUse file
    land_use = np.load(ROOT_FOLDER / dataset / "LANDUSE.npy")
    land_use_torch = torch.from_numpy(land_use).float()

    return {"lr": lr_data_torch, "hr": hr_data_torch, "landuse": land_use_torch}


def get_credentials_path() -> str:
    cred_path = pathlib.Path.home() / ".config/opensr_test/"
    cred_path.mkdir(parents=True, exist_ok=True)
    return cred_path
