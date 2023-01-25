import bisect
import os
from pathlib import Path
from typing import List, Optional
from xml.dom import NotFoundErr

import numpy as np
import torch
from torchvision import transforms

# import re
# import zipfile


def numpy2torch(
    array: np.ndarray,
    device: Optional[str] = None,
    cuda: Optional[bool] = None,
) -> torch.Tensor:
    """Cast an image array to a tensor, ready to be consumed by NU-Net

    Parameters
    ----------
    array : numpy.ndarray
        A numpy array
    device : Optional
        If given, cast the tensor to the specified device. Argument to
        ``torch.Tensor.to()``
    cuda : Optional
        Copy the tensor from CPU to GPU. Ignored if `device` is given.

    Returns
    -------
    torch.Tensor
        Data range is assumed to be UINT8 but the actual dtype will be FLOAT32
        for direct computation.

    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    array = np.asarray(array, dtype=np.float32)
    tensor = transform(array)
    tensor = tensor.unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    elif cuda is not None:
        tensor = tensor.cuda()
    return tensor


def torch2numpy(
    tensor: torch.Tensor,
) -> np.ndarray:
    """Cast pytorch tensor(s) to numpy array(s)

    """
    # (1,c,y,x)
    ch = tensor.size(1)
    tensor = tensor.detach().cpu().squeeze()
    array = tensor.permute(1, 2, 0).numpy() if ch == 3 else tensor.numpy()
    return array

# Renvoie la liste des magnitudes disponibles


def make_sw_list(cfg_folder: Path) -> list:
    sw_list = []
    for cfg_file in os.listdir(cfg_folder):
        sw_list.append(int(Path(cfg_file).stem.split('_sw')[1])/10)
    sw_list.sort()
    return sw_list


def find_sw_cfg(sw: float, cfg_folder: Path):
    pattern = "_sw" + str(int(sw*10))
    for cfg_file in os.listdir(cfg_folder):
        if pattern in cfg_file:
            return cfg_file
    raise NotFoundErr(
        f'No config file was found with this magnitude value : {int(sw*10)}')

# Renvoie si la prédiction nécessite une somme pondérée ou non et les magnitudes des
# deux modèles ainsi que leur poids respectifs dans la somme.


def load_weights(magnitude: float, sw_list: list):
    weighted = True
    sw1 = magnitude
    sw2 = magnitude
    weight1 = 1
    weight2 = 0
    if magnitude in sw_list:
        weighted = False
    else:
        bisect.insort(sw_list, magnitude)
        idx = sw_list.index(magnitude)
        if idx != 0:
            sw1 = sw_list[idx-1]
        else:
            sw1 = None
        if idx != len(sw_list)-1:
            sw2 = sw_list[idx+1]
        else:
            sw2 = None
        sw_list.remove(magnitude)

        if sw1 is not None:
            if sw2 is not None:
                weight1 = abs(magnitude-sw1)/abs(sw2-sw1)
                weight2 = abs(magnitude-sw2)/abs(sw2-sw1)
        else:
            if sw2 is not None:
                weight1 = 0
                weight2 = 1

    return weighted, sw1, sw2, weight1, weight2
