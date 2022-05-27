import re
import os
import bisect
from typing import List, Optional
from pathlib import Path, PureWindowsPath
from xml.dom import NotFoundErr

import numpy as np
import torch
from torchvision import transforms

from nunet.transformer_net import TransformerNet
from nunet.config import Config, SelfConfig

models_folder = Path("C:/Users/hp/Desktop/PRe/nunet/models_filter_slider")


def to_legal_ospath(path: str) -> str:
    """Detect the OS and replace any eventual illegal path character with "_".
    This will allow file transfer without filename check from Linux to Windows.

    Parameters
    ----------
    path: str
        The input path

    Returns
    -------
    legal_path: str
        A path with no illegal windows character but in posix path format
    """
    if os.name == "nt":  # If running on Windows
        # Replace all illegal characters with underscore
        legal_path = re.sub(r"\*|:|<|>|\?|\|", '_', path)
        # If the original path was absolute, we have to put back the ":" after the root directory
        if Path(path).is_absolute():
            legal_path = legal_path.replace("_", ":", 1)
        # Turn the Windows path into a Posix path if it isn't already
        legal_path = PureWindowsPath(legal_path).as_posix()
    elif os.name == "posix":  # If running Linux or MacOS
        legal_path = PureWindowsPath(legal_path).as_posix()
    return(legal_path)


def load_checkpoints(cfg: Config) -> List[TransformerNet]:
    style_models = []
    for p in cfg._saved_model_path:
        p = str(models_folder.parent / p)
        model = torch.load(to_legal_ospath(p))
        state_dict = model['state_dict']
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model = TransformerNet()
        style_model.load_state_dict(state_dict)
        style_models.append(style_model)
    return style_models


def load_model(
    cfg: SelfConfig,
    ind: int = -1
):
    print('len(cfg._saved_model_path):', len(cfg._saved_model_path))
    models = load_checkpoints(cfg)

    common_path = Path(cfg._saved_model_path[0]).parent.stem
    print('common_path:', {common_path})
    models_names = [Path(p).stem for p in cfg._saved_model_path]

    for i, name in enumerate(models_names):
        print(f'{i:3d}, {name}')

    model_name = models_names[ind]
    print(f'Loading {ind}: {model_name}')
    nu_net = models[ind]
    nu_net.cuda()
    nu_net.eval()
    return common_path, model_name, nu_net


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
