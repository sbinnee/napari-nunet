import re
import os
from typing import List, Optional
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
from torchvision import transforms

from .transformer_net import TransformerNet
from .config import Config, SelfConfig

# Normalizing the axes within a dictionnary
# With : X = width , Y = length, Z = depth,  C = number of channels, T = time
axes_dict = {'X': 4, 'Y': 3, 'Z': 2, 'C': 1, 'T': 0, 'YX': [4, 3], 'XYZ': [4, 3, 2], 'ZYX': [
    2, 3, 4], 'YXC': [3, 4, 1], 'CYX': [1, 3, 4], 'TYX': [0, 3, 4], 'YXT': [3, 4, 0]}

# Normalizing image types
img_type_dict = {'XY': '2D GrayScale', 'ZYX': '3D GrayScale',
                 'YXC': '2D RGB', 'TYX': '2D Timed GrayScale'}


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


def detect_axes(img_data: np.ndarray):
    shape = img_data.shape
    ndim = img_data.ndim

    if ndim < 2:
        raise (ValueError('Axes undetected : Image dimensions must not be 0 nor 1'))
    if ndim == 2:
        axes_shape = 'YX'
    elif ndim == 3:
        third_dim = min(shape)
        if third_dim > 3:
            # Assuming that depth will usually be smaller than width and length
            depth_idx = shape.index(third_dim)
            if depth_idx == 0:
                axes_shape = 'ZYX'
            else:
                axes_shape = 'YXZ'
        elif third_dim == 3:
            chan_idx = shape.index(third_dim)
            if chan_idx == 0:
                axes_shape = 'CYX'
            else:
                axes_shape = 'YXC'
        else:
            raise (ValueError(
                'Image type undetected : Image format is wrong or not yet supported'))
    return(axes_shape)


# Normalize the image to TCZYX axis format before processing
def img_reshape_axes(img_data: np.ndarray, axes: str) -> np.ndarray:
    axes = list(axes)
    axes_id = []

    if (len(axes) < 2) or ('Y' not in axes) or ('X' not in axes):
        raise ValueError('An image must at least have Y and X axes')
    elif len(axes) != img_data.ndim:
        raise ValueError(
            'The axes format does not match the number of dimensions of the image')

    else:
        for axis in axes:
            # list with the values of axes from axes_dict (e.g : [4,3,1] for an XYC image)
            axes_id.append(axes_dict[axis])

        print(axes_id)

        normalized_image = img_data
        # Let's sort the axes by ascending order, and move the axes of the ndarray symmetrically
        # Insertion Sort

        for i in range(1, len(axes)):
            k = axes_id[i]
            j = i-1
            while j >= 0 and k < axes_id[j]:
                axes_id[j + 1] = axes_id[j]
                normalized_image = np.moveaxis(normalized_image, j, j+1)
                j -= 1
            axes_id[j + 1] = k

        # Now let's add an axis wherever it's missing compared to the TCZYX format

        # First add the T axis at the beginning
        if axes_id[0] != 0:
            normalized_image = np.expand_dims(normalized_image, 0)
            axes_id.insert(0, 0)

        while normalized_image.ndim != 5:
            for i in range(len(axes)-1):
                if axes_id[i+1]-axes_id[i] > 1:
                    normalized_image = np.expand_dims(normalized_image, i+1)
                    axes_id.insert(i+1, axes_id[i]+1)

    return normalized_image
    # OLD VERSIONS
    # def img_reshape_axes(img_data: np.ndarray, old_axes: str) -> np.ndarray:

    #     ndim = img_data.ndim
    #     old_axes = list(old_axes)
    #     old_axes_id = ()
    #     new_axes_id = ()

    #     if ndim != len(old_axes):
    #         raise (ValueError('Wrong axes specification : axes shape must fit image shape'))
    #     else:
    #         for axis in old_axes:
    #             old_axes_id += (axes_dict[axis],)
    #         if ndim < 2:
    #             raise (ValueError(
    #                 'Axes undetected : Image cannot have less than 2 dimensions'))
    #         if ndim == 2:
    #             img_output = img_data
    #             new_axes_id = old_axes_id
    #         elif ndim == 3:
    #             third_dim_idx = old_axes_id.index(max(old_axes_id))
    #             if third_dim_idx != 2:
    #                 # move it to last dimension
    #                 img_output = np.moveaxis(img_data, third_dim_idx, 2)
    #                 new_axes_id = (
    #                     old_axes_id[1-third_dim_idx], old_axes_id[2], old_axes_id[third_dim_idx])

    #     return(img_output, new_axes_id)


def load_checkpoints(cfg: Config) -> List[TransformerNet]:
    style_models = []
    for p in cfg._saved_model_path:
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
