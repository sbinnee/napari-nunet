from magicgui import magic_factory, magicgui
import torch
import napari
from napari.types import ImageData
from napari.layers import Image
from magicgui.widgets import FunctionGui
import matplotlib.pyplot as plt
import numpy as np
from napari.qt.threading import thread_worker
import time
from pathlib import Path
from typing import List

from nunet.config import SelfConfig
from nunet.utils import load_model, numpy2torch, torch2numpy, img_reshape_axes, detect_axes
from nunet.transformer_net import TransformerNet

cfg_file = Path(
    "C:/Users/hp/Desktop/PRe/nunet/config/self_ultimate_vgg19_lr1e-4_e20_sw100.yml")

lob_logo_path = "C:/Users/hp/Desktop/PRe/napari-nunet/src/resources/Logo_LOB.png"


def grayscale_nunet(img: ImageData, model: TransformerNet):
    tensor = numpy2torch(img).cuda()
    out_tensor = model(tensor)
    out_tensor_clipped = torch.clip(out_tensor, 0, 255)
    out_np_clipped = torch2numpy(out_tensor_clipped)/255.0
    return out_np_clipped


def run_nu_net(img: ImageData, cfg: Path, axes: str):
    cfg = SelfConfig(cfg)
    nu_net = load_model(cfg)[2]
    img = img_reshape_axes(img, axes)  # output in TCZYX format
    shape = img.shape
    img_output = np.empty_like(img, dtype=np.float32)

    with torch.no_grad():
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    img_output[i, j, k, :, :] = grayscale_nunet(
                        img[i, j, k, :, :], nu_net)
    return img_output


def nunet_plugin_wrapper():
    return nunet_plugin


@magicgui(axes=dict(widget_type="LineEdit", label="Axes"), call_button="Run", image=dict(label="Image"), label_head=dict(widget_type="Label", label=f'<h1 align="left"><img src="{lob_logo_path}">NU-Net<h3>Generic segmentation for bioimages</h3></h1>'))
def nunet_plugin(viewer: napari.Viewer, label_head, image: Image, axes) -> ImageData:
    """Widget that applies NU-Net to an image 

    Parameters
    ----------
    img : Image
        Layer data to apply the model to

    Returns
    -------
    output : ImageData
        The transformed image layer
    """
    if image is not None:
        t0 = time.time()
        image_output = run_nu_net(image.data, cfg_file, axes)
        t1 = time.time()
        print(f'Executed in {(t1 - t0) / 60:.2f} minutes')
        return image_output


@nunet_plugin.image.changed.connect
def change_image(new_img: Image):
    nunet_plugin.image.value = new_img
    nunet_plugin.axes.value = detect_axes(nunet_plugin.image.value.data)


@nunet_plugin.axes.changed.connect
def change_axes(new_axes: str):
    if len(new_axes) == nunet_plugin.image.value.data.ndim:
        nunet_plugin.axes.value = new_axes
        print("Axes of the current layer image have been set to", new_axes)
