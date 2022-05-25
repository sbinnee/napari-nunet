import numpy as np

# Normalizing the axes within a dictionnary
# With : X = width , Y = length, Z = depth,  C = number of channels, T = time
axes_dict = {'X': 4, 'Y': 3, 'Z': 2, 'C': 1, 'T': 0, 'YX': [4, 3], 'XYZ': [4, 3, 2], 'ZYX': [
    2, 3, 4], 'YXC': [3, 4, 1], 'CYX': [1, 3, 4], 'TYX': [0, 3, 4], 'YXT': [3, 4, 0]}

# Normalizing image types
img_type_dict = {'XY': '2D GrayScale', 'ZYX': '3D GrayScale',
                 'YXC': '2D RGB', 'TYX': '2D Timed GrayScale'}


def detect_axes(img_data: np.ndarray):
    shape = img_data.shape
    ndim = img_data.ndim

    if ndim < 2 or ndim > 5 or ndim is None:
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
        elif third_dim == 3 or third_dim == 2:
            chan_idx = shape.index(third_dim)
            if chan_idx == 0:
                axes_shape = 'CYX'
            else:
                axes_shape = 'YXC'
        else:
            raise (ValueError(
                'Image type undetected : Image format is wrong or not yet supported'))
    elif ndim == 4:
        min_dim = min(shape)
        if min_dim == 2 or min_dim == 3:  # We guess that a 2-deep or 3-deep dimension is probably the number of channels
            chan_idx = shape.index(min_dim)
            if chan_idx == 1:
                axes_shape = 'TCYX'
            elif chan_idx == 3:
                axes_shape = 'ZYXC'
            else:
                axes_shape = 'CZYX'
        else:
            axes_shape = 'TZYX'
    elif ndim == 5:
        axes_shape = 'TCZYX'
    return axes_shape


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

        normalized_image = img_data
        # Let's sort the axes by ascending order, and move the axes of the ndarray symmetrically
        # Insertion Sort

        for i in range(1, len(axes_id)):
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
            for i in range(len(axes_id)-1):
                if axes_id[i+1]-axes_id[i] > 1:
                    normalized_image = np.expand_dims(normalized_image, i+1)
                    axes_id.insert(i+1, axes_id[i]+1)

    return normalized_image


# Give back the image it's original shape after processing
def img_postprocess_reshape(img_data: np.ndarray, old_axes: str):
    old_axes = [axes_dict[axis] for axis in list(old_axes)]
    print("old_axes =", old_axes)
    current_axes = []
    swap_axes = [0]*len(old_axes)

    for i in range(len(img_data.shape)):
        if img_data.shape[i] != 1:
            current_axes.append(i)
    print("current_axes = ", current_axes)

    for i in current_axes:
        swap_axes[old_axes.index(i)] = current_axes.index(i)

    print(swap_axes)

    img_output = np.transpose(np.squeeze(img_data), axes=swap_axes)

    return img_output

# Check if


def check_input_axes(new_axes: str, img_data: np.ndarray):

    new_axes = new_axes.replace(" ", "").upper()
    for axis in list(new_axes):
        if axis not in ['T', 'C', 'Z', 'Y', 'X'] or new_axes.count(axis) > 1:
            return new_axes, False
    return new_axes, len(new_axes) == img_data.ndim
