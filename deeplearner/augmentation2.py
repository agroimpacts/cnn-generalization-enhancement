import random
import numpy as np
import numpy.ma as ma
import cv2
from collections.abc import Sequence
from skimage import transform as trans
from scipy.ndimage import rotate


def flip(img, lbl, mask, ftype):
    '''
    Synthesize new image chips by flipping the input chip around a user-defined axis.

    Args:
        img (ndarray): Image array with dimensions of (H, W, C)
        lbl (ndarray): Annotation layer with dimensions of (H, W)
        mask (ndarray): Binary mask representing valid pixels in images and label, with dimensions of (H, W)
        ftype (str): Flip type from ['vflip', 'hflip', 'dflip']

    Returns:
        (ndarray, ndarray, ndarray) tuple of flipped image, label, and mask

    Note:
        Provided transformations are:
            1) 'vflip', vertical flip
            2) 'hflip', horizontal flip
            3) 'dflip', diagonal flip (both vertical and horizontal)
    '''

    # Create copies to avoid modifying the input arrays
    img_copy = img.copy()
    lbl_copy = lbl.copy()
    mask_copy = mask.copy()

    # Horizontal flip
    if ftype == 'hflip':
        img_copy = np.flip(img_copy, 0)
        lbl_copy = np.flip(lbl_copy, 0)
        mask_copy = np.flip(mask_copy, 0)

    # Vertical flip
    elif ftype == 'vflip':
        img_copy = np.flip(img_copy, 1)
        lbl_copy = np.flip(lbl_copy, 1)
        mask_copy = np.flip(mask_copy, 1)

    # Diagonal flip (both horizontal and vertical)
    elif ftype == 'dflip':
        img_copy = np.flip(img_copy, (0, 1))
        lbl_copy = np.flip(lbl_copy, (0, 1))
        mask_copy = np.flip(mask_copy, (0, 1))

    else:
        raise ValueError("Invalid flip type")

    return img_copy, lbl_copy, mask_copy


def center_rotate(img, lbl, mask, degree):
    """
    Synthesize new image chips by rotating the input chip around its center.

    Args:

    img (narray): Concatenated variables or brightness value with a dimension of (H, W, C)
    lbl (narray): Ground truth with a dimension of (H,W)
    mask (narray): Binary mask represents valid pixels in images and label, in a dimension of (H,W)
    degree (tuple or list): Range of degree for rotation

    Returns:

    (narray, narray, narray) tuple of rotated image, label and mask

    """
    if isinstance(degree, (tuple, list)):
        if len(degree) == 2:
            rotation_degree = random.uniform(*degree)
        elif len(degree) > 2:
            rotation_degree = random.choice(degree)
        else:
            raise ValueError("Parameter angle needs at least two elements.")
    else:
        raise ValueError(
            "Rotation bound param for augmentation must be a tuple or list."
        )

    img_copy = img.copy()
    lbl_copy = lbl.copy()
    mask_copy = mask.copy()
    
    # Get the dimensions of the image (e.g. number of rows and columns).
    h, w,_ = img_copy.shape

    # Determine the image center.
    center = (w // 2, h // 2)

    # Grab the rotation matrix
    rotMtrx = cv2.getRotationMatrix2D(center, rotation_degree, 1.0)

    # perform the actual rotation for both raw and labeled image.
    img_copy = cv2.warpAffine(img_copy, rotMtrx, (w, h))
    lbl_copy = cv2.warpAffine(lbl_copy, rotMtrx, (w, h))
    mask_copy = cv2.warpAffine(mask_copy, rotMtrx, (w, h))

    return img_copy, np.rint(lbl_copy).astype(lbl.dtype), np.rint(mask_copy).astype(mask.dtype)


def re_scale(img, lbl, mask, scale=(0.75, 1.5), crop_strategy="center"):
    r"""
    Synthesize a new pair of image, label, and mask chips by rescaling the input chips.

    Args:
        img (ndarray) -- Image chip with a dimension of (H,W,C).
        lbl (ndarray) -- Reference annotation layer with a dimension of (H,W).
        mask (ndarray) -- Binary mask with a dimension of (H,W).
        scale (tuple or list) -- A range of scale ratio.
        crop_strategy (str) -- decides whether to crop the rescaled image chip randomly
                               or at the center.

    Returns:
           Tuple[np.ndarray, np.ndarray, np.ndarray] including:
            resampled_img -- A numpy array of rescaled variables or brightness values in the
                             same size as the input chip.
            resampled_label -- A numpy array of rescaled ground truth in the same size as input.
            resampled_mask -- A numpy array of rescaled mask in the same size as input.
    """

    if not all(isinstance(arr, np.ndarray) for arr in [img, lbl, mask]):
        raise ValueError("img, lbl, and mask must be numpy arrays.")
    
    if img.ndim != 3 or lbl.ndim != 2 or mask.ndim != 2:
        raise ValueError(f"Expected image to have 3 dims got {img.ndim}, lbl to have 2 dims got {lbl.ndim}, and mask to have 2 dimes, got {mask.ndim}.")
    
    # channel should be the last dimension (H, W, C)
    if img.shape[0] == min(img.shape):
        img = img.transpose(1, 2, 0)

    h, w, c = img.shape

    if isinstance(scale, Sequence):
        resize_h = round(random.uniform(scale[0], scale[1]) * h)
        resize_w = resize_h
    else:
        raise Exception('Wrong scale type!')

    assert crop_strategy in ["center", "random"], "'crop_strategy' is not recognized."

    # Resampling
    resampled_img = trans.resize(img, (resize_h, resize_w), preserve_range=True)
    resampled_label = trans.resize(lbl, (resize_h, resize_w), order=0, preserve_range=True)
    resampled_mask = trans.resize(mask, (resize_h, resize_w), order=0, preserve_range=True)

    # Calculate cropping or filling offsets
    x_off, y_off = 0, 0
    if crop_strategy == "center":
        x_off = max(0, abs(resize_h - h) // 2)
        y_off = max(0, abs(resize_w - w) // 2)
    elif crop_strategy == "random":
        x_off = random.randint(0, max(0, abs(resize_h - h)))
        y_off = random.randint(0, max(0, abs(resize_w - w)))

    # Initialize canvas
    canvas_img = np.zeros((h, w, c), dtype=img.dtype)
    canvas_label = np.zeros((h, w), dtype=lbl.dtype)
    canvas_mask = np.zeros((h, w), dtype=mask.dtype)

    # Crop or fill the resampled arrays
    if resize_h >= h and resize_w >= w:
        canvas_img = resampled_img[x_off:x_off + h, y_off:y_off + w, :]
        canvas_label = resampled_label[x_off:x_off + h, y_off:y_off + w]
        canvas_mask = resampled_mask[x_off:x_off + h, y_off:y_off + w]
    else:
        canvas_img[x_off:x_off + resize_h, y_off:y_off + resize_w] = resampled_img
        canvas_label[x_off:x_off + resize_h, y_off:y_off + resize_w] = resampled_label
        canvas_mask[x_off:x_off + resize_h, y_off:y_off + resize_w] = resampled_mask

    return canvas_img, np.rint(canvas_label).astype(lbl.dtype), np.rint(canvas_mask).astype(mask.dtype)


def shift_brightness(img, gamma_range=(0.2, 2.0), shift_subset=[4], patch_shift=True):
    """
    Shift image brightness through gamma correction

    Args:
        img (narray): Concatenated variables or brightness value with a dimension of (H, W, C)
        gammaRange (tuple): Range of gamma values
        shiftSubset (tuple): Number of bands or channels for each shift
        patchShift (bool): Whether apply the shift on small patches

     Returns:
        narray, brightness shifted image
    """

    img_copy = img.copy()
    c_start = 0

    if patch_shift:
        # channel should be the last dimension (H, W, C)
        if img_copy.shape[0] == min(img_copy.shape):
            img_copy = img_copy.transpose(1, 2, 0)
        
        for i in shift_subset:
            gamma = random.triangular(gamma_range[0], gamma_range[1], 1)

            h, w, _ = img.shape
            rotMtrx = cv2.getRotationMatrix2D(center=(random.randint(0, h), random.randint(0, w)),
                                              angle=random.randint(0, 90),
                                              scale=random.uniform(1, 2))
            mask = cv2.warpAffine(img_copy[:, :, c_start:c_start + i], rotMtrx, (w, h))
            mask = np.where(mask, 0, 1)
            # apply mask
            img_ma = ma.masked_array(img_copy[:, :, c_start:c_start + i], mask=mask)
            img_copy[:, :, c_start:c_start + i] = ma.power(img_ma, gamma)
            # default extra step -- shift on image
            gamma_full = random.triangular(0.5, 1.5, 1)
            img_copy[:, :, c_start:c_start + i] = np.power(img_copy[:, :, c_start:c_start + i], gamma_full)

            c_start += i

    else:
        if min(img_copy.shape) != img_copy.shape[0]:
            img_copy = np.transpose(img_copy, list(range(img_copy.ndim)[-1:]) + list(range(img_copy.ndim)[:-1]))
        
        for i in shift_subset:
            gamma = random.triangular(gamma_range[0], gamma_range[1], 1)
            
            # if np.isnan(img_copy).any():
            #     print("NaN values are detected before gama corection.")
            
            # if np.isinf(img_copy).any():
            #     print("Inf values are detected before gama corection.")
            
            img_copy[c_start:c_start + i,:,:] = np.power(img_copy[c_start:c_start + i,:,: ], gamma)
            
            # if np.isnan(img_copy).any():
            #     print(f"NaN values are detected after gama corection with gamma {gamma}")

            c_start += i
        
        if min(img_copy.shape) != img_copy.shape[2]:
            img_copy = np.transpose(img_copy, list(range(img_copy.ndim)[-img_copy.ndim + 1:]) + [0])

    return img_copy


def br_manipulation(img, br_type, sigma_range=[0.03, 0.07], br_range=[-0.02, 0.02], contrast_range=[0.9, 1.2], 
                    gamma_range=[0.2, 2.0], shift_subset=[4], patch_shift=True):
    
    img_copy = img.copy()
    
    if br_type == "br_jitter":
        
        sigma = random.uniform(*sigma_range)
        img_copy += np.random.normal(loc=0., scale=sigma, size=img_copy.shape)
        
        min_threshold = 0.0001
        if img_copy.dtype in [np.float32, np.float64]:
            dtype_min = max(0, min_threshold)
            dtype_max = np.finfo(img_copy.dtype).max
        else:
            dtype_min = max(np.iinfo(img_copy.dtype).min, min_threshold)
            dtype_max = np.iinfo(img_copy.dtype).max
        
        img_copy = np.clip(img_copy, dtype_min, dtype_max)
    
    elif br_type == "br_gamma_corection":
        img_copy = shift_brightness(img, gamma_range=gamma_range, shift_subset=shift_subset, 
                                    patch_shift=patch_shift)
    
    elif br_type in ["br_additive", "br_contrast"]:
        dtype = np.float64 if img_copy.dtype in [np.float32, np.float64] else np.iinfo(img_copy.dtype)
        br_val = random.uniform(*br_range) if br_type == "br_additive" else random.uniform(*contrast_range)
        img_copy = np.clip(img_copy.astype(np.float64) * br_val, 0, 1).astype(dtype)

    else:
        raise ValueError("br_type")
    
    return img_copy


