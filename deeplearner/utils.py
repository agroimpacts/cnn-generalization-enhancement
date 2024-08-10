import numpy as np
import pandas as pd
import os
import random
import itertools
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import logging
import pickle
from datetime import datetime

import torch

from .tools import InputError


def load_data(dataPath, clip_val=0, usage="train", window=None, norm_stats_type=None, 
              global_stats=None, nodata_val_ls=None, isLabel=False):
    '''
    Read geographic data into numpy array
    Params:
        dataPath (str): Path of data to load
        usage (str): Usage of the data: "train", "validate", or "predict"
        window (tuple): The view onto a rectangular subset of the data, in the format of
            (column offsets, row offsets, width in pixel, height in pixel)
        norm_stats_type (str): How the normalization statistics is calculated.
        isLabel (binary): Decide whether to saturate data with tested threshold
    Returns:
        narray
    '''

    with rasterio.open(dataPath, "r") as src:

        if isLabel:
            if src.count != 1:
                raise InputError("Label shape not applicable: expected 1 channel")
            img = src.read(1)
        else:
            img_nodata = src.nodata
            nodata_val_ls = list(set(nodata_val_ls + [img_nodata])) if nodata_val_ls else [img_nodata]
            
            assert norm_stats_type in ["mm_local_per_tile", "zv_local_per_tile", "mm_local_per_band", 
                                       "zv_local_per_band", "mm_global_per_band", "zv_global_per_band", 
                                       "mm_global_per_tile", "zv_global_per_tile"]
            
            if norm_stats_type == "mm_local_per_tile":
                img = mmNorm_lab(src.read(), nodata=nodata_val_ls, clip_val=clip_val)
            elif norm_stats_type == "mm_local_per_band":
                img = mmNorm_lpb(src.read(), nodata=nodata_val_ls, clip_val=clip_val) #clip_val=1.5
            elif norm_stats_type == "mm_global_per_band":
                img = mmNorm_gpb(src.read(), nodata=nodata_val_ls, global_stats=global_stats, clip_val=clip_val) #clip_val=1.5
            elif norm_stats_type == "mm_global_per_tile":
                img = mmNorm_gab(src.read(), nodata=nodata_val_ls, global_stats=global_stats, clip_val=clip_val) #clip_val=1.5
            elif norm_stats_type == "zv_local_per_tile":
                img = zvNorm_lab(src.read(), nodata=nodata_val_ls, clip_val=clip_val)
            elif norm_stats_type == "zv_local_per_band":
                img = zvNorm_lpb(src.read(), nodata=nodata_val_ls, clip_val=clip_val) #clip_val=1.5
            elif norm_stats_type == "zv_global_per_band":
                img = zvNorm_gpb(src.read(), nodata=nodata_val_ls, global_stats=global_stats, clip_val=clip_val) #clip_val=1.5
            elif norm_stats_type == "zv_global_per_tile":
                img = zvNorm_gab(src.read(), nodata=nodata_val_ls, global_stats=global_stats, clip_val=clip_val) #clip_val=1.5

            if usage in ['train', 'validate']:
               img = img[:, max(0, window[1]): window[1] + window[3], max(0, window[0]): window[0] + window[2]]

    return img


def get_stacked_img(imgPaths, usage, clip_val=0, norm_stats_type="local_per_tile", 
                    global_stats=None, nodata_val_ls=None, window=None):
    '''
    Read geographic data into numpy array
    Params:
        gsPath (str): Path of growing season image
        osPath (str): Path of off season image
        imgPaths (list): List of paths for imgages
        usage (str): Usage of the image: "train", "validate", or "predict"
        norm_stats_type (str): How the normalization statistics is calculated.
        window (tuple): The view onto a rectangular subset of the data, in the format of
            (column offsets, row offsets, width in pixel, height in pixel)
    Returns:
        narray
    '''

    if len(imgPaths) > 1:
        img_ls = [load_data(m, clip_val, usage, window, norm_stats_type, global_stats, nodata_val_ls) for m in imgPaths]
        img = np.concatenate(img_ls, axis=0).transpose(1, 2, 0)
    else:
        img = load_data(imgPaths[0], clip_val, usage, window, norm_stats_type, global_stats, nodata_val_ls).transpose(1, 2, 0)

    if usage in ["train", "validate"]:
        col_off, row_off, col_target, row_target = window
        row, col, c = img.shape

        if row < row_target or col < col_target:

            row_off = abs(row_off) if row_off < 0 else 0
            col_off = abs(col_off) if col_off < 0 else 0

            canvas = np.zeros((row_target, col_target, c))
            canvas[row_off: row_off + row, col_off : col_off + col, :] = img

            return canvas

        else:
            return img

    elif usage == "predict":
        return img

    else:
        raise ValueError


def get_buffered_window(srcPath, dstPath, buffer):
    '''
    Get bounding box representing subset of source image that overlaps with bufferred destination image, in format
    of (column offsets, row offsets, width, height)

    Params:
        srcPath (str): Path of source image to get subset bounding box
        dstPath (str): Path of destination image as a reference to define the bounding box. Size of the bounding box is
            (destination width + buffer * 2, destination height + buffer * 2)
        buffer (int): Buffer distance of bounding box edges to destination image measured by pixel numbers

    Returns:
        tuple in form of (column offsets, row offsets, width, height)
    '''

    with rasterio.open(srcPath, "r") as src:
        gt_src = src.transform

    with rasterio.open(dstPath, "r") as dst:
        gt_dst = dst.transform
        w_dst = dst.width
        h_dst = dst.height

    col_off = round((gt_dst[2] - gt_src[2]) / gt_src[0]) - buffer
    row_off = round((gt_dst[5] - gt_src[5]) / gt_src[4]) - buffer
    width = w_dst + buffer * 2
    height = h_dst + buffer * 2

    return col_off, row_off, width, height


def get_meta_from_bounds(file, buffer):
    '''
    Get metadata of unbuffered region in given file
    Params:
        file (str):  File name of a image chip
        buffer (int): Buffer distance measured by pixel numbers
    Returns:
        dictionary
    '''

    with rasterio.open(file, "r") as src:

        meta = src.meta
        dst_width = src.width - 2 * buffer
        dst_height = src.height - 2 * buffer

        window = Window(buffer, buffer, dst_width, dst_height)
        win_transform = src.window_transform(window)

    meta.update({
        'width': dst_width,
        'height': dst_height,
        'transform': win_transform,
        'count': 1,
        'nodata': -128,
        'dtype': 'int8'
    })

    return meta



def displayHist(img):
    '''
    Display data distribution of input image in a histogram
    Params:
        img (narray): Image in form of (H,W,C) to display data distribution
    '''

    img = mmNorm1(img)
    im = np.where(img == 0, np.nan, img)

    plt.hist(img.ravel(), 500, [np.nanmin(im), img.max()])
    plt.figure(figsize=(20, 20))
    plt.show()


def mmNorm_lab(img, nodata, clip_val):
    """
    Data normalization with min/max method
    Params:
        img (narray): The targeted image for normalization
    Returns:
        narrray
    """

    #img_tmp = np.where(img == nodata, np.nan, img)
    img_tmp = np.where((img < 0) | np.isin(img, nodata), np.nan, img)
    
    if (clip_val is not None) and clip_val > 0:
        lower_percentiles = np.nanpercentile(img_tmp, clip_val, axis=(1, 2))
        upper_percentiles = np.nanpercentile(img_tmp, 100 - clip_val, axis=(1, 2))
        for b in range(img.shape[0]):
            img[b] = np.clip(img[b], lower_percentiles[b], upper_percentiles[b])
    
    img_max = np.nanmax(img_tmp)
    img_min = np.nanmin(img_tmp)
    
    normalized = (img - img_min)/(img_max - img_min)
    normalized = np.clip(normalized, 0, 1)

    return normalized


def mmNorm_lpb(img, nodata, clip_val=None):
    r"""
    Normalize the input image pixels to [0, 1] ranged based on the
    minimum and maximum statistics of each band per tile.
    Arguments:
            img (numpy array) -- Stacked image bands with a dimension of (C,H,W).
            nodata (str) -- value reserved to represent NoData in the image chip.
            clip_val (int) -- defines how much of the distribution tails to be cut off.
    Returns:
            img (numpy array) -- Normalized image stack of size (C,H,W).
    Note 1: If clip then min, max are calculated from the clipped image.
    """
    #img_tmp = np.where(img == nodata, np.nan, img)
    img_tmp = np.where((img < 0) | np.isin(img, nodata), np.nan, img)
    
    if (clip_val is not None) and clip_val > 0:
        lower_percentiles = np.nanpercentile(img_tmp, clip_val, axis=(1, 2))
        upper_percentiles = np.nanpercentile(img_tmp, 100 - clip_val, axis=(1, 2))
        for b in range(img.shape[0]):
            img[b] = np.clip(img[b], lower_percentiles[b], upper_percentiles[b])
    
    lpb_mins = np.nanmin(img_tmp, axis=(1, 2))
    lpb_maxs = np.nanmax(img_tmp, axis=(1, 2))
    if np.any(lpb_maxs == lpb_mins):
        raise ValueError("Division by zero detected: some bands have identical min and max values.")
    
    normal_img = (img - lpb_mins[:, None, None]) / (lpb_maxs[:, None, None] - lpb_mins[:, None, None])
    normal_img = np.clip(normal_img, 0, 1)
    
    return normal_img


def mmNorm_gpb(img, nodata, global_stats, clip_val=None):
    
    gpb_mins = np.array(global_stats['min'])
    gpb_maxs = np.array(global_stats['max'])
    
    #img_tmp = np.where(img == nodata, np.nan, img)
    img_tmp = np.where((img < 0) | np.isin(img, nodata), np.nan, img)
    
    if (clip_val is not None) and clip_val > 0:
        lower_percentiles = np.nanpercentile(img_tmp, clip_val, axis=(1, 2))
        upper_percentiles = np.nanpercentile(img_tmp, 100 - clip_val, axis=(1, 2))
        for b in range(img.shape[0]):
            img[b] = np.clip(img[b], lower_percentiles[b], upper_percentiles[b])
    
    if np.any(gpb_maxs == gpb_mins):
        raise ValueError("Division by zero detected: some bands have identical min and max values.")
    
    normal_img = (img - gpb_mins[:, None, None]) / (gpb_maxs[:, None, None] - gpb_mins[:, None, None])
    normal_img = np.clip(normal_img, 0, 1)

    return normal_img


def mmNorm_gab(img, nodata, global_stats, clip_val=None):
    
    gpb_mins = np.array(global_stats['min'])
    gpb_maxs = np.array(global_stats['max'])
    gab_min = np.mean(gpb_mins)           
    gab_max = np.mean(gpb_maxs)
    
    if gab_max == gab_min:
        raise ValueError("Division by zero detected: some images have identical min and max values.")
    
    #img_tmp = np.where(img == nodata, np.nan, img)
    img_tmp = np.where((img < 0) | np.isin(img, nodata), np.nan, img)
    
    if (clip_val is not None) and clip_val > 0:
        lower_percentiles = np.nanpercentile(img_tmp, clip_val, axis=(1, 2))
        upper_percentiles = np.nanpercentile(img_tmp, 100 - clip_val, axis=(1, 2))
        for b in range(img.shape[0]):
            img[b] = np.clip(img[b], lower_percentiles[b], upper_percentiles[b])
    
    normal_img = (img - gab_min) / (gab_max - gab_min)
    normal_img = np.clip(normal_img, 0, 1)

    return normal_img


def zvNorm_gpb(img, nodata, global_stats, clip_val=None):
    r"""
    Standardize the input image pixels to have zero mean and unit std based 
    on the values calculated for the whole dataset per band.
    Arguments:
            img (numpy array) -- Stacked image bands with a dimension of (C,H,W).
            nodata (str) -- value reserved to represent NoData in the image chip.
            clip_val (int) -- defines how much of the distribution tails to be cut off.
    Returns:
            img (numpy array) -- Normalized image stack of size (C,H,W).
    
    Note 1: If clip then mean and std values must be provided based on the clipped dataset.
    """
    img_tmp = np.where((img < 0) | np.isin(img, nodata), np.nan, img)
    
    if (clip_val is not None) and clip_val > 0:
        lower_percentiles = np.nanpercentile(img_tmp, clip_val, axis=(1, 2))
        upper_percentiles = np.nanpercentile(img_tmp, 100 - clip_val, axis=(1, 2))
        for b in range(img.shape[0]):
            img[b] = np.clip(img[b], lower_percentiles[b], upper_percentiles[b])
    
    gpb_means = np.array(global_stats['mean'])
    gpb_stds = np.array(global_stats['std'])

    normal_img = (img - gpb_means[:, None, None]) / gpb_stds[:, None, None]
    
    return normal_img


def zvNorm_gab(img, nodata, global_stats, clip_val=None):
    r"""
    Standardize the input image pixels to have zero mean and unit std based 
    on the global values calculated over all bands.
    Arguments:
            img (numpy array) -- Stacked image bands with a dimension of (C,H,W).
            nodata (str) -- value reserved to represent NoData in the image chip.
            clip_val (int) -- defines how much of the distribution tails to be cut off.
    Returns:
            img (numpy array) -- Normalized image stack of size (C,H,W).
    Note 1: If clip then min, max are calculated from the clipped image.
    """
    #img_tmp = np.where(img == nodata, np.nan, img)
    img_tmp = np.where((img < 0) | np.isin(img, nodata), np.nan, img)
    
    if (clip_val is not None) and clip_val > 0:
        lower_percentiles = np.nanpercentile(img_tmp, clip_val, axis=(1, 2))
        upper_percentiles = np.nanpercentile(img_tmp, 100 - clip_val, axis=(1, 2))
        for b in range(img.shape[0]):
            img[b] = np.clip(img[b], lower_percentiles[b], upper_percentiles[b])
    
    gpb_means = np.array(global_stats['mean'])
    gpb_stds = np.array(global_stats['std'])
    
    gab_mean = np.mean(gpb_means)

    num_pixels_per_band = img.shape[1] * img.shape[2]
    squared_std_values = gpb_stds ** 2
    squared_std_values *= num_pixels_per_band
    sum_squared_std = np.sum(squared_std_values)
    total_samples = num_pixels_per_band * len(gpb_stds)
    gab_std = np.sqrt(sum_squared_std / total_samples)

    normal_img = (img - gab_mean) / gab_std
    
    return normal_img


def zvNorm_lpb(img, nodata, clip_val=None):
    r"""
    Standardize the input image pixels to have zero mean and unit std based 
    on the values calculated locally for each tile and per band.
    Arguments:
            img (numpy array) -- Stacked image bands with a dimension of (C,H,W).
            nodata (str) -- value reserved to represent NoData in the image chip.
            clip_val (int) -- defines how much of the distribution tails to be cut off.
    Returns:
            img (numpy array) -- Normalized image stack of size (C,H,W).
    
    Note 1: If clip then mean and std values must be provided based on the clipped dataset.
    """
    img_tmp = np.where((img < 0) | np.isin(img, nodata), np.nan, img)
    
    if (clip_val is not None) and clip_val > 0:
        lower_percentiles = np.nanpercentile(img_tmp, clip_val, axis=(1, 2))
        upper_percentiles = np.nanpercentile(img_tmp, 100 - clip_val, axis=(1, 2))
        for b in range(img.shape[0]):
            img[b] = np.clip(img[b], lower_percentiles[b], upper_percentiles[b])
    
    img_means = np.nanmean(img_tmp, axis=(1, 2))
    img_stds = np.nanstd(img_tmp, axis=(1, 2))
    
    normal_img = (img - img_means[:, None, None]) / img_stds[:, None, None]
    
    return normal_img


def zvNorm_lab(img, nodata, clip_val=None):
    r"""
    Standardize the input image pixels to have zero mean and unit std based 
    on the values calculated locally for each tile and per band.
    Arguments:
            img (numpy array) -- Stacked image bands with a dimension of (C,H,W).
            nodata (str) -- value reserved to represent NoData in the image chip.
            clip_val (int) -- defines how much of the distribution tails to be cut off.
    Returns:
            img (numpy array) -- Normalized image stack of size (C,H,W).
    
    Note 1: If clip then mean and std values must be provided based on the clipped dataset.
    """
    img_tmp = np.where((img < 0) | np.isin(img, nodata), np.nan, img)
    
    if (clip_val is not None) and clip_val > 0:
        lower_percentiles = np.nanpercentile(img_tmp, clip_val, axis=(1, 2))
        upper_percentiles = np.nanpercentile(img_tmp, 100 - clip_val, axis=(1, 2))
        for b in range(img.shape[0]):
            img[b] = np.clip(img[b], lower_percentiles[b], upper_percentiles[b])
    
    img_mean = np.nanmean(img_tmp)
    img_std = np.nanstd(img_tmp)
    normal_img = (img - img_mean) / img_std
    
    return normal_img

def get_chips(img, dsize, buffer):
    '''
    Generate small chips from input images and the corresponding index of each chip The index marks the location of corresponding upper-left pixel of a chip.
    Params:
        img (narray): Image in format of (H,W,C) to be crop, in this case it is the concatenated image of growing
            season and off season
        dsize (int): Cropped chip size
        buffer (int):Number of overlapping pixels when extracting images chips
    Returns:
        list of cropped chips and corresponding coordinates
    '''

    h, w, _ = img.shape
    x_ls = range(0,h - 2 * buffer, dsize - 2 * buffer)
    y_ls = range(0, w - 2 * buffer, dsize - 2 * buffer)

    index = list(itertools.product(x_ls, y_ls))

    img_ls = []
    for i in range(len(index)):
        x, y = index[i]
        img_ls.append(img[x:x + dsize, y:y + dsize, :])

    return img_ls, index



def display(img, label, mask):

    '''
    Display composites and their labels
    Params:
        img (torch.tensor): Image in format of (C,H,W)
        label (torch.tensor): Label in format of (H,W)
        mask (torch.tensor): Mask in format of (H,W)
    '''

    GSimg = (comp432_dis(img, "GS") * 255).permute(1, 2, 0).int()
    OSimg = (comp432_dis(img, "OS") * 255).permute(1, 2, 0).int()


    _, figs = plt.subplots(1, 4, figsize=(20, 20))

    label = label.cpu()

    figs[0].imshow(GSimg)
    figs[1].imshow(OSimg)
    figs[2].imshow(label)
    figs[3].imshow(mask)

    plt.show()


# color composite
def comp432_dis(img, season):
    '''
    Generate false color composites
    Params:
        img (torch.tensor): Image in format of (C,H,W)
        season (str): Season of the composite to generate, be  "GS" or "OS"
    '''

    viewSize = img.shape[1:]

    if season == "GS":

        b4 = mmNorm1(img[3, :, :].cpu().view(1, *viewSize),0)
        b3 = mmNorm1(img[2, :, :].cpu().view(1, *viewSize),0)
        b2 = mmNorm1(img[1, :, :].cpu().view(1, *viewSize),0)

    elif season == "OS":
        b4 = mmNorm1(img[7, :, :].cpu().view(1, *viewSize), 0)
        b3 = mmNorm1(img[6, :, :].cpu().view(1, *viewSize), 0)
        b2 = mmNorm1(img[5, :, :].cpu().view(1, *viewSize), 0)

    else:
        raise ValueError("Bad season value")

    img = torch.cat([b4, b3, b2], 0)

    return img

def make_reproducible(seed=42, cudnn=True):
    """Make all the randomization processes start from a shared seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cudnn:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def pickle_dataset(dataset, filePath):
    with open(filePath, "wb") as fp:
        pickle.dump(dataset, fp)


def load_dataset(filePath):
    return pd.read_pickle(filePath)


def progress_reporter(msg, verbose, logger=None):
    """Helps control print statements and log writes
    Parameters
    ----------
    msg : str
      Message to write out
    verbose : bool
      Prints or not to console
    logger : logging.logger
      logger (defaults to none)
      
    Returns:
    --------  
        Message to console and or log
    """
    
    if verbose:
        print(msg)

    if logger:
        logger.info(msg)


def setup_logger(log_dir, log_name, use_date=False):
    """Create logger
    """
    if use_date:
        dt = datetime.now().strftime("%d%m%Y_%H%M")
        log = "{}/{}_{}.log".format(log_dir, log_name, dt)
    else: 
        log = "{}/{}.log".format(log_dir, log_name)
        
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_format = (
        f"%(asctime)s::%(levelname)s::%(name)s::%(filename)s::"
        f"%(lineno)d::%(message)s"
    )
    logging.basicConfig(filename=log, filemode='w',
                        level=logging.INFO, format=log_format)
    
    return logging.getLogger()


