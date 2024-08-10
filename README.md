## Overview
This repository contains the code used in the following paper:

Khallaghi, S., Abedi, R., Abou Ali, H., Asipunu, M., Alatise, I., Ha, N., Luo, B., Mai, C., Song, L., Wussah, A., Xiong, S., Zhang, Q., Estes, L. (2024). Generalization enhancement strategies to enable cross-year cropland mapping with convolutional neural networks trained using historical samples. ArXiv.

## Supported Models
The package currently supports these semantic segmentation models:
* [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
* [DeepLab v3](https://arxiv.org/pdf/1706.05587.pdf)
* [DeepLab v3+](https://arxiv.org/pdf/1802.02611.pdf)
* [Pyramid Scene Parsing Network (PSPNet)](https://arxiv.org/pdf/1612.01105.pdf)
* [ExFuse](https://arxiv.org/pdf/1804.03821.pdf)
* [Global Convolutional Network (GCN)](https://arxiv.org/pdf/1703.02719.pdf)


## Getting Started
To run the repo you need to: 
1. modify the `config/default_config.yaml` to suit your project's parameters.
2. Follow the step-by-step guide in `notebooks/main.ipynb` to execute the workflow.

## Dataset Preparation

**Our data preparation protocol is designed to work with composite images and is sorted into specific label groups for training efficiency.**

Based on current protocol, the project uses two image composites of 2022 × 2022built separately from growing-season and off-season time series, which are in size of a tile ( 2000 × 2000) plus a buffer of 11 pixels on each side. However, the labels are in grid size of 200 x 200, and was sorted into 4 groups:

* label_group = 0 -- labels that are not reviewed

* label_group = 2 -- labels have both positive and negative categories, while the correctly classified positive category is between 65% and 80%

* label_group = 3 -- labels have both positive and negative categories, while the correctly classified positive category is over 80%

* label_group = 4 -- labels have only negative categories, but it's overall accuracy is 100%


The `deeplearner` package uses csv files to load data. Therefore, two catalogs are required besides the raw images and labels. One is for train and validation, while another one is for prediction:

* catalog for train and validation

  * It contains at least 4 groups of columns:

    * columns for image directories, could be either a relative path to a data folder, or a full path in aws s3, starting with `s3://`
    * a column for label directories, could be either a relative path to a data folder, or a full path in aws s3, starting with `s3://`
    * a column named `usage`, where the usage value is `train` or `validate`
    * a column named `label_group `

  * Here‘s an example of the table format, where `dir_gs` and `dir_os` are directories to images and `dir_label` is directories to labels

    | name      | usage    | dir_gs                                               | dir_os                                               | dir_label                                                    | label_group |
    | --------- | -------- | ---------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ | ----------- |
    | GH0242195 | train    | images/planet/nonfix/GS/tile539785_736815_736967.tif | images/planet/nonfix/OS/tile539785_737029_737118.tif | labels/semantic_segmentation/accurate/GH0242195_3241_5699.tif | 3           |
    | GH0288657 | validate | images/planet/nonfix/GS/tile539959_736815_736967.tif | images/planet/nonfix/OS/tile539959_737029_737118.tif | labels/semantic_segmentation/accurate/GH0288657_3385_5774.tif | 3           |

* catalog for prediction

  * It contains at least 3 groups of columns:

    * columns for image directories, could be either a relative path to a data folder, or a full path in aws s3, starting with `s3://`

  * two columns for naming the output, where output would be named as `score_{col1}_{col2}.tif`

    * a column named `type`, specifying whether each row is a `center` image whose prediction would be written out, or a `neighbor` image

  * Also an example of the table format . Here I use `dir_gs` and `dir_os` as directories to images, and `tile_col` and `tile_row` as the naming columns to keep the naming system consistent with `learner`

    | tile_col | tile_row | dir_gs                                            | dir_os                                            | type     |
    | -------- | -------- | ------------------------------------------------- | ------------------------------------------------- | -------- |
    | 320      | 560      | images/planet/fix/GS/tile539601_736815_736967.tif | images/planet/fix/OS/tile539601_737029_737118.tif | center   |
    | 321      | 560      | images/planet/fix/GS/tile539602_736815_736967.tif | images/planet/fix/OS/tile539602_737029_737118.tif | neighbor |

## Accessing the Dataset
The training dataset, along with csv catalogs for replicating the results in our paper, is available at [dataset access link].





