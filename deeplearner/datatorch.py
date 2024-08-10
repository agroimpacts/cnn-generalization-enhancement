
from datetime import datetime

from torch.utils.data import Dataset, DataLoader

from .utils import *
from .augmentation import *
from .tools import parallelize_df


class planetData(Dataset):
    '''
    Dataset of planet scope image files for pytorch architecture

    '''

    def __init__(self, dataPath, log_dir, catalog, dataSize, buffer, bufferComp, usage, imgPathCols, labelPathCol=None, 
                 labelGroup = [0,1,2,3,4], norm_stats_type="local_per_tile", clip_val=0, global_stats=None, nodata_val_ls=None,
                 catalogIndex=None, deRotate=(-90, 90), bShiftSubs=(4, 4), trans=None):

        '''
        Params:

            dataPath (str): Directory storing files of variables and labels.
            log_dir (str): Directory to save the log file.
            catalog (Pandas.DataFrame): Pandas dataframe giving the list of data and their directories
            dataSize (int): Size of chips that is not buffered, i.e., the size of labels
            buffer (int): Distance to target chips' boundaries measured by number of pixels when extracting images
                (variables), i.e., variables size would be (dsize + buffer) x (dsize + buffer)
            bufferComp (int): Buffer used when creating composite. In the case of Ghana, it is 11.
            usage (str): Usage of the dataset : "train", "validate" or "predict"
            imgPathCols (list): Column names in the catalog referring to image paths
            labelPathCol(str): Column name in the catalog referring to label paths
            labelGroup (list): Group indices of labels to load, where each group corresponds to a specific level of label quality
            catalogIndex (int or None): Row index in catalog to load data for prediction. Only need to be specified when
                usage is "prediction"
            deRotate (tuple or None): Range of degrees for rotation
            bShiftSubs (tuple or list): Number of bands or channels on dataset for each brightness shift
            trans (list): Data augmentation methods: one or multiple elements from ['vflip','hflip','dflip', 'rotate',
                'resize']

        Note:

            Provided transformation are:
                1) 'vflip', vertical flip
                2) 'hflip', horizontal flip
                3) 'dflip', diagonal flip
                4) 'rotate', rotation
                5) 'resize', rescale image fitted into the specified data size
                6) 'shift_brightness', shift brightness of images

            Any value out of the range would cause an error

        Note:

            Catalog for train and validate contrains at least columns for image path, label path and "usage".

            Catalog for prediction contains at least columns for image path, "tile_col", and "tile_row", where the
            "tile_col" and "tile_row" is the relative tile location for naming predictions in Learner

        '''

        self.buffer = buffer
        self.composite_buffer = bufferComp
        self.data_size = dataSize
        self.chip_size = self.data_size+ self.buffer * 2

        self.usage = usage
        self.clip_val = float(clip_val) if clip_val is not None else None
        self.deRotate = deRotate
        self.bshift_subs = bShiftSubs
        self.trans = trans
        self.norm_stats_type = norm_stats_type
        self.global_stats = global_stats
        self.nodata_val_ls = nodata_val_ls

        self.data_path = dataPath
        self.log_dir = log_dir
        self.img_cols = imgPathCols if isinstance(imgPathCols, list) else [imgPathCols]
        self.label_col = labelPathCol
        
        self.logger = setup_logger(self.log_dir, f"{self.usage}_dataset_report", use_date=True)
        start = datetime.now()
        msg = f'started dataset creation process at: {start}'
        progress_reporter(msg, verbose=False, logger=self.logger)

        if self.usage in ["train", "validate"]:
            self.catalog = catalog.loc[
                (catalog['usage'] == self.usage) & 
                (catalog['label_group'].isin(labelGroup))].copy()
            
            self.img, self.label = self.get_train_validate_data()
            print(f'----------{len(self.img)} samples loaded in {self.usage} dataset-----------')

        elif self.usage == "predict":
            self.catalog = catalog.iloc[catalogIndex]
            self.tile = (self.catalog['tile_col'], self.catalog['tile_row'])
            self.year = self.catalog["image_dir"].split("_")[1].split("-")[0]
            #self.year = self.catalog["year"]
            self.img, self.index, self.meta = self.get_predict_data()

        else:
            raise ValueError("Bad usage value")
        
        end = datetime.now()
        msg = f'Completed dataset creation process for {self.usage} at: {end}'    
        progress_reporter(msg, verbose=False, logger=self.logger)


    def get_train_validate_data(self):
        '''
        Get paris of image, label for train and validation

        Returns:
            tuple of list of images and label

        '''

        def load_label(row, data_path):

            buffer = self.buffer

            if type(row[self.label_col]) == str:
              label = row[self.label_col]
            else:
              label = row[self.label_col].iloc[0]
            
            dir_label = row[self.label_col] if row[self.label_col].startswith("s3") \
                else os.path.join(data_path, row[self.label_col])
            label = load_data(dir_label, usage=self.usage, isLabel=True)
            label = np.pad(label, buffer, 'constant')
            msg = f'.. processing lbl sample: {os.path.basename(dir_label)} is complete.'
            progress_reporter(msg, verbose=False, logger=self.logger)

            return label

        def load_img(row, data_path):

            buffer = self.buffer

            dir_label = row['dir_label'] if row['dir_label'].startswith("s3") \
                else os.path.join(data_path, row['dir_label'])
            dir_imgs = [row[m] if row[m].startswith("s3") else os.path.join(data_path, row[m]) for m in self.img_cols]
            window = get_buffered_window(dir_imgs[0], dir_label, buffer)
            img = get_stacked_img(dir_imgs, self.usage, self.clip_val, self.norm_stats_type, 
                                  self.global_stats, self.nodata_val_ls, window=window)
            
            msg = f'.. processing img sample: {os.path.basename(dir_imgs[0])} is complete.'
            progress_reporter(msg, verbose=False, logger=self.logger)

            return img
        
        """
        catalog["img"] = self.catalog.apply(lambda row: load_img(row, data_path=self.data_path), axis=1)
        catalog["label"] = self.catalog.apply(lambda row: load_label(row, data_path=self.data_path), axis=1)
        img_ls = catalog['img'].tolist()
        label_ls = catalog['label'].tolist()
        """

        global list_data # Local function not applicable in parallelism
        def list_data(catalog, data_path):

            catalog["img"] = catalog.apply(lambda row: load_img(row, data_path), axis=1)
            catalog["label"] = catalog.apply(lambda row: load_label(row, data_path), axis=1)

            return catalog.filter(items=['label', 'img'])

        catalog = parallelize_df(self.catalog, list_data, data_path=self.data_path)

        img_ls = catalog['img'].tolist()
        label_ls = catalog['label'].tolist()

        return img_ls, label_ls



    def get_predict_data(self):
        '''
        Get data for prediction

        Returns:
            list of cropped chips
            list of index representing location of each chip in tile
            dictionary of metadata of score map reconstructed from chips

        '''

        dir_imgs = [self.catalog[m] if self.catalog[m].startswith("s3") \
            else os.path.join(self.data_path, self.catalog[m]) for m in self.img_cols]
        img = get_stacked_img(dir_imgs, self.usage, self.clip_val, self.norm_stats_type, 
                              self.global_stats, self.nodata_val_ls)  # entire composite image in (H, W, C)
        buffer_diff = self.buffer - self.composite_buffer
        h,w,c = img.shape

        if buffer_diff > 0:
            canvas = np.zeros((h + buffer_diff * 2, w + buffer_diff * 2, c))

            for i in range(c):
                canvas[:,:,i] = np.pad(img[:,:,i], buffer_diff, mode='reflect')
            img = canvas

        else:
            img = img[buffer_diff:h-buffer_diff, buffer_diff:w-buffer_diff, :]

        meta = get_meta_from_bounds(dir_imgs[0], self.composite_buffer) # meta of composite buffer removed
        img_ls, index_ls = get_chips(img, self.chip_size, self.buffer)

        return img_ls, index_ls, meta


    def __getitem__(self, index):
        """
        Support dataset indexing and apply transformation

        Args:
            index -- Index of each small chips in the dataset

        Returns:
            tuple

        """

        if self.usage in ["train", "validate"]:
            img = self.img[index]
            label = self.label[index]


            if self.usage == "train":
                mask = np.pad(np.ones((self.data_size, self.data_size)), self.buffer, 'constant')
                trans = self.trans
                # trans = None
                deRotate = self.deRotate

                if trans:

                    # 0.5 possibility to flip
                    trans_flip_ls = [m for m in trans if 'flip' in m]
                    if random.randint(0, 1) and len(trans_flip_ls) > 1:
                        trans_flip = random.sample(trans_flip_ls, 1)
                        img, label, mask = flip(img, label, mask, trans_flip[0])

                    # 0.5 possibility to resize
                    if random.randint(0, 1) and 'resize' in trans:
                        img, label, mask = reScale(img, label.astype(np.uint8), mask.astype(np.uint8),
                                                   randResizeCrop=True, diff=True, cenLocate=False)

                    # 0.5 possibility to rotate
                    if random.randint(0, 1) and 'rotate' in trans:
                        img, label, mask = centerRotate(img, label, mask, deRotate)

                    # # 0.5 possibility to shift brightness
                    # if random.randint(0, 1) and 'shift_brightness' in trans:
                    #     img = shiftBrightness(img, gammaRange=(0.2, 2), shiftSubset=self.bshift_subs, patchShift=True)
                    
                    # 0.5 possibility to brightness manipulation
                    trans_br_ls = [m for m in self.trans if 'br_' in m]
                    if random.randint(0, 1) and len(trans_br_ls) >= 1:
                        trans_br = random.sample(trans_br_ls, 1)
                        img = br_manipulation(img, trans_br[0], sigma_range=[0.03, 0.07], 
                                              br_range=[-0.02, 0.02], contrast_range=[0.8, 1.2], 
                                              gamma_range=[0.2, 2.0], shift_subset=self.bshift_subs, 
                                              patch_shift=True)
                    

                # numpy to torch
                label = torch.from_numpy(label).long()
                mask = torch.from_numpy(mask).long()
                img = torch.from_numpy(img.transpose((2, 0, 1))).float()

                # display(img[:, self.buffer:-self.buffer, self.buffer:-self.buffer], label[self.buffer:-self.buffer,self.buffer:-self.buffer], mask[self.buffer:-self.buffer,self.buffer:-self.buffer])
                # display(img, label, mask)

                return img, label, mask

            else:
                # numpy to torch
                label = torch.from_numpy(label).long()
                img = torch.from_numpy(img.transpose((2, 0, 1))).float()

                return img, label

        else:

            img = self.img[index]
            index = self.index[index]

            img = torch.from_numpy(img.transpose((2, 0, 1))).float()

            return img, index


    def __len__(self):
        '''
        Get size of the dataset
        '''

        return len(self.img)


