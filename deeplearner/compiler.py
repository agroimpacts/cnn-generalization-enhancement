from datetime import datetime
from tensorboardX import SummaryWriter
import urllib.parse as urlparse
from rasterio.io import MemoryFile
from pathlib import Path
from torch import optim

from .train import *
from .validate import *
from .evaluate2 import *
from .predict import *
from .models import *
from .utils_aws import *
import torch
import re
import sys
__all__ = ["ModelCompiler"]

def get_optimizer(optimizer, model, params, lr, momentum):

    optimizer = optimizer.lower()
    if optimizer == 'sgd':
        return torch.optim.SGD(params, lr, momentum=momentum)
    elif optimizer == 'nesterov':
        return torch.optim.SGD(params, lr, momentum=momentum, nesterov=True)
    elif optimizer == 'adam':
        return torch.optim.Adam(params, lr)
    elif optimizer == 'amsgrad':
        return torch.optim.Adam(params, lr, amsgrad=True)
    elif optimizer == 'sam':
        base_optimizer = optim.SGD
        return SAM(model.parameters(), base_optimizer, lr=lr, 
                   momentum=momentum)
    else:
        raise ValueError("{} currently not supported, please customize your optimizer in compiler.py".format(optimizer))


def weighted_average_overlay(predDict, overlayPixels):

    if isinstance(predDict, dict):
        key_ls = ["top", "center", "left", "right", "bottom"]
        key_miss_ls = [m for m in predDict.keys() if m not in key_ls]
        if len(key_miss_ls) == 0:
            pass
        else:
            assert "Input must be dictionary containing data for centered image and its 4 neighbors."\
            "Missed {}".format(", ".join(key_miss_ls))
    else:
        assert "Input must be dictionary containing data for centered image and its 4 neighbors, " \
               "including including 'top', 'left', 'right', and  'bottom'"

    target = predDict['center']
    h, w = target.shape
    # top
    if predDict['top'] is not None:
        target_weight = np.array([1. / overlayPixels * np.arange(1, overlayPixels + 1)] * w).transpose(1, 0)
        comp_weight = 1. - target_weight
        # comp = scores_dict["up"][- overlay_pixs : , : ]
        target[:overlayPixels, :] = comp_weight * predDict['top'][- overlayPixels:, :] + \
                                   target_weight * target[:overlayPixels, :]
    else:
        pass
    # bottom
    if predDict['bottom'] is not None:
        target_weight = np.array([1. / overlayPixels * np.flip(np.arange(1, overlayPixels + 1))] * w).transpose(1, 0)
        comp_weight = 1. - target_weight
        target[-overlayPixels:, :] = comp_weight * predDict['bottom'][:overlayPixels, :] + \
                                    target_weight * target[-overlayPixels:, :]
    else:
        pass
    # left
    if predDict['left'] is not None:
        target_weight = np.array([1. / overlayPixels * np.arange(1, overlayPixels + 1)] * h)
        comp_weight = 1 - target_weight
        target[:, :overlayPixels] = comp_weight * predDict['left'][:, -overlayPixels:] + \
                                   target_weight * target[:, :overlayPixels]
    else:
        pass
    # right
    if predDict['right'] is not None:
        target_weight = np.array([1. / overlayPixels * np.flip(np.arange(1, overlayPixels + 1))] * h)
        comp_weight = 1 - target_weight
        target[:, -overlayPixels:] = comp_weight * predDict['right'][:, :overlayPixels] + \
                                    target_weight * target[:, -overlayPixels:]
    else:
        pass

    return target


class ModelCompiler:
    """
    Compiler of specified model

    Args:

        model (''nn.Module''): pytorch model for segmentation
        buffer (int): distance to sample edges not considered in optimization
        gpuDevices (list): indices of gpu devices to use
        params_init (dict object): initial model parameters
        freeze_params (list): list of indices for parameters to keep frozen

    """

    def __init__(self, model, working_dir, out_dir, buffer, class_mapping, gpuDevices=[0], 
                 params_init=None, freeze_params=None):
        
        self.working_dir = working_dir
        self.out_dir = out_dir
        self.class_mapping = class_mapping
        self.num_classes = len(self.class_mapping)
        
        # s3 client
        self.s3_client = boto3.client("s3")

        # model
        self.gpuDevices = gpuDevices

        
        self.model = model

        self.model_name = self.model.__class__.__name__
        
        if params_init:
            self.load_params(params_init, freeze_params)
            print("---------- Pre-trained model compiled successfully ----------")
        else:
            print("---------- Vanilla Model compiled successfully ----------")

        self.buffer = buffer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if self.device.type == "cuda":
            print('----------GPU available----------')
            self.gpu = True
            # GPU setting
            if len(self.gpuDevices) > 1:
                torch.cuda.set_device(gpuDevices[0])
                self.model = torch.nn.DataParallel(self.model, device_ids=gpuDevices)
        else:
            print('----------No GPU available, using CPU instead----------')
            self.gpu = False
        
        num_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print("total number of trainable parameters: {:2.1f}M".format(num_params / 1000000))
            

    def load_params(self, dir_params, freeze_params):

        params_init = urlparse.urlparse(dir_params)
        if params_init.scheme == "s3":
            bucket = params_init.netloc
            params_key = params_init.path
            params_key = params_key[1:] if params_key.startswith('/') else params_key
            _, fn_params = os.path.split(params_key)

            self.s3_client.download_file(Bucket=bucket,
                                         Key=params_key,
                                         Filename=fn_params)
            inparams = torch.load(fn_params, map_location='cuda:{}'.format(self.gpuDevices[0]))

            os.remove(fn_params)  # remove after loaded

        ## or load from local
        else:
            inparams = torch.load(dir_params, map_location=torch.device("cpu"))

        ## overwrite model entries with new parameters
        model_dict = self.model.state_dict()

        if "module" in list(inparams.keys())[0]:
            inparams_filter = {k[7:]: v.cpu() for k, v in inparams.items() if k[7:] in model_dict}

        else:
            inparams_filter = {k: v.cpu() for k, v in inparams.items() if k in model_dict}
        model_dict.update(inparams_filter)
        # load new state dict
        self.model.load_state_dict(model_dict)

        # free some layers
        if freeze_params != None:
            for i, p in enumerate(self.model.parameters()):
                if i in freeze_params:
                    p.requires_grad = False


    def fit(self, trainDataset, valDataset, epochs, optimizer_name, lr_init, lr_policy, criterion, momentum=None, 
            resume=False, resume_epoch=None, aws_bucket=None, aws_prefixout=None, **kwargs):

        # working_dir = config["working_dir"]
        working_dir = self.working_dir
        out_dir = self.out_dir
        
        model_dir = "{}/{}/{}_ep{}".format(working_dir, out_dir, self.model_name, epochs)
        if not os.path.exists(Path(working_dir) / out_dir / model_dir):
            os.makedirs(Path(working_dir) / out_dir / model_dir)
        
        self.checkpoint_dir = Path(working_dir) / out_dir / model_dir / "chkpt"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        os.chdir(Path(working_dir) / out_dir / model_dir)
        
        print('-------------------------- Start training --------------------------')
        start = datetime.now()

        # Tensorboard writer setting
        writer = SummaryWriter('../')

        train_loss = []
        val_loss = []
        lr = lr_init
        # lr_decay = lr_decay if isinstance(lr_decay,tuple) else (lr_decay,1)
        optimizer = get_optimizer(optimizer_name, self.model, filter(lambda p: p.requires_grad, self.model.parameters()), lr,
                                  momentum)

        # initialize different learning rate scheduler
        lr_policy = lr_policy.lower()
        if lr_policy == "StepLR".lower():
            step_size = kwargs.get("step_size", 3)
            gamma = kwargs.get("gamma", 0.98)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma)

        elif lr_policy == "MultiStepLR".lower():
            milestones = kwargs.get("milestones", [15, 25, 35, 50, 70, 90, 120, 150, 200])
            gamma = kwargs.get("gamma", 0.5)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=gamma,
            )

        elif lr_policy == "ReduceLROnPlateau".lower():
            mode = kwargs.get('mode', 'min')
            factor = kwargs.get('factor', 0.8)
            patience = kwargs.get('patience', 3)
            threshold = kwargs.get('threshold', 0.0001)
            threshold_mode = kwargs.get('threshold_mode', 'rel')
            min_lr = kwargs.get('min_lr', 3e-6)
            verbose = kwargs.get('verbose', True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold,
                threshold_mode=threshold_mode, min_lr=min_lr, verbose=verbose
            )

        elif lr_policy == "PolynomialLR".lower():
            max_decay_steps = kwargs.get('max_decay_steps', 100)
            min_learning_rate = kwargs.get('min_learning_rate', 1e-5)
            power = kwargs.get('power', 0.75)
            scheduler = PolynomialLR(
                optimizer, max_decay_steps=max_decay_steps, min_learning_rate=min_learning_rate,
                power=power
            )

        elif lr_policy == "CyclicLR".lower():
            base_lr = kwargs.get('base_lr', 3e-5)
            max_lr = kwargs.get('max_lr', 0.01)
            step_size_up = kwargs.get('step_size_up', 1100)
            mode =  kwargs.get('mode', 'triangular')
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up,
                mode=mode
            )

        else:
            scheduler = None

        if resume:
            model_state_file = os.path.join(
                self.checkpoint_dir,
                f"{resume_epoch}_checkpoint.pth.tar"
            )
            if aws_bucket and aws_prefixout:
                self.s3_client.download_file(Bucket=aws_bucket,
                                             Key=os.path.join(aws_prefixout, f"{resume_epoch}_checkpoint.pth.tar"),
                                             Filename=model_state_file
                                             )
                print(f"Checkpoint file downloaded from s3 and saved to {model_state_file}")
            # Resume the model from the specified checkpoint in the config file.
            if os.path.exists(model_state_file):
                checkpoint = torch.load(model_state_file)
                resume_epoch = checkpoint["epoch"]
                scheduler.load_state_dict(checkpoint["scheduler"])
                self.model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                train_loss = checkpoint["train_loss"]
                val_loss = checkpoint["val_loss"]
            else:
                raise ValueError(f"{model_state_file} does not exist")

        if resume:
            iterable = range(resume_epoch, epochs)
        else:
            iterable = range(epochs)

        for t in iterable:

            print(f"[{t+1}/{epochs}]")

            # start fitting
            start_epoch = datetime.now()
            train(trainDataset, self.model, criterion, optimizer, scheduler, num_classes=self.num_classes, gpu=self.gpu, trainLoss=train_loss)
            validate(valDataset, self.model, criterion, self.buffer, gpu=self.gpu, valLoss=val_loss)

            # Update the scheduler
            if lr_policy in ["StepLR".lower(), "MultiStepLR".lower()]:
                scheduler.step()
                print(f"LR: {scheduler.get_last_lr()}")

            if lr_policy == "ReduceLROnPlateau".lower():
                scheduler.step(val_loss[t])

            if lr_policy == "PolynomialLR".lower():
                scheduler.step(t)
                print(f"LR: {optimizer.param_groups[0]['lr']}")

            # time spent on single iteration
            print('time:', (datetime.now() - start_epoch).seconds)

            # Adjust index and logger to resume status and save checkpoits in defined intervals.
            # index = t-resume_epoch if resume else t

            writer.add_scalars(
                "Loss",
                {"train_loss": train_loss[t],
                 "val_loss": val_loss[t]},
                 t + 1)

            checkpoint_interval = 20 # e.g. save every 10 epochs
            if (t+1) % checkpoint_interval == 0:
                torch.save(
                    {
                        "epoch": t+1,
                        "state_dict": self.model.module.state_dict() if len(self.gpuDevices)>1 else \
                            self.model.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss
                    }, os.path.join(
                        self.checkpoint_dir,
                        f"{t+1}_checkpoint.pth.tar")
                )

        writer.close()

        print(f"-------------------------- Training finished in {(datetime.now() - start).seconds}s --------------------------")


    def accuracy_evaluation(self, eval_dataset, unknown_class_idx=None, filename=None):
        """
        Evaluate the accuracy of the model on the provided evaluation dataset.

        Args:
            eval_dataset (DataLoader): The evaluation dataset.
            filename (str or pathlib object): The filename or path to save the evaluation 
                                              results in the output CSV.
        """

        default_out_dir = Path(self.working_dir) / self.out_dir
        if not os.path.exists(default_out_dir):
            os.makedirs(default_out_dir)
        
        if filename is None:
            output_path = str(default_out_dir / "metrics.csv")
        else:
            if os.path.dirname(filename):
                file_dir = Path(filename).parent
                if not file_dir.exists():
                    os.makedirs(file_dir)
                output_path = filename
            else:
                output_path = str(default_out_dir / filename)
                
        print("---------------- Start evaluation ----------------")

        start = datetime.now()

        evaluate(self.model, eval_dataset, self.num_classes, self.class_mapping, 
                 self.device, unknown_class_idx, self.buffer, output_path)

        duration_in_sec = (datetime.now() - start).seconds
        print(f"--------- Evaluation finished in {duration_in_sec}s ---------")
    

    def predict(self, predDataset, outPrefix, bucket=None, mc_samples=None, predBuffer=None, averageNeighbors=False, 
                shrinkBuffer=0, hardening_threshold=70, filename=""):
        # predDataset must be dictionary containing target and all 4 neighbors if averageNeighbors
        if averageNeighbors == True:
            if isinstance(predDataset, dict):
                key_ls = ["top", "center", "left", "right", "bottom"]
                key_miss_ls = [m for m in predDataset.keys() if m not in key_ls]
                if len(key_miss_ls) == 0:
                    pass
                else:
                    assert "predDataset must be dictionary containing data for centered image and its 4 neighbors when " \
                           "averageNeighbors set to be True. Missed {}".format(", ".join(key_miss_ls))
            else:
                assert "predDataset must be dictionary containing data for centered image and its 4 neighbors when " \
                       "averageNeighbors set to be True, including 'top', 'left', 'right', 'bottom'"
        else:
            pass

        print('-------------------------- Start prediction --------------------------')
        start = datetime.now()
            
        _, meta, tile, year = predDataset["center"] if isinstance(predDataset, dict) else predDataset
        if year is None:
            name_score = f"score_c{tile[0]}_r{tile[1]}{filename}.tif"
        else:
             name_score = f"{year}_score_c{tile[0]}_r{tile[1]}{filename}.tif"

        meta.update({
            'dtype': 'int8'
        })
        meta_uncertanity = meta.copy()
        meta_uncertanity.update({
            "dtype": "float64"
        })
        s3_client = boto3.client("s3")
        prefix_score = os.path.join(outPrefix, "MeanProb_score")
        os.makedirs(prefix_score, exist_ok=True)
        new_buffer = predBuffer - shrinkBuffer
        if averageNeighbors:
            scores_dict = {k: predict(predDataset[k], self.model, predBuffer, gpu=self.gpu, shrinkPixel=shrinkBuffer) if predDataset[k]
                           else None for k in predDataset.keys()}

            nclass = len(list(scores_dict['center']))
            overlay_pixs = new_buffer * 2

            for n in range(nclass):
                score_dict = {k: scores_dict[k][n] if scores_dict[k] else None for k in scores_dict.keys()}
                score = weighted_average_overlay(score_dict, overlay_pixs)
                # write to s3
                score = score[new_buffer: meta['height'] + new_buffer, new_buffer:meta['height'] + new_buffer]
                score = score.astype(meta['dtype'])
                
                with rasterio.open(f"{prefix_score}_{n + 1}_{name_score}", "w", **meta) as dst:
                    dst.write(score, indexes=1)
                
                """
                with MemoryFile() as memfile:
                    with memfile.open(**meta) as src:
                        src.write(score, 1)

                    s3_client.upload_fileobj(Fileobj=memfile,
                                             Bucket=bucket,
                                             Key=os.path.join(prefix_score + "_{}".format(n + 1), name_score))
                """
        # when not averageNeighbors
        else:
            if mc_samples:
                scores = mc_predict(predDataset, self.model, mc_samples, predBuffer, gpu=self.gpu, 
                                    shrinkPixel=shrinkBuffer)
                # write score of each non-background classes into s3
                nclass = len(scores)
                prefix_hardened = os.path.join(outPrefix, "Hardened")
                os.makedirs(prefix_hardened, exist_ok=True)
                prefix_var = os.path.join(outPrefix, "Variance")
                os.makedirs(prefix_var, exist_ok=True)
                prefix_entropy = os.path.join(outPrefix, "Entropy_MI")
                os.makedirs(prefix_entropy, exist_ok=True)
                # subtracting one as I want to ingnore generating results for boundary class.
                # to increase the speed and save space.
                for n in range(nclass-1):
                    canvas = scores[n][:, new_buffer: meta['height'] + new_buffer, new_buffer: meta['width'] + new_buffer]
                
                    mean_pred = np.mean(canvas, axis=0)
                    mean_pred = np.rint(mean_pred)
                
                    mean_pred = mean_pred.astype(meta['dtype'])
                
                    with rasterio.open(os.path.join(prefix_score, f"{n + 1}_{name_score}"), "w", **meta) as dst:
                        dst.write(mean_pred, indexes=1)
                    
                    hardened_score = np.where(mean_pred > hardening_threshold, mean_pred, 0)
                    with rasterio.open(os.path.join(prefix_hardened, f"hardened_{n + 1}_{name_score}"), 
                                   "w", **meta) as dst:
                        dst.write(hardened_score, indexes=1)
                
                    var_pred = np.var(canvas, axis=0)
                    var_pred = var_pred.astype(meta_uncertanity['dtype'])
                    with rasterio.open(os.path.join(prefix_var, f"{n + 1}_{name_score}"), "w", **meta_uncertanity) as dst:
                        dst.write(var_pred, indexes=1)

                    epsilon = sys.float_info.min
                    entropy = -(mean_pred * np.log(mean_pred + epsilon))
                    mutual_info = entropy - np.mean(-canvas * np.log(canvas + epsilon), axis=0)
                    mutual_info = mutual_info.astype(meta_uncertanity['dtype'])
                    with rasterio.open(os.path.join(prefix_entropy, f"{n + 1}_{name_score}"), "w", **meta_uncertanity) as dst:
                        dst.write(mutual_info, indexes=1)
                
                    """
                    # uploading to AWS s3
                    with MemoryFile() as memfile:
                        with memfile.open(**meta) as src:
                            src.write(canvas, 1)
                        s3_client.upload_fileobj(Fileobj=memfile,
                                                 Bucket=bucket,
                                                 Key=os.path.join(prefix_score + "_{}".format(n + 1), name_score))
                    """
            else:
                scores = predict(predDataset, self.model, predBuffer, gpu=self.gpu, shrinkPixel=shrinkBuffer)
                # write score of each non-background classes into s3
                nclass = len(scores)
                prefix_hardened = os.path.join(outPrefix, "Hardened")
                os.makedirs(prefix_hardened, exist_ok=True)
                
                for n in range(nclass-1):
                    canvas = scores[n][new_buffer: meta['height'] + new_buffer, new_buffer: meta['width'] + new_buffer]
                    canvas = canvas.astype(meta['dtype'])
                
                    with rasterio.open(os.path.join(prefix_score, f"{n + 1}_{name_score}"), "w", **meta) as dst:
                        dst.write(canvas, indexes=1)
                    
                    hardened_score = np.where(canvas > hardening_threshold, canvas, 0)
                    with rasterio.open(os.path.join(prefix_hardened, f"hardened_{n + 1}_{name_score}"), 
                                   "w", **meta) as dst:
                        dst.write(hardened_score, indexes=1)

        print('-------------------------- Prediction finished in {}s --------------------------' \
              .format((datetime.now() - start).seconds))

   
    def save_checkpoint(self, bucket, outPrefix, checkpoints):
        '''
            checkpoints: save last n checkpoint files or list of checkpoint to save
        '''
        if type(checkpoints) is list:
            checkpoint_files = [f"{i}_checkpoint.pth.tar" for i in checkpoints]
        else:
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir)]
            # sorted by epoch number
            checkpoint_files = sorted(checkpoint_files, key=lambda x:int(re.findall("\d+", x)[0]), reverse=True)[:checkpoints]

        for f in checkpoint_files:
            file = os.path.join(self.checkpoint_dir,f)
            self.s3_client.upload_file(Filename=file,
                                      Bucket=bucket,
                                      Key=os.path.join(outPrefix, f))
            # os.remove(file)
        print(f"{checkpoints} checkpoint files saved to s3 at {outPrefix}")
    
#     def save(self, bucket=None, outPrefix=None, save_object="params", filename=""):

#         train_loss_dir = "Loss/train_loss"
#         val_loss_dir = "Loss/val_loss"

#         # upload Loss files to s3
#         train_loss = [f for f in os.listdir(train_loss_dir)][0]
#         val_loss = [f for f in os.listdir(val_loss_dir)][0]
        
#         train_loss_out = filename + train_loss
#         val_loss_out = filename + val_loss
        
#         if bucket and outPrefix:
#             self.s3_client.upload_file(Filename=os.path.join(train_loss_dir,train_loss),
#                                         Bucket=bucket,
#                                         Key=outPrefix + f"/{train_loss_dir}/" + train_loss_out)
#             self.s3_client.upload_file(Filename=os.path.join(val_loss_dir,val_loss),
#                                         Bucket=bucket,
#                                         Key=outPrefix+ f"/{val_loss_dir}/" +val_loss_out)
#             print("Loss files uploaded to s3")
#             os.remove(os.path.join(train_loss_dir,train_loss))
#             os.remove(os.path.join(val_loss_dir,val_loss))

        
#         if not filename:
#             filename = self.model_name
        
#         if save_object == "params" :
#             torch.save(self.model.state_dict(),
#                        os.path.join(self.checkpoint_dir, "{}_final_state.pth".format(filename)))
            
#             if bucket and outPrefix:
#                 self.s3_client.upload_file(Filename=fn_params,
#                                       Bucket=bucket,
#                                       Key=os.path.join(outPrefix, fn_params))
#                 print("model parameters uploaded to s3!, at ", outPrefix)
#                 os.remove(fn_params)

#         elif save_object == "model":
#             torch.save(self.model,
#                        os.path.join(self.checkpoint_dir, "{}_final_state.pth".format(filename)))
            
#             if bucket and outPrefix:
#                 self.s3_client.upload_file(Filename=fn_model,
#                                       Bucket=bucket,
#                                       Key=os.path.join(outPrefix, fn_model))
#                 os.remove(fn_model)

#         else:
#             raise ValueError
        
    def save(self, save_object="params", filename=""):
        """
        Saves the state of the model or its parameters to disk.

        This method allows for the saving of either the entire model or just its parameters. 
        The method also allows for specifying a custom filename for the saved file. If no 
        filename is provided, the model's name is used as the default filename.

        Args:
            save_object (str, optional): Determines what to save. If set to "params", only the 
                                         model's parameters are saved. If set to "model", the 
                                         entire model is saved. Defaults to "params".
            filename (str, optional): The name of the file to save the model or parameters to. 
                                      If not provided, the model's name is used as the filename. 
                                      Defaults to an empty string.
        Note:
            The method prints a confirmation message upon successful saving of the model's 
            parameters or the model itself. The saved file will be located in the model's 
            checkpoint directory.
        """

        if not filename:
            filename = self.model_name

        if save_object == "params":
            if len(self.gpuDevices) > 1:
                torch.save(self.model.module.state_dict(),
                           os.path.join(self.checkpoint_dir, "{}_final_state.pth".format(filename)))
            else:
                torch.save(self.model.state_dict(),
                           os.path.join(self.checkpoint_dir, "{}_final_state.pth".format(filename)))

            print("--------------------- Model parameters is saved to disk ---------------------")

        elif save_object == "model":
            torch.save(self.model,
                       os.path.join(self.checkpoint_dir, "{}_final_state.pth".format(filename)))

        else:
            raise ValueError("Improper object type.")
