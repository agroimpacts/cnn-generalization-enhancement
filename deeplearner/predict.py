import os
from rasterio.io import MemoryFile
import numpy as np
import numpy.ma as ma
import boto3

from torch.autograd import Variable
import torch.nn.functional as F


def predict(predData, model, buffer, gpu, shrinkPixel):
    """
    Predict by tile

    Params:
        predData (''DataLoader''): Batch grouped data
        model: Trained model for prediction
        scorePath (str): Directory relative to s3 bucket to save probability maps
        predPath (str): Directory relative to s3 bucket to save hard prediction maps
        buffer (int): Buffer to cut out when writing chips
        gpu (binary,optional): Decide whether to use GPU, default is True

    """
    predData, meta, tile, year = predData
    meta.update({
        'dtype': 'int8'
    })

    model.eval()

    # create dummy tile
    canvas_score_ls = []

    for img, index_batch in predData:

        img = Variable(img, requires_grad=False)

        # GPU setting
        if gpu:
            img = img.cuda()

        out = F.softmax(model(img), 1)
        batch, nclass, height, width = out.size()
        chip_height = height - buffer * 2
        chip_width = width - buffer * 2
        max_index_0 = meta['height'] - chip_height
        max_index_1 = meta['width'] - chip_width

        # new by taking average
        for i in range(batch):
            index = (index_batch[0][i], index_batch[1][i])
            # only score here
            for n in range(nclass - 1):
                out_score = out[:, n + 1, (index[0] != 0) * buffer :
                                          (index[0] != 0) * buffer + chip_height + (index[0]==0 or index[0] == max_index_0) * buffer,
                            (index[1] != 0) * buffer:
                            (index[1] != 0) * buffer + chip_height + (index[1] == 0 or index[1] == max_index_1) * buffer].data[
                                i].cpu().numpy() * 100
                out_score = out_score.astype(meta['dtype'])
                score_height, score_width = out_score.shape

                try:
                    # if canvas_score_ls[n] exists
                    canvas_score_ls[n][index[0] + buffer * (index[0] != 0): index[0] + buffer * (index[0] != 0)+ score_height,
                        index[1]+ buffer * (index[1] != 0): index[1] + buffer * (index[1] != 0)+ score_width] = out_score

                except:
                    # create masked canvas_score_ls[n]
                    canvas_score = np.zeros((meta['height'] + buffer * 2, meta['width'] + buffer * 2), dtype=meta['dtype'])

                    canvas_score[index[0] + buffer * (index[0] != 0): index[0] + buffer * (index[0] != 0)+ score_height,
                        index[1]+ buffer * (index[1] != 0): index[1] + buffer * (index[1] != 0)+ score_width] = out_score
                    canvas_score_ls.append(canvas_score)


    for j in range(len(canvas_score_ls)):
        canvas_score_ls[j] = canvas_score_ls[j][shrinkPixel:meta['height'] + buffer * 2 -shrinkPixel, shrinkPixel:meta['width'] + buffer * 2 - shrinkPixel]

    return canvas_score_ls

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_predict(predData, model, num_mc_trials, buffer, gpu, shrinkPixel):
    """
    Monte Carlo Predict by tile
    Params:
        predData (''DataLoader''): Batch grouped data
        model: Trained model for prediction
        num_mc_trials (int): number of Monte carlo samples.
        scorePath (str): Directory relative to s3 bucket to save probability maps
        predPath (str): Directory relative to s3 bucket to save hard prediction maps
        buffer (int): Buffer to cut out when writing chips
        gpu (binary,optional): Decide whether to use GPU, default is True
    """
    predData, meta, tile, year = predData
    meta.update({
        'dtype': 'int8'
    })

    mc_canvas_score_ls=[]
    mc_score_dict = {}
    
    for mc_trial in range(num_mc_trials):
        
        model.eval()
        enable_dropout(model)
        
        canvas_score_ls = []
        
        for img, index_batch in predData:
            
            img = Variable(img, requires_grad=False)
            if gpu:
                img = img.cuda()

            out = F.softmax(model(img), 1)
            
            batch, nclass, height, width = out.size()
            chip_height = height - buffer * 2
            chip_width = width - buffer * 2
            max_index_0 = meta['height'] - chip_height
            max_index_1 = meta['width'] - chip_width
            
            # new by taking average
            for i in range(batch):
                index = (index_batch[0][i], index_batch[1][i])
               
                for n in range(nclass - 1):
                    out_score = out[:, n + 1, 
                                    (index[0] != 0) * buffer : (index[0] != 0) * buffer + chip_height + (index[0]==0 or index[0] == max_index_0) * buffer,
                                    (index[1] != 0) * buffer: (index[1] != 0) * buffer + chip_height + (index[1] == 0 or index[1] == max_index_1) 
                                    * buffer].data[i].cpu().numpy() * 100
                    out_score = out_score.astype(meta['dtype'])
                    #out_score = np.expand_dims(out_score, axis=0)
                    score_height, score_width = out_score.shape
                    
                    try:
                        # if canvas_score_ls[n] exists
                        canvas_score_ls[n][index[0] + buffer * (index[0] != 0): index[0] + buffer * (index[0] != 0)+ score_height,
                                           index[1]+ buffer * (index[1] != 0): index[1] + buffer * (index[1] != 0)+ score_width] = out_score
                    except:
                        # create masked canvas_score_ls[n]
                        canvas_score = np.zeros((meta['height'] + buffer * 2, meta['width'] + buffer * 2), dtype=meta['dtype'])

                        canvas_score[index[0] + buffer * (index[0] != 0): index[0] + buffer * (index[0] != 0)+ score_height,
                                     index[1]+ buffer * (index[1] != 0): index[1] + buffer * (index[1] != 0)+ score_width] = out_score
                        canvas_score_ls.append(canvas_score)
        
        for j in range(len(canvas_score_ls)):
            canvas_score_ls[j] = canvas_score_ls[j][shrinkPixel:meta['height'] + buffer * 2 -shrinkPixel, shrinkPixel:meta['width'] + buffer * 2 - shrinkPixel]
        
        mc_score_dict[mc_trial] = canvas_score_ls
    
    for i in range(len(list(mc_score_dict.values())[0])):
        mc_canvas_score_ls.append(np.concatenate([np.expand_dims(value_ls[i], 0) for value_ls in mc_score_dict.values()], 0))
    
    del mc_score_dict
    return mc_canvas_score_ls
