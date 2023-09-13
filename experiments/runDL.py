from typing import List
import argparse
import collections
from collections import OrderedDict

from brain_dataset import BrainDataset
from catalyst import metrics
from catalyst.callbacks import CheckpointCallback
from catalyst.contrib.utils.pandas import dataframe_to_list
from catalyst.data import BatchPrefetchLoaderWrapper, ReaderCompose
from catalyst.dl import Runner
from catalyst.metrics.functional._segmentation import dice
from model import MeshNet, UNet
import nibabel as nib
import numpy as np
import pandas as pd
from reader import TensorFixedVolumeNiftiReader, TensorNiftiReader
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from mongoutil import *

open_fn = ReaderCompose(
        [
            TensorFixedVolumeNiftiReader(input_key="images", output_key="images"),
            TensorNiftiReader(input_key="nii_labels", output_key="targets"),
        ]
    )
    
train_data, valid_data, test_data = getMongoDataset()

print(train_data[[0]][7]['subdata'].shape)#subject, subvolume
# train_data = BrainDataset(data=train_data,
#                 list_shape=[256,256,256],
#                 list_sub_shape=[128,128,128],
#                 open_fn=open_fn,
#                 n_subvolumes=8,#n_subvolumes,
#                 mode="train",
#                 input_key="images",
#                 output_key="targets",
#             )
# print(train_data[[0]][7])#subject, subvolume


# def worker_init_fn(worker_id):
#         np.random.seed(np.random.get_state()[1][0] + worker_id)


#train_loader = DataLoader(
#        dataset=train_data,
#        batch_size=1,
#        shuffle=True,
#        worker_init_fn=worker_init_fn,
#         num_workers=16,
#         pin_memory=True,
# )

#train_loader = BatchPrefetchLoaderWrapper(train_loader,
#     num_prefetches=16 )

# print(train_loader[0])