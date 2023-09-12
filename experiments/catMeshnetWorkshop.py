from typing import List
import argparse
import collections
from collections import OrderedDict
from custom_checkpoint_callback import CustomCheckpointCallback
from brain_dataset import BrainDataset
from catalyst import metrics
from catalyst.callbacks import CheckpointCallback
from catalyst.contrib.utils.pandas import dataframe_to_list
from catalyst.data import BatchPrefetchLoaderWrapper, ReaderCompose
from catalyst.dl import Runner, SchedulerCallback

from catalyst.metrics.functional._segmentation import dice
from model import MeshNet, UNet
import nibabel as nib
import numpy as np
import pandas as pd
from reader import TensorFixedVolumeNiftiReader, TensorNiftiReader
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from mongoutil import *
import torch.nn as nn
from mongoutil import MongoDataLoader

import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # assuming you want to use GPU 0

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

    
args = None




# Create an instance of MongoDataLoader with your desired labelnow_choice
loader = MongoDataLoader(labelnow_choice=1)  # Change labelnow_choice as needed



# Now you can use train_loader, valid_loader, and test_loader as needed


def get_loaders(
    random_state: int,
    volume_shape: List[int],
    subvolume_shape: List[int],
    train_subvolumes: int,
    infer_subvolumes: int,
    batch_size: int,  # Add a batch_size parameter with a default value
    num_workers: int,
) -> dict:
    """Get Dataloaders"""
    
    train_loaders = collections.OrderedDict()
    infer_loaders = collections.OrderedDict()
    
    # Call the get_mongo_loaders method to get the DataLoader instances with the specified batch size
    train_loader, valid_loader, test_loader = loader.get_mongo_loaders(batch_size=batch_size, num_workers=num_workers)
    
    
    train_loaders["train"] = BatchPrefetchLoaderWrapper(train_loader,
     num_prefetches=4 )# you GPU memory may limit this. This is the number of
                       # brains that will be fetched to the GPU while it is
                       # still busy with compute of a previous batch
                       
    train_loaders["valid"] = BatchPrefetchLoaderWrapper(valid_loader,
     num_prefetches=4 )# you GPU memory may limit this. This is the number of
                       # brains that will be fetched to the GPU while it is
                       # still busy with compute of a previous batch
    infer_loaders["infer"] = BatchPrefetchLoaderWrapper(test_loader,
     num_prefetches=4 )# you GPU memory may limit this. This is the number of
                       # brains that will be fetched to the GPU while it is
                       # still busy with compute of a previous batch

    return train_loaders, infer_loaders


class CustomRunner(Runner):
    """Custom Runner for demonstrating a NeuroImaging Pipeline"""

    def __init__(self, n_classes: int, coords_generator: CoordsGenerator, batch_size: int):
        """Init."""
        super().__init__()
        self.n_classes = n_classes
        self.epoch = 0
        self.coords_generator = coords_generator

        self.criterion = nn.CrossEntropyLoss()  # for segmentation tasks
        self.batch_size = batch_size  # Store the batch size

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        self._loaders = self._loaders
        return self._loaders

    def predict_batch(self, batch):
        """
        Predicts a batch for an inference dataloader and returns the
        predictions as well as the corresponding slice indices
        """
        batch = batch[0]
        return (
            self.model(batch["images"].float().to(self.device)),
            batch["coords"],
        )

    def on_epoch_end(self, runner):
        """
        Calls scheduler step after an epoch ends using validation dice score
        """
        if runner.loader_key == "valid":  # Checking if it's the validation phase
            print('validation phase and scheduler step update')
            dice_score = self.loader_metrics.get('macro_dice', None)
            print('validation dice_score', dice_score)
            if dice_score is not None:
                self.scheduler.step(dice_score)
            super().on_epoch_end(runner)


    def on_loader_start(self, runner):
        """
        Calls runner methods when the dataloader begins and adds
        metrics for loss and macro_dice
        """
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in ["loss", "macro_dice"]
        }

    def handle_batch(self, batch):
        """
        Custom train/ val step that includes batch unpacking, training, and
        DICE metrics
        """
        if self.criterion == None:
            self.criterion = nn.CrossEntropyLoss()
        
        # Modify the batch size to use the one passed from get_loaders
        x, y = batch  # Assuming that the batch contains a tuple (x, y)
        x = MongoDataLoader.extract_subvolumes(x, self.coords_generator)
        y = MongoDataLoader.extract_label_subvolumes(y, self.coords_generator).long()

        for x, y in zip(x, y):
            if self.is_train_loader:
                self.optimizer.zero_grad()

            y_hat = self.model(x.cuda())
            loss = self.criterion(y_hat.cuda(), y.cuda())

            one_hot_targets = (
                torch.nn.functional.one_hot(y, self.n_classes).permute(0, 4, 1, 2, 3).cuda()
            )

            if self.is_train_loader:
                loss.backward()
                self.optimizer.step()

            logits_softmax = F.softmax(y_hat)
            macro_dice = dice(logits_softmax, one_hot_targets, mode="macro")

            self.batch_metrics.update({"loss": loss, "macro_dice": macro_dice})

            for key in ["loss", "macro_dice"]:
                self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

    def on_loader_end(self, runner):
        """
        Calls runner methods when a dataloader finishes running and updates
        metrics
        """
        for key in ["loss", "macro_dice"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)


def voxel_majority_predict_from_subvolumes(loader, n_classes, segmentations=None):
    """
    Predicts Brain Segmentations given a dataloader class and a optional dict
    to contain the outputs. Returns a dict of brain segmentations.
    """
    if segmentations is None:
        for subject in range(loader.dataset.subjects):
            segmentations[subject] = torch.zeros(
                tuple(np.insert(loader.volume_shape, 0, n_classes)), dtype=torch.uint8,
            ).cpu()

    prediction_n = 0
    for inference in tqdm(runner.predict_loader(loader=loader)):
        coords = inference[1].cpu()
        _, predicted = torch.max(F.log_softmax(inference[0].cpu(), dim=1), 1)
        for j in range(predicted.shape[0]):
            c_j = coords[j][0]
            subj_id = prediction_n // loader.dataset.n_subvolumes
            for c in range(n_classes):
                segmentations[subj_id][
                    c, c_j[0, 0] : c_j[0, 1], c_j[1, 0] : c_j[1, 1], c_j[2, 0] : c_j[2, 1],
                ] += (predicted[j] == c)
            prediction_n += 1

    for i in segmentations.keys():
        segmentations[i] = torch.max(segmentations[i], 0)[1]
    return segmentations


def get_model_memory_size(model):
    params = sum(p.numel() for p in model.parameters())
    tensors = [p for p in model.parameters()]
    
    float_size = 4  # for float32
    total_memory = sum([np.prod(t.size()) * float_size for t in tensors])
    return total_memory / (1024 ** 2)  # convert bytes to megabytes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T1 segmentation Training")
    parser.add_argument("--n_classes", default=3, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--train_subvolumes",
        default=64,
        type=int,
        metavar="N",
        help="Number of total subvolumes to sample from one brain",
    )
    parser.add_argument(
        "--infer_subvolumes",
        default=256,
        type=int,
        metavar="N",
        help="Number of total subvolumes to sample from one brain",
    )
    parser.add_argument("--sv_w", default=38, type=int, metavar="N", help="Width of subvolumes")
    parser.add_argument(
        "--sv_h", default=38, type=int, metavar="N", help="Height of subvolumes",
    )
    parser.add_argument("--sv_d", default=38, type=int, metavar="N", help="Depth of subvolumes")
    parser.add_argument("--model", default="meshnet")
    parser.add_argument(
        "--dropout", default=0.1, type=float, metavar="N", help="dropout probability for meshnet",
    )
    parser.add_argument("--large", default=False)
    parser.add_argument(
        "--n_epochs", default=5, type=int, metavar="N", help="number of total epochs to run",
    )
    parser.add_argument('--subvolume_size', type=int, required=True)
    parser.add_argument('--patch_size', type=int, required=True)
    parser.add_argument('--n_layers', type=int, required=True)
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--d_ff', type=int, required=True)
    parser.add_argument('--n_heads', type=int, required=True)
    parser.add_argument('--d_encoder', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)

    
    args = parser.parse_args()
    print("{}".format(args))

    volume_shape = [256, 256, 256]
    #subvolume_shape = [args.sv_h, args.sv_w, args.sv_d]
    subvolume_shape = [args.train_subvolumes, args.train_subvolumes, args.train_subvolumes]
    train_loaders, infer_loaders = get_loaders(
        0,
        volume_shape,
        subvolume_shape,
        args.train_subvolumes,
        args.infer_subvolumes,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    if args.model == "meshnet":
        print('creating meshnet.')
        net = MeshNet(
            n_channels=1, n_classes=args.n_classes, large=args.large, dropout_p=args.dropout,
        ).cuda()
    else:
        print('creating unet')
        net = UNet(n_channels=1, n_classes=args.n_classes).cuda()

    logdir = "logs/{model}_gmwm".format(model=args.model)
    if args.large:
        logdir += "_large"

    if args.dropout:
        logdir += "_dropout"

    logdir+= "_sv{sv}".format(sv=args.train_subvolumes)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)


    runner = CustomRunner(n_classes=args.n_classes, 
            coords_generator = CoordsGenerator(
                list_shape=[256, 256, 256],
                list_sub_shape=[args.train_subvolumes, args.train_subvolumes, args.train_subvolumes]),
                batch_size=args.batch_size
            )
    runner.train(
        model=net,
        optimizer=optimizer,
        loaders=train_loaders,
        num_epochs=args.n_epochs,
        scheduler=scheduler,
        callbacks=[CustomCheckpointCallback(
            metric_name="macro_dice", 
            minimize_metric=False, 
            save_n_best=5, 
            logdir=logdir)
                   ],
        logdir=logdir,
        verbose=True,
    )
