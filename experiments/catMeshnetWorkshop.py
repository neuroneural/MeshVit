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

args = None




def get_loaders(
    random_state: int,
    volume_shape: List[int],
    subvolume_shape: List[int],
    train_subvolumes: int = 128,
    infer_subvolumes: int = 256,
    in_csv_train: str = None,
    in_csv_valid: str = None,
    in_csv_infer: str = None,
    batch_size: int = 16,
    num_workers: int = 10,
) -> dict:
    """Get Dataloaders"""
    
    
    train_loaders = collections.OrderedDict()
    infer_loaders = collections.OrderedDict()
    
    train_loader, valid_loader, test_loader = getMongoLoaders()
    
    
    
    train_loaders["train"] = BatchPrefetchLoaderWrapper(train_loader,
     num_prefetches=16 )# you GPU memory may limit this. This is the number of
                       # brains that will be fetched to the GPU while it is
                       # still busy with compute of a previous batch
                       
    train_loaders["valid"] = BatchPrefetchLoaderWrapper(valid_loader,
     num_prefetches=16 )# you GPU memory may limit this. This is the number of
                       # brains that will be fetched to the GPU while it is
                       # still busy with compute of a previous batch
    infer_loaders["infer"] = BatchPrefetchLoaderWrapper(test_loader,
     num_prefetches=16 )# you GPU memory may limit this. This is the number of
                       # brains that will be fetched to the GPU while it is
                       # still busy with compute of a previous batch

    return train_loaders, infer_loaders


class CustomRunner(Runner):
    """Custom Runner for demonstrating a NeuroImaging Pipeline"""

    def __init__(self, n_classes: int):
        """Init."""
        super().__init__()
        self.n_classes = n_classes

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        self._loaders = self._loaders
        return self._loaders

    def predict_batch(self, batch):
        """
        Predicts a batch for an inference dataloader and returns the
        predictions as well as the corresponding slice indices
        """
        import logging
        
        logging.basicConfig(filename='debug.log', level=logging.DEBUG)
        logging.debug(f'batch: {batch}, type: {type(batch)}')
        
        # model inference step
        batch = batch[0]
        return (
            self.model(batch["images"].float().to(self.device)),
            batch["coords"],
        )

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
        
        # model train/valid step
        #batch = batch[0]
        x, y = batch#batch["images"].float(), batch["targets"]

        if self.is_train_loader:
            self.optimizer.zero_grad()

        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            scheduler.step()

        one_hot_targets = (
            torch.nn.functional.one_hot(y, self.n_classes).permute(0, 4, 1, 2, 3).cuda()
        )

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
    # parser.add_argument(
    #     "--train_path",
    #     metavar="PATH",
    #     default="./data/dataset_train.csv",
    #     help="Path to list with brains for training",
    # )
    # parser.add_argument(
    #     "--validation_path",
    #     metavar="PATH",
    #     default="./data/dataset_valid.csv",
    #     help="Path to list with brains for validation",
    # )
    # parser.add_argument(
    #     "--inference_path",
    #     metavar="PATH",
    #     default="./data/dataset_infer.csv",
    #     help="Path to list with brains for inference",
    # )
    parser.add_argument("--n_classes", default=31, type=int)
    parser.add_argument(
        "--train_subvolumes",
        default=128,
        type=int,
        metavar="N",
        help="Number of total subvolumes to sample from one brain",
    )
    parser.add_argument(
        "--infer_subvolumes",
        default=512,
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
        "--dropout", default=0, type=float, metavar="N", help="dropout probability for meshnet",
    )
    parser.add_argument("--large", default=False)
    parser.add_argument(
        "--n_epochs", default=30, type=int, metavar="N", help="number of total epochs to run",
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
    subvolume_shape = [128, 128, 128]
    train_loaders, infer_loaders = get_loaders(
        0,
        volume_shape,
        subvolume_shape,
        args.train_subvolumes,
        args.infer_subvolumes,
        "",
        "",
        "",#TODO: CLEANUP 
    )

    if args.model == "meshnet":
        net = MeshNet(
            n_channels=1, n_classes=args.n_classes, large=args.large, dropout_p=args.dropout,
        )
    else:
        net = UNet(n_channels=1, n_classes=args.n_classes)

    logdir = "logs/{model}_gmwm".format(model=args.model)
    if args.large:
        logdir += "_large"

    if args.dropout:
        logdir += "_dropout"

    optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
    scheduler = OneCycleLR(
        optimizer, max_lr=0.02, epochs=args.n_epochs, steps_per_epoch=len(train_loaders["train"]),
    )

    runner = CustomRunner(n_classes=args.n_classes)
    runner.train(
        model=net,
        optimizer=optimizer,
        loaders=train_loaders,
        num_epochs=args.n_epochs,
        scheduler=scheduler,
        callbacks=[CheckpointCallback(logdir=logdir)],
        logdir=logdir,
        verbose=True,
    )

    segmentations = {}
    for subject in range(infer_loaders["infer"].dataset.subjects):
        segmentations[subject] = torch.zeros(
            tuple(np.insert(volume_shape, 0, args.n_classes)), dtype=torch.uint8,
        )

    segmentations = voxel_majority_predict_from_subvolumes(
        infer_loaders["infer"], args.n_classes, segmentations
    )
    subject_metrics = []
    for subject, subject_data in enumerate(tqdm(infer_loaders["infer"].dataset.data)):
        seg_labels = nib.load(subject_data["nii_labels"]).get_fdata()
        segmentation_labels = torch.nn.functional.one_hot(
            torch.from_numpy(seg_labels).to(torch.int64), args.n_classes
        )

        inference_dice = (
            dice(
                torch.nn.functional.one_hot(segmentations[subject], args.n_classes).permute(
                    0, 3, 1, 2
                ),
                segmentation_labels.permute(0, 3, 1, 2),
            )
            .detach()
            .numpy()
        )
        macro_inference_dice = (
            dice(
                torch.nn.functional.one_hot(segmentations[subject], args.n_classes).permute(
                    0, 3, 1, 2
                ),
                segmentation_labels.permute(0, 3, 1, 2),
                mode="macro",
            )
            .detach()
            .numpy()
        )
        subject_metrics.append((inference_dice, macro_inference_dice))

    per_class_df = pd.DataFrame([metric[0] for metric in subject_metrics])
    macro_df = pd.DataFrame([metric[1] for metric in subject_metrics])
    print(per_class_df, macro_df)
