import easybar
import numpy as np 
from catalyst.data import BatchPrefetchLoaderWrapper

from torch.utils.data import DataLoader, Dataset

from mongoslabs.gencoords import CoordsGenerator
from mongoslabs.mongoloader import (
    create_client,
    collate_subcubes,
    mcollate,
    MBatchSampler,
    MongoDataset,
    MongoClient,
    mtransform,
)


import sys
sys.path.append('/data/users2/washbee/MeshVit') #change this path
import torch
import torch.nn as nn

from meshnet import enMesh_checkpoint, enMesh
import json

from segmenter.segm.model.decoder3d import MaskTransformer3d
from segmenter.segm.model.segmenter3d import Segmenter3d
from segmenter.segm.model.vit3d import VisionTransformer3d

from subvolume.utils import extract_subvolumes
import torch.nn.functional as F

import argparse
import os

from dice import *

filename = '/data/users2/washbee/MeshVit/experiments/hyperparamsearch_V100.log'

def log_hyperparams(subvolume_size, patch_size, n_layers, d_model, d_ff, n_heads, d_encoder, lr, modelsize, completed, msg,loss):
    # Check if the file exists
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            # Write the header
            file.write("subvolume_size,patch_size,n_layers,d_model,d_ff,n_heads,d_encoder,lr,modelsize,completed,msg,dice_val_avg\n")
    
    # Append the data
    with open(filename, 'a') as file:
        file.write(f"{subvolume_size},{patch_size},{n_layers},{d_model},{d_ff},{n_heads},{d_encoder},{lr},{modelsize},{completed},{msg},{loss}\n")


def get_model_memory_size(model):
    params = sum(p.numel() for p in model.parameters())
    tensors = [p for p in model.parameters()]
    
    float_size = 4  # for float32
    total_memory = sum([np.prod(t.size()) * float_size for t in tensors])
    return total_memory / (1024 ** 2)  # convert bytes to megabytes


parser = argparse.ArgumentParser(description='Hyperparameters for 3D Vision Transformer')
parser.add_argument('--subvolume_size', type=int, required=True)
parser.add_argument('--patch_size', type=int, required=True)
parser.add_argument('--n_layers', type=int, required=True)
parser.add_argument('--d_model', type=int, required=True)
parser.add_argument('--d_ff', type=int, required=True)
parser.add_argument('--n_heads', type=int, required=True)
parser.add_argument('--d_encoder', type=int, required=True)
parser.add_argument('--lr', type=float, required=True)

args = parser.parse_args()

# Replace original hyperparameter declarations with the following
subvolume_size = args.subvolume_size
patch_size = args.patch_size
n_layers = args.n_layers
d_model = args.d_model
d_ff = args.d_ff
n_heads = args.n_heads
d_encoder = args.d_encoder
lr = args.lr



volume_shape = [256]*3
subvolume_shape = [32]*3 # if you are sampling subcubes within the
                         # volume. Since you will have to generate patches for
                         # transformer, check implementation of the collate
                         # function and try to make yours such that it is
                         # effient in splitting the volume into patches your
                         # way. Just make it efficient as your utilization will
                         # be affected by your choices big time!

# The `sublabel` option selects the 104 DKT Atlas as the classification
# labels. The `gwmlabel` option offers a 3-class classification for gray
# matter, white matter, and non-brain regions. The `50label` option utilizes a
# carefully chosen pattern to fuse the labels of the DKT atlas across
# hemispheres.
LABELNOW=["sublabel", "gwmlabel", "50label"][0]
# The host that runs the database
MONGOHOST = "arctrdcn018.rs.gsu.edu"
# The name of our database containing neuroimaging data prepared for training
DBNAME = 'MindfulTensors'
# This is a very dirty but large collection of (T1 image, label cube) pairs
COLLECTION = 'MRNslabs'
# A much cleaner and thus less conducive to generalization dataset
#COLLECTION = "HCP"

batch_size = 1

# Do not modify the following block
INDEX_ID = "subject"
VIEWFIELDS = ["subdata", LABELNOW, "id", "subject"]
coord_generator = CoordsGenerator(volume_shape, subvolume_shape)
batched_subjs = 1

def createclient(x):
    return create_client(x, dbname=DBNAME,
                         colname=COLLECTION,
                         mongohost=MONGOHOST)

def mycollate_full(x):
    return mcollate(x, labelname=LABELNOW)

def mytransform(x):
    return mtransform(x, label=LABELNOW)


client = MongoClient("mongodb://" + MONGOHOST + ":27017")
db = client[DBNAME]
posts = db[COLLECTION]
num_examples = int(posts.find_one(sort=[(INDEX_ID, -1)])[INDEX_ID] + 1)

train_size = int(0.8 * num_examples)
valid_size = int(0.1 * num_examples)
test_size = num_examples - train_size - valid_size

indices = list(range(num_examples))
#np.random.shuffle(indices)  # Ensure you import numpy as np
print('!!!removed stochasticity')
temp_train_size = int(.001*num_examples)
temp_valid_size = int(.001*num_examples)
#train_indices = indices[:train_size]
train_indices = indices[:temp_train_size]#change this
print ("!!!change the above")
valid_indices = indices[train_size:(train_size + temp_valid_size)] ##change back from temp!!!
print("change back from temp!!!")
test_indices = indices[(train_size + valid_size):]

# For the training set:
train_dataset = MongoDataset(
    train_indices,     
    mytransform,
    None,
    id=INDEX_ID,
    fields=VIEWFIELDS,
    )
train_sampler = MBatchSampler(train_dataset, batch_size=1)
train_dataloader = BatchPrefetchLoaderWrapper(
    DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=mycollate_full, # always fetched full brains without dicing
        pin_memory=True,
        worker_init_fn=createclient,
        num_workers=8, # remember your cores and memory are limited, do not
                       # make this parameter more than you have cores and
                       # larger than num_prefetches does not make sense either
        ),
     num_prefetches=16 # you GPU memory may limit this. This is the number of
                       # brains that will be fetched to the GPU while it is
                       # still busy with compute of a previous batch
     )
valid_dataset = MongoDataset(
    valid_indices,     
    mytransform,
    None,
    id=INDEX_ID,
    fields=VIEWFIELDS,
    )
valid_sampler = MBatchSampler(valid_dataset, batch_size=1)
valid_dataloader = BatchPrefetchLoaderWrapper(
    DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        collate_fn=mycollate_full, # always fetched full brains without dicing
        pin_memory=True,
        worker_init_fn=createclient,
        num_workers=8, # remember your cores and memory are limited, do not
                       # make this parameter more than you have cores and
                       # larger than num_prefetches does not make sense either
        ),
     num_prefetches=16 # you GPU memory may limit this. This is the number of
                       # brains that will be fetched to the GPU while it is
                       # still busy with compute of a previous batch
     )

test_dataset = MongoDataset(
    test_indices,     
    mytransform,
    None,
    id=INDEX_ID,
    fields=VIEWFIELDS,
    )
test_sampler = MBatchSampler(test_dataset, batch_size=1)
test_dataloader = BatchPrefetchLoaderWrapper(
    DataLoader(
        test_dataset,
        sampler=test_sampler,
        collate_fn=mycollate_full, # always fetched full brains without dicing
        pin_memory=True,
        worker_init_fn=createclient,
        num_workers=8, # remember your cores and memory are limited, do not
                       # make this parameter more than you have cores and
                       # larger than num_prefetches does not make sense either
        ),
     num_prefetches=16 # you GPU memory may limit this. This is the number of
                       # brains that will be fetched to the GPU while it is
                       # still busy with compute of a previous batch
     )


# tdataset = MongoDataset(
#     range(num_examples),
#     mytransform,
#     None,
#     id=INDEX_ID,
#     fields=VIEWFIELDS,
#     )
# tsampler = (
#     MBatchSampler(tdataset, batch_size=1)
#     )
# tdataloader = BatchPrefetchLoaderWrapper(
#     DataLoader(
#         tdataset,
#         sampler=tsampler,
#         collate_fn=mycollate_full, # always fetched full brains without dicing
#         pin_memory=True,
#         worker_init_fn=createclient,
#         num_workers=8, # remember your cores and memory are limited, do not
#                        # make this parameter more than you have cores and
#                        # larger than num_prefetches does not make sense either
#         ),
#      num_prefetches=16 # you GPU memory may limit this. This is the number of
#                        # brains that will be fetched to the GPU while it is
#                        # still busy with compute of a previous batch
#      )

config_file = 'modelAE.json'

device = "cuda"
#now as args above:
#subvolume_size = 64
image_size = (subvolume_size,subvolume_size,subvolume_size)
#patch_size = 16 
#n_layers = 12
#d_model = 128
#d_ff = 128
#n_heads = 8
n_cls = 104 #per voxel number of classes
                         #image_size, patch_size, n_layers, d_model, d_ff, n_heads, n_cls,
vit = VisionTransformer3d(image_size, patch_size, n_layers, d_model, d_ff, n_heads, n_cls, 
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=1)

#d_encoder = 128
#...
drop_path_rate=0.0
dropout=0.1
loss = None
completed = False
modelsize = None
try:
    #                           n_cls,patch_size,d_encoder,n_layers,n_heads,d_model,d_ff,drop_path_rate,dropout,
    decoder = MaskTransformer3d(n_cls, patch_size,d_encoder, n_layers, n_heads, d_model, d_ff, drop_path_rate, dropout)
    model = Segmenter3d(vit, decoder, n_cls=n_cls).to(device)
    modelsize = get_model_memory_size(model)
    print("modelsize is ", modelsize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  # for segmentation tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_epochs = 1  # or however many you want

    model.train()  # set the model to training mode

    coordinates = coord_generator.get_coordinates(mode="train")
    train_dataloader = test_dataloader##remove this later!!!!!!!!!!!. 
    print('remove the above line!!!')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        for i, (x, y) in enumerate(train_dataloader):
            # Move data to GPU if available
            x, y = x.cuda(), y.cuda()
            
            

            subvolumesx = extract_subvolumes(x, subvolume_size)
            subvolumesy = extract_subvolumes(y, subvolume_size)
            # Forward pass
            for subvolx,subvoly in zip(subvolumesx,subvolumesy):
                
                outputs = model(subvolx)

                #print('compare',outputs.shape,subvoly.long().shape, y.shape)
                loss = criterion(outputs, subvoly.long())  # Ensure the target tensor is of type long
                # Print some statistics
                #if (i + 1) % 10 == 0:  # print every 10 batches for example
                print(f"Batch {i+1}, Loss: {loss.item()}")
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            #exit()
            # Print some statistics
            if (i + 1) % 10 == 0:  # print every 10 batches for example
                print(f"Batch {i+1}, Loss: {loss.item()}")
                print(x.shape,y.shape)
                
            easybar.print_progress(i, len(train_dataloader))
    
    validcount = 0.0
    validDicetotal=0.0
    model = model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_dataloader):
                # Move data to GPU if available
                x, y = x.cuda(), y.cuda()
                subvolumesx = extract_subvolumes(x, subvolume_size)
                subvolumesy = extract_subvolumes(y, subvolume_size)
                # Forward pass
                for subvolx,subvoly in zip(subvolumesx,subvolumesy):
                    
                    outputs = model(subvolx)
                    validcount+=1.0
                    #loss = criterion(outputs, subvoly.long())  # Ensure the target tensor is of type long
                    #validlosstotal+=loss.item()
                    predicted = torch.argmax(outputs,dim=1)
                    validDicetotal += torch.mean(faster_dice(predicted, subvoly.long(), range(104))).item()
                    
                # Print some statistics
                if (i + 1) % 10 == 0:  # print every 10 batches for example
                    print(f"Validation Batch {i+1}, Loss: {loss.item()}")
                    print(x.shape,y.shape)
                    
                easybar.print_progress(i, len(valid_dataloader))

        log_hyperparams(subvolume_size, patch_size, n_layers, d_model, d_ff, n_heads, d_encoder, lr, modelsize, True, "Completed",validDicetotal/validcount)
        completed = True

except torch.cuda.OutOfMemoryError as e:
    print("torch.cuda.OutOfMemoryError caught")
except Exception as e:
    print('exception caught')
except BaseException as e:
    print('base exception caught')
finally:
    print("finally")
    if not completed:
        log_hyperparams(subvolume_size, patch_size, n_layers, d_model, d_ff, n_heads, d_encoder, lr, modelsize, False, "Not Completed","NA")
        
    exit()