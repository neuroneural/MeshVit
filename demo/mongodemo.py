import easybar

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

tdataset = MongoDataset(
    range(num_examples),
    mytransform,
    None,
    id=INDEX_ID,
    fields=VIEWFIELDS,
    )
tsampler = (
    MBatchSampler(tdataset, batch_size=1)
    )
tdataloader = BatchPrefetchLoaderWrapper(
    DataLoader(
        tdataset,
        sampler=tsampler,
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

import torch
import torch.nn as nn


import unet
model = unet.UNet3D(
    in_channels=1, # 1 input channel (assuming grayscale MRI)
    out_classes=104, # Assuming binary segmentation for simplicity
    dimensions=3, # 3D
    num_encoding_blocks=2, # Absolute minimal depth
    out_channels_first_layer=4, # Absolute minimal number of channels
    normalization=None, # No normalization
    padding=1
)




# Instantiate
#model = UNet3D(in_channels=1, out_channels=104)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



criterion = nn.CrossEntropyLoss()  # for segmentation tasks
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 1  # or however many you want

model.train()  # set the model to training mode
import torch.nn.functional as F

def downsample_volume(volume, factor=4):
    # Calculate the new size
    new_size = (64,64,64)
    
    # Use trilinear interpolation to downsample
    downsampled = F.interpolate(volume, size=new_size, mode='trilinear', align_corners=False)
    
    return downsampled

import torch.nn.functional as F

def upsample_volume(volume, output_size=(256,256,256)):
    # Use trilinear interpolation to upsample
    upsampled = F.interpolate(volume, size=output_size, mode='trilinear', align_corners=False)
    
    return upsampled



for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    for i, (x, y) in enumerate(tdataloader):
        # Move data to GPU if available
        x = downsample_volume(x)
        y = y.float()
        y =y.unsqueeze(1)
        y = downsample_volume(y)
        y = y.squeeze(1)
        x, y = x.cuda(), y.cuda()
        
        # Forward pass
        outputs = model(x)

        # Compute loss
        loss = criterion(outputs, y.long())  # Ensure the target tensor is of type long

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print some statistics
        if (i + 1) % 10 == 0:  # print every 10 batches for example
            print(f"Batch {i+1}, Loss: {loss.item()}")
            print(x.shape,y.shape)
            x = upsample_volume(x)
            y =y.unsqueeze(1)
            y = upsample_volume(y)
            y = y.squeeze(1)
            print(x.shape,y.shape)

        easybar.print_progress(i, len(tdataloader))
