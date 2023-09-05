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

config_file = 'modelAE.json'

device = "cuda"
image_size = (32,32,32)
patch_size = 4 
n_layers = 12
d_model = 128
d_ff = 128
n_heads = 8
n_cls = 104 #per voxel number of classes
vit = VisionTransformer3d(image_size, patch_size, n_layers, d_model, d_ff, n_heads, n_cls, 
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=1)

d_encoder = 128
#n_layers,
#n_heads,
#d_model=128
#d_ff=128
drop_path_rate=0.0
dropout=0.1
decoder = MaskTransformer3d(n_cls, patch_size,d_encoder, n_layers, n_heads, d_model, d_ff, drop_path_rate, dropout)
model = Segmenter3d(vit, decoder, n_cls=n_cls).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()  # for segmentation tasks
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1  # or however many you want

model.train()  # set the model to training mode

coordinates = coord_generator.get_coordinates(mode="train")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    for i, (x, y) in enumerate(tdataloader):
        # Move data to GPU if available
        x, y = x.cuda(), y.cuda()
        
        subvolume_size = 32

        subvolumesx = extract_subvolumes(x, subvolume_size)
        subvolumesy = extract_subvolumes(y, subvolume_size)
        # Forward pass
        for subvolx,subvoly in zip(subvolumesx,subvolumesy):
            
            outputs = model(subvolx)

            # Compute loss
            print('min max', torch.min(x),torch.max(x),torch.min(y),torch.max(y),torch.min(outputs), torch.max(outputs),torch.min(subvolx),torch.max(subvolx), torch.min(subvoly), torch.max(subvoly))
            print('minsubx, maxsubx, minout, maxout', torch.min(subvolx), torch.max(subvolx), torch.min(outputs), torch.max(outputs))

            print('compare',outputs.shape,subvoly.long().shape, y.shape)

            indices = torch.nonzero(outputs > 0)
            indicessubx = torch.nonzero(subvolx > 0)
            print(f"Size of indicessubx: {indicessubx.size(0)}")
            print(f"Size of indices: {indices.size(0)}")
            if indices.size(0)>0:
                print("!!!")


            #exit()
            #loss = criterion(outputs, subvoly.long())  # Ensure the target tensor is of type long

            # Print some statistics
            #if (i + 1) % 10 == 0:  # print every 10 batches for example
            #print(f"Batch {i+1}, Loss: {loss.item()}")
            print('subvx,subvy', subvolx.shape,subvoly.shape)
            #exit()
            # Backward pass and optimization
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()


        exit()
        # Print some statistics
        if (i + 1) % 10 == 0:  # print every 10 batches for example
            print(f"Batch {i+1}, Loss: {loss.item()}")
            print(x.shape,y.shape)
            
        easybar.print_progress(i, len(tdataloader))
