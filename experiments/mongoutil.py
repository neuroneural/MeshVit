from torch.utils.data import DataLoader

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


# Do not modify the following block
INDEX_ID = "subject"
VIEWFIELDS = ["subdata", LABELNOW, "id", "subject"]



def createclient(x):
    global DBNAME
    global COLLECTION
    global MONGOHOST
    return create_client(x, dbname=DBNAME,
                         colname=COLLECTION,
                         mongohost=MONGOHOST)

def mycollate_full(x):
    global LABELNOW
    return mcollate(x, labelname=LABELNOW)

def mytransform(x):
    global LABELNOW
    return mtransform(x, label=LABELNOW)


#def getMongoLoaders(subvolume_size,patch_size,n_layers,d_model,d_ff,n_heads,d_encoder,lr):
def getMongoLoaders():
    

    # volume_shape = [256]*3
    # subvolume_shape = [32]*3 # if you are sampling subcubes within the
    #                         # volume. Since you will have to generate patches for
    #                         # transformer, check implementation of the collate
    #                         # function and try to make yours such that it is
    #                         # effient in splitting the volume into patches your
    #                         # way. Just make it efficient as your utilization will
    #                         # be affected by your choices big time!
    global LABELNOW
    global MONGOHOST
    global DBNAME
    global COLLECTION
    global INDEX_ID
    global VIEWFIELDS
    batch_size = 1

    batched_subjs = 1
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
    train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            collate_fn=mycollate_full, # always fetched full brains without dicing
            pin_memory=True,
            worker_init_fn=createclient,
            num_workers=4, # remember your cores and memory are limited, do not
                        # make this parameter more than you have cores and
                        # larger than num_prefetches does not make sense either
            )
    valid_dataset = MongoDataset(
        valid_indices,     
        mytransform,
        None,
        id=INDEX_ID,
        fields=VIEWFIELDS,
        )
    valid_sampler = MBatchSampler(valid_dataset, batch_size=1)
    valid_loader = DataLoader(
            valid_dataset,
            sampler=valid_sampler,
            collate_fn=mycollate_full, # always fetched full brains without dicing
            pin_memory=True,
            worker_init_fn=createclient,
            num_workers=4, # remember your cores and memory are limited, do not
                        # make this parameter more than you have cores and
                        # larger than num_prefetches does not make sense either
            )

    test_dataset = MongoDataset(
        test_indices,     
        mytransform,
        None,
        id=INDEX_ID,
        fields=VIEWFIELDS,
        )
    test_sampler = MBatchSampler(test_dataset, batch_size=1)
    test_loader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            collate_fn=mycollate_full, # always fetched full brains without dicing
            pin_memory=True,
            worker_init_fn=createclient,
            num_workers=4, # remember your cores and memory are limited, do not
                        # make this parameter more than you have cores and
                        # larger than num_prefetches does not make sense either
            )
    return train_loader, valid_loader, test_loader
##############################

def getMongoDataset():
    


    # volume_shape = [256]*3
    # subvolume_shape = [32]*3 # if you are sampling subcubes within the
    #                         # volume. Since you will have to generate patches for
    #                         # transformer, check implementation of the collate
    #                         # function and try to make yours such that it is
    #                         # effient in splitting the volume into patches your
    #                         # way. Just make it efficient as your utilization will
    #                         # be affected by your choices big time!

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
    batched_subjs = 1
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
        posts,
        id=INDEX_ID,
        fields=VIEWFIELDS,
        )
    # train_sampler = MBatchSampler(train_dataset, batch_size=1)
    # train_loader = DataLoader(
    #         train_dataset,
    #         sampler=train_sampler,
    #         collate_fn=mycollate_full, # always fetched full brains without dicing
    #         pin_memory=True,
    #         worker_init_fn=createclient,
    #         num_workers=4, # remember your cores and memory are limited, do not
    #                     # make this parameter more than you have cores and
    #                     # larger than num_prefetches does not make sense either
    #         )
    valid_dataset = MongoDataset(
        valid_indices,     
        mytransform,
        posts,
        id=INDEX_ID,
        fields=VIEWFIELDS,
        )
    # valid_sampler = MBatchSampler(valid_dataset, batch_size=1)
    # valid_loader = DataLoader(
    #         valid_dataset,
    #         sampler=valid_sampler,
    #         collate_fn=mycollate_full, # always fetched full brains without dicing
    #         pin_memory=True,
    #         worker_init_fn=createclient,
    #         num_workers=4, # remember your cores and memory are limited, do not
    #                     # make this parameter more than you have cores and
    #                     # larger than num_prefetches does not make sense either
    #         )

    test_dataset = MongoDataset(
        test_indices,     
        mytransform,
        posts,
        id=INDEX_ID,
        fields=VIEWFIELDS,
        )
    # test_sampler = MBatchSampler(test_dataset, batch_size=1)
    # test_loader = DataLoader(
    #         test_dataset,
    #         sampler=test_sampler,
    #         collate_fn=mycollate_full, # always fetched full brains without dicing
    #         pin_memory=True,
    #         worker_init_fn=createclient,
    #         num_workers=4, # remember your cores and memory are limited, do not
    #                     # make this parameter more than you have cores and
    #                     # larger than num_prefetches does not make sense either
    #         )
    return train_dataset, valid_dataset, test_dataset


import torch

def extract_subvolumes(tensor: torch.Tensor, coord_generator: CoordsGenerator) -> torch.Tensor:
    """
    Extracts subvolumes from a tensor using a coordinate generator.

    Args:
    - tensor: A tensor from which to extract subvolumes.
    - coord_generator: An instance of the CoordsGenerator class for generating coordinates.

    Returns:
    - subvolumes: A tensor containing the extracted subvolumes.
    """
    tensor = tensor.cpu()
    
    # Get coordinates for non-overlapping subvolumes
    
    coords = coord_generator.get_coordinates(mode="test")  # Use "test" mode to get non-overlapping subvolumes

    # Extract subvolumes based on the generated coordinates
    subvolumes_list = []
    for (z_start, z_end), (y_start, y_end), (x_start, x_end) in coords:
        subvolume = tensor[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
        subvolumes_list.append(subvolume.unsqueeze(0))

    # Stack the subvolumes to get the final tensor
    subvolumes = torch.cat(subvolumes_list, dim=0)

    return subvolumes

# # Test
# print('hi')
# tensor = torch.randn(1, 1, 256, 256, 256)
# coord_generator = CoordsGenerator(list_shape=[256, 256, 256], list_sub_shape=[128, 128, 128])
# subvolumes = extract_subvolumes(tensor, coord_generator)
# print(subvolumes.shape)  # Expected output: torch.Size([8, 1, 128, 128, 128])

import torch

def extract_label_subvolumes(tensor: torch.Tensor, coord_generator: CoordsGenerator) -> torch.Tensor:
    """Extracts subvolumes from a tensor using a coordinates generator.
    
    Args:
        tensor (torch.Tensor): A tensor from which subvolumes will be extracted. Shape: (1, 256, 256, 256)
        coord_generator (CoordsGenerator): An instance of CoordsGenerator to provide subvolume coordinates.
        
    Returns:
        torch.Tensor: A tensor containing the extracted subvolumes.
    """
    tensor = tensor.cpu()
    assert tensor.dim() == 4, "Expected tensor of 4 dimensions (1, 256, 256, 256)"
    
    coords = coord_generator.get_coordinates(mode="test")  # Use "test" mode to get non-overlapping subvolumes
    subvolumes_list = []
    for (z_start, z_end), (y_start, y_end), (x_start, x_end) in coords:
        subvolume = tensor[:, z_start:z_end, y_start:y_end, x_start:x_end]
        subvolumes_list.append(subvolume.unsqueeze(0))

    # Stack the subvolumes to get the final tensor
    subvolumes = torch.cat(subvolumes_list, dim=0)
    return subvolumes

# Example
# tensor = torch.randint(0, 104, (1, 256, 256, 256))  # Random tensor with values between 0 and 103
# coord_generator = CoordsGenerator(list_shape=[256, 256, 256], list_sub_shape=[128, 128, 128])
# subvolumes = extract_label_subvolumes(tensor, coord_generator)

# print(subvolumes.shape)  # Expected shape: (number_of_subvolumes, 128, 128, 128)
