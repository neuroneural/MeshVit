from torch.utils.data import DataLoader
from fixed_coords_generator import FixedCoordGenerator

from mongo_loader import (
    create_client,
    collate_subcubes,
    mcollate,
    MBatchSampler,
    MongoDataset,
    MongoClient,
    mtransform,
)
import torch
import random 
import easybar

from torch.utils.data import DataLoader, Dataset, RandomSampler, BatchSampler



from torch.utils.data import DataLoader, Dataset

class MongoDataLoader:
    # Do not modify the following block
    
    def __init__(self, batch_size=1, labelnow_choice=1, COLLECTION="HCP"):
        self.LABELNOW=["sublabel", "gwmlabel", "50label"][labelnow_choice]
        self.MONGOHOST = "arctrdcn018.rs.gsu.edu"
        self.DBNAME = 'MindfulTensors'
        self.COLLECTION=COLLECTION
        #self.COLLECTION = 'MRNslabs'
        # A much cleaner and thus less conducive to generalization dataset
        #COLLECTION = "HCP"
        
        self.batch_size = batch_size

        self.INDEX_ID = "subject"
        self.VIEWFIELDS = ["subdata", self.LABELNOW, "id", "subject"]
        self.batched_subjs = self.batch_size
        self.client = MongoClient("mongodb://" + self.MONGOHOST + ":27017")
        self.db = self.client[self.DBNAME]
        self.posts = self.db[COLLECTION]
        self.num_examples = int(self.posts.find_one(sort=[(self.INDEX_ID, -1)])[self.INDEX_ID] + 1)
        self.train_size = int(0.8 * self.num_examples)
        self.valid_size = int(0.1 * self.num_examples)
        self.test_size = self.num_examples - self.train_size - self.valid_size

        indices = list(range(self.num_examples))
        self.train_indices = indices[:self.train_size]
        self.valid_indices = indices[self.train_size:(self.train_size + self.valid_size)]
        self.test_indices = indices[(self.train_size + self.valid_size):]
    
    
    def createclient(self,x):
        return create_client(x, dbname=self.DBNAME,
                            colname=self.COLLECTION,
                            mongohost=self.MONGOHOST)
    
    def mycollate_full(self, x):
        return mcollate(x, labelname=self.LABELNOW)

    def mytransform(self,x):
        return mtransform(x, label=self.LABELNOW)
    
    def get_mongo_loaders(self, num_workers=4):
        train_dataset = MongoDataset(
            self.train_indices,
            self.mytransform,
            None,
            id=self.INDEX_ID,
            fields=self.VIEWFIELDS,
        )
        random_sampler = RandomSampler(train_dataset)
        batch_sampler = BatchSampler(random_sampler, batch_size=self.batch_size, drop_last=False)

        train_loader = DataLoader(
            train_dataset,
            sampler=batch_sampler,
            collate_fn=self.mycollate_full,
            pin_memory=True,
            worker_init_fn=self.createclient,
            num_workers=num_workers
        )
        
        
        valid_dataset = MongoDataset(
            self.valid_indices,
            self.mytransform,
            None,
            id=self.INDEX_ID,
            fields=self.VIEWFIELDS,
        )
        random_sampler = RandomSampler(valid_dataset)
        batch_sampler = BatchSampler(random_sampler, batch_size=self.batch_size, drop_last=False)

        valid_loader = DataLoader(
            valid_dataset,
            sampler=batch_sampler,
            collate_fn=self.mycollate_full,
            pin_memory=True,
            worker_init_fn=self.createclient,
            num_workers=num_workers
        )
        
        test_dataset = MongoDataset(
            self.test_indices,
            self.mytransform,
            None,
            id=self.INDEX_ID,
            fields=self.VIEWFIELDS,
        )
        random_sampler = RandomSampler(test_dataset)
        batch_sampler = BatchSampler(random_sampler, batch_size=self.batch_size, drop_last=False)

        test_loader = DataLoader(
            test_dataset,
            sampler=batch_sampler,
            collate_fn=self.mycollate_full,
            pin_memory=True,
            worker_init_fn=self.createclient,
            num_workers=num_workers
        )
        
        return train_loader, valid_loader, test_loader

    @staticmethod
    def extract_subvolumes(mri_tensor, label_tensor, coord_generator):
        # Generate coordinates
        batch_size = mri_tensor.shape[0]
        coords = coord_generator.get_coordinates(mode="train")  # Assuming you want a random subcube
        cl = []
        for b in range(batch_size): 
            cl.append(random.choice(coords))
        coords = cl

        # Extract MRI subvolumes
        assert mri_tensor.dim() == 5, "Expected MRI tensor of 5 dimensions (batch, channel, depth, height, width)"
        mri_subvolumes = MongoDataLoader._extract_from_tensor(mri_tensor, coords, channel_dim=True)

        # Extract label subvolumes
        assert label_tensor.dim() == 4, "Expected label tensor of 4 dimensions (batch, depth, height, width)"
        label_subvolumes = MongoDataLoader._extract_from_tensor(label_tensor, coords, channel_dim=False)

        return mri_subvolumes, label_subvolumes
    
    @staticmethod
    def _extract_from_tensor(tensor, coords, channel_dim=False):
        tensor = tensor.cpu()
        batch_size = tensor.shape[0]
        subvolumes_list = []
        for b in range(batch_size):
            (z_start, z_end), (y_start, y_end), (x_start, x_end) = coords[b]
            if channel_dim:
                subvolume = tensor[b, :, z_start:z_end, y_start:y_end, x_start:x_end]  # Keep the channel dimension
            else:
                subvolume = tensor[b, z_start:z_end, y_start:y_end, x_start:x_end]
            subvolumes_list.append(subvolume.unsqueeze(0))
        
        subvolumes = torch.cat(subvolumes_list, dim=0)
        return subvolumes
    
    @staticmethod
    def extract_all_subvolumes(mri_tensor, label_tensor, coord_generator):
        # Generate coordinates
        all_coords = coord_generator.get_coordinates(mode="train")

        # Extract MRI subvolumes
        assert mri_tensor.dim() == 5, "Expected MRI tensor of 5 dimensions (batch, channel, depth, height, width)"
        mri_subvolumes_list = [(MongoDataLoader._extract_from_tensor(mri_tensor, [coord], channel_dim=True), coord) for coord in all_coords]

        # Extract label subvolumes
        assert label_tensor.dim() == 4, "Expected label tensor of 4 dimensions (batch, depth, height, width)"
        label_subvolumes_list = [(MongoDataLoader._extract_from_tensor(label_tensor, [coord], channel_dim=False), coord) for coord in all_coords]

        return mri_subvolumes_list, label_subvolumes_list

    @staticmethod
    def reconstitute_volume(subvolumes_list, original_shape):
        # Initialize a zero tensor of the original shape
        reconstructed = torch.zeros(original_shape, dtype=subvolumes_list[0][0].dtype)

        # Iterate over subvolumes and their coordinates to place them back into the volume
        for (subvolume, coord) in subvolumes_list:
            (z_start, z_end), (y_start, y_end), (x_start, x_end) = coord
            reconstructed[z_start:z_end, y_start:y_end, x_start:x_end] += subvolume.squeeze()

        return reconstructed
