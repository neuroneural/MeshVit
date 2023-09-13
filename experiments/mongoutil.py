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
import torch

class MongoDataLoader:
    dbname = 'MindfulTensors'
    #colname = 'MRNslabs'
    mongohost = 'arctrdcn018.rs.gsu.edu'
    index_id = 'subject'
    LABELNOW = ["sublabel", "gwmlabel", "50label"]

    def __init__(self, labelnow_choice=0,colname="HCP"):
        self.labelnow_choice = labelnow_choice
        self.view_fields = ['subdata', self.LABELNOW[labelnow_choice], 'id', 'subject']
        self.colname=colname
    
    def create_client_wrapper(self, x):
        return create_client(x, dbname=self.dbname, colname=self.colname, mongohost=self.mongohost)

    def mcollate_full(self, x):
        return mcollate(x, labelname=self.LABELNOW[self.labelnow_choice])

    def mtransform_wrapper(self, x):
        return mtransform(x, label=self.LABELNOW[self.labelnow_choice])

    def get_mongo_loaders(self, batch_size, num_workers):

        client = MongoClient("mongodb://" + self.mongohost + ":27017")
        db = client[self.dbname]
        posts = db[self.colname]
        num_examples = int(posts.find_one(sort=[(self.index_id, -1)])[self.index_id] + 1)

        train_size = int(0.8 * num_examples)
        valid_size = int(0.1 * num_examples)
        test_size = num_examples - train_size - valid_size

        indices = list(range(num_examples))
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:(train_size + valid_size)]
        test_indices = indices[(train_size + valid_size):]

        train_dataset = MongoDataset(
            train_indices,
            self.mtransform_wrapper,
            None,
            id=self.index_id,
            fields=self.view_fields,
        )
        train_sampler = MBatchSampler(train_dataset, batch_size=batch_size)
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            collate_fn=self.mcollate_full,
            pin_memory=True,
            worker_init_fn=self.create_client_wrapper,
            num_workers=num_workers,
            batch_size=batch_size
        )

        valid_dataset = MongoDataset(
            valid_indices,
            self.mtransform_wrapper,
            None,
            id=self.index_id,
            fields=self.view_fields,
        )
        valid_sampler = MBatchSampler(valid_dataset, batch_size=batch_size)
        valid_loader = DataLoader(
            valid_dataset,
            sampler=valid_sampler,
            collate_fn=self.mcollate_full,
            pin_memory=True,
            worker_init_fn=self.create_client_wrapper,
            num_workers=num_workers,
            batch_size=batch_size
        )

        test_dataset = MongoDataset(
            test_indices,
            self.mtransform_wrapper,
            None,
            id=self.index_id,
            fields=self.view_fields,
        )
        test_sampler = MBatchSampler(test_dataset, batch_size=batch_size)
        test_loader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            collate_fn=self.mcollate_full,
            pin_memory=True,
            worker_init_fn=self.create_client_wrapper,
            num_workers=num_workers,
            batch_size=batch_size
        )

        return train_loader, valid_loader, test_loader

    # Rest of the class remains the same

    @staticmethod
    def extract_subvolumes(tensor, coord_generator):
        tensor = tensor.cpu()
        coords = coord_generator.get_coordinates(mode="test")
        subvolumes_list = []
        for (z_start, z_end), (y_start, y_end), (x_start, x_end) in coords:
            subvolume = tensor[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
            subvolumes_list.append(subvolume.unsqueeze(0))
        subvolumes = torch.cat(subvolumes_list, dim=0)
        return subvolumes

    @staticmethod
    def extract_label_subvolumes(tensor, coord_generator):
        tensor = tensor.cpu()
        assert tensor.dim() == 4, "Expected tensor of 4 dimensions (1, 256, 256, 256)"
        coords = coord_generator.get_coordinates(mode="test")
        subvolumes_list = []
        for (z_start, z_end), (y_start, y_end), (x_start, x_end) in coords:
            subvolume = tensor[:, z_start:z_end, y_start:y_end, x_start:x_end]
            subvolumes_list.append(subvolume.unsqueeze(0))
        subvolumes = torch.cat(subvolumes_list, dim=0)
        return subvolumes
