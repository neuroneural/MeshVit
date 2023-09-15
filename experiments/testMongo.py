import unittest
from unittest.mock import Mock
from mongoslabs.mongoloader import (
    create_client,
    collate_subcubes,
    mcollate,
    MBatchSampler,
    MongoDataset,
    MongoClient,
    mtransform,
)

import unittest
from unittest.mock import MagicMock, patch
import torch
from torch.utils.data import DataLoader
from mongoutil import *
from unittest.mock import patch, Mock
from catalyst.data import BatchPrefetchLoaderWrapper, ReaderCompose






class TestDataloaderWithMBatchSampler(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8

        self.mdl = MongoDataLoader(batch_size=8)
        a,b,c = self.mdl.get_mongo_loaders()
        # Wrapping it all in a DataLoader
        self.dataloader = BatchPrefetchLoaderWrapper(a,
     num_prefetches=4 )

    def test_dataloader_iteration(self):
        batch_count = 0
        subject_count = 0

        # Iterating over the DataLoader
        for x,y in self.dataloader:
            batch_count += 1
            print('x.shape y.shape', x.shape,y.shape)
            subject_count += x.shape[0]

        print('batch_count',batch_count)
        print('subject_count',subject_count)
        self.assertGreater(batch_count, self.mdl.train_size//self.mdl.batch_size -1 )
        self.assertEqual(subject_count, self.mdl.train_size)


if __name__ == "__main__":
    unittest.main()


