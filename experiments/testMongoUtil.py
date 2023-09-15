import unittest
from mongoutil import *
from catalyst.data import BatchPrefetchLoaderWrapper, ReaderCompose

import easybar
class TestMongoDataLoader(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8
        self.num_workers = 4
        self.mongo_loader = MongoDataLoader(batch_size=self.batch_size)
        
    def test_mongo_loaders(self):
        # Get the train, valid, and test loaders
        train_loader,valid_loader, test_loader = self.mongo_loader.get_mongo_loaders(
            num_workers=self.num_workers
        )
        
    #     train_loader = BatchPrefetchLoaderWrapper(train_loader,
    #  num_prefetches=4 )

    #     valid_loader = BatchPrefetchLoaderWrapper(valid_loader,
    #  num_prefetches=4 )
        
    #     test_loader = BatchPrefetchLoaderWrapper(test_loader,
    #  num_prefetches=4 )

        # Check if loaders are not None
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(valid_loader)
        self.assertIsNotNone(test_loader)

        loaders = {
            "Train": train_loader,
            "Valid": valid_loader,
            "Test": test_loader
        }
        
        it = 0
        subjs = [615,76,78]
        for name, loader in loaders.items():
            print(name, 'length loader',len(loader))
            num_batches = 0
            num_subjects = 0
            for batch in loader:
                # assuming the subjects are in the first dimension of the batch
                num_subjects += batch[0].shape[0]
                num_batches += 1
                easybar.print_progress(num_subjects, len(loader))


            print(f"{name} Loader:")
            print(f"Number of batches: {num_batches}")
            print(f"Number of subjects: {num_subjects}")
            print("-------------------------------")

            # For example, if you want to ensure that each loader has at least one batch
            self.assertGreater(num_batches, 0)
            
            assert num_subjects == subjs[it]
            it+=1
    
            #print("hcp has 769 total")
            
            # Add more assertions as necessary
            

if __name__ == '__main__':
    unittest.main()
