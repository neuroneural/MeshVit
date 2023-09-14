import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from brain_dataset import BrainDataset
from coords_generator import CoordsGenerator

class TestBrainDataset(unittest.TestCase):

    def setUp(self):
        # Improved mock setup for slicer to have get_fdata method
        slicer_mock = MagicMock()
        slicer_mock.get_fdata = MagicMock(return_value=np.random.rand(10, 10, 10))

        self.mock_open_fn = MagicMock(return_value={
            "images": np.random.rand(10, 10, 10),
            "targets": slicer_mock  # Using slicer_mock instead of the previously mocked object
        })

        self.mock_coords_generator = MagicMock()
        self.mock_coords_generator.get_coordinates = MagicMock(return_value=[[(0, 5), (0, 5), (0, 5)]])

        self.list_data = [0, 1]
        self.list_shape = [10, 10, 10]
        self.list_sub_shape = [5, 5, 5]

        self.dataset = BrainDataset(self.list_data, self.list_shape, self.list_sub_shape, self.mock_open_fn, n_subvolumes=2)

    @patch('brain_dataset.CoordsGenerator', return_value=MagicMock())
    def test_init(self, mock_coords_generator):
        self.assertIsNotNone(self.dataset.data)
        self.assertEqual(self.dataset.mode, "train")
        self.assertIsNotNone(self.dataset.generator)

    def test_len(self):
        expected_len = len(self.list_data) * 2
        self.assertEqual(len(self.dataset), expected_len)

    def test_getitem(self):
        with patch.object(self.dataset, 'generator', self.mock_coords_generator):
            item = self.dataset[0]
            self.assertIsNotNone(item)

    def test_crop(self):
        with patch.object(self.dataset, 'generator', self.mock_coords_generator):
            dict_data = self.mock_open_fn(0)
            coords = [(0, 5), (0, 5), (0, 5)]
            cropped = self.dataset._crop(dict_data, [coords])
            self.assertIn("images", cropped)
            self.assertIn("targets", cropped)
            self.assertIn("coords", cropped)
