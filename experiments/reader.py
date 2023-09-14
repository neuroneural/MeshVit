from typing import List, Optional
import torch
import nibabel as nib
import numpy as np
from catalyst.contrib.data.reader import IReader


class TensorToNiftiConverter:
    """
    A utility class to convert PyTorch tensors to Nifti images.
    """

    @staticmethod
    def convert(tensor: torch.Tensor) -> nib.Nifti1Image:
        # Convert tensor to numpy
        array = tensor.cpu().numpy()
        # Convert to Nifti Image
        nifti_image = nib.Nifti1Image(array, affine=np.eye(4))
        return nifti_image


class TensorNiftiReader(IReader):
    """
    Converts a PyTorch tensor to a Nifti image.
    """

    def __init__(self, input_key: str, output_key: Optional[str] = None):
        super().__init__(input_key, output_key or input_key)

    def __call__(self, element) -> dict:
        tensor = element[self.input_key]
        nifti_image = TensorToNiftiConverter.convert(tensor)
        output = {self.output_key: nifti_image}
        return output


class TensorFixedVolumeNiftiReader(TensorNiftiReader):
    """
    Converts a PyTorch tensor to a Nifti image with a fixed volume.
    """

    def __init__(self, input_key: str, output_key: str, volume_shape: List = None):
        super().__init__(input_key, output_key)
        if volume_shape is None:
            volume_shape = [256, 256, 256]
        self.volume_shape = volume_shape

    def __call__(self, element) -> dict:
        import logging

        logging.basicConfig(filename='debug.log', level=logging.DEBUG)


        #logging.debug(f'element: {element}, type: {type(element)}')
        #logging.debug(f'self.input_key: {self.input_key}')
        #logging.debug(f'element: {element[[0]]}, type: {type(element)}')
        #logging.debug(f'self.input_key: {self.input_key}')

        tensor = element[self.input_key]
        array = tensor.cpu().numpy()
        array = (array - array.min()) / (array.max() - array.min())
        new_array = np.zeros(self.volume_shape)
        new_array[: array.shape[0], : array.shape[1], : array.shape[2]] = array
        nifti_image = TensorToNiftiConverter.convert(torch.tensor(new_array))
        output = {self.output_key: nifti_image}
        return output
