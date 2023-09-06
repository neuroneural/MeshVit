from subvolume.utils import extract_subvolumes
import torch

subvolume_size = 64

mri_volume = torch.rand((1, 1, 256, 256, 256))  # Dummy MRI data
subvolumes = extract_subvolumes(mri_volume, subvolume_size)
print(f"Shape of each MRI Volume: {mri_volume.shape}")
print(f"Total Subvolumes: {len(subvolumes)}")
print(f"Shape of each Subvolume: {subvolumes[0].shape}")

seg_volume = torch.rand((1, 256, 256, 256))  # Dummy MRI data
subvolumes = extract_subvolumes(seg_volume, subvolume_size)
print(f"Shape of each Seg Volume: {seg_volume.shape}")
print(f"Total Subvolumes: {len(subvolumes)}")
print(f"Shape of each Subvolume: {subvolumes[0].shape}")


