import torch

def extract_subvolumes(volume, subvolume_size):
    """
    Extracts subvolumes from the given 3D MRI volume.

    Args:
    - volume (torch.Tensor): The MRI volume of shape (C, D, H, W)
    - subvolume_size (int): The size of the desired subvolume (assuming cubic shape)

    Returns:
    - list of torch.Tensor: A list of subvolumes
    """
    squeezed = len(volume.shape ) == 5
    if squeezed :
        assert volume.shape[0] == 1
        assert volume.shape[1] == 1
        volume = volume.squeeze(0)

    C, D, H, W = volume.shape

    # Ensuring the MRI volume can be evenly divided by the subvolume size
    assert D % subvolume_size == 0
    assert H % subvolume_size == 0
    assert W % subvolume_size == 0

    subvolumes = []

    for d in range(0, D, subvolume_size):
        for h in range(0, H, subvolume_size):
            for w in range(0, W, subvolume_size):
                subvolume = volume[:, d:d+subvolume_size, h:h+subvolume_size, w:w+subvolume_size]
                if squeezed:
                    subvolume = subvolume.unsqueeze(0)
                subvolumes.append(subvolume)

    return subvolumes

# Example usage:
# subvolume_size = 64

# mri_volume = torch.rand((1, 1, 256, 256, 256))  # Dummy MRI data
# subvolumes = extract_subvolumes(mri_volume, subvolume_size)
# print(f"Shape of each MRI Volume: {mri_volume.shape}")
# print(f"Total Subvolumes: {len(subvolumes)}")
# print(f"Shape of each Subvolume: {subvolumes[0].shape}")

# seg_volume = torch.rand((1, 256, 256, 256))  # Dummy MRI data
# subvolumes = extract_subvolumes(seg_volume, subvolume_size)
# print(f"Shape of each Seg Volume: {seg_volume.shape}")
# print(f"Total Subvolumes: {len(subvolumes)}")
# print(f"Shape of each Subvolume: {subvolumes[0].shape}")


