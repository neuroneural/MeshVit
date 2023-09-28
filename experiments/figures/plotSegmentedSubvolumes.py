import sys
import torch
import os
sys.path.append('/data/users2/washbee/MeshVit/experiments') #change this path

import matplotlib.pyplot as plt
from mongoutil import *
from fixed_coords_generator import FixedCoordGenerator
# 1. Get 5 MRI volumes

mongo_loader = MongoDataLoader(batch_size=5)
_, _, test_loader = mongo_loader.get_mongo_loaders()


data_iter = iter(test_loader)
mri_tensors, label_tensors = next(data_iter)

print("MRI Tensor dtype:", mri_tensors.dtype)
print("Label Tensor dtype:", label_tensors.dtype)
 
# 2. Extract subvolumes

coord_gen = FixedCoordGenerator(256, 128)
label_subvolumes_list = []

print("shapes",mri_tensors.shape,label_tensors.shape)
for i in range(5):
    _, label_subvolumes = mongo_loader.extract_all_subvolumes(mri_tensors[i].unsqueeze(0), label_tensors[i].unsqueeze(0), coord_gen)
    print("MRI subvolumes dtype:", _[0][0].dtype)
    print("Label subvolumes dtype:",  label_subvolumes[0][0].dtype)

    label_subvolumes_list.append(label_subvolumes)

print('label_subvolumes_list', len(label_subvolumes_list))

# 3. Reassemble label subvolumes into volume
reconstructed_labels_list = []
for j in range(5):
    reconstructed_labels_list.append(mongo_loader.reconstitute_volume(label_subvolumes_list[j], (256, 256, 256)))

print("reconstructed_labels_list", len(reconstructed_labels_list))
for i in range(5):
    print("reconstructed_labels_list[i].shape", reconstructed_labels_list[i].shape )
    print("label_tensors[i,...].shape",label_tensors[i,...].shape)
    print('equality', torch.equal(reconstructed_labels_list[i], label_tensors[i,...]))
    assert torch.equal(reconstructed_labels_list[i], label_tensors[i,...]), "tensors are not equal"

#all assertions for labels_tensors pass. 
# chatgpt, I need to revise the below code given how the above code has been written. 
# 4. Display
def display_slices(mri, original_label, reconstituted_label, slice_num, save_dir='./saved_figs'):
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))    
    axes[0].imshow(mri[slice_num], cmap='gray')
    axes[0].set_title("MRI")
    axes[1].imshow(original_label[slice_num], cmap='gray')
    axes[1].set_title("Original Label")
    axes[2].imshow(reconstituted_label[slice_num], cmap='gray')
    axes[2].set_title("Reconstituted Label")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()

    # Save the figure as SVG
    save_path = os.path.join(save_dir, f'slice_{slice_num}.svg')
    fig.savefig(save_path, format='svg')
    plt.close(fig)  # Close the figure to free up memory

# Displaying the first MRI volume's slice 128
# Remember that the MRI tensor has an extra channel dimension, so it's indexed differently than labels.
# display_slices(mri_tensors[0][0].cpu().numpy(), label_tensors[0].cpu().numpy(), reconstructed_labels_list[0].cpu().numpy(), 128)
# # Displaying the first MRI volume's slice 128
# display_slices(mri_tensors[0][0][128].numpy(), label_tensors[0][128].numpy(), reconstructed_labels[0][128].numpy())