import sys
import torch
import os
sys.path.append('/data/users2/washbee/MeshVit/experiments') #change this path

import matplotlib.pyplot as plt
from mongoutil import *
from fixed_coords_generator import FixedCoordGenerator
# 1. Get 5 MRI volumes
batch_size = 1 
mongo_loader = MongoDataLoader(batch_size=batch_size)
_, _, test_loader = mongo_loader.get_mongo_loaders()


data_iter = iter(test_loader)
mri_tensors, label_tensors = next(data_iter)

print("MRI Tensor dtype:", mri_tensors.dtype)
print("Label Tensor dtype:", label_tensors.dtype)
 
# 2. Extract subvolumes

coord_gen = FixedCoordGenerator(256, 128)
label_subvolumes_list = []

print("shapes",mri_tensors.shape,label_tensors.shape)
for i in range(batch_size):
    _, label_subvolumes = mongo_loader.extract_all_subvolumes(mri_tensors[i].unsqueeze(0), label_tensors[i].unsqueeze(0), coord_gen)
    print("MRI subvolumes dtype:", _[0][0].dtype)
    print("Label subvolumes dtype:",  label_subvolumes[0][0].dtype)

    label_subvolumes_list.append(label_subvolumes)

print('label_subvolumes_list', len(label_subvolumes_list))

# 3. Reassemble label subvolumes into volume
reconstructed_labels_list = []
for j in range(batch_size):
    reconstructed_labels_list.append(mongo_loader.reconstitute_volume(label_subvolumes_list[j], (256, 256, 256)))

print("reconstructed_labels_list", len(reconstructed_labels_list))
for i in range(batch_size):
    print("reconstructed_labels_list[i].shape", reconstructed_labels_list[i].shape )
    print("label_tensors[i,...].shape",label_tensors[i,...].shape)
    print('equality', torch.equal(reconstructed_labels_list[i], label_tensors[i,...]))
    assert torch.equal(reconstructed_labels_list[i], label_tensors[i,...]), "tensors are not equal"

import torch
import monai.networks.nets as nets

model_path = "/data/users2/washbee/MeshVit/experiments/logs/monaiunet_e100_gmwm_sv128_be768142-3d25-49a0-b088-1f4bef64846b/best_full.pth"


checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
print('checkpoint keys')
#for key in checkpoint:
#    print(key)

#exit()
model = nets.UNet(
    dimensions=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
    )
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# If you're using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3.2 Run predictions
with torch.no_grad():
    mri_tensors = mri_tensors.to(device)
    predictions = model(mri_tensors)
    predictions = predictions.cpu()  # Move predictions to CPU for further processing

print("predictions.shape", predictions.shape)
# Apply softmax to the predictions along the channel dimension
softmax_predictions = torch.nn.functional.softmax(predictions, dim=1)

# Get the class prediction by taking the argmax along the channel dimension
class_predictions = torch.argmax(softmax_predictions, dim=1)

# 4. Display MRI, Original Label, and Prediction
def display_slices(mri, original_label, predicted_label, slice_num, save_dir='./saved_figs'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))    
    axes[0].imshow(mri[slice_num], cmap='gray')
    axes[0].set_title("MRI")
    axes[1].imshow(original_label[slice_num], cmap='gray')
    axes[1].set_title("Original Label")
    axes[2].imshow(predicted_label[slice_num], cmap='gray')
    axes[2].set_title("Predicted Label")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'slice_{slice_num}.svg')
    fig.savefig(save_path, format='svg')
    plt.close(fig)

# Displaying the first MRI volume's slice 128
display_slices(mri_tensors[0][0].cpu().numpy(), 
               label_tensors[0].cpu().numpy(), 
               class_predictions[0].cpu().numpy(), 
               128, 
               save_dir='./unet_figs')
