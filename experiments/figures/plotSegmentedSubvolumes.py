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
copy_mri = mri_tensors.clone()
assert torch.equal(copy_mri,mri_tensors)
print('Database')
print("mri_tensors.shape","label_tensors.shape",mri_tensors.shape,label_tensors.shape)
#print('Fake')
#mri_tensors = torch.zeros((1,1,256,256,256),dtype=torch.float32)
#label_tensors = torch.zeros((1,256,256,256),dtype=torch.int64)
#print("mri_tensors.shape","label_tensors.shape",mri_tensors.shape,label_tensors.shape)
assert mri_tensors.shape == (1,1,256,256,256) 
assert label_tensors.shape == (1,256,256,256) 
print("MRI Tensor dtype:", mri_tensors.dtype)
print("Label Tensor dtype:", label_tensors.dtype)
 
# 2. Extract subvolumes

coord_gen = FixedCoordGenerator(256, 128)
label_subvolumes_list = []
mri_subvolumes_list = []
print("shapes",mri_tensors.shape,label_tensors.shape)
for i in range(batch_size):
    mri_subvolumes, label_subvolumes = mongo_loader.extract_all_subvolumes(mri_tensors[i].unsqueeze(0), label_tensors[i].unsqueeze(0), coord_gen)
    print("MRI subvolumes dtype:", mri_subvolumes[0][0].dtype)
    print("Label subvolumes dtype:",  label_subvolumes[0][0].dtype)

    label_subvolumes_list.append(label_subvolumes)
    mri_subvolumes_list.append(mri_subvolumes)
    print(type(mri_subvolumes))
    print(len(mri_subvolumes))
    print(type(label_subvolumes))
    print(len(label_subvolumes))
    print(type(mri_subvolumes[0]))
    print(len(mri_subvolumes[0]))
    print(type(label_subvolumes[0]))
    print(len(label_subvolumes[0]))
    print(type(mri_subvolumes[0][0]))
    print(mri_subvolumes[0][0].shape)
    print(type(label_subvolumes[0][0]))
    print(label_subvolumes[0][0].shape)
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


# 4. Display MRI, Original Label, and Prediction
def display_slices(mri, original_label, predicted_label, slice_num, save_dir='./saved_figs',name = ""):
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

    save_path = os.path.join(save_dir, f'{name}_slice_{slice_num}.svg')
    fig.savefig(save_path, format='svg')
    save_path = os.path.join(save_dir, f'{name}_slice_{slice_num}.png')
    fig.savefig(save_path, format='png')
    plt.close(fig)


model_path = "/data/users2/washbee/MeshVit/experiments/logs/monaiunet_e100_gmwm_sv128_be768142-3d25-49a0-b088-1f4bef64846b/best_full.pth"

checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
model = nets.UNet(
            dimensions=3,
            in_channels=1,
            out_channels=3,#3 classes
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# If you're using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3.2 Run predictions
predictions_list = []
with torch.no_grad():
    subvolumes = mri_subvolumes_list[0]#.to(device)
    
    for subvolume, coord in subvolumes:
        print("subvolume.shape",subvolume.shape)
        prediction = model(subvolume)
        prediction = prediction.cpu()  # Move predictions to CPU for further processing
        softmax_prediction = torch.nn.functional.softmax(prediction, dim=1)
        # Get the class prediction by taking the argmax along the channel dimension
        class_prediction = torch.argmax(softmax_prediction, dim=1)
        print("class_predictions.shape,coord",class_prediction.shape,coord)
        predictions_list.append((class_prediction,coord))

reconstructed_prediction = mongo_loader.reconstitute_volume(predictions_list, (256, 256, 256))

print("reconstructed_prediction.shape", reconstructed_prediction.shape)
assert reconstructed_prediction.shape == (256,256,256)

assert torch.equal(copy_mri,mri_tensors)

display_slices(mri_tensors[0][0].cpu().numpy(), 
               label_tensors[0].cpu().numpy(), 
               reconstructed_prediction.cpu().numpy(), 
               128, 
               save_dir='./unet_figs',name='unet')

import sys
sys.path.append('/data/users2/washbee/MeshVit') #change this path

from segmenter.segm.model.decoder3d import MaskTransformer3d  #check sys.path.append
from segmenter.segm.model.segmenter3d import Segmenter3d #check sys.path.append
from segmenter.segm.model.vit3d import VisionTransformer3d #check sys.path.append

subvolume_shape = [128,128,128]
patch_size = 16
n_layers = 8
d_model = 128
d_ff = 64
n_heads = 64
n_classes = 3
vit = VisionTransformer3d(subvolume_shape, patch_size, n_layers, d_model, d_ff, n_heads, n_classes, 
            dropout=0.1,
            drop_path_rate=0.0,
            distilled=False,
            channels=1)
drop_path_rate=0.0
dropout=0.1
d_encoder = 128
decoder = MaskTransformer3d(n_classes, patch_size, d_encoder, n_layers, n_heads, d_model, d_ff, drop_path_rate, dropout)
model = Segmenter3d(vit, decoder, n_cls=n_classes)
        
model_path = "/data/users2/washbee/MeshVit/experiments/logs/3DVit_e100_gmwm_sv128_8d5db215-95fc-40cd-98aa-824c89a6a03f/best_full.pth"
checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
    
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# If you're using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3.2 Run predictions
predictions_list = []
with torch.no_grad():
    subvolumes = mri_subvolumes_list[0]#.to(device)
    
    for subvolume, coord in subvolumes:
        print("subvolume.shape",subvolume.shape)
        prediction = model(subvolume)
        prediction = prediction.cpu()  # Move predictions to CPU for further processing
        softmax_prediction = torch.nn.functional.softmax(prediction, dim=1)
        # Get the class prediction by taking the argmax along the channel dimension
        class_prediction = torch.argmax(softmax_prediction, dim=1)
        print("class_predictions.shape,coord",class_prediction.shape,coord)
        predictions_list.append((class_prediction,coord))

reconstructed_prediction = mongo_loader.reconstitute_volume(predictions_list, (256, 256, 256))

print("reconstructed_prediction.shape", reconstructed_prediction.shape)
assert reconstructed_prediction.shape == (256,256,256)

assert torch.equal(copy_mri,mri_tensors)

display_slices(mri_tensors[0][0].cpu().numpy(), 
               label_tensors[0].cpu().numpy(), 
               reconstructed_prediction.cpu().numpy(), 
               128, 
               save_dir='./3DVit_figs',name='3DVit')


from model import MeshNet, UNet
model = MeshNet(
n_channels=1, n_classes=3, large=True, dropout_p=0.1
)
model_path = "/data/users2/washbee/MeshVit/experiments/logs/meshnet_e100_gmwm_large_dropout_sv128_927e1a09-6b00-4434-bb1d-2e9568a5abad/best_full.pth"
checkpoint = torch.load(model_path,map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# If you're using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3.2 Run predictions
predictions_list = []
with torch.no_grad():
    subvolumes = mri_subvolumes_list[0]#.to(device)
    for subvolume, coord in subvolumes:
        print("subvolume.shape",subvolume.shape)
        prediction = model(subvolume)
        prediction = prediction.cpu()  # Move predictions to CPU for further processing
        softmax_prediction = torch.nn.functional.softmax(prediction, dim=1)
        # Get the class prediction by taking the argmax along the channel dimension
        class_prediction = torch.argmax(softmax_prediction, dim=1)
        print("class_predictions.shape,coord",class_prediction.shape,coord)
        predictions_list.append((class_prediction,coord))

reconstructed_prediction = mongo_loader.reconstitute_volume(predictions_list, (256, 256, 256))

print("reconstructed_prediction.shape", reconstructed_prediction.shape)
assert reconstructed_prediction.shape == (256,256,256)
assert torch.equal(copy_mri,mri_tensors)

display_slices(mri_tensors[0][0].cpu().numpy(), 
    label_tensors[0].cpu().numpy(), 
    reconstructed_prediction.cpu().numpy(), 
    128, 
    save_dir='./meshnet_figs',name='meshnet')

