import torch
import torchvision
import time
import monai.networks.nets as nets
import torch.nn as nn
import sys
sys.path.append('/data/users2/washbee/MeshVit/experiments/') #change this path
sys.path.append('/data/users2/washbee/MeshVit/') #change this path
from model import MeshNet, UNet

from segmenter.segm.model.decoder3d import MaskTransformer3d  #check sys.path.append
from segmenter.segm.model.segmenter3d import Segmenter3d #check sys.path.append
from segmenter.segm.model.vit3d import VisionTransformer3d #check sys.path.append

modeltrain = False 
criterion = None
if modeltrain:
    criterion = nn.CrossEntropyLoss()

model_choice = 5
subvolume_size = 2
model_list = ['unet','meshnet_large','meshnet', '3dvit_large','3dvit_small']

if model_choice == 1:
    model = nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=3,#3 classes
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).cuda()
elif model_choice == 2:
    model = MeshNet(
            n_channels=1, n_classes=3, large=True, dropout_p=0.1,
        ).cuda()
elif model_choice == 3:
    model = MeshNet(
            n_channels=1, n_classes=3, large=False, dropout_p=0.1,
        ).cuda()
elif model_choice == 4:
    #128,16,8,128,64,64,128
    subvolume_shape = [128,128,128]
    if subvolume_size == 2:
        subvolume_shape = [256,256,256]
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
    model = Segmenter3d(vit, decoder, n_cls=n_classes).cuda()
elif model_choice == 5:
    #128,16,16,32,16,16,32
    subvolume_shape = [128,128,128]
    if subvolume_size == 2:
        subvolume_shape = [256,256,256]
        
    patch_size = 16
    n_layers = 16
    d_model = 32
    d_ff = 16
    n_heads = 16
    n_classes = 3
    vit = VisionTransformer3d(subvolume_shape, patch_size, n_layers, d_model, d_ff, n_heads, n_classes, 
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=1)
    drop_path_rate=0.0
    dropout=0.1
    d_encoder = 32
    decoder = MaskTransformer3d(n_classes, patch_size, d_encoder, n_layers, n_heads, d_model, d_ff, drop_path_rate, dropout)
    model = Segmenter3d(vit, decoder, n_cls=n_classes).cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # initial learning rate

# Load your model (assuming a pretrained resnet50 for demonstration purposes)
if not modeltrain:
    print('model eval')
    model = model.eval()  # Set the model to evaluation mode
else:
    print('model train')
    model = model.train()
batch_size = 1#powers of 2. unet maxed out 32. 
num_epochs = 10
for i in range(num_epochs):
    # Generate some dummy data for inference
    dummy_data = None
    dummy_labels = None
    if subvolume_size == 1:
        dummy_data = torch.randn(batch_size,1,128,128,128).cuda()
        dummy_labels = torch.ones(batch_size,128,128,128).long().cuda()
    elif subvolume_size == 2:
        dummy_data = torch.randn(batch_size,1,256,256,256).cuda()
        dummy_labels = torch.ones(batch_size,256,256,256).long().cuda()
        
    # Measure initial GPU memory consumption
    initial_memory_allocated = torch.cuda.memory_allocated()
    initial_memory_cached = torch.cuda.memory_cached()

    # Time the inference process
    start_time = time.time()
    try:
        if not modeltrain:
            with torch.no_grad():
                outputs = model(dummy_data)
        else:
            outputs = model(dummy_data)
            loss = criterion(outputs,dummy_labels)
            loss.backward()
            optimizer.step()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('exception occurrred') 
        exit()       
    end_time = time.time()
    inference_time = end_time - start_time

    # Measure final GPU memory consumption
    final_memory_allocated = torch.cuda.memory_allocated()
    final_memory_cached = torch.cuda.memory_cached()

    memory_increase_allocated = final_memory_allocated - initial_memory_allocated
    memory_increase_cached = final_memory_cached - initial_memory_cached

    print(f"Inference Time: {inference_time:.4f} seconds")
    print(f"GPU Memory Increase (allocated): {memory_increase_allocated / (1024 ** 2):.2f} MB")
    print(f"GPU Memory Increase (cached): {memory_increase_cached / (1024 ** 2):.2f} MB")

    # Cleanup to release GPU memory
    del dummy_data
    torch.cuda.empty_cache()

# # Generate some dummy data for inference
# dummy_data = torch.randn(2, 1, 128,128,128).cuda()

# # Measure initial GPU memory consumption
# initial_memory_allocated = torch.cuda.memory_allocated()
# initial_memory_cached = torch.cuda.memory_cached()

# Time the inference process
# start_time = time.time()
# with torch.no_grad():
#     outputs = model(dummy_data)
# end_time = time.time()
# inference_time = end_time - start_time

# Measure final GPU memory consumption
# final_memory_allocated = torch.cuda.memory_allocated()
# final_memory_cached = torch.cuda.memory_cached()

# memory_increase_allocated = final_memory_allocated - initial_memory_allocated
# memory_increase_cached = final_memory_cached - initial_memory_cached

# print(f"Inference Time: {inference_time:.4f} seconds")
# print(f"GPU Memory Increase (allocated): {memory_increase_allocated / (1024 ** 2):.2f} MB")
# print(f"GPU Memory Increase (cached): {memory_increase_cached / (1024 ** 2):.2f} MB")

# # Cleanup to release GPU memory
# del dummy_data
# torch.cuda.empty_cache()
