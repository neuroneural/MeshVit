import torch
import torchvision
import time

# Load your model (assuming a pretrained resnet50 for demonstration purposes)
model = torchvision.models.resnet50(pretrained=True).cuda()
model.eval()  # Set the model to evaluation mode

# Generate some dummy data for inference
batch_size = 64
dummy_data = torch.randn(batch_size, 3, 224, 224).cuda()

# Measure initial GPU memory consumption
initial_memory_allocated = torch.cuda.memory_allocated()
initial_memory_cached = torch.cuda.memory_cached()

# Time the inference process
start_time = time.time()
with torch.no_grad():
    outputs = model(dummy_data)
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
import torch
import torchvision
import time

# Load your model (assuming a pretrained resnet50 for demonstration purposes)
model = torchvision.models.resnet50(pretrained=True).cuda()
model.eval()  # Set the model to evaluation mode

# Generate some dummy data for inference
batch_size = 128
dummy_data = torch.randn(batch_size, 3, 224, 224).cuda()

# Measure initial GPU memory consumption
initial_memory_allocated = torch.cuda.memory_allocated()
initial_memory_cached = torch.cuda.memory_cached()

# Time the inference process
start_time = time.time()
with torch.no_grad():
    outputs = model(dummy_data)
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
