from torch.profiler import profile, record_function, ProfilerActivity
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained VGG16 model from torchvision
model = models.vgg16(pretrained=False).to(device)  # Set to `True` for pretrained weights, but here we'll use random weights

# Hyperparameters
learning_rate = 0.001
num_epochs = 1
batch_size = 1
num_classes = 10  # Default for VGG16
input_shape = (3, 224, 224)  # Default input shape for VGG16

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Generate random data
num_samples = 1  # Just a random number for this example
X_train = torch.randn(num_samples, *input_shape).to(device)
y_train = torch.randint(0, num_classes, (num_samples,)).to(device)

pf =None 
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
    # Your PyTorch code here, for example:
    
    # Training loop
    for epoch in range(num_epochs):
        for i in range(0, num_samples, batch_size):
            inputs = X_train[i:i+batch_size]
            labels = y_train[i:i+batch_size]

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 10 batches (just for demonstration)
            if (i // batch_size) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i//batch_size + 1}/{num_samples//batch_size}], Loss: {loss.item():.4f}")
    print('a')


stats = prof.key_averages()
print('b')
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=1))

# Define the CSV file name
filename = "profiler_output.csv"
# Open the CSV file in write mode
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write the headers to the CSV file
    headers = ["Name", "Self CPU %", "Self CPU", "CPU time %", "CPU time", "CPU Memory", "Self CUDA %", "Self CUDA", "CUDA time %", "CUDA time", "CUDA Memory"]
    csvwriter.writerow(headers)
    # for avg in prof.key_averages():
    #     print(avg.key)
    #     exit()  # Just to print the attributes/methods for the first object in the list

    # Write the profiler values to the CSV file
    total_cpu_time = sum([avg.self_cpu_time_total for avg in prof.key_averages()])
    # Calculate the total CUDA time
    total_cuda_time = sum([avg.self_cuda_time_total for avg in prof.key_averages()])
    
    for avg in stats:
        #print('c')

        csvwriter.writerow([
            avg.key, 
            avg.self_cpu_time_total / total_cpu_time * 100,
            avg.self_cpu_time_total,
            avg.cpu_time / total_cpu_time * 100,
            avg.cpu_time,
            avg.cpu_memory_usage,
            avg.self_cuda_time_total / total_cuda_time * 100 if total_cuda_time > 0 else 0,
            avg.self_cuda_time_total,
            avg.cuda_time / total_cuda_time * 100 if total_cuda_time > 0 else 0,
            avg.cuda_time,
            avg.cuda_memory_usage
        ])
print("Training complete!")
