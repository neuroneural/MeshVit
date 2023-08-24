# MeshVit
(Dilated) Vision Transformer for Gray/White Matter Segmentation

# neuro_monai_sbatch.sh

This script:

- Specifies Slurm job parameters (job name, outputs, allocated resources, etc.)
- Activates a Conda environment named `neuro`
- Sets library paths
- Runs a python script (`minimal_monai_torchio_example.py`) with specified parameters

# neuro_monai_sbatch_meshnet.sh

## MeshNet Training

- **Platform**: Slurm cluster
- **Compute**: v100 GPUs, High Memory Nodes
- **Framework**: Conda with `neuro` environment
- **Codebase**: Python, utilizing `minimal_monai_torchio_example.py`
- **Dataset**: Located within `./data/afedorov_T1_c_atlas_data/`

# neuro_mongo_sbatch.sh

## Vit3D Training Workflow

- **Platform**: Slurm cluster
- **Compute**: v100 GPUs, High Memory Nodes
- **Environment**: Managed via Conda, specifically the `neuro` environment.
- **Project Root**: `/data/users2/bbaker/projects/MeshVit/neuro2`
- **Logging**: Slurm logs stored in `/data/users2/bbaker/projects/MeshVit/slurm/` with error and output streams separated.
- **Notifications**: Emails sent to `bbaker43@gsu.edu` on all job events.
- **Script Execution**: Using the Python script `minimal_mongo.py` in the `training` directory.
  - **Model**: `segmenter`
  - **Epochs**: Short runs of 10 epochs for quick iteration and testing.
  - **Classes**: 3 class segmentation.
  - **Logging & Results**: Stored under `../vit3d_results/`.
- **Dataset Configuration**: Custom subvolume and patch sizes set for dataset processing (`tsv`, `sv`, `ps`).

Developer Notes:
- Utilizing Conda for environment management ensures consistent dependency versions across runs.
- Slurm configuration allows leveraging GPU resources and simplifying parallel execution.
- Emphasis on monitoring via email notifications and dedicated logging to keep a tab on training progress and issues.

# neuro_mongo_sbatch_meshnet.sh  


## Vit3D MeshNet Training Workflow

- **Platform**: Slurm cluster
- **Compute Specs**: v100 GPUs, High Memory Nodes
- **Environment Management**: Conda with `neuro` environment.
- **Project Directory**: `/data/users2/bbaker/projects/MeshVit/neuro2`
- **Logging**: Slurm logs stored in `/data/users2/bbaker/projects/MeshVit/slurm/`.
- **Notifications**: Configured for `bbaker43@gsu.edu`.
- **Main Script**: `minimal_mongo.py` within the `training` folder.
  - **Model**: `meshnet`
  - **Training Duration**: Quick iterations with 10 epochs.
  - **Segmentation Classes**: Three distinct classes.
  - **Results Directory**: `../vit3d_results/`.

Development Insights:
- The Conda environment ensures a consistent working environment.
- The use of Slurm streamlines GPU utilization and job management.
- Regular email notifications and log separation for quick troubleshooting and monitoring.
- Dataset customization via specific subvolume and patch sizes to cater to specific data nuances.

# test.py  

## 3D Segmenter with Vision Transformer

- **Framework**: PyTorch
- **Image Size**: 3D volume with dimensions 38x38x38
- **Device**: CUDA (GPU acceleration)

### Model Components:
- **Vision Transformer**: `VisionTransformer3d` 
  - Patches of size: 12
  - Embedding size: 128
  - Depth: 8
  - Number of heads: 3
  - Input channels: 1
  
- **Decoder**: `MaskTransformer3d` 
  - Heads: 2
  - Dropout: 0.0 and 0.1

- **Segmenter**: Combines the Vision Transformer and Decoder for 3D segmentation.

### Experiment:
- Dummy 3D data is generated to feed into the `Segmenter3d`.
- The trainable parameters of the model are then counted and printed.

Development Note:
This script showcases the integration of a 3D Vision Transformer with a custom decoder for segmentation tasks. The current configuration is indicative and may need tweaking based on the dataset and desired results.

# meshvit/MaskTransformer3d.py

## Mask Transformer for Image Segmentation

This module focuses on image segmentation using transformer-based architectures. Here's a breakdown of its architecture and components:

- **Framework**: PyTorch.

### Primary Components:

- **MaskTransformer (Class)**: 
  - **Input Parameters**:
    - `n_cls`: Number of classes.
    - `patch_size`: Size of each image patch.
    - `d_encoder`: Encoder depth.
    - `n_layers`: Number of transformer layers.
    - `n_heads`: Number of attention heads.
    - `d_model`: Depth of the model.
    - `d_ff`: Depth of the feed-forward network.
    - `drop_path_rate`: Dropout rate.
    - `dropout`: Dropout value.
  - **Attributes**:
    - `blocks`: Sequential transformer blocks.
    - `cls_emb`: Embeddings for the classes.
    - `proj_dec`: Projection for the decoder.
    - `proj_patch`: Projection for patches.
    - `proj_classes`: Projection for the classes.
    - `decoder_norm`: Layer normalization for the decoder.
    - `mask_norm`: Layer normalization for the mask.
  - **Methods**:
    - `forward`: Propagates the input through the model, returning masks.
    - `get_attention_map`: Retrieves the attention map for a given layer ID.

- **Utilities and Blocks**:
  - `Block`: Basic transformer block used in the sequence.
  - `FeedForward`: Basic feed-forward network block.
  - `init_weights`: Helper function to initialize weights.
  - `trunc_normal_`: Truncated normal initialization from `timm`.

### Developer Insights:

- The `MaskTransformer` is tailored to segment images, extracting features from the encoder output and predicting segmentation masks.
- It combines traditional transformer blocks with custom projections to cater to segmentation specifics.
- Attention maps can be retrieved for specific layers, aiding in model interpretability and understanding.

# meshvit/ViT2d.py

## Visual Transformer using Linformer

This code module provides an implementation of the Visual Transformer (ViT) model that leverages the efficient attention mechanism of Linformer for image classification.

- **Frameworks & Libraries**: PyTorch, Linformer, ViT from `vit_pytorch`.

### Key Features:

- **build_vit Function**: 
  - **Input Parameters**:
    - `dim`: Dimensionality of the token embeddings.
    - `seq_len`: Length of the sequence, determined by image and patch size.
    - `depth`: Number of transformer layers.
    - `heads`: Number of attention heads.
    - `k`: Context window size for Linformer.
    - `image_size`: Dimension (height/width) of the input image.
    - `patch_size`: Size of each image patch.
    - `num_classes`: Number of output classes.
    - `channels`: Number of input channels (e.g., 3 for RGB images).
  - **Description**: Constructs the ViT model using Linformer as the transformer backbone.
  - **Return**: The ViT model.

- **Efficient Attention Mechanism**:
  - The Linformer reduces the self-attention computation from O(n^2) to O(nk) by approximating the full attention matrix with a fixed size context window.
  
### Usage:

- When executed as a main script, the module builds a default ViT model and prints its modules.

### Developer Insights:

- The integration of Linformer in ViT paves the way for handling larger images or sequences without a significant increase in computation.
- The `seq_len` represents the total number of patches plus a class token, which is used for classification in the ViT paradigm.
- By using efficient transformers like Linformer in computer vision models, developers can harness the power of transformers while maintaining computational feasibility.


# meshvit/ViT3d.py

## 3D Visual Transformer using Linformer

The module provides a 3D variant of the Visual Transformer (ViT) for processing volumetric data using the Linformer's efficient attention mechanism.

- **Frameworks & Libraries**: PyTorch, Linformer, `einops`.

### Key Features:

- **ViT3d Class**:
  - **Description**: A 3D version of the Visual Transformer, modified to handle volumetric data.
  - **Key Variables**:
    - `pos_embedding`: Positional embeddings for patches.
    - `cls_token`: Special classification token.
    - `transformer`: Underlying transformer, Linformer in this case.
    - `mlp_head`: The MLP layer for final classification.
  - **Methods**:
    - `forward`: Processes an input volume and returns model predictions.

- **build_vit Function**: 
  - **Input Parameters**:
    - `dim`: Dimensionality of the token embeddings.
    - `seq_len`: Length of the sequence, determined by image and patch size.
    - `depth`: Number of transformer layers.
    - `heads`: Number of attention heads.
    - `k`: Context window size for Linformer.
    - `image_size`: Dimension (height/width/depth) of the input volume.
    - `patch_size`: Size of each image patch.
    - `num_classes`: Number of output classes.
    - `channels`: Number of input channels (e.g., 1 for grayscale volumes).
    - `output_shape`: The desired shape of the model's output. Useful for segmentations or other spatial outputs.
  - **Description**: Constructs the 3D ViT model using Linformer as its transformer backbone.
  - **Return**: The 3D ViT model.

- **Efficient Attention Mechanism**:
  - The Linformer reduces the self-attention computation by approximating the full attention matrix with a fixed size context window.

### Usage:

- When executed as a main script, the module creates an instance of the 3D ViT model, processes a random input volume, and prints the output shape.

### Developer Insights:

- The use of Linformer in the 3D ViT model allows for efficient handling of larger volumes.
- The `seq_len` represents the total number of patches plus a class token, adapted for the 3D setting.
- This architecture is ideal for 3D image tasks like medical image analysis where the data is naturally volumetric.
