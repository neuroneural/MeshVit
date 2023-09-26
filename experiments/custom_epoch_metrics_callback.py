from catalyst.dl import Callback, CallbackOrder
import torch
import numpy as np 
import os

class CustomEpochMetricsCallback(Callback):
    def __init__(self, metric_name, subvolume_size, patch_size, n_layers, d_model, d_ff, n_heads, d_encoder, modelsize, filename=None,logdir=None):
        super().__init__(order=CallbackOrder.Metric)
        self.metric_name = metric_name
        self.subvolume_size = subvolume_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.d_encoder = d_encoder
        self.modelsize = modelsize
        self.filename=filename
        self.logdir=logdir

    def on_epoch_end(self, runner):
        # Obtain your metric value from the runner's state
        if runner.loader_key.startswith("valid"):
            metric_value = runner.loader_metrics.get('macro_dice', None)
            print('logging hyperparameters and metric in callback')
            if self.filename is None and self.logdir is None:
                CustomEpochMetricsCallback.log_hyperparams(self.subvolume_size, self.patch_size, self.n_layers, self.d_model, self.d_ff, self.n_heads, self.d_encoder, .0005, self.modelsize, True, "Completed",metric_value)
            elif self.filename is not None and self.logdir is None:
                CustomEpochMetricsCallback.log_hyperparams(self.subvolume_size, self.patch_size, self.n_layers, self.d_model, self.d_ff, self.n_heads, self.d_encoder, .0005, self.modelsize, True, "Completed",metric_value, self.filename)
            elif self.filename is not None and self.logdir is not None:
                CustomEpochMetricsCallback.log_hyperparams(self.subvolume_size, self.patch_size, self.n_layers, self.d_model, self.d_ff, self.n_heads, self.d_encoder, .0005, self.modelsize, True, "Completed",metric_value, self.filename,self.logdir)
    @staticmethod
    def log_hyperparams(subvolume_size, patch_size, n_layers, d_model, d_ff, n_heads, d_encoder, lr, modelsize, completed, msg,loss, 
                        filename = '/data/users2/washbee/MeshVit/experiments/3DVit_hsearch_V100.log',
                        logdir="NA"):
        # Check if the file exists
        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                # Write the header
                file.write("subvolume_size,patch_size,n_layers,d_model,d_ff,n_heads,d_encoder,lr,modelsize,completed,msg,dice_val_avg,logdir\n")
        
        # Append the data
        with open(filename, 'a') as file:
            file.write(f"{subvolume_size},{patch_size},{n_layers},{d_model},{d_ff},{n_heads},{d_encoder},{lr},{modelsize},{completed},{msg},{loss},{logdir}\n")


    # Now you can use train_loader, valid_loader, and test_loader as needed
    @staticmethod
    def get_model_memory_size(model):
        params = sum(p.numel() for p in model.parameters())
        tensors = [p for p in model.parameters()]
        
        float_size = 4  # for float32
        total_memory = sum([np.prod(t.size()) * float_size for t in tensors])
        return total_memory / (1024 ** 2)  # convert bytes to megabytes
