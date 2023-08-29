from meshnet import enMesh_checkpoint, enMesh

model = enMesh_checkpoint(in_channels=1,
            n_classes=self.n_classes,
            channels=model_channels,
            config_file=config_file)

if you really need to go to 256
        model = enMesh(
            in_channels=1,
            n_classes=self.n_classes,
            channels=model_channels,
            config_file=config_file,
        )
use this class, but this class requires a careful treatment in muti-GPU training  and has a special forward mechanism :slightly_smiling_face:
                loss, y_hat = self.model.forward(
                    x=sample, y=label, loss=self.criterion, verbose=False
You do not need to call backward if you use this class




