from catalyst.dl import CheckpointCallback
import os

class CustomCheckpointCallback(CheckpointCallback):

    def __init__(self,metric_name="", minimize_metric=False, **kwargs):
        super().__init__(**kwargs)
        self.metric_name = metric_name
        self.minimize_metric = minimize_metric
        # Initialize the best_main_metric based on whether we want to minimize or maximize
        self.best_main_metric = float('inf') if self.minimize_metric else float('-inf')

    def on_epoch_end(self, runner):
        metric_value = None
        if runner.loader_key == "valid":  # Checking if it's the validation phase
            print('checkpointing from validation')
            dice_score = runner.loader_metrics.get('macro_dice', None)
            print('validation dice_score', dice_score)
            if dice_score is not None:
                metric_value = dice_score

        is_best = (
            (self.minimize_metric and metric_value < self.best_main_metric) or 
            (not self.minimize_metric and metric_value > self.best_main_metric)
        )
        if is_best:
            self.best_main_metric = metric_value
            checkpoint_data = {
                "model_state_dict": runner.model.state_dict(),
                "optimizer_state_dict": runner.optimizer.state_dict(),
                "epoch": runner.epoch,
                "valid_metrics": runner.epoch_metrics,
            }
            checkpoint_path = os.path.join(self.logdir, f"checkpoint_epoch_{runner.epoch}.pth")
            runner.engine.save_checkpoint(checkpoint_data, checkpoint_path)

            # The next line should use `self.logdir` as the path
            runner.engine.save_checkpoint(checkpoint_data, checkpoint_path)

