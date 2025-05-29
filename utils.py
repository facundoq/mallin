import matplotlib.pyplot as plt
import pytorch_lightning as pl


from pathlib import Path


class LossPlotter(pl.Callback):
    def __init__(self,filepath:Path):
        super().__init__()
        self.train_loss_values = []
        self.filepath=filepath

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_loss_values.append(trainer.callback_metrics['train_loss'].item())

    def on_train_end(self, trainer, pl_module):
        plt.plot(self.train_loss_values)
        plt.xlabel('Training Steps')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Curve')
        plt.savefig(self.filepath)
        plt.close()