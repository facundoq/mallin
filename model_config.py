from pathlib import Path
import darts.models as dsm
from abc import abstractmethod
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
import matplotlib.pyplot as plt
class ModelConfig:

    def __init__(self,covariate=False):
        self.covariate=covariate
        
    @property
    @abstractmethod
    def id(self):
        pass
    
    
    
    @abstractmethod
    def params(self,model):
        pass
    
    @abstractmethod
    def make(self):
        pass
    
    @property
    def fit_kwargs(self):
        return {}
    
class AutoARIMAFactory(ModelConfig):
    id = "AutoARIMA"
    def make(self):
        return dsm.AutoARIMA(season_length=12)
    def params(self,model):
        return {}

class AutoETSFactory(ModelConfig):
    id = "AutoETS"
    def make(self):
        return dsm.AutoETS(season_length=12)
    def params(self,model):
        return {}        



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
        
class LSTMFactory(ModelConfig):
    
    def __init__(self, covariate=False, training_length=12,input_chunk_length=12,hidden_dim=16):
        self.input_chunk_length=input_chunk_length
        self.training_length=training_length
        self.hidden_dim=hidden_dim
        super().__init__(covariate)
    
    
    @property
    def id(self,):
        return f"LSTM(icl={self.input_chunk_length},tl={self.training_length},cov={self.covariate})"    
    fit_kwargs = {"verbose":False}
    def make(self):
        my_stopper = EarlyStopping(
            monitor="train_loss",  # "val_loss",
            patience=5,
            min_delta=0.05,
            mode='min',
        )
        loss_plotter = LossPlotter(Path(f"results/{self.id}/loss.png"))
        pl_trainer_kwargs = {"callbacks": [my_stopper,loss_plotter]}
        model= dsm.RNNModel(
                                model="LSTM",
                                hidden_dim=self.hidden_dim,
                                dropout=0,
                                batch_size=32,
                                n_epochs=300,
                                optimizer_kwargs={"lr": 1e-4},
                                model_name="NVDI_RNN",
                                random_state=42,
                                training_length=self.training_length,
                                input_chunk_length=self.input_chunk_length,
                                force_reset=True,
                                save_checkpoints=False,
                                pl_trainer_kwargs=pl_trainer_kwargs
                            )
        return model
         
    def params(self,model):
        return {}    
    
