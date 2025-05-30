from pathlib import Path
import darts.models as dsm
from abc import abstractmethod
from pytorch_lightning.callbacks import EarlyStopping

from utils import LossPlotter

class ModelConfig:
    def __init__(self,future_covariates=[],past_covariates=[],**model_kwargs):
        self.future_covariates=future_covariates
        self.past_covariates=past_covariates
        self.model_kwargs = model_kwargs
    def model_kwargs_id(self):
        def safe(s:str): return str(s).replace("[","(").replace("]",")")
        return ",".join([f"{safe(k)}={safe(v)}" for k,v in self.model_kwargs.items()])
    def future_covariates_id(self):
        return  ",fcov="+(','.join(self.future_covariates)) if len(self.future_covariates) >0  else ""
    def past_covariates_id(self):
        return  ",pcov="+(','.join(self.past_covariates)) if len(self.past_covariates) >0  else ""
    def covariates_id(self):
        return self.future_covariates_id()+self.past_covariates_id()
    @property
    @abstractmethod
    def id(self):
        pass
    
    def desc(self,model):
        pass
    
    @abstractmethod
    def make(self):
        pass
    
    @property
    def fit_kwargs(self):
        return {}


class XGBFactory(ModelConfig):


    @property
    def id(self):
        return f"XGB({self.model_kwargs_id()}{self.covariates_id()})"
    def make(self):
        return dsm.XGBModel(**self.model_kwargs)
    def desc(self,model):
        return {}
    


class VARIMAFactory(ModelConfig):
    def __init__(self, future_covariates=[],past_covariates=[],p=1,d=0,q=0):
        super().__init__(future_covariates=future_covariates,past_covariates=past_covariates)
        self.p=p
        self.d=d
        self.q=q
    @property
    def id(self):
        return f"VARIMA(p={self.p},d={self.d},q={self.q}{self.covariates_id()})"
    def make(self):
        return dsm.VARIMA(p=self.p,q=self.q,d=self.d)
    def desc(self,model):
        return {}
        
class AutoARIMAFactory(ModelConfig):
        
    @property
    def id(self): return f"AutoARIMA({self.model_kwargs_id()}{self.covariates_id()})"
    def make(self):
        return dsm.AutoARIMA(**self.model_kwargs)
    def desc(self,model):
        return {}

class AutoETSFactory(ModelConfig):
    @property
    def id(self): return f"AutoETS({self.model_kwargs_id()}{self.covariates_id()})"
    def make(self):
        return dsm.AutoETS(**self.model_kwargs)
    def desc(self,model):
        return {}        

class BlockRNNFactory(ModelConfig):
    
    def __init__(self, future_covariates=[],past_covariates=[], training_length=12,input_chunk_length=12,hidden_dim=16,variant="LSTM"):
        self.variant=variant
        self.input_chunk_length=input_chunk_length
        self.training_length=training_length
        self.hidden_dim=hidden_dim
        super().__init__(future_covariates=future_covariates,past_covariates=past_covariates)
    
    
    @property
    def id(self,):

        return f"Block{self.variant}(icl={self.input_chunk_length},tl={self.training_length},hd={self.hidden_dim}{self.covariates_id()})"    
    
    fit_kwargs = {"verbose":None,
                  #"show_warnings":False,
                }

    def make(self):
        my_stopper = EarlyStopping(
            monitor="train_loss",  # "val_loss",
            patience=10,
            min_delta=0.01,
            mode='min',
        )
        loss_plotter = LossPlotter(Path(f"results/{self.id}/loss.png"))
        pl_trainer_kwargs = {"callbacks": [my_stopper,loss_plotter]}
        model= dsm.BlockRNNModel(
                                model="LSTM",
                                hidden_dim=self.hidden_dim,
                                dropout=0,
                                batch_size=4,
                                n_epochs=600,
                                optimizer_kwargs={"lr": 1e-4},
                                model_name="NVDI_RNN",
                                random_state=42,
                                # training_length=self.training_length,
                                input_chunk_length=self.input_chunk_length,
                                output_chunk_length=4,
                                force_reset=True,
                                save_checkpoints=False,
                                pl_trainer_kwargs=pl_trainer_kwargs
                            )
        return model
         
    def desc(self,model):
        return {}    
    
