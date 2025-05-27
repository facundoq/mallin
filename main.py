from pathlib import Path
from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import darts as ds
import darts.models as dsm
from darts.metrics import mape,mae,rmse
import preprocessing

output_folderpath = Path("results")
output_folderpath.mkdir(exist_ok=True,parents=True)

from sklearn.metrics import mean_squared_error
def eval_model(model,train:ds.TimeSeries,test:ds.TimeSeries,metrics:dict[str,callable]):
    test_pred = model.predict(len(test))
    train_pred = model.historical_forecasts(train,start=pd.Timestamp("2022-01-01"),forecast_horizon=12,last_points_only=True,stride = 12)
    results = {}
    series = {
        "train":(train,train_pred),
        "test":(test,test_pred)
    }
    for m_name,metric in metrics.items():
        for s_name,(y_true,y_pred) in series.items():
            result_id = f"{s_name}_{m_name}"    
            results[result_id] = metric(y_true,y_pred)

    return results,train_pred,test_pred

def plot_diagnostics(output_folderpath:Path,group:str,train:ds.TimeSeries,train_pred:ds.TimeSeries,test:ds.TimeSeries,test_pred:ds.TimeSeries):
    train.plot(label="train")
    train_pred.plot(label="train_forecast")    
    plt.savefig(output_folderpath/f"{group}_train.svg")
    plt.close()
    train.plot(label="train")
    test.plot(label="test")
    test_pred.plot(label="test_forecast")
    plt.savefig(output_folderpath/f"{group}_prediction.svg")
    plt.close()



def test_model_group(df:pl.DataFrame,test_start:pd.Timestamp,group_id:str,model,model_name:str,diagnostics=False):
    model_folderpath = output_folderpath/ model_name
    model_folderpath.mkdir(exist_ok=True,parents=True)
    model_filepath = model_folderpath/f"{group_id}.pkl"
    print(f"Running experiments for {group_id}")
    df = df.to_pandas()
    df = df.set_index("Dates")
    df = df.asfreq("MS",method='bfill')
    ts = ds.TimeSeries.from_dataframe(df,value_cols="NDVI")
    ts.plot()
    plt.savefig(f"{group_id}_series.svg")
    plt.close()
    train,test =ts.split_after(test_start)
    
    if model_filepath.exists():
        model = model.__class__.load(str(model_filepath.absolute()))
    else:
        model.fit(train,verbose=False)
        model.save(str(model_filepath.absolute()))
    
   
    metrics = {
        "mape":mape,
        "mae":mae,
        "rmse":rmse,
    }
    metrics_results,train_pred,test_pred = eval_model(model,train,test,metrics)

    if diagnostics: plot_diagnostics(model_folderpath,group_id,train,train_pred,test,test_pred)

    results = {"group":group_id}
    results.update(metrics_results)
    return results
  
    

    
def test_models(df:pl.DataFrame,models:dict[str,Callable],x:list[str],y:str,test_start:pd.Timestamp):
    groups = df.select(pl.col("Mallin")).unique().sort(by="Mallin")
    
    all_results = []
    for model_name,model_generator in models.items():
        print(f"Running tests for {model_name}")
        results_filepath = output_folderpath/f"{model_name}.csv"
        if results_filepath.exists():
            all_results.append(pl.read_csv(results_filepath))
            break
        results = []
        for group_name, in groups.iter_rows():
            p = pl.col("Mallin")==group_name
            df_g = df.filter(p)
            model=model_generator()
            result = test_model_group(df_g,test_start,group_name,model,model_name,diagnostics=True)
            result["group"]=group_name
            result["model"]=model_name
            results.append(result)
        results_df = pl.from_dicts(results)
        results_df.write_csv(output_folderpath/f"{model_name}.csv")
        all_results.append(results_df)
    return all_results

from pytorch_lightning.callbacks import EarlyStopping

def get_models():
    my_stopper = EarlyStopping(
        monitor="train_loss",  # "val_loss",
        patience=5,
        min_delta=0.05,
        mode='min',
    )
    pl_trainer_kwargs = {"callbacks": [my_stopper]}

    models = {
        
        "LSTM": lambda: dsm.RNNModel(
                                model="LSTM",
                                hidden_dim=20,
                                dropout=0,
                                batch_size=16,
                                n_epochs=300,
                                optimizer_kwargs={"lr": 1e-3},
                                model_name="NVDI_RNN",
                                random_state=42,
                                training_length=20,
                                input_chunk_length=14,
                                force_reset=True,
                                save_checkpoints=False,
                                pl_trainer_kwargs=pl_trainer_kwargs
                            ),
        "AutoARIMA": lambda: dsm.AutoARIMA(),
    }
    return models
def main():
    # Read the .rds file
    df = preprocessing.read_dataset("data.rds")
    df = preprocessing.preprocess(df)
    # train,test = preprocessing.train_test_split(df,2023)
    
    models = get_models()
    results = test_models(df,models,[],"NDVI",pd.Timestamp("2024-01-01"))
    for model_results in results:
        print(model_results)
        for col in ["train_rmse","test_rmse"]:
            v = model_results.mean().select(pl.col(col))
            print(f"Average {col}: {v}")

    


if __name__ == "__main__":
    main()
