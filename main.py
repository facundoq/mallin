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
import model_config 

    
output_folderpath = Path("results")
output_folderpath.mkdir(exist_ok=True,parents=True)

from sklearn.metrics import mean_squared_error
Metric = Callable[[ds.TimeSeries,ds.TimeSeries],float]

def eval_model(model,train:ds.TimeSeries,test:ds.TimeSeries,train_covariates:ds.TimeSeries,test_covariates:ds.TimeSeries,metrics:dict[str,Metric],config:model_config.ModelConfig,backtest_config:dict):
    test_pred = model.predict(len(test),**test_covariates)
    print("\t\t Backtesting...")
    train_pred = model.historical_forecasts(train,**train_covariates,**backtest_config,**config.fit_kwargs)
    results = {}
    series = {
        "backtest":(train,train_pred),
        "test":(test,test_pred)
    }
    for m_name,metric in metrics.items():
        for s_name,(y_true,y_pred) in series.items():
            result_id = f"{s_name}_{m_name}"    
            results[result_id] = metric(y_true,y_pred)

    return results,train_pred,test_pred

def plot_diagnostics(output_folderpath:Path,group:str,train:ds.TimeSeries,train_pred:ds.TimeSeries,test:ds.TimeSeries,test_pred:ds.TimeSeries):
    train.plot(label="train")
    train_pred.plot(label="backtest_forecast")    
    plt.savefig(output_folderpath/f"{group}_backtest.svg")
    plt.close()
    train.plot(label="train")
    test.plot(label="test")
    test_pred.plot(label="test_forecast")
    plt.savefig(output_folderpath/f"{group}_prediction.svg")
    plt.close()

def get_covariates(config:model_config.ModelConfig,df:pd.DataFrame,test_start:pd.Timestamp):
    train_covariates = {}
    test_covariates = {}
    for name,columns in [("past_covariates",config.past_covariates),("future_covariates",config.future_covariates)]:
        if len(columns)>0:
            covariates = ds.TimeSeries.from_dataframe(df,value_cols=columns)
            before,after = covariates.split_after(test_start)
            train_covariates[name]=covariates
            test_covariates[name]=covariates
    
    return train_covariates,test_covariates
    
def test_model_group(df:pd.DataFrame,test_start:pd.Timestamp,group_id:str,config:model_config.ModelConfig,backtest_config:dict,diagnostics=False):
    model=config.make()
    model_folderpath = output_folderpath/ config.id
    model_folderpath.mkdir(exist_ok=True,parents=True)
    model_filepath = model_folderpath/f"{group_id}.pkl"
    print(f"\tGroup {group_id}, samples: {len(df)}")

    ts = ds.TimeSeries.from_dataframe(df,value_cols="NDVI")
    
    train_covariates, test_covariates = get_covariates(config,df,test_start)
    
    if model_filepath.exists():
        model = model.__class__.load(str(model_filepath.absolute()))
    else:
        train,test =ts.split_after(test_start)
        model.fit(train,**train_covariates,**config.fit_kwargs)
        model.save(str(model_filepath.absolute()))
    
   
    metrics = {
        "mape":mape,
        "mae":mae,
        "rmse":rmse,
    }
    metrics_results,train_pred,test_pred = eval_model(model,train,test,train_covariates,test_covariates,metrics,config,backtest_config)

    if diagnostics: plot_diagnostics(model_folderpath,group_id,train,train_pred,test,test_pred)

    results = {"group":group_id}
    results.update(metrics_results)
    return results
  
    

    
def test_models(df:pl.DataFrame,models:list[model_config.ModelConfig],x:list[str],y:str,test_start:pd.Timestamp,backtest_config:dict):
    groups = df.select(pl.col("Mallin")).unique().sort(by="Mallin")

    
    all_results = []
    for model in models:
        print(f"Model {model.id}")
        results_filepath = output_folderpath/f"{model.id}.csv"
        if results_filepath.exists():
            results_df = pl.read_csv(results_filepath)
            print(f"Skipping; results already exist in {results_filepath}")
        else:
            results = []
            for group_name, in groups.iter_rows():
                df_g = preprocessing.get_mallin_timeseries_pd(df,group_name)
                result = test_model_group(df_g,test_start,group_name,model,backtest_config,diagnostics=True)
                result["group"]=group_name
                result["model"]=model.id
                results.append(result)
                print("[test_models] debugging 1 mallin")
                break
            results_df = pl.from_dicts(results)
            results_df.write_csv(output_folderpath/f"{model.id}.csv")
        all_results.append(results_df)
        
    return all_results
    
def main():
    # Read the .rds file
    df = preprocessing.read_dataset("data/scaled.rds")
    df = preprocessing.preprocess(df)
    season = 24
   
    past_covariates = ["month","Snow","Evap","Tmax","Tmin"]
    # train,test = preprocessing.train_test_split(df,2023)
    future_covariates = ["month","Tmax","Tmin","Snow"]
    models = [
        model_config.AutoARIMAFactory(season_length=season),
        model_config.AutoARIMAFactory(season_length=season,future_covariates=future_covariates),
        model_config.AutoETSFactory(season_length=season),
        model_config.AutoETSFactory(season_length=season,future_covariates=future_covariates),
        model_config.XGBFactory(lags=3, lags_past_covariates=3, output_chunk_length=3,past_covariates=past_covariates),
        model_config.XGBFactory(lags=12, lags_past_covariates=12, output_chunk_length=6,past_covariates=past_covariates),
        model_config.XGBFactory(lags=12, lags_past_covariates=12,lags_future_covariates=[0,1,2,3,4], output_chunk_length=6,past_covariates=past_covariates,future_covariates=future_covariates),
        model_config.BlockRNNFactory(training_length=6,input_chunk_length=6),            
        model_config.BlockRNNFactory(training_length=12,input_chunk_length=12),
        model_config.BlockRNNFactory(training_length=12,input_chunk_length=12,hidden_dim=64),
        model_config.BlockRNNFactory(training_length=12,input_chunk_length=12,hidden_dim=256),
        model_config.BlockRNNFactory(training_length=12,input_chunk_length=12,past_covariates=past_covariates),
    ]
    test_cutoff = pd.Timestamp("2024-01-01")
    backtest_config = {
        "start":pd.Timestamp("2022-01-01"),
        "forecast_horizon":season,
        "last_points_only":True,
        "stride" : season,
    }
    results = test_models(df,models,[],"NDVI",test_cutoff,backtest_config)
    for model,model_results in zip(models,results):
        print(model_results)
    all_results:pl.DataFrame = pl.concat(results)
    averages = all_results.group_by("model").mean().drop("group").sort("model")
    print(averages)
    averages.write_csv(output_folderpath/"results.csv")
    


if __name__ == "__main__":
    main()
