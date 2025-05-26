from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
import pmdarima as pm
import joblib
import pickle
import polars as pl

from statsmodels.tsa.arima.model import ARIMA


def read_dataset(filepath):
    result = pyreadr.read_r(filepath)

    # Access the DataFrame
    pdf = result[None]
        
    pdf.convert_dtypes(dtype_backend="pyarrow")
    df = pl.from_pandas(pdf)
    df = df.drop_nulls()
    return df
def train_test_split(df:pl.DataFrame,year_cutoff:int):
    p = pl.col("Dates").dt.year()<=year_cutoff
    train = df.filter(p)
    test = df.filter(~p)
    return train,test

from sklearn.metrics import mean_squared_error


def train_arima_hyperopt(x_train:np.ndarray,y_train:np.ndarray):

    order = (2,1,1)
    model = ARIMA(y_train, order=order)
    results = model.fit()
    return results,model

def test_arima_group(train:pl.DataFrame,test:pl.DataFrame,x:str,y:str,group:str,diagnostics=False):
    output_folderpath = Path("results/ARIMA/")
    output_folderpath.mkdir(exist_ok=True,parents=True)
    model_filepath = output_folderpath/f"{group}.pkl"
    print(f"Running experiments for {group}")
    y_train = train.select(pl.col(y)).to_numpy()
    y_test = test.select(pl.col(y)).to_numpy()
    
    if model_filepath.exists():
        with open(model_filepath, 'rb') as pkl:
            model = pickle.load(pkl)
    else:
        model = pm.auto_arima(y_train, seasonal=True, m=12)
        with open(model_filepath, 'wb') as pkl:
            pickle.dump(model, pkl)

    if diagnostics:
        model.summary()
        model.plot_diagnostics()
        plt.savefig(output_folderpath/f"{group}_diagnostics.svg")
        plt.close()
        print(model.summary())
    train_rmse = np.sqrt(model.oob())
    y_test_pred = 0# results.get_prediction(y_test)
    test_rmse = 0# np.sqrt(mean_squared_error(y_test,y_test_pred))
    order = model.params()
    return {"group":group,
            "params":order,
            "train_rmse":train_rmse,
            "test_rmse":test_rmse,
            }
  
    

    
def test_arima(train:pl.DataFrame,test:pl.DataFrame,x:str,y:str):
    groups = train.select(pl.col("Mallin")).unique().sort(by="Mallin")
    results = []
    for g, in groups.iter_rows():
        p = pl.col("Mallin")==g
        train_g = train.filter(p)
        test_g = test.filter(p)
        results.append(test_arima_group(train_g,test_g,x,y,g))
    return results
    

def main():
    # Read the .rds file
    df = read_dataset("data.rds")
    
    train,test = train_test_split(df,2023)
    
    
    results = test_arima(train,test,"NDVI","NDVI")
    results_df = pl.from_dicts(results)
    print(results_df)
    for col in ["train_rmse","test_rmse"]:
        v = results_df.mean().select(pl.col(col))
        print(f"Average {col}: {v}")
    
    
    

    


if __name__ == "__main__":
    main()
