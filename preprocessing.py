import pyreadr
import polars as pl

def read_dataset(filepath):
    result = pyreadr.read_r(filepath)

    # Access the DataFrame
    pdf = result[None]
        
    pdf.convert_dtypes(dtype_backend="pyarrow")
    df = pl.from_pandas(pdf)
    df = df.drop_nulls()
    return df

def preprocess(df:pl.DataFrame):
    d = pl.col("Dates")
    df =df.with_columns(
        d.dt.ordinal_day().alias("day"),
        d.dt.month().alias("month"),
        d.dt.year().alias("year"),
        d.dt.to_string("%Y").alias("year_str"),
        pl.concat_str([pl.col("Mallin"),d.dt.year()],separator="-").alias("id")
    )
    df = df.drop_nulls()
    y = pl.col("year")
    df = df.filter(y<2025).filter(y>2000)
    return df

def train_test_split(df:pl.DataFrame,year_cutoff:int):
    p = pl.col("Dates").dt.year()<=year_cutoff
    train = df.filter(p)
    test = df.filter(~p)
    return train,test

def get_mallin_timeseries_pd(df:pl.DataFrame,mallin:str):
    # convert to pandas and resample to
    # make the timeseries have a define frequency 
    dfm = df.filter(pl.col("Mallin")==mallin)
    pdf = dfm.to_pandas()
    pdf = pdf.set_index("Dates")
    return pdf.asfreq("SMS",method='bfill')


if __name__ == "__main__":
    pass
