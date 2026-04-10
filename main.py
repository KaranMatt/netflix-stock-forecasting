import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from fastapi import FastAPI
from pydantic import BaseModel,Field,ConfigDict
from contextlib import asynccontextmanager
from typing import List

class StockInput(BaseModel):
    returns_list: List[float]
    model_config=ConfigDict(populate_by_name=True)


class Returns(BaseModel):
    returns_output:float

model=None
@asynccontextmanager
async def lifespan(app:FastAPI):
    global model

    model=tf.keras.models.load_model('Models/gru_multi_returns.keras',safe_mode=False)
    print('Model Loaded')

    yield
    print('Shutdown')
    model=None

app=FastAPI(title='Netflix Stock Forecasting API',lifespan=lifespan)

def preprocessing_returns(returns_list:List[float]):
    df=pd.DataFrame(returns_list,columns=['returns'])
    df['ema_20']=df['returns'].ewm(span=20,adjust=False).mean()
    df['ema_5']=df['returns'].ewm(span=5,adjust=False).mean()
    ema_12=df['returns'].ewm(span=12,adjust=False).mean()
    ema_26=df['returns'].ewm(span=26,adjust=False).mean()
    macd=ema_12-ema_26
    signal_line=macd.ewm(span=9,adjust=False).mean()
    df['macd_histogram']=macd-signal_line
    
    for i in range(7):
       df[f'returns {i+1}']=df['returns'].shift(periods=i+1)
       df[f'ema_20 {i+1}']=df['ema_20'].shift(periods=i+1)
       df[f'ema_5 {i+1}']=df['ema_5'].shift(periods=i+1)
       df[f'macd_histogram {i+1}']=df['macd_histogram'].shift(periods=i+1)

    df=df.dropna()
    input_cols=['returns 1', 'ema_20 1',
       'ema_5 1', 'macd_histogram 1', 'returns 2', 'ema_20 2', 'ema_5 2',
       'macd_histogram 2', 'returns 3', 'ema_20 3', 'ema_5 3',
       'macd_histogram 3', 'returns 4', 'ema_20 4', 'ema_5 4',
       'macd_histogram 4', 'returns 5', 'ema_20 5', 'ema_5 5',
       'macd_histogram 5', 'returns 6', 'ema_20 6', 'ema_5 6',
       'macd_histogram 6', 'returns 7', 'ema_20 7', 'ema_5 7',
       'macd_histogram 7']
    df=df[input_cols]
    return df

@app.get('/root')
def root():
    return {'Message':'Welcome to Stock API'}

@app.get('/heath')
def health():
    if model:
        return {'Model_Loaded':'True'}
    else:
        return {'Model_Loaded':'False'}
    

@app.post('/predict')
def predict_stock(data:StockInput):
    features=preprocessing_returns(data.returns_list)
    preds=model.predict(features)[0][0]
    return Returns(returns_output=preds)
