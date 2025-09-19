""" Imports """

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset

import os
import time

import onnx
import onnxruntime as ort

""" Load Data """

pd.set_option('display.max_columns', None)

import kagglehub

path = kagglehub.dataset_download("fridrichmrtn/e-commerce-churn-dataset-rees46")

print("Path to dataset files:", path)
for filename in os.listdir(path):
    print(filename)

""" Explor Data """

df=pd.read_csv(path+'/rees46_customer_model.csv')


df.drop('row_id',axis=1,inplace=True)
df.drop('user_id',axis=1,inplace=True)

""" Quick analysis for features """

def more_info(data):
  n_unique=[]
  uniques=[]
  data_type=[]
  missing_values=[]
  missing_percentage=[]
  columns=[]
  max_values=[]
  min_values=[]

  for col in data.columns:
    columns.append(col)
    n_unique.append(data[col].nunique())
    uniques.append(data[col].unique())
    data_type.append(data[col].dtype)
    missing_values.append(data[col].isnull().sum())
    missing_percentage.append((data[col].isnull().sum()/data.shape[0])*100)
    if data[col].dtype!='object':
      max_values.append(data[col].max())
      min_values.append(data[col].min())
    else:
      max_values.append(np.nan)
      min_values.append(np.nan)
  more_info_df = pd.DataFrame({'column': columns,
                                'n_unique': n_unique,
                                'uniques': uniques,
                                'data_type': data_type,
                                'missing_values': missing_values,
                                'missing_percentage': missing_percentage,
                               'max_values':max_values,
                               'min_values':min_values})
  return more_info_df

info=more_info(df)
info

info[info['n_unique']==1]

df.drop(columns=info[info['n_unique']==1]['column'],axis=1,inplace=True)

""" Remove the features that have one unique value only """

df.duplicated().sum()

"""check duplicates"""

info[info['missing_values']>0]

"""check missing values"""

info[info['data_type']=='object']


""" Spliting """

x=df.drop(['target_revenue','target_customer_value','target_actual_profit','target_customer_value_lag1'],axis=1)
y=df['target_customer_value']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

""" Scaling """

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

"""
Linear Regression
"""

import time
start = time.time()
lr_model=LinearRegression()
lr_model.fit(x_train,y_train)
y_pred=lr_model.predict(x_test)
MSE_lr=mean_squared_error(y_test,y_pred)
MAE_lr=mean_absolute_error(y_test,y_pred)
R2_lr=r2_score(y_test,y_pred)
print(f'MSE: {MSE_lr}')
print(f'MAE: {MAE_lr}')
print(f'R2: {R2_lr}')
lr_time=time.time() - start
print("RF train time:", lr_time, "sec")

"""
Random Forest Regressor
"""

start = time.time()
rf_model=RandomForestRegressor(n_estimators=100)
rf_model.fit(x_train,y_train)
y_pred=rf_model.predict(x_test)
MSE_rf=mean_squared_error(y_test,y_pred)
MAE_rf=mean_absolute_error(y_test,y_pred)
R2_rf=r2_score(y_test,y_pred)
print(f'MSE: {MSE_rf}')
print(f'MAE: {MAE_rf}')
print(f'R2: {R2_rf}')
Rf_time=time.time() - start
print("RF train time:", Rf_time, "sec")

"""
 Deep Learning (ANN)

 Convert Data to tensor
"""

x_train_t=torch.tensor(x_train,dtype=torch.float32)

x_test_t=torch.tensor(x_test,dtype=torch.float32)

y_train_t=torch.tensor(y_train.values,dtype=torch.float32).reshape(-1, 1)

y_test_t=torch.tensor(y_test.values,dtype=torch.float32).reshape(-1, 1)


""" Build Pytorch ANN """

class NN_model(nn.Module):
  def __init__(self):
     super(NN_model,self).__init__()
     self.model=nn.Sequential(
        nn.Linear(x_train.shape[1],512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512,256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,1)
      )
  def forward(self,x):
    return self.model(x)

model= NN_model()

""" Build Early Stopping Class """

class EarlyStopping:
    def __init__(self, patience=5,verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose
        self.counter = 0
        self.best_model_state = None
        self.path = path
        self.restore_best_weights = True

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

""" Build Optimizer , Loss & Early Stopping """

loss_F=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
early_stopping = EarlyStopping(patience=20, delta=0.01)

"""Train & Test ANN"""

start = time.time()
val_loss_train=[]
val_loss_test=[]
R2_scores=[]
MAE_scores=[]
epochs=2000
for epoch in range(epochs):
  model.train()
  y_pred=model(x_train_t)
  loss=loss_F(y_pred,y_train_t)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  model.eval()
  with torch.no_grad():
    y_pred_test=model(x_test_t)
    test_loss=loss_F(y_pred_test,y_test_t)
    val_loss_train.append(loss.item())
    val_loss_test.append(test_loss.item())
    R2_scores.append(r2_score(y_test_t,y_pred_test))
    MAE_scores.append(mean_absolute_error(y_test_t,y_pred_test))
    MSE=mean_squared_error(y_test_t,y_pred_test)
    MAE=mean_absolute_error(y_test_t,y_pred_test)
    R2=r2_score(y_test_t,y_pred_test)
    if epoch%10==0:
      print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, MSE: {MSE:.4f}, MAE: {MAE:.4f}, R2: {R2:.4f}')

    early_stopping(test_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
model.load_state_dict(torch.load('checkpoint.pt'))
ANN_time=time.time() - start

pred=model(x_test_t).detach().numpy()
MSE_ANN=mean_squared_error(y_test_t,pred)
MAE_ANN=mean_absolute_error(y_test_t,pred)
R2_ANN=r2_score(y_test_t,pred)

print(f'MSE: {MSE_ANN:.4f}, MAE: {MAE_ANN:.4f}, R2: {R2_ANN:.4f}')
print("ANN train time:", ANN_time, "sec")


""" Save As Onnx & Onnx Run Time"""

input_dim=x_train_t.shape[1]
torch.save(model.state_dict(), "model_weights.pt")
dummy_input = torch.randn(1, input_dim, dtype=torch.float32)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'},
                  'output': {0: 'batch_size'}},
    opset_version=12
)
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name  = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
preds = sess.run([output_name], {input_name: x_test_np})[0]
