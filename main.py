import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import plotly.io as pi
import plotly.graph_objs as go

# 시계열 예측 라이브러리 
from prophet import Prophet
from prophet.plot import plot_plotly, add_changepoints_to_plot

# read file
df = pd.read_csv('연도별평균기온.csv')
df = df.astype({'Temperature' : 'float'})

# data formating
df['Year'] = pd.to_datetime(df['Year'], format = '%Y')

df_prophet = df.rename(columns={
    'Year': 'ds',
    'Temperature': 'y'
})

# def parameter
m = Prophet(
    changepoint_prior_scale=0.2, # 디폴트값 = 0.05
    changepoint_range=0.98,      
    yearly_seasonality=True,   
    weekly_seasonality=False,    
    daily_seasonality=False,     
    seasonality_mode='additive' 
)

# model learning
m.fit(df_prophet)

# predict range
future = pd.DataFrame({'ds' : [y for y in range (1909, 2031)]})

# prdict
forecast = m.predict(future)

# graph
fig = plot_plotly(m, forecast)
pi.write_image(fig, file = './fig/Prophet_fig.png',format='png',engine='kaleido')

plt.clf()
# change point graph
fig2 = m.plot(forecast)
a = add_changepoints_to_plot(fig2.gca(), m, forecast)
plt.savefig('./fig/change_point_fig.png')

plt.clf()
# detail graph
fig3 = m.plot_components(forecast)
plt.savefig('./fig/detail_fig.png')

print(forecast[['ds', 'yhat']].tail(7))
