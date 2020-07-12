import warnings
import streamlit as st
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('seaborn')
import plotly.graph_objs as go

import pandas as pd
import matplotlib.pyplot as plt
import fbprophet
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def prophet(df_final):

    df = pd.read_csv('Ani1.csv',parse_dates=['Date'],infer_datetime_format=True)
    df= df.drop('Unnamed: 0',axis=1)

    df_warehouse_s = df[['Date', 'Order_Demand_Whse_S']]
    df_s = df_warehouse_s.rename(columns={"Date": "ds", "Order_Demand_Whse_S": "y"})
    prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,)
    prophet.fit(df_s)
    future = prophet.make_future_dataframe(periods=12, freq='M')
    df_forecast = prophet.predict(future)
    st.markdown('## Demand Forecasting using FbProphet')
    trace = go.Scatter(
        name = 'Actual Demand Quantity',
        mode = 'markers',
        x = list(df_s['ds']),
        y = list(df_s['y']),
        marker=dict(
            color='#FFBAD2',
            line=dict(width=1)
        )
    )
    trace1 = go.Scatter(
        name = 'trend',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat']),
        marker=dict(
            color='red',
            line=dict(width=3)
        )
    )
    upper_band = go.Scatter(
        name = 'upper band',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_upper']),
        line= dict(color='#57b88f'),
        fill = 'tonexty'
    )
    lower_band = go.Scatter(
        name= 'lower band',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_lower']),
        line= dict(color='#1705ff')
    )
    tracex = go.Scatter(
        name = 'Actual price',
       mode = 'markers',
       x = list(df_s['ds']),
       y = list(df_s['y']),
       marker=dict(
          color='black',
          line=dict(width=2)
       )
    )
    data = [trace1, lower_band, upper_band, trace]

    layout = dict(title='Order Demand Forecasting - Warehouse S',
                 xaxis=dict(title = 'Dates', ticklen=2, zeroline=True),
                  yaxis=dict(title = 'Order Quantity', ticklen=2, zeroline=True))

    fig=dict(data=data,layout=layout)
    # plt.savefig('btc03.html')
    st.plotly_chart(fig)


    df_warehouse_j = df[['Date', 'Order_Demand_Whse_J']]
    df_j = df_warehouse_j.rename(columns={"Date": "ds", "Order_Demand_Whse_J": "y"})
    prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,)
    prophet.fit(df_j)
    future = prophet.make_future_dataframe(periods=12, freq='M')
    df_forecast = prophet.predict(future)

    trace = go.Scatter(
        name = 'Actual Demand Quantity',
        mode = 'markers',
        x = list(df_j['ds']),
        y = list(df_j['y']),
        marker=dict(
            color='#FFBAD2',
            line=dict(width=1)
        )
    )
    trace1 = go.Scatter(
        name = 'trend',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat']),
        marker=dict(
            color='red',
            line=dict(width=3)
        )
    )
    upper_band = go.Scatter(
        name = 'upper band',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_upper']),
        line= dict(color='#57b88f'),
        fill = 'tonexty'
    )
    lower_band = go.Scatter(
        name= 'lower band',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_lower']),
        line= dict(color='#1705ff')
    )
    tracex = go.Scatter(
        name = 'Actual price',
       mode = 'markers',
       x = list(df_j['ds']),
       y = list(df_j['y']),
       marker=dict(
          color='black',
          line=dict(width=2)
       )
    )
    data = [trace1, lower_band, upper_band, trace]

    layout = dict(title='Order Demand Forecasting - Warehouse J',
                 xaxis=dict(title = 'Dates', ticklen=2, zeroline=True),
                  yaxis=dict(title = 'Order Quantity', ticklen=2, zeroline=True))

    fig=dict(data=data,layout=layout)
    # plt.savefig('btc03.html')
    st.plotly_chart(fig)
    return 0
