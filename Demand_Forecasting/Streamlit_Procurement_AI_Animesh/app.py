import pandas as pd
import streamlit as st
import about
import itertools
import statsmodels.api as sm #for decomposing the trends, seasonality etc.
from statsmodels.tsa.statespace.sarimax import SARIMAX #the big daddy
from plotly.offline import iplot, init_notebook_mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
import plotly
import plotly.graph_objs as go
import os
import warnings
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('seaborn')


# def main_about():
#     st.title('About')
#     st.markdown('---')
#     #Display About section

def summary_plots(df_final):
    #Running Boxplot on Warehouse J

    lst_warehouse=list(df_final.columns)[2:]
    traces = []

    for whouse in lst_warehouse:
        s = df_final[whouse].to_frame().reset_index(drop=True)
        trace = go.Box(y= s[whouse], name= 'Warehouse {}'.format(whouse), jitter=0.8, whiskerwidth=0.2, marker=dict(size=2), line=dict(width=1))
        traces.append(trace)

    layout = go.Layout(
    title='Order Demand Boxplot Across Different Warehouses',
    yaxis=dict(
        autorange=True, showgrid=True, zeroline=True,
        gridcolor='rgb(233,233,233)', zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2, gridwidth=1),
    xaxis=dict(tickangle=15),
    margin=dict(l=40, r=30, b=80, t=100), showlegend=False,
    width=900,
                height=500
    )


    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig)

    st.markdown("### Since **Warehouse J** has the maximum demand, summarizing results for it ")

    #Running ARIMA on Warehouse J
    st.cache(persist=True)
    df_month = df_final.resample('M', on = 'Date').sum()
    df_month = df_month.reset_index()



    df_month = df_month.set_index('Date')
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    train = df_month.iloc[:len(df_month)-12]
    test = df_month.iloc[len(df_month)-12:]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            mod = sm.tsa.statespace.SARIMAX(df_month['Order_Demand_Whse_J'], order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))


    mod = sm.tsa.statespace.SARIMAX(train['Order_Demand_Whse_A'],
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    # st.write(pd.DataFrame(results.summary().tables[1]),header=None)
    #
    # results.plot_diagnostics(figsize=(20, 12))
    # st.pyplot()


    start = len(train)
    end = len(train) + len(test) - 1

    # Predictions for one-year against the test set
    predictions = results.predict(start, end,
                                 typ = 'levels').rename("Predictions")

    # trace0 = go.Scatter(
    #     x = predictions.index,
    #     y = predictions,
    #     mode = 'lines+markers',
    #     name = 'Predicted Values',
    #     text= predictions
    #
    #
    # )
    #
    # trace1 = go.Scatter(
    #     x = test['Order_Demand_Whse_J'].index,
    #     y = test['Order_Demand_Whse_J'],
    #     mode = 'lines+markers',
    #     name = 'Actual Values',
    #     text= predictions
    #
    #
    # )
    #
    # data = [trace0, trace1]
    #
    #
    # layout = dict(title = 'Warehouse J - Actual vs Predicted',
    #               xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
    #               yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
    #               width=900,
    #             height=500
    #
    #              )
    # fig = dict(data = data, layout = layout)
    # st.plotly_chart(fig)

    from sklearn.metrics import mean_squared_error
    from statsmodels.tools.eval_measures import rmse

    # Calculate root mean squared error
    rmse(test["Order_Demand_Whse_J"], predictions)
    df_warha = df_month['Order_Demand_Whse_J']
    mod = sm.tsa.statespace.SARIMAX(df_month['Order_Demand_Whse_J'],
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    pred = results.get_prediction(start=pd.to_datetime('2016-04-30'), dynamic=True)
    pred_ci = pred.conf_int()


    trace0 = go.Scatter(
        x = pred_ci.index,
        y = pred_ci.iloc[:, 1],
        mode = 'lines',
        name = 'CI Upper Limit',
        text= pred_ci.iloc[:, 0])


    trace1 = go.Scatter(
        x = pred.predicted_mean.index,
        y = pred.predicted_mean,
        mode = 'lines',
        name = 'Predicted Values',
        text= predictions,
        fill = 'tonexty'
    )

    trace2 = go.Scatter(
        x = pred_ci.index,
        y = pred_ci.iloc[:, 0],
        mode = 'lines',
        name = 'CI Lower Limit',
        text= pred_ci.iloc[:, 0],
        fill = 'tonexty')



    trace3 = go.Scatter(
        x = df_warha['2015-06':].index,
        y = df_warha['2015-06':],
        mode = 'lines',
        name = 'Observed Values',
        text= df_warha['2015-06':]
    #     fill = 'tonexty'
    )





    pred_uc = results.get_forecast(steps=12)
    pred_ci1 = pred_uc.conf_int()

    trace4 = go.Scatter(
        x =pred_ci1.index,
        y = pred_ci1.iloc[:, 1],
        mode = 'lines',
        name = 'Upper CI',
    #     fill = 'tonexty'



    )

    trace5 = go.Scatter(
        x =pred_uc.predicted_mean.index,
        y = pred_uc.predicted_mean,
        mode = 'lines',
        name = 'Forecast',
        fill = 'tonexty')

    trace6 = go.Scatter(
        x =pred_ci1.index,
        y = pred_ci1.iloc[:, 0],
        mode = 'lines',
        name = 'Lower CI',
        fill = 'tonexty'



    )

    data = [trace0, trace1,trace2,trace3,trace4,trace5,trace6]


    layout = dict(title = 'Warehouse J Forecast SARIMA',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                height=500

                 )
    fig = dict(data = data, layout = layout)
    st.plotly_chart(fig)

    st.cache(persist=True)
    #Running Prophet on Warehouse J
    df_warehouse_j = df_final[['Date', 'Order_Demand_Whse_J']]
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
            line=dict(width=1),
            size=2
        )
    )
    trace1 = go.Scatter(
        name = 'trend',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat']),
        marker=dict(
            color='red',
            line=dict(width=10)
        )
    )
    upper_band = go.Scatter(
        name = 'upper band',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_upper']),
        line= dict(color='#57b88f'),
        fill = 'tonexty'
        # fillcolor='#E0F2F3',
        # opacity=0.1
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

    layout = dict(title='Order Demand Forecasting FbProphet- Warehouse J',
                 xaxis=dict(title = 'Dates', ticklen=2, zeroline=True),
                  yaxis=dict(title = 'Order Quantity', ticklen=2, zeroline=True),
                  width=900,
                height=500)

    fig=dict(data=data,layout=layout)
    # plt.savefig('btc03.html')
    st.plotly_chart(fig)
    return 0





def EDA_Warehouse_Demnds(df_final):

    st.markdown('## Plotting the **Time Series** Data for warehouses')

    lst_warehouse=list(df_final.columns)[2:]
    traces = []

    for whouse in lst_warehouse:
        s = df_final[whouse].to_frame().reset_index(drop=True)
        trace = go.Box(y= s[whouse], name= 'Warehouse {}'.format(whouse), jitter=0.8, whiskerwidth=0.2, marker=dict(size=2), line=dict(width=1))
        traces.append(trace)

    layout = go.Layout(
    title='Order Demand Boxplot Across Different Warehouses',
    yaxis=dict(
        autorange=True, showgrid=True, zeroline=True,
        gridcolor='rgb(233,233,233)', zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2, gridwidth=1),
    xaxis=dict(tickangle=15),
    margin=dict(l=40, r=30, b=80, t=100), showlegend=False,
    width=900,
                height=500
    )


    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig)

    # Create traces
    trace0 = go.Scatter(
        x = df_final['Date'],
        y = df_final['Order_Demand_Whse_A'],
        mode = 'lines',
        name = 'Warehouse A',
        text= df_final['Order_Demand_Whse_A']


    )

    trace1 = go.Scatter(
        x = df_final['Date'],
        y = df_final['Order_Demand_Whse_C'],
        mode = 'lines',
        name = 'Warehouse C',
        text= df_final['Order_Demand_Whse_C']
    )

    trace2 = go.Scatter(
        x = df_final['Date'],
        y = df_final['Order_Demand_Whse_J'],
        mode = 'lines',
        name = 'Warehouse J',
        text= df_final['Order_Demand_Whse_J']
    )

    trace3 = go.Scatter(
        x = df_final['Date'],
        y = df_final['Order_Demand_Whse_S'],
        mode = 'lines',
        name = 'Warehouse S',
        text= df_final['Order_Demand_Whse_S']

    )

    data = [trace0, trace1, trace2, trace3]


    layout = dict(title = 'Warehouse Demands - Daily',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                  height=500

                 )
    fig = dict(data = data, layout = layout)
    st.plotly_chart(fig)


    df_resamp = df_final.resample('W', on = 'Date').sum()
    df_resamp = df_resamp.reset_index()

    # st.header('Weekly Warhouse Demand')

    # Create traces
    trace0 = go.Scatter(
        x = df_resamp['Date'],
        y = df_resamp['Order_Demand_Whse_A'],
        mode = 'lines',
        name = 'Warehouse A',
        text= df_resamp['Order_Demand_Whse_A']


    )

    trace1 = go.Scatter(
        x = df_resamp['Date'],
        y = df_resamp['Order_Demand_Whse_C'],
        mode = 'lines',
        name = 'Warehouse C',
        text= df_resamp['Order_Demand_Whse_C']
    )

    trace2 = go.Scatter(
        x = df_resamp['Date'],
        y = df_resamp['Order_Demand_Whse_J'],
        mode = 'lines',
        name = 'Warehouse J',
        text= df_resamp['Order_Demand_Whse_J']
    )

    trace3 = go.Scatter(
        x = df_resamp['Date'],
        y = df_resamp['Order_Demand_Whse_S'],
        mode = 'lines',
        name = 'Warehouse S',
        text= df_resamp['Order_Demand_Whse_S']

    )

    data = [trace0, trace1, trace2, trace3]


    layout = dict(title = 'Warehouse Demands - Weekly',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                  height=500

                 )
    fig = dict(data = data, layout = layout)
    st.plotly_chart(fig)

    # st.header ('Monthly Warehouse Demand')

    df_month = df_final.resample('M', on = 'Date').sum()
    df_month = df_month.reset_index()
    # Create traces
    trace0 = go.Scatter(
        x = df_month['Date'],
        y = df_month['Order_Demand_Whse_A'],
        mode = 'lines',
        name = 'Warehouse A',
        text= df_month['Order_Demand_Whse_A']


    )

    trace1 = go.Scatter(
        x = df_month['Date'],
        y = df_month['Order_Demand_Whse_C'],
        mode = 'lines',
        name = 'Warehouse C',
        text= df_month['Order_Demand_Whse_C']
    )

    trace2 = go.Scatter(
        x = df_month['Date'],
        y = df_month['Order_Demand_Whse_J'],
        mode = 'lines',
        name = 'Warehouse J',
        text= df_month['Order_Demand_Whse_J']
    )

    trace3 = go.Scatter(
        x = df_month['Date'],
        y = df_month['Order_Demand_Whse_S'],
        mode = 'lines',
        name = 'Warehouse S',
        text= df_month['Order_Demand_Whse_S']

    )

    data = [trace0, trace1, trace2, trace3]


    layout = dict(title = 'Warehouse Demands - Monthly',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                  height=500

                 )
    fig = dict(data = data, layout = layout)
    st.plotly_chart(fig)



    return 0

def seasonality(df_final):

    df_month = df_final.resample('M', on = 'Date').sum()
    df_month = df_month.reset_index()



    df_month = df_month.set_index('Date')

    st.markdown(' ## Seasonality Plot for different warehouses')

    decomposition = sm.tsa.seasonal_decompose(df_month['Order_Demand_Whse_A'], model='additive')

    trace0 = go.Scatter(
        x = decomposition.observed.index,
        y = decomposition.observed,
        mode = 'lines',
        name = 'Observed',
        text= decomposition.observed


    )

    trace1 = go.Scatter(
        x = decomposition.trend.index,
        y = decomposition.trend,
        mode = 'lines',
        name = 'Trend',
        text= decomposition.trend


    )

    trace2 = go.Scatter(
        x = decomposition.seasonal.index,
        y = decomposition.seasonal,
        mode = 'lines',
        name = 'Seasonal',
        text= decomposition.seasonal


    )

    trace3 = go.Scatter(
        x = decomposition.resid.index,
        y = decomposition.resid,
        mode = 'lines',
        name = 'Residual',
        text= decomposition.resid


    )

    data = [trace0, trace1, trace2, trace3]


    layout = dict(title = 'Plot for Warehouse A - Seasonality',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                  height=500

                 )
    fig = dict(data = data, layout = layout)
    st.plotly_chart(fig)


    # st.header('Seasonality Plot for Warehouse J')

    # Create traces
    decomposition = sm.tsa.seasonal_decompose(df_month['Order_Demand_Whse_J'], model='additive')

    trace0 = go.Scatter(
        x = decomposition.observed.index,
        y = decomposition.observed,
        mode = 'lines',
        name = 'Observed',
        text= decomposition.observed


    )

    trace1 = go.Scatter(
        x = decomposition.trend.index,
        y = decomposition.trend,
        mode = 'lines',
        name = 'Trend',
        text= decomposition.trend


    )

    trace2 = go.Scatter(
        x = decomposition.seasonal.index,
        y = decomposition.seasonal,
        mode = 'lines',
        name = 'Seasonal',
        text= decomposition.seasonal


    )

    trace3 = go.Scatter(
        x = decomposition.resid.index,
        y = decomposition.resid,
        mode = 'lines',
        name = 'Residual',
        text= decomposition.resid


    )

    data = [trace0, trace1, trace2, trace3]


    layout = dict(title = 'Plot for Warehouse J - Seasonality',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                  height=500

                 )
    fig = dict(data = data, layout = layout)
    st.plotly_chart(fig)

    # st.header('Seasonality Plot for Warehouse C')

    # Create traces
    decomposition = sm.tsa.seasonal_decompose(df_month['Order_Demand_Whse_C'], model='additive')

    trace0 = go.Scatter(
        x = decomposition.observed.index,
        y = decomposition.observed,
        mode = 'lines',
        name = 'Observed',
        text= decomposition.observed


    )

    trace1 = go.Scatter(
        x = decomposition.trend.index,
        y = decomposition.trend,
        mode = 'lines',
        name = 'Trend',
        text= decomposition.trend


    )

    trace2 = go.Scatter(
        x = decomposition.seasonal.index,
        y = decomposition.seasonal,
        mode = 'lines',
        name = 'Seasonal',
        text= decomposition.seasonal


    )

    trace3 = go.Scatter(
        x = decomposition.resid.index,
        y = decomposition.resid,
        mode = 'lines',
        name = 'Residual',
        text= decomposition.resid


    )

    data = [trace0, trace1, trace2, trace3]


    layout = dict(title = 'Plot for Warehouse C - Seasonality',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                  height=500

                 )
    fig = dict(data = data, layout = layout)
    st.plotly_chart(fig)

    # st.header('Seasonality Plot for Warehouse S')
    # Create traces
    decomposition = sm.tsa.seasonal_decompose(df_month['Order_Demand_Whse_S'], model='additive')

    trace0 = go.Scatter(
        x = decomposition.observed.index,
        y = decomposition.observed,
        mode = 'lines',
        name = 'Observed',
        text= decomposition.observed


    )

    trace1 = go.Scatter(
        x = decomposition.trend.index,
        y = decomposition.trend,
        mode = 'lines',
        name = 'Trend',
        text= decomposition.trend


    )

    trace2 = go.Scatter(
        x = decomposition.seasonal.index,
        y = decomposition.seasonal,
        mode = 'lines',
        name = 'Seasonal',
        text= decomposition.seasonal


    )

    trace3 = go.Scatter(
        x = decomposition.resid.index,
        y = decomposition.resid,
        mode = 'lines',
        name = 'Residual',
        text= decomposition.resid


    )

    data = [trace0, trace1, trace2, trace3]


    layout = dict(title = 'Plot for Warehouse S - Seasonality',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                  height=500

                 )
    fig = dict(data = data, layout = layout)
    st.plotly_chart(fig)
    return 0

def ARIMA_model(df_final):

    st.markdown('## SARIMA on Warehouse J')
    df_month = df_final.resample('M', on = 'Date').sum()
    df_month = df_month.reset_index()



    df_month = df_month.set_index('Date')
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    train = df_month.iloc[:len(df_month)-12]
    test = df_month.iloc[len(df_month)-12:]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            mod = sm.tsa.statespace.SARIMAX(df_month['Order_Demand_Whse_J'], order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))


    mod = sm.tsa.statespace.SARIMAX(train['Order_Demand_Whse_A'],
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    st.write(pd.DataFrame(results.summary().tables[1]),header=None)

    results.plot_diagnostics(figsize=(20, 12))
    st.pyplot()


    start = len(train)
    end = len(train) + len(test) - 1

    # Predictions for one-year against the test set
    predictions = results.predict(start, end,
                                 typ = 'levels').rename("Predictions")

    trace0 = go.Scatter(
        x = predictions.index,
        y = predictions,
        mode = 'lines+markers',
        name = 'Predicted Values',
        text= predictions


    )

    trace1 = go.Scatter(
        x = test['Order_Demand_Whse_J'].index,
        y = test['Order_Demand_Whse_J'],
        mode = 'lines+markers',
        name = 'Actual Values',
        text= predictions


    )

    data = [trace0, trace1]


    layout = dict(title = 'Warehouse J - Actual vs Predicted',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                height=500

                 )
    fig = dict(data = data, layout = layout)
    st.plotly_chart(fig)

    from sklearn.metrics import mean_squared_error
    from statsmodels.tools.eval_measures import rmse

    # Calculate root mean squared error
    rmse(test["Order_Demand_Whse_J"], predictions)
    df_warha = df_month['Order_Demand_Whse_J']
    mod = sm.tsa.statespace.SARIMAX(df_month['Order_Demand_Whse_J'],
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    pred = results.get_prediction(start=pd.to_datetime('2016-04-30'), dynamic=True)
    pred_ci = pred.conf_int()


    trace0 = go.Scatter(
        x = pred_ci.index,
        y = pred_ci.iloc[:, 1],
        mode = 'lines',
        name = 'CI Upper Limit',
        text= pred_ci.iloc[:, 0])


    trace1 = go.Scatter(
        x = pred.predicted_mean.index,
        y = pred.predicted_mean,
        mode = 'lines',
        name = 'Predicted Values',
        text= predictions,
        fill = 'tonexty'
    )

    trace2 = go.Scatter(
        x = pred_ci.index,
        y = pred_ci.iloc[:, 0],
        mode = 'lines',
        name = 'CI Lower Limit',
        text= pred_ci.iloc[:, 0],
        fill = 'tonexty')



    trace3 = go.Scatter(
        x = df_warha['2015-06':].index,
        y = df_warha['2015-06':],
        mode = 'lines',
        name = 'Observed Values',
        text= df_warha['2015-06':]
    #     fill = 'tonexty'
    )





    pred_uc = results.get_forecast(steps=12)
    pred_ci1 = pred_uc.conf_int()

    trace4 = go.Scatter(
        x =pred_ci1.index,
        y = pred_ci1.iloc[:, 1],
        mode = 'lines',
        name = 'Upper CI',
    #     fill = 'tonexty'



    )

    trace5 = go.Scatter(
        x =pred_uc.predicted_mean.index,
        y = pred_uc.predicted_mean,
        mode = 'lines',
        name = 'Forecast',
        fill = 'tonexty')

    trace6 = go.Scatter(
        x =pred_ci1.index,
        y = pred_ci1.iloc[:, 0],
        mode = 'lines',
        name = 'Lower CI',
        fill = 'tonexty'



    )



    data = [trace0, trace1,trace2,trace3,trace4,trace5,trace6]


    layout = dict(title = 'Warehouse J Forecast',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                height=500

                 )
    fig = dict(data = data, layout = layout)
    st.plotly_chart(fig)
    return 0

def prophet(df_final):

    # df = pd.read_csv(df_final,parse_dates=['Date'],infer_datetime_format=True)
    # df= df.drop('Unnamed: 0',axis=1)

    df_warehouse_s = df_final[['Date', 'Order_Demand_Whse_J']]
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
            line=dict(width=1),
            size=2
        )
    )
    trace1 = go.Scatter(
        name = 'trend',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat']),
        marker=dict(
            color='red',
            line=dict(width=10)
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
                  yaxis=dict(title = 'Order Quantity', ticklen=2, zeroline=True),
                   width=900,
                height=500)

    fig=dict(data=data,layout=layout)
    # plt.savefig('btc03.html')
    st.plotly_chart(fig)


    df_warehouse_j = df_final[['Date', 'Order_Demand_Whse_J']]
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
            line=dict(width=1),
            size=2
        )
    )
    trace1 = go.Scatter(
        name = 'trend',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat']),
        marker=dict(
            color='red',
            line=dict(width=10)
        )
    )
    upper_band = go.Scatter(
        name = 'upper band',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_upper']),
        line= dict(color='#57b88f'),
        fill = 'tonexty'
        # fillcolor='#E0F2F3',
        # opacity=0.1
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
                  yaxis=dict(title = 'Order Quantity', ticklen=2, zeroline=True),
                  width=900,
                height=500)

    fig=dict(data=data,layout=layout)
    # plt.savefig('btc03.html')
    st.plotly_chart(fig)
    return 0

def demand_forecast(file_csv):

    print(file_csv)

    selected_filename = st.selectbox('Load a file to execute',file_csv)
    st.write('You selected `%s`' % selected_filename + '. To perform operations on this file, select your desired operation')
    df_final = pd.read_csv(selected_filename,parse_dates=['Date'],infer_datetime_format=True)

    buttons = ['View EDA','View Seasonality' ,'Forecast using SARIMA','Forecast using FbProphet','Summary']
    check=st.selectbox('Select Operation', buttons, index=0, key=None)

    if check==('View EDA'):
        st.spinner('Execution has started, you can monitor the stats in the command prompt.')
        EDA_Warehouse_Demnds(df_final)
    if check==('View Seasonality'):
        st.spinner('Execution has started, you can monitor the stats in the command prompt.')
        seasonality(df_final)
    if check==('Forecast using SARIMA'):
        st.spinner('Execution has started, you can monitor the stats in the command prompt.')
        ARIMA_model(df_final)
    if check==('Forecast using FbProphet'):
        st.spinner('Execution has started, you can monitor the stats in the command prompt.')
        prophet(df_final)

    if check==('Summary'):
        st.spinner('Execution has started, you can monitor the stats in the command prompt.')
        summary_plots(df_final)

    return 0




def main():


    file_csv=[]
    for f in os.listdir("."):
        if f.endswith('.csv'):
            file_csv.append(f)
    menu_list = ['Home','Category Management','Contract Management','Procure-to-Pay','Strategic Sourcing']
    # Display options in Sidebar
    st.sidebar.title('Navigation')
    menu_sel = st.sidebar.radio('', menu_list, index=0, key=None)


    # Display text in Sidebar
    about.display_sidebar()

    # Selecting About Menu
    if menu_sel == 'Home':
        about.display_about()

    if menu_sel == 'Category Management':
        # st.markdown('# Category Management')
        html_temp = """
        <div style="background-color:#7DCEA0;padding:10px">
        <h2 style="color:black;text-align:center;">Category Management </h2>
           </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)
        cat_level1 =['Category Analytics','Spend Analytics','Savings Lifecycle Analytics']
        a = st.radio('', cat_level1,index=0,key=None)
        if a == 'Category Analytics':
            cat_level2 =['Classification','Consumption Analysis','None']

            b=st.selectbox('Select Sublevel', cat_level2,index=0)
            st.write('You selected `%s`' % b)

            if b == 'Consumption Analysis':
                st.header('Demand Forecasting')
                st.markdown('''
                For the monthly demand for each product in different central  warehouse
                - Products are manufactured in different loaction all over the world
                - Takes more than one month to ship products via ocean to different central ware houses
                
                The task is to do a **Demand Forecast** across multiple warehouses
                ''')
                demand_forecast(file_csv)

        if a=='Spend Analytics':
                cat_level3=['Spend Classification','Spend Forecasting']
                st.selectbox('Select Sublevel', cat_level3,index=0)

        if a=='Savings Lifecycle Analytics':
                cat_level4=['Cost-Savings','Spend vs Budget']
                st.selectbox('Select Sublevel', cat_level4,index=0)



    return 0



if __name__== "__main__":
    main()
