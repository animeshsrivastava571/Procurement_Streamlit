import pandas as pd
import streamlit as st
import about
import itertools
import statsmodels.api as sm #for decomposing the trends, seasonality etc.
import plotly.graph_objs as go
import os
import deepAr
from fbprophet import Prophet



def summary_plots(df_final):
    #Running Boxplot on Warehouse J

    lst_warehouse=list(df_final.columns)[2:]
    traces = []

    for whouse in lst_warehouse:
        s = df_final[whouse].to_frame().reset_index(drop=True)
        trace = go.Box(y= s[whouse], name= 'Warehouse {}'.format(whouse), jitter=0.8, whiskerwidth=0.2, marker=dict(size=2), line=dict(width=1))
        traces.append(trace)

    layout = go.Layout(
    title='Order Demand Boxplot Across Different Warehouses - Weekly',
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
        name = 'CI upper limit',
        text= pred_ci.iloc[:, 0])


    trace1 = go.Scatter(
        x = pred.predicted_mean.index,
        y = pred.predicted_mean,
        mode = 'lines',
        name = 'predicted values',
        text= predictions,
        fill = 'tonexty'
    )

    trace2 = go.Scatter(
        x = pred_ci.index,
        y = pred_ci.iloc[:, 0],
        mode = 'lines',
        name = 'CI lower limit',
        text= pred_ci.iloc[:, 0],
        fill = 'tonexty')



    trace3 = go.Scatter(
        x = df_warha['2015-06':].index,
        y = df_warha['2015-06':],
        mode = 'lines',
        name = 'actual demand quantity',
        text= df_warha['2015-06':]
    #     fill = 'tonexty'
    )





    pred_uc = results.get_forecast(steps=12)
    pred_ci1 = pred_uc.conf_int()

    trace4 = go.Scatter(
        x =pred_ci1.index,
        y = pred_ci1.iloc[:, 1],
        mode = 'lines',
        name = 'upper CI',
    #     fill = 'tonexty'



    )

    trace5 = go.Scatter(
        x =pred_uc.predicted_mean.index,
        y = pred_uc.predicted_mean,
        mode = 'lines',
        name = 'forecast',
        fill = 'tonexty')

    trace6 = go.Scatter(
        x =pred_ci1.index,
        y = pred_ci1.iloc[:, 0],
        mode = 'lines',
        name = 'lower CI',
        fill = 'tonexty'



    )

    data = [trace0, trace1,trace2,trace3,trace4,trace5,trace6]


    layout = dict(title = 'Warehouse J Forecast SARIMA - Weekly',
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
        name = 'actual demand quantity',
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
        name = 'predicted values',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat']),
        marker=dict(
            color='red',
            line=dict(width=10)
        )
    )
    upper_band = go.Scatter(
        name = 'CI upper limit',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_upper']),
        line= dict(color='#57b88f'),
        fill = 'tonexty'
        # fillcolor='#E0F2F3',
        # opacity=0.1
    )
    lower_band = go.Scatter(
        name= 'CI lower limit',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_lower']),
        line= dict(color='#1705ff')
    )
    tracex = go.Scatter(
        name = 'actual demand quantity',
       mode = 'markers',
       x = list(df_j['ds']),
       y = list(df_j['y']),
       marker=dict(
          color='black',
          line=dict(width=2)
       )
    )
    data = [trace1, lower_band, upper_band, trace]

    layout = dict(title='Warehouse J forecast FbProphet - Weekly',
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
    # Plot 1: BoxPlot
    lst_warehouse=list(df_final.columns)[2:]
    traces = []

    for whouse in lst_warehouse:
        s = df_final[whouse].to_frame().reset_index(drop=True)
        trace = go.Box(y= s[whouse], name= 'Warehouse {}'.format(whouse), jitter=0.8, whiskerwidth=0.2, marker=dict(size=2), line=dict(width=1))
        traces.append(trace)

    layout = go.Layout(
    title=' Weekly Order Demand Boxplot Across Different Warehouses',
    yaxis=dict(
        autorange=True, showgrid=True, zeroline=True,
        gridcolor='rgb(233,233,233)', zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2, gridwidth=1),
    xaxis=dict(tickangle=15),
    margin=dict(l=40, r=30, b=80, t=100), showlegend=False,
    width=900,
                height=500,
    hoverlabel=dict(
        bgcolor="white",
        font_size=10,
        font_family="Rockwell"

    )
    )


    fig1 = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig1)

    #Plot 2: Simple Time Series
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


    layout = dict(title = 'Warehouse Demand - Daily',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                  height=500,
                  hoverlabel=dict(
                                    bgcolor="white",
                                    font_size=10,
                                    font_family="Rockwell"

    )

    )
    fig2 = dict(data = data, layout = layout)

    st.plotly_chart(fig2)

    #Plot 3: Weekly Simple Time Series
    df_week = df_final.resample('W', on = 'Date').sum()
    df_week = df_week.reset_index()

    # st.header('Weekly Warhouse Demand')

    # Create traces
    trace0 = go.Scatter(
        x = df_week['Date'],
        y = df_week['Order_Demand_Whse_A'],
        mode = 'lines',
        name = 'Warehouse A',
        text= df_week['Order_Demand_Whse_A']


    )

    trace1 = go.Scatter(
        x = df_week['Date'],
        y = df_week['Order_Demand_Whse_C'],
        mode = 'lines',
        name = 'Warehouse C',
        text= df_week['Order_Demand_Whse_C']
    )

    trace2 = go.Scatter(
        x = df_week['Date'],
        y = df_week['Order_Demand_Whse_J'],
        mode = 'lines',
        name = 'Warehouse J',
        text= df_week['Order_Demand_Whse_J']
    )

    trace3 = go.Scatter(
        x = df_week['Date'],
        y = df_week['Order_Demand_Whse_S'],
        mode = 'lines',
        name = 'Warehouse S',
        text= df_week['Order_Demand_Whse_S']

    )

    data = [trace0, trace1, trace2, trace3]


    layout = dict(title = 'Warehouse Demand - Weekly',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                  height=500,
                  hoverlabel=dict(
                                    bgcolor="white",
                                    font_size=10,
                                    font_family="Rockwell"

    )

                 )
    fig3 = dict(data = data, layout = layout)
    st.plotly_chart(fig3)

    # Plot 4: Monthly Time Series

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


    layout = dict(title = 'Warehouse Demand - Monthly',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                  height=500,
                  hoverlabel=dict(
                                    bgcolor="white",
                                    font_size=10,
                                    font_family="Rockwell"

    )

                 )
    fig4 = dict(data = data, layout = layout)
    st.plotly_chart(fig4)
    return (fig1,fig2,fig3,fig4)

def seasonality(df_final):

    df_month = df_final.resample('M', on = 'Date').sum()
    df_month = df_month.reset_index()



    df_month = df_month.set_index('Date')

    st.markdown(' ## Seasonality Plot for different warehouses')

    def plot_seasonality(decomposition,whouse):

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


        layout = dict(title = 'Plot for Warehouse {} - Seasonality'.format(whouse),
                      xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                      yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                      width=900,
                      height=500,
                      hoverlabel=dict(
                                    bgcolor="white",
                                    font_size=10,
                                    font_family="Rockwell"

    )

                     )
        fig = dict(data = data, layout = layout)
        st.plotly_chart(fig)
        return 0

    decomposition = sm.tsa.seasonal_decompose(df_month['Order_Demand_Whse_A'], model='additive')
    plot_seasonality(decomposition,'A')

    # Create traces
    decomposition = sm.tsa.seasonal_decompose(df_month['Order_Demand_Whse_J'], model='additive')
    plot_seasonality(decomposition,'J')

    decomposition = sm.tsa.seasonal_decompose(df_month['Order_Demand_Whse_S'], model='additive')
    plot_seasonality(decomposition,'S')

    decomposition = sm.tsa.seasonal_decompose(df_month['Order_Demand_Whse_C'], model='additive')
    plot_seasonality(decomposition,'C')

    return 0

def ARIMA_model(df_final):
    st.markdown("## About SARIMA")
    st.markdown(
        """
        In statistics and econometrics, and in particular in time series analysis, an autoregressive 
        integrated moving average (ARIMA) model is a generalization of an autoregressive moving average (ARMA) 
        model. Both of these models are fitted to time series data either to better understand the data or to 
        predict future points in the series (forecasting). ARIMA models are applied in some cases where 
        data show evidence of non-stationarity, where an initial differencing step 
        (corresponding to the "integrated" part of the model) can be applied one or more times to 
        eliminate the non-stationarity. **SARIMA** is an extension of the ARIMA model which includes the seasonality
        component.
        """
    )

    st.markdown('## SARIMA on Warehouse J - Weekly Demand')
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

    # for param in pdq:
    #     for param_seasonal in seasonal_pdq:
    #         mod = sm.tsa.statespace.SARIMAX(df_month['Order_Demand_Whse_J'], order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
    #         results = mod.fit()
    #         print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))


    mod = sm.tsa.statespace.SARIMAX(train['Order_Demand_Whse_J'],
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    df_results=(pd.DataFrame(results.summary().tables[1]))
    header=df_results.iloc[0]
    df_results.columns=header
    df_results =df_results.iloc[1:,:]
    st.table(df_results)

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


    layout = dict(title = 'Warehouse J - Actual vs Predicted - Weekly Demand',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                height=500,
                hoverlabel=dict(
                                    bgcolor="white",
                                    font_size=10,
                                    font_family="Rockwell"

    )

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


    layout = dict(title = 'Warehouse J Forecast - Weekly',
                  xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                  yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),
                  width=900,
                height=500,
                hoverlabel=dict(
                                    bgcolor="white",
                                    font_size=10,
                                    font_family="Rockwell"

    )

                 )
    fig = dict(data = data, layout = layout)
    st.plotly_chart(fig)
    return 0

def prophet(df_final):

    st.markdown("## About FbProphet")
    st.markdown(
        """
        Developed by **Facebook's core Data Science team**, Prophet is a procedure for forecasting time series data based on an additive model where non-linear 
        trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best
         with time series that have strong seasonal effects and several seasons of historical data. 
         Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.
         - Accurate and fast
         - Fully automatic
         - Tunable Forecast
        """
    )

    df_warehouse_s = df_final[['Date', 'Order_Demand_Whse_S']]
    df_s = df_warehouse_s.rename(columns={"Date": "ds", "Order_Demand_Whse_S": "y"})
    prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,)
    prophet.fit(df_s)
    future = prophet.make_future_dataframe(periods=12, freq='M')
    df_forecast = prophet.predict(future)
    st.markdown('## Demand Forecasting using FbProphet - Weekly')
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
        name = 'predicted',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat']),
        marker=dict(
            color='red',
            line=dict(width=10)
        )
    )
    upper_band = go.Scatter(
        name = 'CI upper limit',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_upper']),
        line= dict(color='#57b88f'),
        fill = 'tonexty'
    )
    lower_band = go.Scatter(
        name= 'CI lower limit',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_lower']),
        line= dict(color='#1705ff')
    )

    data = [trace1, lower_band, upper_band, trace]

    layout = dict(title='Order Demand Forecasting - Warehouse S',
                 xaxis=dict(title = 'Dates', ticklen=2, zeroline=True),
                  yaxis=dict(title = 'Order Quantity', ticklen=2, zeroline=True),
                   width=900,
                height=500,
                  hoverlabel=dict(
                                    bgcolor="white",
                                    font_size=10,
                                    font_family="Rockwell"

    ))

    fig=dict(data=data,layout=layout)
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
        name = 'predicted',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat']),
        marker=dict(
            color='red',
            line=dict(width=10)
        )
    )
    upper_band = go.Scatter(
        name = 'CI upper limit',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_upper']),
        line= dict(color='#57b88f'),
        fill = 'tonexty'
        # fillcolor='#E0F2F3',
        # opacity=0.1
    )
    lower_band = go.Scatter(
        name= 'CI lower limit',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_lower']),
        line= dict(color='#1705ff')
    )

    data = [trace1, lower_band, upper_band, trace]

    layout = dict(title='Order Demand Forecasting - Warehouse J',
                 xaxis=dict(title = 'Dates', ticklen=2, zeroline=True),
                  yaxis=dict(title = 'Order Quantity', ticklen=2, zeroline=True),
                  width=900,
                height=500,
                  hoverlabel=dict(
                                    bgcolor="white",
                                    font_size=10,
                                    font_family="Rockwell"

    ))

    fig=dict(data=data,layout=layout)
    # plt.savefig('btc03.html')
    st.plotly_chart(fig)
    return 0

def demand_forecast(file_csv):

    print(file_csv)

    selected_filename = st.selectbox('Load a file to execute',file_csv)
    st.write('You selected `%s`' % selected_filename + '. To perform analysis on this file, select your desired operation')
    df_final = pd.read_csv(selected_filename,parse_dates=['Date'],infer_datetime_format=True)

    buttons = ['View EDA','View Seasonality' ,'Forecast using SARIMA','Forecast using FbProphet','Forecast using DeepAR','Summary']
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
    if check==('Forecast using DeepAR'):
        st.spinner('Execution has started, you can monitor the stats in the command prompt.')
        deepAr.forecast()

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
        <div style="background-color:#8E1047;padding:10px">
        <h2 style="color:white;text-align:center;">Category Management </h2>
           </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)
        cat_level1 =['Category Analytics','Spend Analytics','Savings Lifecycle Analytics']
        a = st.radio('', cat_level1,index=0,key=None)
        if a == 'Category Analytics':
            cat_level2 =['Classification','Consumption Analysis']

            b=st.selectbox('Select Sublevel', cat_level2,index=1)
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
