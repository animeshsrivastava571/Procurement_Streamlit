import pandas as pd
import streamlit as st
import about
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import norm, skew #for some statistics
from scipy import stats #qqplot
import statsmodels.api as sm #for decomposing the trends, seasonality etc.
from statsmodels.tsa.statespace.sarimax import SARIMAX #the big daddy
from plotly.offline import iplot, init_notebook_mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
import plotly
import plotly.graph_objs as go
import os

# Get JMETER_HOME environment variable
# print('ANi')
def main_about():
    st.title('About')
    st.markdown('---')
    #Display About section


# def EDA():

def demand_forecast(file_csv):

    print(file_csv)

    selected_filename = st.selectbox('Load a file to execute',file_csv)
    st.write('You selected `%s`' % selected_filename + '. To execute this test plan, click on Run button as shown below.')

    if st.button('Run'):
        st.info('Execution has started, you can monitor the stats in the command prompt.')
        # jmeter_execute(selected_filename)
        df_final = pd.read_csv(selected_filename,parse_dates=['Date'],infer_datetime_format=True)

        # st.title('Order Demand Forecasting')
        # st.text('Created by animesh.sr@hcl.com')
        # st.markdown('<style>h1{color: green;}</style>', unsafe_allow_html=True)

        # img = Image.open('Image1.jpg')
        # st.image(img,width=800)

        st.header ('Plotting the Time Series Data for different warehouses')

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


        layout = dict(title = 'Plot for Warehouse Demands',
                      xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                      yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),

                     )
        fig = dict(data = data, layout = layout)
        st.plotly_chart(fig)


        df_resamp = df_final.resample('W', on = 'Date').sum()
        df_resamp = df_resamp.reset_index()

        st.header('Weekly Warhouse Demand')

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


        layout = dict(title = 'Plot for Warehouse Demands - Weekly',
                      xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                      yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),

                     )
        fig = dict(data = data, layout = layout)
        st.plotly_chart(fig)

        st.header ('Monthly Warehouse Demand')

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


        layout = dict(title = 'Plot for Warehouse Demands - Monthly',
                      xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                      yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),

                     )
        fig = dict(data = data, layout = layout)
        st.plotly_chart(fig)

        df_month = df_month.set_index('Date')

        st.header('Seasonality Plot for Warehouse A')

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

                     )
        fig = dict(data = data, layout = layout)
        st.plotly_chart(fig)


        st.header('Seasonality Plot for Warehouse J')

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

                     )
        fig = dict(data = data, layout = layout)
        st.plotly_chart(fig)

        st.header('Seasonality Plot for Warehouse C')

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

                     )
        fig = dict(data = data, layout = layout)
        st.plotly_chart(fig)

        st.header('Seasonality Plot for Warehouse S')
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

                     )
        fig = dict(data = data, layout = layout)
        st.plotly_chart(fig)

        st.header('ARIMA on Warehouse A')
        import itertools
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
                mod = sm.tsa.statespace.SARIMAX(df_month['Order_Demand_Whse_A'], order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
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
            x = test['Order_Demand_Whse_A'].index,
            y = test['Order_Demand_Whse_A'],
            mode = 'lines+markers',
            name = 'Actual Values',
            text= predictions


        )

        data = [trace0, trace1]


        layout = dict(title = 'Plot for Warehouse A - SARIMAX',
                      xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                      yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),

                     )
        fig = dict(data = data, layout = layout)
        st.plotly_chart(fig)

        from sklearn.metrics import mean_squared_error
        from statsmodels.tools.eval_measures import rmse

        # Calculate root mean squared error
        rmse(test["Order_Demand_Whse_A"], predictions)
        df_warha = df_month['Order_Demand_Whse_A']
        mod = sm.tsa.statespace.SARIMAX(df_month['Order_Demand_Whse_A'],
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


        layout = dict(title = 'Warehouse A Forecast',
                      xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
                      yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False),

                     )
        fig = dict(data = data, layout = layout)
        st.plotly_chart(fig)

    return 0




def main():
    print("Ani_main")

    file_csv=[]
    for f in os.listdir("."):
        if f.endswith('.csv'):
            file_csv.append(f)
    menu_list = ['Home','Category Management','Contract Management','Procure-to-Pay','Strategic Sourcing']
    # Display options in Sidebar
    st.sidebar.title('Navigation')
    menu_sel = st.sidebar.radio('', menu_list, index=1, key=None)


    # Display text in Sidebar
    about.display_sidebar()

    # Selecting About Menu
    if menu_sel == 'Home':
        about.display_about()

    # Selecting Execute Menu
    if menu_sel == 'Demand Forecast':
    #jmeter_run = st.radio('Select',('Default','Execute','Analyze'))
    #if jmeter_run == 'Execute':
        st.title('Demand Forecasting')
        demand_forecast(file_csv)
    return 0

    #
    # #if jmeter_run == 'Analyze':
    # if menu_sel == 'Analyze JMeter Test Results':
    #     st.title('Analyze JMeter Test Results')
    #
    #     filename = jmeter_analyze()
    #     st.write('You selected `%s`' % filename)
    #     #DATA_URL = ('C:\\Users\\Navee\\OneDrive\\Documents\\Tools\\apache-jmeter-5.2\\bin\\Run2.csv')
    #     DATA_URL = filename
    #
    #     st.markdown('')
    #     # Show Graphs Checkbox
    #     show_graphs = st.checkbox('Show Graphs')
    #
    #     # Show Profiling Report
    #     profile_report = st.button('Generate Profiling Report')
    #
    #     # Generate Profiling Report
    #
    #     if profile_report:
    #         st.write('Generating Report for ', filename)
    #         pd_profile(filename)
    #
    #
    #     st.title('Apache JMeter Load Test Results')
    #     data = pd.read_csv(DATA_URL)
    #
    #     #Display Start Time
    #     startTime = data['timeStamp'].iloc[0]/1000
    #     startTime = datetime.datetime.fromtimestamp(startTime).strftime('%Y-%m-%d %H:%M:%S')
    #     st.write('Start Time ', startTime)
    #
    #     endTime = data['timeStamp'].iloc[-1]/1000
    #     endTime = datetime.datetime.fromtimestamp(endTime).strftime('%Y-%m-%d %H:%M:%S')
    #     st.write('End Time ', endTime)
    #
    #     FMT = '%Y-%m-%d %H:%M:%S'
    #     delta = datetime.datetime.strptime(endTime, FMT) - datetime.datetime.strptime(startTime, FMT)
    #
    #     st.write('Total duration of the test (HH:MM:SS) is ', delta)
    #
    #     st.subheader('Summary Report - Response Time')
    #     st.write(data.groupby('label')['elapsed'].describe(percentiles=[0.75,0.95,0.99]))
    #
    #     st.subheader('Error Count')
    #     errCount = data.groupby(['label','responseCode'])['responseCode'].count()
    #     st.write(errCount)
    #
    #     if show_graphs:
    #         chart_data = pd.DataFrame(data,columns=['timeStamp','Latency','label','responseCode','elapsed','Connect','bytes'])
    #
    #         st.subheader("Graph between Timestamp and Latency")
    #
    #         st.vega_lite_chart(chart_data, {
    #             "mark": {"type": "bar", "color": "maroon"},
    #             "selection": {
    #                 "grid": {
    #                 "type": "interval", "bind": "scales"
    #                 }
    #             },
    #             'encoding': {
    #                 "tooltip": [
    #             {"field": "timeStamp", "type": "temporal"},
    #             {"field": "label", "type": "nominal"},
    #             {"field": "Latency", "type": "quantitative"}
    #             ],
    #             'x': {'field': 'timeStamp', 'type': 'temporal'},
    #             'y': {'field': 'Latency', 'type': 'quantitative'},
    #             },
    #             })
    #
    #         st.subheader("Graph between Timestamp and Response Code")
    #         st.vega_lite_chart(chart_data, {
    #             "mark": {"type": "bar", "color": "aqua"},
    #             "selection": {
    #                 "grid": {
    #                 "type": "interval", "bind": "scales"
    #                 }
    #             },
    #             'encoding': {
    #                 "tooltip": [
    #             {"field": "timeStamp", "type": "temporal"},
    #             {"field": "label", "type": "nominal"},
    #             {"field": "responseCode", "type": "quantitative"}
    #             ],
    #             'x': {'field': 'timeStamp', 'type': 'temporal'},
    #             'y': {'field': 'responseCode', 'type': 'quantitative'},
    #             },
    #             })
    #
    #         st.subheader("Graph between Timestamp and Response Time")
    #         st.vega_lite_chart(chart_data, {
    #             "mark": {"type": "bar", "color": "orange"},
    #             "selection": {
    #                 "grid": {
    #                 "type": "interval", "bind": "scales"
    #                 }
    #             },
    #             'encoding': {
    #                 "tooltip": [
    #             {"field": "timeStamp", "type": "temporal"},
    #             {"field": "label", "type": "nominal"},
    #             {"field": "elapsed", "type": "quantitative"}
    #             ],
    #             'x': {'field': 'timeStamp', 'type': 'temporal'},
    #             'y': {'field': 'elapsed', 'type': 'quantitative'},
    #             },
    #             })
    #
    #         st.subheader("Graph between Timestamp and Connect Time")
    #         st.vega_lite_chart(chart_data, {
    #             "mark": {"type": "bar", "color": "darkgreen"},
    #             "selection": {
    #                 "grid": {
    #                 "type": "interval", "bind": "scales"
    #                 }
    #             },
    #             'encoding': {
    #                 "tooltip": [
    #             {"field": "timeStamp", "type": "temporal"},
    #             {"field": "label", "type": "nominal"},
    #             {"field": "Connect", "type": "quantitative"}
    #             ],
    #             'x': {'field': 'timeStamp', 'type': 'temporal'},
    #             'y': {'field': 'Connect', 'type': 'quantitative'},
    #             },
    #             })
    #
    #         st.subheader("Graph between Timestamp and bytes")
    #         st.vega_lite_chart(chart_data, {
    #             "mark": {"type": "bar", "color": "darkblue"},
    #             "selection": {
    #                 "grid": {
    #                 "type": "interval", "bind": "scales"
    #                 }
    #             },
    #             'encoding': {
    #                 "tooltip": [
    #             {"field": "timeStamp", "type": "temporal"},
    #             {"field": "label", "type": "nominal"},
    #             {"field": "bytes", "type": "quantitative"}
    #             ],
    #             'x': {'field': 'timeStamp', 'type': 'temporal'},
    #             'y': {'field': 'bytes', 'type': 'quantitative'},
    #             },
    #             })
    #
    #         st.subheader("Graph between Timestamp and Response Time - Line Chart")
    #         st.vega_lite_chart(chart_data, {
    #         "mark": "line",
    #     "encoding": {
    #         "tooltip": [
    #             {"field": "timeStamp", "type": "temporal"},
    #             {"field": "label", "type": "nominal"},
    #             {"field": "elapsed", "type": "quantitative"}
    #             ],
    #         "x": {"field": "timeStamp", "type": "temporal"},
    #         "y": {"field": "elapsed", "type": "quantitative"},
    #         "color": {"field": "label", "type": "nominal"}
    #     },
    #             })
    #
    #         st.subheader("Graph between Timestamp and Response Time - Bar Chart")
    #         st.vega_lite_chart(chart_data, {
    #         "mark": "bar",
    #     "encoding": {
    #         "tooltip": [
    #             {"field": "timeStamp", "type": "temporal"},
    #             {"field": "label", "type": "nominal"},
    #             {"field": "elapsed", "type": "quantitative"}
    #             ],
    #         "x": {"field": "timeStamp", "type": "temporal"},
    #         "y": {"field": "elapsed", "type": "quantitative"},
    #         "color": {"field": "label", "type": "nominal"}
    #     },
    #             })
    #
    #         st.subheader("Histogram")
    #         st.vega_lite_chart(chart_data, {
    #             "transform": [{
    #             "filter": {"and": [
    #             {"field": "timeStamp", "valid": True},
    #             {"field": "elapsed", "valid": True}
    #             ]}
    #         }],
    #         "mark": "rect",
    #         "width": 300,
    #         "height": 200,
    #         "encoding": {
    #             "x": {
    #             "field": "timeStamp",
    #             "type": "temporal"
    #             },
    #             "y": {
    #             "field": "elapsed",
    #             "type": "quantitative"
    #             },
    #             "color": {
    #             "aggregate": "count",
    #             "type": "quantitative"
    #             }
    #         },
    #         "config": {
    #             "view": {
    #             "stroke": "transparent"
    #             }
    #         }
    #                 })
    #
    #         st.subheader("Histogram")
    #         st.vega_lite_chart(chart_data, {
    #             "transform": [{
    #             "filter": {"and": [
    #             {"field": "timeStamp", "valid": True},
    #             {"field": "Connect", "valid": True}
    #             ]}
    #         }],
    #         "mark": "rect",
    #         "width": 300,
    #         "height": 200,
    #         "encoding": {
    #             "x": {
    #             "field": "timeStamp",
    #             "type": "temporal"
    #             },
    #             "y": {
    #             "field": "Connect",
    #             "type": "quantitative"
    #             },
    #             "color": {
    #             "aggregate": "count",
    #             "type": "quantitative"
    #             }
    #         },
    #         "config": {
    #             "view": {
    #             "stroke": "transparent"
    #             }
    #         }
    #                 })
    #
    #         st.subheader("Scatter Plot between Timestamp and Response Time")
    #         st.vega_lite_chart(chart_data, {
    #
    #             "selection": {
    #             "grid": {
    #             "type": "interval", "bind": "scales"
    #             }
    #         },
    #         "mark": "circle",
    #         "encoding": {
    #             "tooltip": [
    #                 {"field": "timeStamp", "type": "temporal"},
    #                 {"field": "label", "type": "nominal"},
    #                 {"field": "elapsed", "type": "quantitative"}
    #                 ],
    #             "x": {
    #             "field": "timeStamp", "type": "temporal"    },
    #             "y": {
    #             "field": "elapsed", "type": "quantitative"    },
    #             "size": {"field": "label", "type": "nominal"}
    #         },
    #                 })

if __name__== "__main__":
    main()
