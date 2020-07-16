#!/usr/bin/env python
# coding: utf-8

# In[64]:

import streamlit as st
import pandas as pd

def forecast():

    st.markdown("## About DeepAR")
    st.markdown(
        """
        The **Amazon SageMaker DeepAR** forecasting algorithm is a supervised learning algorithm for forecasting 
        scalar (one-dimensional) time series using recurrent neural networks (RNN). Classical forecasting 
        methods, such as autoregressive integrated moving average (ARIMA) or exponential smoothing (ETS), 
        fit a single model to each individual time series. They then use that model to extrapolate the time 
        series into the future. The basic algorithm uses **Recurrent Neural Networks.**
        
        In many applications, however, you have many similar time series across a set of cross-sectional units. For example, 
        you might have time series groupings for demand for different products, server loads, and requests for webpages. 
        For this type of application, you can benefit from training a single model jointly over all of the time series. 
        DeepAR takes this approach. When your dataset contains hundreds of related time series, DeepAR outperforms the 
        standard ARIMA and ETS methods. You can also use the trained model to generate forecasts for new time series 
        that are similar to the ones it has been trained on.
        
        
        """
    )
    st.markdown('## Monthly Forecast using DeepAR')
    df = pd.read_csv('Warehouse_monthly.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.info()
    df.tail()

    df_ex = df.set_index('Date')
    custom_dataset = df_ex.to_numpy().T


    # In[68]:
    prediction_length = 12
    freq = "M"
    #custom_dataset = np.random.normal(size=(N, T))
    start = pd.Timestamp("31-01-2012", freq=freq)


    # In[69]:


    from gluonts.dataset.common import ListDataset
    train_ds = ListDataset([{'target': x, 'start': start}
                            for x in custom_dataset[:, :-prediction_length]],
                           freq=freq)
    # test dataset: use the whole dataset, add "target" and "start" fields
    test_ds = ListDataset([{'target': x, 'start': start}
                           for x in custom_dataset],
                          freq=freq)


    # In[70]:


    from gluonts.model.deepar import DeepAREstimator
    from gluonts.trainer import Trainer

    estimator = DeepAREstimator(freq= freq, prediction_length=prediction_length, trainer=Trainer(epochs=10))
    predictor = estimator.train(training_data=train_ds)


    # In[75]:


    from gluonts.evaluation.backtest import make_evaluation_predictions
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )


    # In[76]:


    forecasts = list(forecast_it)
    tss = list(ts_it)


    # In[77]:


    import numpy as np
    ts_entry = tss[0]
    np.array(ts_entry[:5]).reshape(-1,)


    # In[78]:


    #forecast_entry = forecasts[0]
    def plot_prob_forecasts(ts_entry, forecast_entry, wareh_name):
        plot_length = 150
        prediction_intervals = (50.0, 90.0)
        legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

        fig, ax = plt.subplots(1, 1, figsize=(30 ,16))
        ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
        forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
        plt.grid(which="major")
        plt.legend(legend, loc="upper left",fontsize=40)
        plt.title(wareh_name,fontsize=50)
        plt.xlabel('Date',fontsize=40)
        plt.ylabel('Order Demand',fontsize=40)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        st.pyplot()


    # In[79]:


    import matplotlib.pyplot as plt

    ts_entry = tss[0]
    forecast_entry = forecasts[0]
    # matplotlib.rcParams.update({'font.size': 40})
    # get_ipython().run_line_magic('matplotlib', 'inline')
    plot_prob_forecasts(ts_entry, forecast_entry, 'Warehouse A Forecast' )


    # In[80]:


    ts_entry = tss[1]
    forecast_entry = forecasts[1]
    plot_prob_forecasts(ts_entry, forecast_entry, 'Warehouse C Forecast' )


    # In[81]:


    ts_entry = tss[2]
    forecast_entry = forecasts[2]
    plot_prob_forecasts(ts_entry, forecast_entry, 'Warehouse J Forecast' )


    # In[82]:


    ts_entry = tss[3]
    forecast_entry = forecasts[3]
    plot_prob_forecasts(ts_entry, forecast_entry, 'Warehouse S Forecast' )


    # In[83]:


    from gluonts.evaluation import Evaluator
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))


    # In[24]:


    # item_metrics


    metric_df = item_metrics[['MAPE']]
    list_w = ['Warehouse A', 'Warehouse C', 'Warehouse J', 'Warehouse S']
    se = pd.Series(list_w)
    metric_df['Warehouse'] = se.values

    st.markdown("## Look at the MAPE values for different warehouses")
    st.table(metric_df)
    df_wa = df[['Date', 'Order_Demand_Whse_A']]
    #df_wa = df_wa.set_index('Date')
    df_wa = df_wa.rename(columns={"Order_Demand_Whse_A": "y"})

    df_wa = df_wa.set_index('Date')

    from gluonts.dataset.common import ListDataset
    training_data = ListDataset(
        [{"start": df_wa.index[0], "target": df_wa.y[:"2015-12-31"]}],
        freq = "M"
    )


    # In[91]:


    from gluonts.model.deepar import DeepAREstimator
    from gluonts.trainer import Trainer

    estimator = DeepAREstimator(freq="M", prediction_length=12, trainer=Trainer(epochs=10))
    predictor = estimator.train(training_data=training_data)


    # In[92]:


    # df_wa.index[48]


    # In[93]:


    import matplotlib.pyplot as plt
    from pylab import rcParams
    rcParams['figure.figsize'] = 10, 8
    test_data = ListDataset(
        [{"start": df_wa.index[0], "target": df_wa.y[:"2016-12-31"]}],
        freq = "M"
    )


    # In[94]:


    from gluonts.evaluation.backtest import make_evaluation_predictions
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data,  # test dataset
        predictor=predictor,  # predictor
        num_samples=12,  # number of sample paths we want for evaluation
    )


    # In[95]:


    forct = predictor.predict(test_data)


    # In[96]:


    forct = next(iter(forct))


    # In[97]:


    # forct.mean_ts


    # In[98]:


    forecasts = list(forecast_it)
    tss = list(ts_it)


    # In[99]:


    ts_entry = tss[0]


    # In[100]:


    forecast_entry = forecasts[0]


    # In[101]:


    # forecast_entry.mean_ts


    # In[102]:


    def plot_prob_forecasts(ts_entry, forecast_entry):
        plot_length = 150
        prediction_intervals = (50.0, 90.0)
        legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

        fig, ax = plt.subplots(1, 1, figsize=(30, 16))
        ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
        forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
        plt.grid(which="major")
        plt.legend(legend, loc="upper left",fontsize=40)
        plt.title(wareh_name,fontsize=50)
        plt.xlabel('Date',fontsize=40)
        plt.ylabel('Order Demand',fontsize=40)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        st.pyplot()


    # In[103]:


    # plot_prob_forecasts(ts_entry, forecast_entry)


    # In[104]:


    # from gluonts.evaluation import Evaluator
    # evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    # agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))

    # def plot_prob_forecasts1(ts_entry, forecast_entry, wareh_name):
    #     plot_length = 150
    #     prediction_intervals = (50.0, 90.0)
    #     legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    #
    #     fig, ax = plt.subplots(1, 1, figsize=(30, 16))
    #     ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    #     forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    #     plt.grid(which="major")
    #     plt.legend(legend, loc="upper left",fontsize=40)
    #     plt.title(wareh_name,fontsize=50)
    #     plt.xlabel('Date',fontsize=40)
    #     plt.ylabel('Order Demand',fontsize=40)
    #     plt.xticks(fontsize=30)
    #     plt.yticks(fontsize=30)
    #     st.pyplot()
    #
    # # rcParams['figure.figsize'] = 8, 8
    # def my_func_week(df_wa, wareh_name):
    #     training_data = ListDataset(
    #     [{"start": df_wa.index[0], "target": df_wa.y[:"2016-01-03"]}],
    #     freq = "W")
    #     estimator = DeepAREstimator(freq="W", prediction_length=52, trainer=Trainer(epochs=10))
    #     predictor = estimator.train(training_data=training_data)
    #     test_data = ListDataset(
    #     [{"start": df_wa.index[0], "target": df_wa.y[:"2017-01-01"]}],
    #     freq = "W")
    #     forecast_it, ts_it = make_evaluation_predictions(dataset=test_data,  predictor=predictor, num_samples=52,)
    #     forecasts = list(forecast_it)
    #     tss = list(ts_it)
    #     ts_entry = tss[0]
    #     forecast_entry = forecasts[0]
    #     plot_prob_forecasts1(ts_entry, forecast_entry, wareh_name)
    #     evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    #     agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))
    #     forecasts = list(forecast_it)
    #     mape = agg_metrics['MAPE']
    #     return mape
    #
    #     df = pd.read_csv('warehouse_weekly_data.csv', index_col = 0)
    #     df['Date'] = pd.to_datetime(df['Date'])
    #     df[df['Date'] >='2015-12-26']
    #     df_wj = df[['Date', 'Order_Demand_Whse_J']]
    #     df_wj = df_wj.rename(columns={"Order_Demand_Whse_J": "y"})
    #     df_wj = df_wj.set_index('Date')
    #     wh = 'Warehouse J forecast'
    #     mape_a = my_func_week(df_wj, wh)


    print("Success")





