import streamlit as st
import pandas as pd
import numpy as np
# from fbprophet import Prophet
# from fbprophet.diagnostics import performance_metrics
# from fbprophet.diagnostics import cross_validation
# from fbprophet.plot import plot_cross_validation_metric
import base64
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import datetime
from sklearn.metrics import mean_squared_error
import plotly.express as px
st.title('Time Series Forecasting Using Streamlit')

st.write("IMPORT DATA")
st.write("Import the time series csv file. It should have two columns labelled as 'ds' and 'y'.The 'ds' column should be of datetime format  by Pandas. The 'y' column must be numeric representing the measurement to be forecasted.")

data = st.file_uploader('Upload here',type='csv')
# data = pd.read_csv('dataset_for_static_model.csv')

if data is not None:
    appdata = pd.read_csv(data)
    appdata['Datetime'] = appdata['Datetime'].apply(lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x, "%d-%m-%Y %H:%M"), "%Y-%m-%d %H:%M"))    
    st.write(data)
    
    max_date = appdata['Datetime'].max()

st.write("SELECT FORECAST PERIOD")

periods_input = st.number_input('How many days forecast do you want?',
min_value = 1, max_value = 365)

# if data is not None:
#     obj = Prophet()
#     obj.fit(appdata)

st.write("VISUALIZE FORECASTED DATA")
st.write("The following plot shows future predicted values. 'yhat' is the predicted value; upper and lower limits are 80% confidence intervals by default")

if data is not None:
    col='Total'
#     offset = df[df['Datetime']<'2022-05-21'].shape[0]   ## for train test split
    offset=len(appdata)-periods_input
    typeOfDay_cat = appdata[['typeOfDay']]

    typeOfDay_cat_encoder = OneHotEncoder(sparse=False)
    typeOfDay_cat_1hot = typeOfDay_cat_encoder.fit_transform(typeOfDay_cat)

    Festival_cat = appdata[['Festival']]

    Festival_cat_encoder = OneHotEncoder(sparse=False)
    Festival_cat_1hot = Festival_cat_encoder.fit_transform(Festival_cat)

    inning_cat = appdata[['inning']]

    inning_cat_encoder = OneHotEncoder(sparse=False)
    inning_cat_1hot = inning_cat_encoder.fit_transform(inning_cat)

#     region_cat = df[['Region']]

#     region_cat_encoder = OneHotEncoder(sparse=False)
#     region_cat_1hot = region_cat_encoder.fit_transform(region_cat)

    timeOfDay_cat = appdata[['timeOfDay']]

    timeOfDay_cat_encoder = OneHotEncoder(sparse=False)
    timeOfDay_cat_1hot = timeOfDay_cat_encoder.fit_transform(timeOfDay_cat)

    df_cat = pd.concat([pd.DataFrame(typeOfDay_cat_encoder.transform(appdata[['typeOfDay']]), columns=typeOfDay_cat_encoder.get_feature_names_out(),
                index = appdata.index),
                pd.DataFrame(Festival_cat_encoder.transform(appdata[['Festival']]), columns=Festival_cat_encoder.get_feature_names_out(),
                index = appdata.index), 
                pd.DataFrame(inning_cat_encoder.transform(appdata[['inning']]), columns=inning_cat_encoder.get_feature_names_out(),
                index = appdata.index),
#                 pd.DataFrame(region_cat_encoder.transform(df[['Region']]), columns=region_cat_encoder.get_feature_names_out(),
#                 index = df.index),
                pd.DataFrame(timeOfDay_cat_encoder.transform(appdata[['timeOfDay']]), columns=timeOfDay_cat_encoder.get_feature_names_out(),
                index = appdata.index)], axis=1)


    df_num = appdata[['Balls', 'team1Fanbase', 'team2Fanbase', 'AvgFirstInningsScore',col]]
    X = pd.concat([df_num.iloc[:,:-1], df_cat], axis=1)
    y = pd.DataFrame(df_num.loc[:,col])

    X_train = X.loc[0:offset-1,:]
    X_test = X.loc[offset:,:]

    y_train = y.loc[0:offset-1,:]
    y_test = y.loc[offset:,:]

    feature_scaler = StandardScaler()
    X_train[['Balls', 'team1Fanbase', 'team2Fanbase', 'AvgFirstInningsScore']] = feature_scaler.fit_transform(X_train[['Balls', 'team1Fanbase', 'team2Fanbase', 'AvgFirstInningsScore']])

    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train)

    X_test[['Balls', 'team1Fanbase', 'team2Fanbase', 'AvgFirstInningsScore']] = feature_scaler.transform(X_test[['Balls', 'team1Fanbase', 'team2Fanbase', 'AvgFirstInningsScore']])
    y_test = target_scaler.transform(y_test)

    svr = SVR()

    # train the model on training data
    svr.fit(X_train.fillna(0), y_train)

    # make predictions on test data
    y_pred = svr.predict(X_test.fillna(0))
    # y_pred=1.33*y_pred
    # calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    y_pred1=svr.predict(X_train.fillna(0))
    # mse = mean_squared_error(y_train, y_pred1)
    # print(mse)
    # print mean squared error

    total_pred=np.concatenate([y_pred1, y_pred])
    total=np.concatenate([y_train, y_test])
#     mse = mean_squared_error(total, total_pred)

#     # print mean squared error
#     print("Mean squared error: ", mse)
#     df['Pred_'+col]=target_scaler.inverse_transform(total_pred.reshape(-1, 1))                     
#     print("For the col ",col," mean squared error: ", mse)


#     forecast = fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

#     forecast_filtered =  forecast[forecast['ds'] > max_date]    
    st.write(total_pred)

    
    st.write("The next visual shows the actual (red line) and predicted (blue line) values over time.")    

    appdata['pred']=target_scaler.inverse_transform(total_pred.reshape(-1, 1))

    figure1 =px.line(
        data_frame =appdata,
                x = appdata['Datetime'],
            #     x = test_set['Datetime'].astype(str),
            #         y=["rat%_Universe scaled", "rr_reqRR_ratio", "maxSR scaled"],
                y=["Total","pred"],
            color_discrete_sequence=['red', "blue"])
            #     symbol=sample_df['timeOfDay'],
            #     symbol_sequence=['circle-open', 'square'],
            #         y = 'rat%_Universe',
            #     opacity = 0.9,
            #     orientation = "v",
            #     barmode = 'group',
            #     text="match_name"


            # fig2.update_traces(textposition=sample_df['textPosition'])

            #     fig2.add_scatter(x=sample_df['Start Time'], y=sample_df['RR scaled'], name="run rate")

            #     fig2.add_trace(go.Table(cells={"values":df.T.values}, header={"values":df.columns}), row=1,col=1)


            # fig2.update_xaxes(tickangle=290)
    figure1.update_layout(showlegend=True,font=dict(family="Courier New",size=12,color='Black'),
                                   title=f"SVR with linear kernel model prediction",
                                   xaxis_title="Time of day",
                                   yaxis_title="Predicted Viewership",
                                   width=1000,height=500)

    st.write(figure1)
 
    
#     st.write("The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.")
      

#     figure2 = obj.plot_components(fcst)
#     st.write(figure2)
