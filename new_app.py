import streamlit as st
import pandas as pd
import numpy as np
# from fbprophet import Prophet
# from fbprophet.diagnostics import performance_metrics
# from fbprophet.diagnostics import cross_validation
# from fbprophet.plot import plot_cross_validation_metric
import base64
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import datetime
from sklearn.metrics import mean_squared_error
import plotly.express as px
st.title('Time Series Forecasting Using Streamlit')

st.write("IMPORT DATA")
st.write("Import the time series csv file. It should have two columns labelled as 'ds' and 'y'.The 'ds' column should be of datetime format  by Pandas. The 'y' column must be numeric representing the measurement to be forecasted.")

data = st.file_uploader('Upload first file here',type='csv')
# data = pd.read_csv('dataset_for_static_model.csv')


data2 = st.file_uploader('Upload second file here',type='csv')



def model_run(appdata_main,select_region,select_team1,select_team2):
    appdata=appdata[appdata['Region']==select_region]
#         appdata=appdata[(appdata['matchName'].str.contains(select_team1)) & (appdata['matchName'].str.contains(select_team2))]    
    df_cat = pd.concat([pd.DataFrame(typeOfDay_cat_encoder.transform(appdata[['typeOfDay']]), columns=typeOfDay_cat_encoder.get_feature_names_out(),
                index = appdata.index),
                pd.DataFrame(Festival_cat_encoder.transform(appdata[['Festival']]), columns=Festival_cat_encoder.get_feature_names_out(),
                index = appdata.index), 
                pd.DataFrame(inning_cat_encoder.transform(appdata[['inning']]), columns=inning_cat_encoder.get_feature_names_out(),
                index = appdata.index),
                pd.DataFrame(region_cat_encoder.transform(appdata[['Region']]), columns=region_cat_encoder.get_feature_names_out(),
                index = appdata.index),
                pd.DataFrame(timeOfDay_cat_encoder.transform(appdata[['timeOfDay']]), columns=timeOfDay_cat_encoder.get_feature_names_out(),
                index = appdata.index)], axis=1)


    df_num = appdata[['Balls', 'team1Fanbase', 'team2Fanbase', 'AvgFirstInningsScore','Pred',col]]
    X = pd.concat([df_num.iloc[:,:-1], df_cat], axis=1)
    y = pd.DataFrame(df_num.loc[:,col])
    offset=len(X)-periods_input
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

#     svr = SVR()

    # train the model on training data
#     svr.fit(X_train.fillna(0), y_train)

    # make predictions on test data
    y_pred = svr.predict(X_test.fillna(0))
    # y_pred=1.33*y_pred
    # calculate mean squared error
#     mse = mean_squared_error(y_test, y_pred)
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
#     st.write(total_pred)


#     st.write("The next visual shows the actual (red line) and predicted (blue line) values over time.") 
    total_pred=target_scaler.inverse_transform(total_pred.reshape(-1, 1))
    total=target_scaler.inverse_transform(total.reshape(-1, 1))
    return total_pred,total


if data is not None and data2 is not None:
    df_new = pd.read_csv(data)
    extra=pd.read_csv(data2)
    appdata_main=extra.merge(df_new, on=['Datetime','inning','matchName','timeOfDay'],how='left',suffixes=('', '_y'))


    appdata_main['Datetime'] = appdata_main['Datetime'].apply(lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x, "%d-%m-%Y %H:%M"), "%Y-%m-%d %H:%M"))    
#     st.write(data)

    max_date = appdata_main['Datetime'].max()


    select_tg = st.sidebar.selectbox('What TG Level?',
                                    ['2-12_male','2-12_female','13-21_male','13-21_female',
                                     '22-30_male','22-30_female','31-40_male','31-40_female',
                                     '41-50_male','41-50_female','51-60_male','51-60_female',
                                     '61+_male','61+_female'])    



    with open('model_'+select_tg+'.pkl', 'rb') as f:
        svr = pickle.load(f)

    team_list1=['CSK', 'MI', 'RCB', 'LSG', 'RR', 'KKR', 'PBKS', 'GT', 'DC', 'SRH']    
    select_team1 = st.sidebar.selectbox('Select Team 1',
                                    team_list1)    
    team_list2=['CSK', 'MI', 'RCB', 'LSG', 'RR', 'KKR', 'PBKS', 'GT', 'DC', 'SRH']    
    team_list2.remove(select_team1)
    select_team2 = st.sidebar.selectbox('Select Team 2',
                                    team_list2)   

    region_list=['AP / Telangana', 'Assam / North East / Sikkim', 'Bihar/Jharkhand',
           'Delhi', 'Guj / D&D / DNH', 'Har/HP/J&K', 'Karnataka', 'Kerala',
           'MP/Chhattisgarh', 'Mah / Goa', 'Odisha', 'Pun/Cha', 'Rajasthan',
           'TN/Pondicherry', 'UP/Uttarakhand', 'West Bengal']

    select_region= st.sidebar.selectbox('Select Region',
                                    region_list)



    periods_input = st.number_input('How many days forecast do you want?',
    min_value = 1, max_value = 365)  
#     select_region = st.text_input('Which Region?')  

    st.write("VISUALIZE FORECASTED DATA")
    st.write("The following plot shows future predicted values. 'yhat' is the predicted value; upper and lower limits are 80% confidence intervals by default")





# if data is not None:
    col=select_tg
#     offset = df[df['Datetime']<'2022-05-21'].shape[0]   ## for train test split

    appdata=appdata_main
    typeOfDay_cat = appdata[['typeOfDay']]

    typeOfDay_cat_encoder = OneHotEncoder(sparse=False)
    typeOfDay_cat_1hot = typeOfDay_cat_encoder.fit_transform(typeOfDay_cat)

    Festival_cat = appdata[['Festival']]

    Festival_cat_encoder = OneHotEncoder(sparse=False)
    Festival_cat_1hot = Festival_cat_encoder.fit_transform(Festival_cat)

    inning_cat = appdata[['inning']]

    inning_cat_encoder = OneHotEncoder(sparse=False)
    inning_cat_1hot = inning_cat_encoder.fit_transform(inning_cat)

    region_cat = appdata[['Region']]

    region_cat_encoder = OneHotEncoder(sparse=False)
    region_cat_1hot = region_cat_encoder.fit_transform(region_cat)

    timeOfDay_cat = appdata[['timeOfDay']]

    timeOfDay_cat_encoder = OneHotEncoder(sparse=False)
    timeOfDay_cat_1hot = timeOfDay_cat_encoder.fit_transform(timeOfDay_cat)
    
    total_pred,total=model_run(appdata,select_region,select_team1,select_team2)
     
    new_new=appdata[appdata['Region']==select_region]
    new_new['pred']=total_pred
    new_new['new_total']=total

    figure1 =px.line(
        data_frame =new_new,
                x = new_new['Datetime'],
            #     x = test_set['Datetime'].astype(str),
            #         y=["rat%_Universe scaled", "rr_reqRR_ratio", "maxSR scaled"],
                y=["new_total","pred"],
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
