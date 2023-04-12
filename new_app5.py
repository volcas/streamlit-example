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
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
import plotly.express as px
st.title('Viewership Forecasting using Regression')

# st.write("IMPORT DATA")
# st.write("Import the time series csv file. It should have two columns labelled as 'ds' and 'y'.The 'ds' column should be of datetime format  by Pandas. The 'y' column must be numeric representing the measurement to be forecasted.")

# data = st.file_uploader('Upload first file here',type='csv')
# data='TS_Reg.csv'


# data2 = st.file_uploader('Upload second file here',type='csv')

### FOR THE RESULT FILE

from google.oauth2 import service_account
from gsheetsdb import connect

# Create a connection object.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
    ],
)
conn = connect(credentials=credentials)

# Perform SQL query on the Google Sheet.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_resource(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows

# BARC MODEL(Model 1) Prediction
sheet_url = st.secrets["private_gsheets_url_1"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')
model1_data=pd.DataFrame.from_records(rows)

model1_data.columns=['Date', 'Time', 'team1', 'team2', 'Venue', 'Stadium','Balls', 
       'team1Fanbase', 'team2Fanbase', 'typeOfDay', 'Festival', 'inning',
       'timeOfDay', 'AvgFirstInningsScore', 'Universe_total_prediction']

# sheet_url = st.secrets["private_gsheets_url_2"]
# rows = run_query(f'SELECT * FROM "{sheet_url}"')

# innings1_result=pd.DataFrame([['Universe_total',0.095298871,0.162203503]], columns=['col', 'MSE', 'MAPE'])

# innings1_result=pd.DataFrame.from_records(rows)
# innings1_result.columns=['col', 'MSE', 'MAPE']

sheet_url = st.secrets["private_gsheets_url_2"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')
actual_data=pd.DataFrame.from_records(rows)
actual_data.columns=['Time', 'Concurrency Source', 'Mux', 'Last9', 'Manual Override',
       'Reported Concurrency', 'Final Concurrency',
       'Final Concurrency Display', 'Final Concurrency Source', 'Normalized',
       'Datetime']


# BARC MODEL+ IPL Model(Model 2) Prediction
sheet_url = st.secrets["private_gsheets_url_3"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')
model2_data=pd.DataFrame.from_records(rows)

model2_data.columns=['Date', 'Time', 'team1', 'team2', 'Venue', 'Stadium','Balls', 
       'team1Fanbase', 'team2Fanbase', 'typeOfDay', 'Festival', 'inning',
       'timeOfDay', 'AvgFirstInningsScore', 'Universe_total_prediction']




# region_list=['AP / Telangana', 'Assam / North East / Sikkim', 'Bihar/Jharkhand',
#        'Delhi', 'Guj / D&D / DNH', 'Har/HP/J&K', 'Karnataka', 'Kerala',
#        'MP/Chhattisgarh', 'Mah / Goa', 'Odisha', 'Pun/Cha', 'Rajasthan',
#        'TN/Pondicherry', 'UP/Uttarakhand', 'West Bengal']

# select_region= st.sidebar.selectbox('Select Region',
#                                     region_list)

# select_tg = st.sidebar.selectbox('What TG Level?',
#                                 ['2-12_male','2-12_female','13-21_male','13-21_female',
#                                  '22-30_male','22-30_female','31-40_male','31-40_female',
#                                  '41-50_male','41-50_female','51-60_male','51-60_female',
#                                  '61+_male','61+_female'])    



match_schedule={'2023-03-31': ['19:30:00'],
  '2023-04-01': ['19:30:00', '15:30:00'],
 '2023-04-02': ['19:30:00', '15:30:00'],
 '2023-04-03': ['19:30:00'],
 '2023-04-04': ['19:30:00'],
 '2023-04-05': ['19:30:00'],
 '2023-04-06': ['19:30:00'],
 '2023-04-07': ['19:30:00'],
 '2023-04-08': ['15:30:00', '19:30:00'],
 '2023-04-09': ['19:30:00', '15:30:00'],
 '2023-04-10': ['19:30:00'],
 '2023-04-11': ['19:30:00'],
 '2023-04-12': ['19:30:00'],
 '2023-04-13': ['19:30:00'],
 '2023-04-14': ['19:30:00'],
 '2023-04-15': ['15:30:00', '19:30:00'],
 '2023-04-16': ['19:30:00', '15:30:00'],
 '2023-04-17': ['19:30:00'],
 '2023-04-18': ['19:30:00'],
 '2023-04-19': ['19:30:00'],
 '2023-04-20': ['15:30:00', '19:30:00'],
 '2023-04-21': ['19:30:00'],
 '2023-04-22': ['19:30:00', '15:30:00'],
 '2023-04-23': ['15:30:00', '19:30:00'],
 '2023-04-24': ['19:30:00'],
 '2023-04-25': ['19:30:00'],
 '2023-04-26': ['19:30:00'],
 '2023-04-27': ['19:30:00'],
 '2023-04-28': ['19:30:00'],
 '2023-04-29': ['19:30:00', '15:30:00'],
 '2023-04-30': ['15:30:00', '19:30:00'],
 '2023-05-01': ['19:30:00'],
 '2023-05-02': ['19:30:00'],
 '2023-05-03': ['19:30:00'],
 '2023-05-04': ['19:30:00', '15:30:00'],
 '2023-05-05': ['19:30:00'],
 '2023-05-06': ['15:30:00', '19:30:00'],
 '2023-05-07': ['20:00:00', '19:30:00'],
 '2023-05-08': ['19:30:00'],
 '2023-05-09': ['19:30:00'],
 '2023-05-10': ['19:30:00'],
 '2023-05-11': ['19:30:00'],
 '2023-05-12': ['19:30:00'],
 '2023-05-13': ['15:30:00', '19:30:00'],
 '2023-05-14': ['19:30:00', '15:30:00'],
 '2023-05-15': ['19:30:00'],
 '2023-05-16': ['19:30:00'],
 '2023-05-17': ['19:30:00'],
 '2023-05-18': ['19:30:00'],
 '2023-05-19': ['19:30:00'],
 '2023-05-20': ['19:30:00', '15:30:00'],
 '2023-05-21': ['19:30:00', '15:30:00']}

date_list=list(match_schedule.keys())    
select_date = st.sidebar.selectbox('Select Date',
                                date_list)  

time_list=list(match_schedule[select_date])    
select_time = st.sidebar.selectbox('Select Time',
                                time_list)   



# Combining actual with predictions Model 1

actual_data['Datetime']=pd.to_datetime(actual_data['Time'])
# st.write(model1_data['Date'].dtype)

model1_data['Date']=pd.to_datetime(model1_data['Date']).dt.strftime('%Y-%m-%d')
# st.write(model1_data['Date'])

model1_data['Time']=pd.to_datetime(model1_data['Time']).dt.time
model1_data['Time']=model1_data['Time'].astype(str)

model1_data['Datetime']=pd.to_datetime(model1_data['Date'] + " " + model1_data['Time'], format="%Y-%m-%d %H:%M:%S")

# st.write(model1_data['Datetime'])


# Removing the first match
# model1_data=model1_data[model1_data['Date']!='31-03-2023']


combined_df=model1_data.merge(actual_data,on='Datetime',how='left',suffixes=('', '_y'))
# st.write(actual_data['Datetime'])

# Removing matches yet to happen
combined_df.dropna(inplace=True)


appdata=combined_df.copy()

timeOfDay='evening'
if select_time=='15:30:00':
    timeOfDay='afternoon'


appdata=appdata[(appdata['Date'].str.contains(select_date)) & (appdata['timeOfDay']==timeOfDay)] 
# appdata=appdata[appdata['tg_col']==col]                          
appdata=appdata.reset_index().drop('index',1)

if not appdata.empty:

    st.write("Match:",appdata['team1'][0]," vs ",appdata['team2'][0], " on ",appdata['Date'][0])
#     st.write(":blue[",appdata['team1'][0]," vs ",appdata['team2'][0], " on ",appdata['Date'][0],"]")    

    cc1=appdata["Mux"].max()
    st.write("The peak MUX concurrency of the chosen match:",cc1)
#     '$This text is italicized $'
    st.markdown('*Note: The peak MUX concurrency is wrt to the five minute interval, the actual peak may differ*')
#     cc2=appdata["Last9"].max()
#     st.write("The peak MUX concurrency of the chosen match:",cc2)
    
    cc3=appdata["Universe_total_prediction"].max()
    st.write("The peak prediction(BARC Model) concurrency of the chosen match:",cc3)

    #     st.header("Model result metrics for the TG: Innings 1")
    #     st.write(resultdata[['col','MAPE']])



    # for date in np.unique(combined_df['Date']):
    #     for time in ['afternoon', 'evening']:
    #         for inning in ['inning1','inning2']:
    #         new=combined_df[(combined_df['Date']==date) & (combined_df['timeOfDay']==time)]
    mape = round(mean_absolute_percentage_error(appdata['Mux'], appdata['Universe_total_prediction']),2)

    figure1 =px.line(
                data_frame =appdata,
                        x = appdata['Datetime'],
                        y=["Mux","Universe_total_prediction"],
        color_discrete_sequence=['green','blue'],
    #                     text=mape
    )

                    # fig2.update_traces(textposition=sample_df['textPosition'])

                    #     fig2.add_scatter(x=sample_df['Start Time'], y=sample_df['RR scaled'], name="run rate")

                    #     fig2.add_trace(go.Table(cells={"values":df.T.values}, header={"values":df.columns}), row=1,col=1)


                    # fig2.update_xaxes(tickangle=290)
    figure1.update_layout(showlegend=True,font=dict(family="Courier New",size=12,color='Black'),
                                   title="MAPE:"+str(mape),
                                   xaxis_title="Time of day",
                                   yaxis_title="Concurrency",
                                   width=800,height=500)

    st.write(figure1)

else:
    appdata=model1_data.copy()

    timeOfDay='evening'
    if select_time=='15:30:00':
        timeOfDay='afternoon'


    appdata=appdata[(appdata['Date'].str.contains(select_date)) & (appdata['timeOfDay']==timeOfDay)] 
    # appdata=appdata[appdata['tg_col']==col]                          
    appdata=appdata.reset_index().drop('index',1)
    
    if not appdata.empty:

            st.write("Match:",appdata['team1'][0]," vs ",appdata['team2'][0], " on ",appdata['Date'][0])
#             st.markdown(":blue[",appdata['team1'][0]," vs ",appdata['team2'][0], " on ",appdata['Date'][0],"]")    

#             cc1=appdata["Mux"].max()
#             st.write("The peak MUX concurrency of the chosen match:",cc1)
#             st.write("No actual available data for the chosen match")
            st.markdown(":blue[No actual available data for the chosen match]")


            cc2=appdata["Universe_total_prediction"].max()
            st.write("The peak prediction(BARC Model) concurrency of the chosen match:",cc2)

            #     st.header("Model result metrics for the TG: Innings 1")
            #     st.write(resultdata[['col','MAPE']])



            # for date in np.unique(combined_df['Date']):
            #     for time in ['afternoon', 'evening']:
            #         for inning in ['inning1','inning2']:
            #         new=combined_df[(combined_df['Date']==date) & (combined_df['timeOfDay']==time)]
#             mape = mean_absolute_percentage_error(appdata['Mux'], appdata['Universe_total_prediction'])

            figure1 =px.line(
                        data_frame =appdata,
                                x = appdata['Datetime'],
                                y=["Universe_total_prediction"],
                color_discrete_sequence=["blue"],
            #                     text=mape
            )

                            # fig2.update_traces(textposition=sample_df['textPosition'])

                            #     fig2.add_scatter(x=sample_df['Start Time'], y=sample_df['RR scaled'], name="run rate")

                            #     fig2.add_trace(go.Table(cells={"values":df.T.values}, header={"values":df.columns}), row=1,col=1)


                            # fig2.update_xaxes(tickangle=290)
            figure1.update_layout(showlegend=True,font=dict(family="Courier New",size=12,color='Black'),
                                           title="BARC Model Prediction",
                                           xaxis_title="Time of day",
                                           yaxis_title="Concurrency",
                                           width=800,height=500)

            st.write(figure1)
            
            
    else:
        st.markdown(":blue[Neither Predictions nor actual data is available]")
        

st.header("MODEL 2")        
# FOR MODEL 2        

# Combining actual with predictions Model 2

# actual_data['Datetime']=pd.to_datetime(actual_data['Time'])
# st.write(model1_data['Date'].dtype)

model2_data['Date']=pd.to_datetime(model2_data['Date']).dt.strftime('%Y-%m-%d')
# st.write(model1_data['Date'])

model2_data['Time']=pd.to_datetime(model2_data['Time']).dt.time
model2_data['Time']=model2_data['Time'].astype(str)

model2_data['Datetime']=pd.to_datetime(model2_data['Date'] + " " + model2_data['Time'], format="%Y-%m-%d %H:%M:%S")

# st.write(model1_data['Datetime'])


# Removing the first match
# model1_data=model1_data[model1_data['Date']!='31-03-2023']


combined_df2=model2_data.merge(actual_data,on='Datetime',how='left',suffixes=('', '_y'))
# st.write(actual_data[actual_data['Datetime']==''])

# Removing matches yet to happen
combined_df2.dropna(inplace=True)


appdata2=combined_df2.copy()

timeOfDay='evening'
if select_time=='15:30:00':
    timeOfDay='afternoon'

# st.write(appdata2)
# st.write(select_date)
# st.write(timeOfDay)

appdata2=appdata2[(appdata2['Date'].str.contains(select_date)) & (appdata2['timeOfDay']==timeOfDay)] 
# appdata=appdata[appdata['tg_col']==col]                          
appdata2=appdata2.reset_index().drop('index',1)
# st.write(appdata2)
if not appdata2.empty:

    st.write("Match:",appdata2['team1'][0]," vs ",appdata2['team2'][0], " on ",appdata2['Date'][0])
#     st.write(":blue[",appdata['team1'][0]," vs ",appdata['team2'][0], " on ",appdata['Date'][0],"]")    

    cc1=appdata2["Mux"].max()
    st.write("The peak MUX concurrency of the chosen match:",cc1)

#     cc2=appdata["Last9"].max()
#     st.write("The peak MUX concurrency of the chosen match:",cc2)
    
    cc3=appdata2["Universe_total_prediction"].max()
    st.write("The peak prediction(BARC + IPL Model) concurrency of the chosen match:",cc3)

    #     st.header("Model result metrics for the TG: Innings 1")
    #     st.write(resultdata[['col','MAPE']])



    # for date in np.unique(combined_df['Date']):
    #     for time in ['afternoon', 'evening']:
    #         for inning in ['inning1','inning2']:
    #         new=combined_df[(combined_df['Date']==date) & (combined_df['timeOfDay']==time)]
    mape = round(mean_absolute_percentage_error(appdata2['Mux'], appdata2['Universe_total_prediction']),2)

    figure1 =px.line(
                data_frame =appdata2,
                        x = appdata2['Datetime'],
                        y=["Mux","Universe_total_prediction"],
        color_discrete_sequence=['green','blue'],
    #                     text=mape
    )

                    # fig2.update_traces(textposition=sample_df['textPosition'])

                    #     fig2.add_scatter(x=sample_df['Start Time'], y=sample_df['RR scaled'], name="run rate")

                    #     fig2.add_trace(go.Table(cells={"values":df.T.values}, header={"values":df.columns}), row=1,col=1)


                    # fig2.update_xaxes(tickangle=290)
    figure1.update_layout(showlegend=True,font=dict(family="Courier New",size=12,color='Black'),
                                   title="MAPE:"+str(mape),
                                   xaxis_title="Time of day",
                                   yaxis_title="Concurrency",
                                   width=800,height=500)

    st.write(figure1)

else:
    appdata2=model2_data.copy()

    timeOfDay='evening'
    if select_time=='15:30:00':
        timeOfDay='afternoon'


    appdata2=appdata2[(appdata2['Date'].str.contains(select_date)) & (appdata2['timeOfDay']==timeOfDay)] 
    # appdata=appdata[appdata['tg_col']==col]                          
    appdata2=appdata2.reset_index().drop('index',1)
    
    if not appdata2.empty:

            st.write("Match:",appdata2['team1'][0]," vs ",appdata2['team2'][0], " on ",appdata2['Date'][0])
#             st.markdown(":blue[",appdata['team1'][0]," vs ",appdata['team2'][0], " on ",appdata['Date'][0],"]")    

#             cc1=appdata["Mux"].max()
#             st.write("The peak MUX concurrency of the chosen match:",cc1)
#             st.write("No actual available data for the chosen match")
            st.markdown(":blue[No actual available data for the chosen match]")


            cc2=appdata2["Universe_total_prediction"].max()
            st.write("The peak prediction(BARC Model) concurrency of the chosen match:",cc2)

            #     st.header("Model result metrics for the TG: Innings 1")
            #     st.write(resultdata[['col','MAPE']])



            # for date in np.unique(combined_df['Date']):
            #     for time in ['afternoon', 'evening']:
            #         for inning in ['inning1','inning2']:
            #         new=combined_df[(combined_df['Date']==date) & (combined_df['timeOfDay']==time)]
#             mape = mean_absolute_percentage_error(appdata['Mux'], appdata['Universe_total_prediction'])

            figure1 =px.line(
                        data_frame =appdata2,
                                x = appdata2['Datetime'],
                                y=["Universe_total_prediction"],
                color_discrete_sequence=["blue"],
            #                     text=mape
            )

                            # fig2.update_traces(textposition=sample_df['textPosition'])

                            #     fig2.add_scatter(x=sample_df['Start Time'], y=sample_df['RR scaled'], name="run rate")

                            #     fig2.add_trace(go.Table(cells={"values":df.T.values}, header={"values":df.columns}), row=1,col=1)


                            # fig2.update_xaxes(tickangle=290)
            figure1.update_layout(showlegend=True,font=dict(family="Courier New",size=12,color='Black'),
                                           title="BARC Model Prediction",
                                           xaxis_title="Time of day",
                                           yaxis_title="Concurrency",
                                           width=800,height=500)

            st.write(figure1)
            
            
    else:
        st.markdown(":blue[Either Predictions for Model 2 or actual data is not available]")
    
    st.header("DATA SOURCE:")
    url = "https://docs.google.com/spreadsheets/d/1VrIBuVTzp44jmh1ssqfj9F0_3d1xdQTfbAyA6X9jiEA/edit?usp=sharing"
    st.write("You can view all the underlying data [HERE](%s)" % url)

#     st.markdown(":blue[Either Predictions for Model 2 or actual data is not available]")
    
