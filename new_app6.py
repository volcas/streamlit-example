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

# st.write("IMPORT DATA")
# st.write("Import the time series csv file. It should have two columns labelled as 'ds' and 'y'.The 'ds' column should be of datetime format  by Pandas. The 'y' column must be numeric representing the measurement to be forecasted.")

# data = st.file_uploader('Upload first file here',type='csv')
# data='TS_Reg.csv'


# data2 = st.file_uploader('Upload second file here',type='csv')

### FOR THE RESULT FILE

from google.oauth2 import service_account
from gsheetsdb import connect


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

sheet_url = st.secrets["private_gsheets_url_1"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')


innings1_data=pd.DataFrame.from_records(rows)
innings1_data.columns=['Datetime', 'Description', '13-21_female', '13-21_female_prediction',
       '13-21_female_MAPE', '13-21_female_AP / Telangana_actuals',
       '13-21_female_AP / Telangana_predictions',
       '13-21_female_AP / Telangana_MAPE',
       '13-21_female_Assam / North East / Sikkim_actuals',
       '13-21_female_Assam / North East / Sikkim_predictions',
       ...
       'Universe_total_Rajasthan_MAPE',
       'Universe_total_TN/Pondicherry_actuals',
       'Universe_total_TN/Pondicherry_predictions',
       'Universe_total_TN/Pondicherry_MAPE',
       'Universe_total_UP/Uttarakhand_actuals',
       'Universe_total_UP/Uttarakhand_predictions',
       'Universe_total_UP/Uttarakhand_MAPE',
       'Universe_total_West Bengal_actuals',
       'Universe_total_West Bengal_predictions',
       'Universe_total_West Bengal_MAPE']

sheet_url = st.secrets["private_gsheets_url_2"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')

innings2_data=pd.DataFrame.from_records(rows)
innings2_data.columns=['Datetime', 'Description', '13-21_female', '13-21_female_prediction',
       '13-21_female_MAPE', '13-21_female_AP / Telangana_actuals',
       '13-21_female_AP / Telangana_predictions',
       '13-21_female_AP / Telangana_MAPE',
       '13-21_female_Assam / North East / Sikkim_actuals',
       '13-21_female_Assam / North East / Sikkim_predictions',
       ...
       'Universe_total_Rajasthan_MAPE',
       'Universe_total_TN/Pondicherry_actuals',
       'Universe_total_TN/Pondicherry_predictions',
       'Universe_total_TN/Pondicherry_MAPE',
       'Universe_total_UP/Uttarakhand_actuals',
       'Universe_total_UP/Uttarakhand_predictions',
       'Universe_total_UP/Uttarakhand_MAPE',
       'Universe_total_West Bengal_actuals',
       'Universe_total_West Bengal_predictions',
       'Universe_total_West Bengal_MAPE']



region_list=['AP / Telangana', 'Assam / North East / Sikkim', 'Bihar/Jharkhand',
       'Delhi', 'Guj / D&D / DNH', 'Har/HP/J&K', 'Karnataka', 'Kerala',
       'MP/Chhattisgarh', 'Mah / Goa', 'Odisha', 'Pun/Cha', 'Rajasthan',
       'TN/Pondicherry', 'UP/Uttarakhand', 'West Bengal']

select_region= st.sidebar.selectbox('Select Region',
                                    region_list)

select_tg = st.sidebar.selectbox('What TG Level?',
                                ['2-12_male','2-12_female','13-21_male','13-21_female',
                                 '22-30_male','22-30_female','31-40_male','31-40_female',
                                 '41-50_male','41-50_female','51-60_male','51-60_female',
                                 '61+_male','61+_female'])    

team_list1=['CSK', 'MI', 'RCB', 'LSG', 'RR', 'KKR', 'PBKS', 'GT', 'DC', 'SRH']    
select_team1 = st.sidebar.selectbox('Select Team 1',
                                team_list1)    
team_list2=['CSK', 'MI', 'RCB', 'LSG', 'RR', 'KKR', 'PBKS', 'GT', 'DC', 'SRH']    
team_list2.remove(select_team1)
select_team2 = st.sidebar.selectbox('Select Team 2',
                                team_list2)   



# periods_input = st.number_input('How many days forecast do you want?',
# min_value = 1, max_value = 365)  
#     select_region = st.text_input('Which Region?')  

st.write("VISUALIZE FORECASTED DATA")

# appdata_main = pd.read_csv(data)
# result="ResultOP.csv"
# result_main=pd.read_csv(result)
# appdata_main=extra.merge(df_new, on=['Datetime','inning','matchName','timeOfDay'],how='left',suffixes=('', '_y'))


# appdata_main['Datetime'] = appdata_main['Datetime'].apply(lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x, "%d-%m-%Y %H:%M"), "%Y-%m-%d %H:%M"))    


# innings1_data=pd.read_csv("C:/Work/IPL Prediction/TG_level_predictions_inning1.csv")
# innings1_result=pd.read_csv("C:/Work/IPL Prediction/Innings1_result.csv")
# innings1_sum=pd.read_csv("C:/Work/IPL Prediction/mean_viewership_team_innings1.csv")


# innings2_data=pd.read_csv("C:/Work/IPL Prediction/TG_level_predictions_inning2.csv")
# innings2_result=pd.read_csv("C:/Work/IPL Prediction/Innings2_result.csv")
# innings2_sum=pd.read_csv("C:/Work/IPL Prediction/mean_viewership_team_innings2.csv")
# appdata_main=appdata_main[appdata_main['Datetime']<"2022-05-24"]


# if data is not None:

#     offset = df[df['Datetime']<'2022-05-21'].shape[0]   ## for train test split



# appdata=appdata[appdata['Region']==select_region]
cols=innings1_data.columns
reg_cols=[k for k in cols if select_region in k]
tg_cols=[k for k in reg_cols if select_tg in k]
tg_cols.append('Datetime')
tg_cols.append('Description')
appdata=innings1_data.copy()
appdata=appdata[(appdata['Description'].str.contains(select_team1)) & (appdata['Description'].str.contains(select_team2))]                 
appdata=appdata[tg_cols]
# appdata=appdata.drop_duplicates(['Datetime','Description'])
appdata=appdata.reset_index().drop('index',1)

                          



# sumdata=innings1_sum.copy()
# sumdata=sumdata[(sumdata['matchName'].str.contains(select_team1)) & (sumdata['matchName'].str.contains(select_team2))] 
# st.write(sumdata[['matchName',col]])


# st.write(resultdata[['col','MAPE']])
# metric_cols=[k for k in reg_cols if 'MAPE' in k]     
try:
    resultdata=appdata[tg_cols[2]].values[0]
    st.header("Innings 1")
    st.write("MAPE:",resultdata)

    for date in np.unique(appdata['Datetime'].astype(str).str.split().str[0]):
        new=appdata[appdata['Datetime'].astype(str).str.contains(date)]
        figure1 =px.line(
            data_frame =new,
                    x = new['Datetime'],
                    y=tg_cols[:2],
            color_discrete_sequence=['red', "blue"])

                # fig2.update_traces(textposition=sample_df['textPosition'])

                #     fig2.add_scatter(x=sample_df['Start Time'], y=sample_df['RR scaled'], name="run rate")

                #     fig2.add_trace(go.Table(cells={"values":df.T.values}, header={"values":df.columns}), row=1,col=1)


                # fig2.update_xaxes(tickangle=290)
        figure1.update_layout(showlegend=True,font=dict(family="Courier New",size=12,color='Black'),
                                       title=f"Prediction for "+ max(new['Description'])+ " on "+ date+ " (Innings1)",
                                       xaxis_title="Time of day",
                                       yaxis_title="Predicted Viewership",
                                       width=800,height=600)

        st.write(figure1)
        st.write("The above plot shows the predicted and actual ratings of the selected TGs on the left dropdown")
        
        #INNINGS 2
#         appdata=innings2_data.copy()

        # appdata=appdata[appdata['Region']==select_region]
        cols=innings2_data.columns
        reg_cols=[k for k in cols if select_region in k]
        tg_cols=[k for k in reg_cols if select_tg in k]
        tg_cols.append('Datetime')
        tg_cols.append('Description')
        appdata=innings2_data.copy()
        appdata=appdata[(appdata['Description'].str.contains(select_team1)) & (appdata['Description'].str.contains(select_team2))]                 
        appdata=appdata[tg_cols]
        # appdata=appdata.drop_duplicates(['Datetime','Description'])
        appdata=appdata.reset_index().drop('index',1)

        st.header("Innings 2")
        resultdata=appdata[tg_cols[2]].values[0]
        st.write("MAPE:",resultdata)

        for date in np.unique(appdata['Datetime'].astype(str).str.split().str[0]):
            new=appdata[appdata['Datetime'].astype(str).str.contains(date)]
            figure1 =px.line(
                data_frame =new,
                        x = new['Datetime'],
                        y=tg_cols[:2],
                color_discrete_sequence=['red', "blue"])

                    # fig2.update_traces(textposition=sample_df['textPosition'])

                    #     fig2.add_scatter(x=sample_df['Start Time'], y=sample_df['RR scaled'], name="run rate")

                    #     fig2.add_trace(go.Table(cells={"values":df.T.values}, header={"values":df.columns}), row=1,col=1)


                    # fig2.update_xaxes(tickangle=290)
            figure1.update_layout(showlegend=True,font=dict(family="Courier New",size=12,color='Black'),
                                           title=f"Prediction for "+ max(new['Description'])+ " on "+ date+ " (Innings2)",
                                           xaxis_title="Time of day",
                                           yaxis_title="Predicted Viewership",
                                           width=800,height=600)

            st.write(figure1)
            st.write("The above plot shows the predicted and actual ratings of the selected TGs on the left dropdown")



except:
#     st.write("No matchups between these two happened after 1st may(TEST Sample).Kindly choose another matchup")  
    st.markdown(":blue[No matchups between these two happened after 15th may(TEST Sample).Kindly choose another matchup]")

# st.header("All the TG results at a glance")
# st.write(innings1_result[['col','MAPE']])
    
