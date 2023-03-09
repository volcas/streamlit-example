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
data='TS_Reg.csv'


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
# @st.cache_data(ttl=600)
# @st.cache_data(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows

sheet_url = st.secrets["private_gsheets_url_1"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')

# Print results.
for row in rows:
    st.write(f"{row.name} has a :{row.pet}:")
# def load_data(sheets_url):
#     csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
#     return pd.read_csv(csv_url)

# appdata_main = load_data(st.secrets["private_gsheets_url_1"])

# result_main = load_data(st.secrets["private_gsheets_url_2"])



# if data is not None and data2 is not None:


# with open('model_'+select_tg+'.pkl', 'rb') as f:
#     svr = pickle.load(f)


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

appdata_main=appdata_main[appdata_main['Datetime']<"2022-05-24"]


# if data is not None:
col=select_tg
#     offset = df[df['Datetime']<'2022-05-21'].shape[0]   ## for train test split

appdata=appdata_main.copy()

appdata=appdata[appdata['Region']==select_region]
appdata=appdata[(appdata['match_name'].str.contains(select_team1)) & (appdata['match_name'].str.contains(select_team2))] 
appdata=appdata.reset_index().drop('index',1)

resultdata=result_main[(result_main['key'].str.contains(select_region)) & (result_main['key'].str.contains(col))] 

st.write("Model result metrics for the TG")
st.write(resultdata)
st.write("The following plot shows the predicted and actual ratings of the selected TGs on the left dropdown")

for date in np.unique(appdata['Datetime'].str.split().str[0]):
    new=appdata[appdata['Datetime'].str.contains(date)]
    figure1 =px.line(
        data_frame =new,
                x = new['Datetime'],
                y=["Actual_"+ col,"Predicted_"+col],
        color_discrete_sequence=['red', "blue"])

            # fig2.update_traces(textposition=sample_df['textPosition'])

            #     fig2.add_scatter(x=sample_df['Start Time'], y=sample_df['RR scaled'], name="run rate")

            #     fig2.add_trace(go.Table(cells={"values":df.T.values}, header={"values":df.columns}), row=1,col=1)


            # fig2.update_xaxes(tickangle=290)
    figure1.update_layout(showlegend=True,font=dict(family="Courier New",size=12,color='Black'),
                                   title=f"SVR with Time Series model prediction for "+ max(new['match_name'])+ " on "+ date,
                                   xaxis_title="Time of day",
                                   yaxis_title="Predicted Viewership",
                                   width=1000,height=500)

    st.write(figure1)

    
    
st.write("All the TG results at a glance")
st.write(result_main)
    
