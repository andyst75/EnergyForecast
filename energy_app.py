import datetime

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pandas.plotting import register_matplotlib_converters

from energy import Energy
import plotly.express as px

register_matplotlib_converters()
plt.style.use('default')

st.set_page_config(layout='centered')

st.title('Energy forecast app')

st.markdown('''
This app retrieves cryptocurrency prices for the top 100 cryptocurrency from the **CoinMarketCap**!
''')

expander_bar = st.beta_expander("About")
expander_bar.markdown("""
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, BeautifulSoup, requests, json, time
* **Data source:** [Minenergo](http://https://minenergo.gov.ru/).
""")
DAYS_BACK = 200
PRED_HORIZON = 3
WIN_LEN = 30
TODAY = datetime.datetime.now().date()
MIN_DATE = TODAY - datetime.timedelta(days=DAYS_BACK) + datetime.timedelta(days=WIN_LEN)
MAX_DATE = datetime.datetime(2019, 7, 1).date()

st.sidebar.header('User Input Features')

col1 = st.sidebar
col1.header('Input Options')

random_date = col1.checkbox('Random date', value=False)

if random_date:
    delta = int((MAX_DATE - MIN_DATE).days * np.random.random())
    predict_from = MIN_DATE + datetime.timedelta(days=delta)
else:
    predict_from = MAX_DATE
predict_from = col1.date_input('Predict from', value=predict_from, min_value=MIN_DATE, max_value=MAX_DATE)

col1.button('Update')


@st.cache
def load_data():
    return Energy()


data_df = load_data()

df_with_consumption = data_df.get_data_with_consumption(str(predict_from), predict_days=PRED_HORIZON - 1)

#ax = df_with_consumption['consumption'].plot(figsize=(10, 5), label='predict')
#ax.set(ylabel=f'consumption', xlabel='', title=f'Prediction of the consumption')
#plt.legend()
#st.pyplot(plt)

fig = px.line(df_with_consumption[['fact', 'consumption' ]], labels={'DATE':'Период', 'value':'Среднечасовое потребление, МВт'})

fig.update_layout( 
    xaxis=dict( 
        rangeselector=dict( 
            buttons=list([ 
                dict(count=14, 
                     step="day", 
                     stepmode="backward"), 
                dict(count=30, 
                     step="day", 
                     stepmode="backward"), 
                dict(count=6, 
                     step="month", 
                     stepmode="backward"), 
            ]) 
        ), 
        rangeslider=dict( 
            visible=True
        ), 
    ) 
) 

fig.update_layout(showlegend=False,)

fig.update_layout(
    autosize=False,
    width=800,
    height=600,)

st.plotly_chart(fig, use_container_width=True)


with st.beta_expander("See explanation"):
    st.write("""
           The chart above shows some numbers I picked for you.
            I rolled actual dice for these, so they're *guaranteed* to
            be random.
         """)
    st.image("https://static.streamlit.io/examples/dice.jpg")

df_consumption = df_with_consumption[['consumption']].reset_index()
st.dataframe(df_consumption[['consumption']].describe(), 500, 500)
