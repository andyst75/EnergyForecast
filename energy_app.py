import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from pandas.plotting import register_matplotlib_converters

from energy import Energy

register_matplotlib_converters()
plt.style.use('default')

st.set_page_config(layout='wide', page_title='Energy forecast app')
st.title('Energy forecast app')

st.markdown('''
Short-term forecast system for electricity consumption ***OES of the Middle Volga***, with the possibility of modeling
electricity consumption depending on changes in external factors.
''')

st.markdown("""
* **Python libraries:** scikit-learn, pandas, requests, streamlit, plotly, seaborn, matplotlib, seaborn, numpy
* **Data source:** [Minenergo](https://minenergo.gov.ru/).
""")
DAYS_BACK = 1000
TODAY = datetime.datetime.now().date()
MIN_DATE = TODAY - datetime.timedelta(days=DAYS_BACK)
MAX_DATE = datetime.datetime(2020, 7, 1).date()

col1 = st.sidebar
st.sidebar.image(image=Image.open(f'{Path().absolute()}/resources/made.png'), width=200)
st.sidebar.header('Options')

random_date = st.sidebar.checkbox('Random date', value=False)

if random_date:
    delta = int((MAX_DATE - MIN_DATE).days * np.random.random())
    predict_from = MIN_DATE + datetime.timedelta(days=delta)
else:
    predict_from = MIN_DATE
predict_from = st.sidebar.date_input(
    label='Predict from',
    value=MIN_DATE,
    min_value=MIN_DATE,
    max_value=MAX_DATE)
pred_horizon = st.sidebar.slider(
    label="Predict period (days)",
    value=2,
    min_value=1,
    max_value=5)
temperature_delta = st.sidebar.slider(
    label="Choose variation of temperature",
    value=0,
    min_value=-10,
    max_value=10)
consumption_index_delta = st.sidebar.slider(
    label="Choose variation of consumer activity index",
    value=0,
    min_value=-20,
    max_value=20)
isolation_index_delta = st.sidebar.slider(
    label="Choose variation of isolation index",
    value=0.,
    min_value=-1.,
    max_value=1.,
    step=0.5)

st.sidebar.button('Update')


@st.cache
def load_energy_data():
    return Energy()


data_df = load_energy_data()

df_with_consumption = data_df.get_data_with_consumption(str(predict_from),
                                                        predict_days=pred_horizon - 1,
                                                        temperature_delta=temperature_delta,
                                                        consumption_index_delta=consumption_index_delta,
                                                        isolation_index_delta=isolation_index_delta)

shift_index_data = pd.date_range(df_with_consumption[['consumption']].index[-1] + pd.DateOffset(1),
                                 periods=pred_horizon, freq='D')
df_with_consumption_with_shift = df_with_consumption.append(pd.DataFrame(index=shift_index_data))
fact_df = df_with_consumption_with_shift[['fact']]
consumption_df = df_with_consumption_with_shift[['consumption']].shift(pred_horizon)
data_plot = pd.concat([fact_df, consumption_df], axis=1)

fig = px.line(data_plot,
              labels={'DATE': 'Период', 'value': 'Среднечасовое потребление, МВт'})

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

fig.update_layout(showlegend=False)

fig.update_layout(
    autosize=False,
    width=800,
    height=800
)

st.plotly_chart(fig, use_container_width=True)

st.header("Statistic")
df_consumption = df_with_consumption[['consumption']].reset_index()
st.dataframe(df_consumption[['consumption']].describe().applymap('{:,.1f}'.format).T)

st.header("Data")
df_consumption = df_with_consumption.reset_index()
st.dataframe(df_consumption.head(10))
