import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
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

expander_bar = st.beta_expander("About")
expander_bar.markdown("""
* **Team:** Andrei Starikov, Nikolay Dyakin, Ilya Avilov, Orkhan Gadzhily, Evgenii Munin
* **Python libraries:** scikit-learn, pandas, requests, streamlit, plotly, seaborn, matplotlib, seaborn, numpy, base64
* **Data source:** [Minenergo](https://minenergo.gov.ru/).
""")


@st.cache
def load_energy_data():
    return Energy()


energy_obj = load_energy_data()

col1 = st.sidebar
col1.image(image=Image.open(f'{Path().absolute()}/resources/made.png'), width=200)
col1.header('Options')

pred_horizon = col1.slider(
    label="Predict period, days",
    value=2,
    min_value=1,
    max_value=5)

col1.subheader("What If Prediction")

temperature_delta = col1.slider(
    label="Variation of temperature, Δt°C",
    value=0,
    min_value=-10,
    max_value=10)
consumption_index_delta = col1.slider(
    label="Variation of consumer activity index, Δt°C",
    value=0,
    min_value=-20,
    max_value=20)
isolation_index_delta = col1.slider(
    label="Variation of isolation index",
    value=0.,
    min_value=-1.,
    max_value=1.,
    step=0.5)

MIN_DATE, MAX_DATE = energy_obj.get_period()
random_period = st.sidebar.checkbox('Random period', value=False)
if random_period:
    delta = int((MAX_DATE - MIN_DATE).days * np.random.random())
    period_from = MIN_DATE + datetime.timedelta(days=delta)
    delta = int((MAX_DATE - period_from).days * np.random.random())
    period_to = period_from + datetime.timedelta(days=delta)
else:
    period_from = MIN_DATE
    period_to = MAX_DATE

period_from = st.sidebar.date_input(
    label='Period from',
    value=period_from,
    min_value=MIN_DATE,
    max_value=MAX_DATE)
period_to = st.sidebar.date_input(
    label='Period to',
    value=period_to,
    min_value=MIN_DATE,
    max_value=MAX_DATE)

if period_to < period_from:
    period_from = period_to - pd.DateOffset(1)

st.sidebar.button('Update')


# Download CSV data
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="energy.csv">Download CSV File</a>'
    return href


predicted_df, metric_df, original_predicted_df, original_metric_df = energy_obj.what_if_predict(
    period_from,
    period_to,
    temperature_delta=temperature_delta,
    consumption_index_delta=consumption_index_delta,
    isolation_index_delta=isolation_index_delta
)
PREDICT_COL = 'predict'


def extract_predict(df):
    cols = ['PRED_1', 'PRED_2', 'PRED_3', 'PRED_4', 'PRED_5']
    shift_date = pd.DataFrame(
        df.loc[pd.to_datetime(period_to), cols[:pred_horizon]].to_numpy(),
        index=pd.date_range(
            period_to + pd.DateOffset(1),
            periods=pred_horizon, freq='D'
        ),
        columns=[PREDICT_COL]
    )
    return shift_date


def build_full_predict(df):
    cols = ['fact', 'PRED_1']
    predicted_df = df[cols].rename(columns={'PRED_1': PREDICT_COL})
    predicted_df[PREDICT_COL] = predicted_df[PREDICT_COL].shift(1)
    shift_date = extract_predict(df)
    predicted_df = predicted_df.append(shift_date)
    return predicted_df


plot_df = build_full_predict(predicted_df)
fig = px.line(plot_df, labels={'value': 'Average hourly consumption, MW'})
fig.update_layout(
    height=600,
    xaxis=dict(
        title='',
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
                dict(count=1,
                     step="year",
                     stepmode="backward"),
                dict(step="all"),
            ])
        ),
        rangeslider=dict(
            visible=True,
        ),
        type="date",
    ),
    legend=dict(
        yanchor="top", y=0.99,
        xanchor="left", x=0.01,
        title='',
    )
)

st.plotly_chart(fig, use_container_width=True)
st.markdown(filedownload(plot_df.reset_index()), unsafe_allow_html=True)
st.dataframe(metric_df.loc[:pred_horizon, ['MAPE']].T)
st.subheader("What If Prediction")

delta_df = pd.DataFrame({
    'base': extract_predict(original_predicted_df)[PREDICT_COL],
    'with_changes': extract_predict(predicted_df)[PREDICT_COL],
},
)
delta_df['delta_hour'] = delta_df['with_changes'] - delta_df['base']
delta_df['delta_day'] = delta_df['delta_hour'] * 24
delta_df = delta_df.reset_index()
delta_df['index'] = delta_df['index'].dt.date
delta_df = delta_df.set_index('index')

st.dataframe(delta_df.applymap('{:,.1f}'.format).T)

total = delta_df['delta_day'].sum()
st.markdown(f'Total delta consumption for scenario, MW: **{total:,.1f}**')
