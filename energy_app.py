import sys
import numpy as np
import pandas as pd

import streamlit as st
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import datetime
import base64
import time
import os
import pickle

from Energy import Energy

register_matplotlib_converters()
plt.style.use('default')

st.set_page_config(layout='centered')

st.title('Energy forecast app')

energy = Energy()

df = energy.get_data()

df['USE_FACT'].plot(figsize=(10, 5), label='past')

st.pyplot(plt)
