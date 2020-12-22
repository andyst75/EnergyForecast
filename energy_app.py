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

import Energy

register_matplotlib_converters()
plt.style.use('default')

st.set_page_config(layout="wide")

st.title('Energy forecast app')

