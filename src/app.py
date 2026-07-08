import joblib
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

model = joblib.load("../models/model.pkl")

st.title("Can You Survive The Titanic?")
