import streamlit as st
import pandas as pd
import os
import google.genai as genai
## hello
st.set_page_config(page_title="Micron Streamlit Dummy", layout="wide")

DATASET_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'Datasets')

if "page" not in st.session_state:
    st.session_state.page = "upload"
if "overview_shift" not in st.session_state:
    st.session_state.overview_shift = "total"
if "df" not in st.session_state:
    st.session_state.df = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "selected_shift" not in st.session_state:
    st.session_state.selected_shift = "total"
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None
if "ai_summary_key" not in st.session_state:
    st.session_state.ai_summary_key = None
if "util_threshold" not in st.session_state:
    st.session_state.util_threshold = 95
if "_thresh_slider" not in st.session_state:
    st.session_state._thresh_slider = 95
if "_thresh_input" not in st.session_state:
    st.session_state._thresh_input = 95

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_MODEL = st.secrets["GEMINI_MODEL"]

@st.cache_resource
def get_gemini_client():
    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={"timeout": 30},
    )