import streamlit as st
import pandas as pd

import sklearn

st.title("Personal Machine Learninig app")

st.caption("sth")
st.info(':smile: This app builds a machine learning model for Bu')
import pandas as pd
import streamlit as st

with st.expander('📂 Raw Data Preview'):
    st.write("### Raw Input Data")
    df = pd.read_csv("rawdata.csv", sep=';')   # Vancouver Open Data CSVs often use ';'
    
    # show a sample instead of the whole dataset
    st.dataframe(df.sample(n=min(20, len(df)), random_state=42))

    st.write("### All Column Names")
    st.write(df.columns.tolist())

# Target column
target = "zoning_classification"

# Features we keep
useful_features = [
    "legal_type",
    "current_land_value",
    "current_improvement_value",
    "previous_land_value",
    "previous_improvement_value",
    "tax_levy",
    "year_built",
    "big_improvement_year",
    "tax_assessment_year",
    "neighbourhood_code",
    "property_postal_code"
]

# Build X and y
X = df[useful_features].copy()
y = df[target].copy()

with st.expander('🎯 Features and Target'):
    st.write("### Features selected for prediction")
    st.dataframe(X.sample(n=min(20, len(X)), random_state=42))

    st.write("### Target (zoning_classification)")
    st.write(y.value_counts().head(20))  # top 20 most common classes


