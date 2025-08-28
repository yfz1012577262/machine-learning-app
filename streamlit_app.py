import streamlit as st
import pandas as pd
import altair as alt
import sklearn
import numpy as np
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

# Quick look at data distribution
with st.expander('Scatterplot for zoninig type'):
    
    plot_df = df[["current_land_value","tax_levy","zoning_classification"]].dropna().copy()
    plot_df["land_log"] = np.log1p(plot_df["current_land_value"])
    plot_df["levy_log"] = np.log1p(plot_df["tax_levy"])

    scatter = (
        alt.Chart(plot_df)
        .mark_circle(size=35, opacity=0.35)
        .encode(
            x=alt.X("land_log:Q", title="log1p(current_land_value)"),
            y=alt.Y("levy_log:Q", title="log1p(tax_levy)"),
            color=alt.Color("zoning_classification:N", legend=alt.Legend(columns=2)),
            tooltip=[
                alt.Tooltip("current_land_value:Q", format=",.0f"),
                alt.Tooltip("tax_levy:Q", format=",.0f"),
                "zoning_classification:N"
         ],
        )
        .interactive()  # zoom & pan
    )
    st.altair_chart(scatter, use_container_width=True)

# Data Preparation
with st.sidebar:
    st.header('Selected Input features')
with st.expander('🔧 Data Preprocessing'):
    st.write("### Missing Values Analysis")
    missing_info = X.isnull().sum()
    missing_df = pd.DataFrame({
        'Feature': missing_info.index,
        'Missing Count': missing_info.values,
        'Missing %': (missing_info.values / len(X) * 100).round(2)
    })
    st.dataframe(missing_df[missing_df['Missing Count'] > 0])
    
    # Remove rows with missing target
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]
    
    # Simple train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    st.write(f"Training set: {len(X_train)} samples")
    st.write(f"Test set: {len(X_test)} samples")