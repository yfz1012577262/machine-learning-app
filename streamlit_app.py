from sklearn.calibration import LabelEncoder
import streamlit as st
import pandas as pd
import altair as alt
import sklearn
import numpy as np
from sklearn import preprocessing
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

with st.expander('Features and Target'):
    st.write("## Features selected for prediction")
    st.dataframe(X.sample(n=min(20, len(X)), random_state=42))

    st.write("### Target (zoning_classification)")
    st.write(y.value_counts().head(30))  # top 30 most common classes

# Quick look at data distribution
with st.expander('Scatterplot for zoning type'):

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
with st.expander('Data Preprocessing'):
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

    # Adding pipeline and separate numeric and categorical features
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    st.write(f"Numeric features: {numeric_features}")
    st.write(f"Categorical features: {categorical_features}")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    


    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    st.write(f"acceptable missing%")
    st.write(f"Training set: {len(X_train)} samples")
    st.write(f"Test set: {len(X_test)} samples")

with st.expander('Model Training v2 with the Preprocessing pipeline'):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Preprocess the data

    if st.button('Train Model with Pipeline'):
        Pipeline_rf=Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
        Pipeline_rf.fit(X_train, y_train)
        
        y_pred = Pipeline_rf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model Accuracy: {accuracy:.3%}")
        st.write("### Sample Predictions")
        sample_results = pd.DataFrame({
            'Actual': y_test.iloc[:20],
            'Predicted': y_pred[:20]
        })
        st.dataframe(sample_results)
