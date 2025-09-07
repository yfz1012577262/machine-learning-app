from sklearn.calibration import LabelEncoder
import streamlit as st
import pandas as pd
import altair as alt
import sklearn
import numpy as np
from sklearn import preprocessing

#Tab and page config
st.set_page_config(
    page_title="Vancouver Property ML Zoning Classifier App",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏠 Vancouver Property Zoning Classification")
st.markdown("---")

# Load data
df = pd.read_csv("rawdata.csv", sep=';')

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

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = None

tb1, tb2, tb3, tb4 = st.tabs([
    "Data Overview",
    "EDA & Preprocessing",
    "Model Training",
    "Evaluation & Comparison"
])

with tb1:
    with st.expander('📂 Raw Data Preview'):
        st.write("### Raw Input Data")
        st.dataframe(df.sample(n=min(20, len(df)), random_state=42))
        st.write("### All Column Names")
        st.write(df.columns.tolist())

    with st.expander('Features and Target'):
        st.write("## Features selected for prediction")
        st.dataframe(X.sample(n=min(20, len(X)), random_state=42))
        st.write("### Target (zoning_classification)")
        st.write(y.value_counts().head(30))

with tb2:
    with st.expander('Scatterplot for zoning type'):
        # Sample the data BEFORE processing to reduce memory usage
        sample_size = st.slider("Sample size for plot", 100, 5000, 1000)
        
        # Get only the columns we need and sample first
        plot_cols = ["current_land_value", "tax_levy", "zoning_classification"]
        temp_df = df[plot_cols].dropna()
        
        # Sample the data
        if len(temp_df) > sample_size:
            plot_df = temp_df.sample(n=sample_size, random_state=42).copy()
            st.caption(f"Showing {sample_size} of {len(temp_df)} data points for performance")
        else:
            plot_df = temp_df.copy()
            st.caption(f"Showing all {len(plot_df)} data points")
        
        # Apply transformations only to sampled data
        plot_df["land_log"] = np.log1p(plot_df["current_land_value"].clip(lower=0))
        plot_df["levy_log"] = np.log1p(plot_df["tax_levy"].clip(lower=0))
        
        # Only show top zoning classes to reduce legend size
        top_zones = plot_df['zoning_classification'].value_counts().head(10).index
        plot_df_filtered = plot_df[plot_df['zoning_classification'].isin(top_zones)]
        
        # Create simplified chart
        scatter = (
            alt.Chart(plot_df_filtered)
            .mark_circle(size=30, opacity=0.5)  # Smaller circles, more opacity
            .encode(
                x=alt.X("land_log:Q", title="log(Land Value)"),
                y=alt.Y("levy_log:Q", title="log(Tax Levy)"),
                color=alt.Color("zoning_classification:N", 
                              legend=alt.Legend(columns=1, symbolLimit=10)),
                tooltip=["zoning_classification:N"]  # Simplified tooltip
            )
            .properties(
                width=600,  # Fixed width
                height=400  # Fixed height
            )
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)

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
        
        from sklearn.model_selection import train_test_split
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

with tb3:
    with st.expander('Model Training v3 with the Multiple Models for Comparison'):
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score

        # Define models to compare
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }

        # Model comparison
        selected_model = st.multiselect(
            "Select models to compare",
            options=list(models.keys()),
            default=["Random Forest"]
        )

        if st.button("Train and Compare Models"):
            if len(selected_model) == 0:
                st.warning("Please select at least one model.")
            else:
                results = {}
                progress_bar = st.progress(0)

                for i, model_name in enumerate(selected_model):
                    progress_bar.progress((i + 1) / len(selected_model))
                    with st.spinner(f"Training {model_name}..."):
                        Pipeline_model = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('classifier', models[model_name])
                        ])
                        Pipeline_model.fit(X_train, y_train)
                        y_pred = Pipeline_model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        results[model_name] = accuracy

                    st.success(f"{model_name} trained with accuracy: {accuracy:.4%}")

                st.write("### Training Completed")
                progress_bar.empty()
                
                # Store results in session state
                st.session_state.results = results

with tb4:
    st.write("## Evaluation & Comparison")
    
    if st.session_state.results is not None:
        results = st.session_state.results
        
        # Dataframe for comparison of models
        results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
        
        # Display results
        st.write("### Model Performance Comparison Results")
        st.dataframe(results_df.style.format({"Accuracy": "{:.4%}"}))

        # Best model
        best_model_name = results_df.iloc[0]['Model']
        best_model_accuracy = results_df.iloc[0]['Accuracy']
        st.write(f"### Best Model: {best_model_name} with Accuracy: {best_model_accuracy:.4%}")

        # Bar chart for comparison
        Bar_chart = alt.Chart(results_df).mark_bar().encode(
            x=alt.X('Accuracy:Q', scale=alt.Scale(domain=[0, 1]), title='Accuracy'),
            y=alt.Y('Model:N', sort='-x', title='Model'),
            color=alt.Color('Accuracy:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=['Model', alt.Tooltip('Accuracy:Q', format='.4%')]
        ).properties(
            title='Model Accuracy Comparison',
            width=700,
            height=400
        )
        st.altair_chart(Bar_chart, use_container_width=True)
    else:
        st.info("Please train models in the 'Model Training' tab first to see results here.")

# Sidebar
with st.sidebar:
    st.header('Selected Input features')