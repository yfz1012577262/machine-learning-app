from sklearn.preprocessing import LabelEncoder
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

# Load data/cache data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, sep=';')
    return data

try:
    df = load_data('rawdata.csv')
except Exception as e:
    st.error(f"Error loading data: {e}")

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
        from sklearn.pipeline import Pipeline

        # 0) Speed / compute controls
        st.write("#### Speed / compute controls")
        col_speed1, col_speed2 = st.columns(2)
        with col_speed1:
            subsample_frac = st.slider(
                "Training subsample fraction", min_value=0.10, max_value=1.00,
                value=1.00, step=0.05
            )
        with col_speed2:
            rf_cores = st.slider("RandomForest cores (n_jobs)", 1, 4, 1, 1)

        # 1) Choose which algorithms to train
        algo_options = ["Random Forest", "Gradient Boosting", "Logistic Regression", "Decision Tree"]
        selected_model = st.multiselect(
            "Select models to compare",
            options=algo_options,
            default=["Random Forest"]
        )

        # 2) Show hyperparameters ONLY for selected algorithms (numeric controls as sliders)
        c1, c2 = st.columns(2)

        # -Random Forest (constrained) 
        if "Random Forest" in selected_model:
            with c1:
                st.markdown("**Random Forest**")
                rf_n_estimators = st.slider("RF n_estimators", 50, 300, 120, 10, key="rf_n")
                rf_use_max_depth = st.checkbox("RF use max_depth", value=True, key="rf_use_md")
                rf_max_depth = st.slider("RF max_depth", 3, 30, 12, 1, key="rf_md") if rf_use_max_depth else None
                rf_max_features = st.selectbox("RF max_features", ["sqrt", "log2"], index=0, key="rf_mf")

        # Gradient Boosting (constrained)
        if "Gradient Boosting" in selected_model:
            with c1:
                st.markdown("**Gradient Boosting**")
                gb_n_estimators = st.slider("GB n_estimators", 50, 300, 100, 10, key="gb_n")
                gb_learning_rate = st.slider("GB learning_rate", 0.01, 0.5, 0.10, 0.01, key="gb_lr")
                gb_use_max_depth = st.checkbox("GB use max_depth", value=True, key="gb_use_md")
                gb_max_depth = st.slider("GB max_depth", 2, 6, 3, 1, key="gb_md") if gb_use_max_depth else None
                gb_early_stop = st.checkbox("GB early stopping", value=True, key="gb_es")

        # Logistic Regression (constrained)
        if "Logistic Regression" in selected_model:
            with c2:
                st.markdown("**Logistic Regression**")
                lr_C = st.slider("LR C", 0.01, 10.0, 1.0, 0.01, key="lr_c")
                lr_max_iter = st.slider("LR max_iter", 100, 500, 200, 50, key="lr_mi")
                lr_solver = st.selectbox("LR solver", ["lbfgs", "liblinear", "saga"], index=0, key="lr_sol")

        #  Decision Tree (constrained) 
        if "Decision Tree" in selected_model:
            with c2:
                st.markdown("**Decision Tree**")
                dt_use_max_depth = st.checkbox("DT use max_depth", value=True, key="dt_use_md")
                dt_max_depth = st.slider("DT max_depth", 3, 40, 12, 1, key="dt_md") if dt_use_max_depth else None
                dt_min_samples_split = st.slider("DT min_samples_split", 2, 50, 10, 1, key="dt_mss")

        # 3) Build models dict using only selected algorithms (with tuned params)
        models = {}

        if "Random Forest" in selected_model:
            rf_kwargs = {"n_estimators": int(rf_n_estimators), "random_state": 42, "n_jobs": int(rf_cores)}
            if rf_use_max_depth:
                rf_kwargs["max_depth"] = int(rf_max_depth)
            rf_kwargs["max_features"] = rf_max_features
            models["Random Forest"] = RandomForestClassifier(**rf_kwargs)

        if "Gradient Boosting" in selected_model:
            gb_kwargs = {"n_estimators": int(gb_n_estimators), "learning_rate": float(gb_learning_rate), "random_state": 42}
            if gb_use_max_depth:
                gb_kwargs["max_depth"] = int(gb_max_depth)
            if gb_early_stop:
                gb_kwargs.update({"n_iter_no_change": 5, "validation_fraction": 0.1})
            models["Gradient Boosting"] = GradientBoostingClassifier(**gb_kwargs)

        if "Logistic Regression" in selected_model:
            models["Logistic Regression"] = LogisticRegression(
                C=float(lr_C), max_iter=int(lr_max_iter), solver=lr_solver, random_state=42
            )

        if "Decision Tree" in selected_model:
            dt_kwargs = {"random_state": 42, "min_samples_split": int(dt_min_samples_split)}
            if dt_use_max_depth:
                dt_kwargs["max_depth"] = int(dt_max_depth)
            models["Decision Tree"] = DecisionTreeClassifier(**dt_kwargs)

        # 4) Train & compare (unchanged flow, with optional subsample)
        if st.button("Train and Compare Models"):
            if len(models) == 0:
                st.warning("Please select at least one model.")
            else:
                results = {}
                progress_bar = st.progress(0)

                # Optional subsample for speed
                if subsample_frac < 1.0:
                    # use a consistent random_state for repeatability
                    X_train_use = X_train.sample(frac=subsample_frac, random_state=42)
                    y_train_use = y_train.loc[X_train_use.index]
                else:
                    X_train_use, y_train_use = X_train, y_train

                for i, model_name in enumerate(models.keys()):
                    progress_bar.progress((i + 1) / max(1, len(models)))
                    with st.spinner(f"Training {model_name}..."):
                        Pipeline_model = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('classifier', models[model_name])
                        ])
                        Pipeline_model.fit(X_train_use, y_train_use)
                        y_pred = Pipeline_model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        results[model_name] = accuracy

                    st.success(f"{model_name} trained with accuracy: {accuracy:.4%}")

                st.write("### Training Completed")
                progress_bar.empty()
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