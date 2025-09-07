import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Page config
st.set_page_config(
    page_title="Vancouver Property ML Zoning Classifier App",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("⚙️ Data & Performance")
    use_only_useful_cols = st.toggle("Load only useful columns", value=True)
    max_rows = st.number_input("Max rows to load (0 = all)", min_value=0, value=100_000, step=10_000)
    include_postal_code = st.toggle("Include property_postal_code (high cardinality!)", value=False)

# Target column
target = "zoning_classification"

# Features we keep
base_features = [
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
]
useful_features = base_features + (["property_postal_code"] if include_postal_code else [])

@st.cache_data(show_spinner=True)
def load_data(path, usecols=None, nrows=None):
    dtype_map = {
        "legal_type": "category",
        "neighbourhood_code": "category",
        "property_postal_code": "category",
    }
    cols = list(set(usecols + [target])) if usecols else None
    return pd.read_csv(path, sep=';', usecols=cols, dtype=dtype_map, nrows=(None if nrows == 0 else nrows))

# Load data
df = load_data("rawdata.csv", usecols=useful_features if use_only_useful_cols else None, nrows=max_rows)

st.title("🏠 Vancouver Property Zoning Classification")
st.markdown("---")

if target not in df.columns:
    st.error(f"Column '{target}' not found in data.")
    st.stop()

# Build X and y
X = df[useful_features].copy()
y = df[target].copy()

# Remove rows with missing target
mask = ~y.isnull()
X = X[mask]
y = y[mask]

# Define features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Create preprocessor
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y if y.nunique() > 1 else None
)

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = None

# Create tabs
tb1, tb2, tb3, tb4 = st.tabs([
    "Data Overview",
    "EDA & Preprocessing",
    "Model Training",
    "Evaluation & Comparison"
])

# TAB 1: DATA OVERVIEW
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
        vc = y.value_counts()
        st.write(vc.head(30))
        if len(vc) > 30:
            st.caption(f"(Showing top 30 of {len(vc)} classes)")

# TAB 2: EDA & PREPROCESSING
with tb2:
    with st.expander('Scatterplot for zoning type'):
        plot_df = df[["current_land_value","tax_levy","zoning_classification"]].dropna().sample(
            n=min(20_000, len(df)), random_state=42
        ).copy()
        plot_df["land_log"] = np.log1p(plot_df["current_land_value"].clip(lower=0))
        plot_df["levy_log"] = np.log1p(plot_df["tax_levy"].clip(lower=0))
        alt.data_transformers.disable_max_rows()
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
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)

    with st.expander('Data Preprocessing Info'):
        st.write("### Missing Values Analysis")
        missing_info = X.isnull().sum()
        missing_df = pd.DataFrame({
            'Feature': missing_info.index,
            'Missing Count': missing_info.values,
            'Missing %': (missing_info.values / len(X) * 100).round(2)
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])
        st.write(f"Numeric features: {numeric_features}")
        st.write(f"Categorical features: {categorical_features}")
        st.write(f"Training set: {len(X_train)} samples")
        st.write(f"Test set: {len(X_test)} samples")

# TAB 3: MODEL TRAINING
with tb3:
    with st.expander('Model Training v3 with Multiple Models for Comparison'):
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=200, n_jobs=-1, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }

        # Model selection
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
                    with st.spinner(f'Training {model_name}...'):
                        pipeline = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('classifier', models[model_name])
                        ])
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        results[model_name] = float(accuracy)
                    st.success(f"✅ {model_name} complete: {accuracy:.2%} accuracy")
                st.session_state.results = results
                progress_bar.empty()

# TAB 4: EVALUATION & COMPARISON
with tb4:
    st.write("## Evaluation & Comparison")
    if st.session_state.get('results'):
        results = st.session_state['results']
        results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy']).sort_values('Accuracy', ascending=False)
        st.write("### Model Performance Comparison Results")
        st.dataframe(results_df.style.format({"Accuracy": "{:.4%}"}))
        best_model_name = results_df.iloc[0]['Model']
        best_model_accuracy = results_df.iloc[0]['Accuracy']
        st.write(f"### 🏆 Best Model: {best_model_name} with Accuracy: {best_model_accuracy:.4%}")
        bar_chart = alt.Chart(results_df).mark_bar().encode(
            x=alt.X('Accuracy:Q', scale=alt.Scale(domain=[0, 1]), title='Accuracy'),
            y=alt.Y('Model:N', sort='-x', title='Model'),
            color=alt.Color('Accuracy:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=['Model', alt.Tooltip('Accuracy:Q', format='.4%')]
        ).properties(title='Model Accuracy Comparison', width=700, height=400)
        st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.info("📊 Please train models in the 'Model Training' tab first to see results here.")

# Sidebar
with st.sidebar:
    st.write("---")
    st.write("**Dataset Info:**")
    st.write(f"• Total samples (loaded): {len(df)}")
    st.write(f"• Features used: {len(useful_features)}")
    st.write(f"• Target classes: {y.nunique()}")
    st.write("---")
    st.write("**Current Split:**")
    st.write(f"• Training: {len(X_train)}")
    st.write(f"• Testing: {len(X_test)}")
    st.write("• Test ratio: 30%")
