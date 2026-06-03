import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

st.set_page_config(page_title="HepaGuard: Liver Disease XAI", layout="wide")

# ==========================================
# 1. Load Artifacts (Graceful Fallback)
# ==========================================
@st.cache_resource
def load_artifacts():
    # Load core components
    model = joblib.load("streamlit_artifacts/final_ensemble.pkl")
    
    # Try to load RF pipeline for SHAP
    rf_pipeline = None
    scaler = None
    rf_clf = None
    
    if os.path.exists("streamlit_artifacts/rf_pipeline.pkl"):
        rf_pipeline = joblib.load("streamlit_artifacts/rf_pipeline.pkl")
        scaler = rf_pipeline.named_steps['scaler']
        rf_clf = rf_pipeline.named_steps['clf']
    
    with open("streamlit_artifacts/selected_features.json", "r") as f:
        features = json.load(f)
        
    with open("streamlit_artifacts/optimal_threshold.json", "r") as f:
        threshold = json.load(f)["threshold"]
        
    # Initialize SHAP Explainer on the EXTRACTED classifier
    shap_explainer = None
    if rf_clf is not None:
        shap_explainer = shap.TreeExplainer(rf_clf)
    
    # Try to load LIME explainer or background data
    lime_explainer = None
    if os.path.exists("streamlit_artifacts/lime_explainer.pkl"):
        lime_explainer = joblib.load("streamlit_artifacts/lime_explainer.pkl")
    elif os.path.exists("streamlit_artifacts/shap_background.npy"):
        background_data = np.load("streamlit_artifacts/shap_background.npy")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=background_data,
            feature_names=features,
            class_names=['No Disease', 'Liver Disease'],
            mode='classification',
            discretize_continuous=True,
            random_state=42
        )
    
    return model, rf_pipeline, scaler, features, threshold, shap_explainer, lime_explainer

model, rf_pipeline, scaler, selected_features, optimal_threshold, shap_explainer, lime_explainer = load_artifacts()

# ==========================================
# 2. Feature Engineering Function
# ==========================================
def engineer_features(df):
    df = df.copy()
    epsilon = 1e-6
    
    df['Bilirubin_Ratio'] = df['Direct_Bilirubin'] / (df['Total_Bilirubin'] + epsilon)
    df['AST_ALT_Ratio'] = df['Aspartate_Aminotransferase'] / (df['Alamine_Aminotransferase'] + epsilon)
    df['Globulin'] = df['Total_Protiens'] - df['Albumin']
    df['Enzyme_Load'] = df['Alkaline_Phosphotase'] + df['Alamine_Aminotransferase'] + df['Aspartate_Aminotransferase']
    df['Bili_AST_Interaction'] = df['Total_Bilirubin'] * df['Aspartate_Aminotransferase']
    df['Albumin_Protein_Ratio'] = df['Albumin'] / (df['Total_Protiens'] + epsilon)
    df['Is_Elder'] = (df['Age'] >= 60).astype(int)
    df['ALT_Albumin_Ratio'] = df['Alamine_Aminotransferase'] / (df['Albumin'] + epsilon)
    df['Bilirubin_Protein'] = df['Total_Bilirubin'] * df['Total_Protiens']
    
    for col in ['Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase']:
        df[f'{col}_log'] = np.log1p(df[col])
        
    return df

# ==========================================
# 3. Streamlit UI
# ==========================================
st.title("🩺 HepaGuard: Liver Disease Prediction & XAI")
st.markdown("Enter patient clinical data below to get an AI-powered risk assessment with **SHAP** and **LIME** explanations.")

st.sidebar.header("📋 Patient Data Input")

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
total_bilirubin = st.sidebar.number_input("Total Bilirubin", min_value=0.0, value=1.0, step=0.1)
direct_bilirubin = st.sidebar.number_input("Direct Bilirubin", min_value=0.0, value=0.3, step=0.1)
alkaline_phosphotase = st.sidebar.number_input("Alkaline Phosphotase", min_value=0.0, value=150.0, step=1.0)
alamine_aminotransferase = st.sidebar.number_input("Alamine Aminotransferase (ALT)", min_value=0.0, value=40.0, step=1.0)
aspartate_aminotransferase = st.sidebar.number_input("Aspartate Aminotransferase (AST)", min_value=0.0, value=45.0, step=1.0)
total_protiens = st.sidebar.number_input("Total Proteins", min_value=0.0, value=7.0, step=0.1)
albumin = st.sidebar.number_input("Albumin", min_value=0.0, value=4.0, step=0.1)
albumin_globulin_ratio = st.sidebar.number_input("Albumin and Globulin Ratio", min_value=0.0, value=1.5, step=0.1)

if st.sidebar.button("🔍 Analyze Risk & Explain"):
    with st.spinner("Running models and generating explanations..."):
        # 1. Create DataFrame
        input_data = {
            'Age': age, 'Gender': 1 if gender == "Male" else 0,
            'Total_Bilirubin': total_bilirubin, 'Direct_Bilirubin': direct_bilirubin,
            'Alkaline_Phosphotase': alkaline_phosphotase, 'Alamine_Aminotransferase': alamine_aminotransferase,
            'Aspartate_Aminotransferase': aspartate_aminotransferase, 'Total_Protiens': total_protiens,
            'Albumin': albumin, 'Albumin_and_Globulin_Ratio': albumin_globulin_ratio
        }
        df = pd.DataFrame([input_data])
        
        # 2. Engineer Features
        df_engineered = engineer_features(df)
        
        # 3. Align with selected features
        for col in selected_features:
            if col not in df_engineered.columns:
                df_engineered[col] = 0
                
        X_input = df_engineered[selected_features]
        
        # 4. Predict
        probas = model.predict_proba(X_input)[0]
        prob_disease = probas[1]
        
        prediction = 1 if prob_disease >= optimal_threshold else 0
        prediction_label = "⚠️ Liver Disease Risk Detected" if prediction == 1 else "✅ No Liver Disease Risk Detected"
        risk_color = "red" if prediction == 1 else "green"
        
        # ==========================================
        # 5. Display Results
        # ==========================================
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", prediction_label, delta=None, delta_color=risk_color)
        with col2:
            st.metric("Disease Probability", f"{prob_disease:.2%}")
        with col3:
            st.metric("Optimal Threshold Used", f"{optimal_threshold:.2%}")
            
        st.subheader("📊 Processed Features")
        st.dataframe(df_engineered[selected_features].T.rename(columns={0: "Value"}), use_container_width=True)
        
                # ==========================================
        # 6. XAI: SHAP Explanation ( Matplotlib Waterfall)
        # ==========================================
        if shap_explainer is not None and scaler is not None:
            st.subheader("🧠 SHAP Explanation")
            st.markdown("Shows how each feature pushed the prediction away from the base rate. *(Values shown are scaled Z-scores).*")
            
            # Scale the input data
            X_input_scaled = scaler.transform(X_input)
            
            # Get SHAP values
            shap_values = shap_explainer.shap_values(X_input_scaled)
            expected_vals = shap_explainer.expected_value
            
            # Extract class 1 (Liver Disease) values and base value robustly
            if isinstance(shap_values, list):
                # List format: [class_0_shap, class_1_shap]
                shap_vals = np.array(shap_values[1])
                base_val = expected_vals[1] if isinstance(expected_vals, (list, np.ndarray)) and len(expected_vals) > 1 else expected_vals
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # 3D array format: (samples, features, classes)
                shap_vals = shap_values[0, :, 1]
                base_val = expected_vals[1] if isinstance(expected_vals, (list, np.ndarray)) and len(expected_vals) > 1 else expected_vals
            else:
                # 2D array format or single class
                shap_vals = np.array(shap_values)
                base_val = expected_vals

            # Ensure shap_vals is 1D (for a single prediction)
            if shap_vals.ndim > 1:
                shap_vals = shap_vals[0]
                
            # CRITICAL FIX: Safely convert base_val to a standard Python float scalar
            # If it's an array like [0.65], flatten it and grab the first item.
            if isinstance(base_val, (list, np.ndarray)):
                base_val = float(np.array(base_val).flatten()[0])
            else:
                base_val = float(base_val)

            X_scaled_1d = X_input_scaled[0]

            # Use SHAP Waterfall plot (Native Matplotlib, 100% Streamlit compatible)
            try:
                explanation = shap.Explanation(
                    values=shap_vals,
                    base_values=base_val,
                    data=X_scaled_1d,
                    feature_names=selected_features
                )
                shap.plots.waterfall(explanation, show=False)
                st.pyplot(plt.gcf())
                plt.clf() # Clear figure to prevent overlapping on next run
            except Exception:
                # Fallback to standard matplotlib force plot if waterfall fails
                shap.force_plot(
                    base_val,
                    shap_vals,
                    X_scaled_1d,
                    feature_names=selected_features,
                    matplotlib=True,
                    show=False
                )
                st.pyplot(plt.gcf())
                plt.clf()
        # ==========================================
        # 7. XAI: LIME Explanation
        # ==========================================
        if lime_explainer is not None:
            st.subheader("🔬 LIME Explanation (Local Surrogate)")
            st.markdown("Approximates the model's behavior locally. LIME uses the **original, unscaled values** for easy interpretation.")
            
            X_np = X_input.values
            
            exp = lime_explainer.explain_instance(
                X_np[0], 
                rf_pipeline.predict_proba, 
                num_features=len(selected_features),
                top_labels=1
            )
            
            available_labels = list(exp.local_exp.keys())

            target_label = 'Liver Disease' if 'Liver Disease' in available_labels else available_labels[0]

            fig = exp.as_pyplot_figure(label=target_label)
            st.pyplot(fig)
            fig.set_size_inches(10, 6)
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf() # Clear figure
        else:
            st.info("ℹ️ LIME explainer is not available. Run the artifact generation script in your Jupyter Notebook to enable LIME.")
            
        st.info("**Medical Disclaimer:** This is an AI-assisted screening tool. Always consult a hepatologist or medical professional for a definitive diagnosis.")