#!/usr/bin/env python3
"""
PPR501 - Machine Learning Homework 3
Interactive Streamlit Web Dashboard

This application allows users to input patient medical clinical parameters,
performs preprocessing through the backend data pipeline, runs predictions
across 8 different machine learning classifiers, and visualizes the results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import model_backend

# Page configurations
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide"
)

# Custom premium dashboard styling (White Background, Dark Charcoal accents)
st.markdown("""
<style>
    /* Page background */
    .stApp {
        background-color: #black !important;
        color: white !important;
    }
            
    /* Button */ 
    div.stButton > button {
        background-color: #262730;
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        background: #3a3b47;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load data and setup session states
X_train, y_train, X_val, y_val, X_test, y_test = model_backend.load_data()
examples = model_backend.get_example_patients()

# Pre-populate session state with Example 1 if empty
if 'age' not in st.session_state:
    for k, v in examples["Example 1 (No Heart Disease)"].items():
        st.session_state[k] = v


def load_example():
    """Callback when example selection changes."""
    sel = st.session_state.example_select
    if sel in examples:
        for k, v in examples[sel].items():
            st.session_state[k] = v


# Compute global feature importances using Random Forest once
@st.cache_resource
def get_global_feature_importance():
    return model_backend.compute_feature_importance(X_train, y_train)


feature_importance = get_global_feature_importance()


# Cache model training on 13 features
@st.cache_resource
def get_cached_models(k_features):
    selected_features = list(feature_importance.index[:k_features])
    results = model_backend.train_and_evaluate_models(
        X_train, y_train, X_val, y_val, X_test, y_test, selected_features
    )
    return results, selected_features


# Set hyperparameter features to match all 13 clinical features
results, selected_features = get_cached_models(k_features=13)

# Build Layout columns
left_col, right_col = st.columns([1.1, 0.9])

# --- Left Column: Enter Patient Features ---
with left_col:
    # 13 Clinical inputs inside the expander container
    st.text("✍️ Enter Patient Features")
    # Row 1
    with st.container(border=True):
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
        with r1_c1:
            st.number_input("age (years)", min_value=1, max_value=120, step=1, key="age")
        with r1_c2:
            st.selectbox(
                "sex (0=female, 1=male)", 
                options=[0, 1], 
                format_func=lambda x: "1" if x==1 else "0",
                key="sex"
            )
        with r1_c3:
            st.selectbox("cp (chest pain type 1..4)", options=[1, 2, 3, 4], key="cp")
        with r1_c4:
            st.number_input("trestbps (resting BP mmHg)", min_value=50, max_value=250, step=1, key="trestbps")
    
    # Row 2
    with st.container(border=True):
        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
        with r2_c1:
            st.number_input("chol (serum cholesterol mg/dl)", min_value=50, max_value=600, step=1, key="chol")
        with r2_c2:
            st.selectbox("fbs (>120 mg/dl? 1/0)", options=[0, 1], key="fbs")
        with r2_c3:
            st.selectbox("restecg (0..2)", options=[0, 1, 2], key="restecg")
        with r2_c4:
            st.number_input("thalach (max heart rate)", min_value=50, max_value=250, step=1, key="thalach")
    
    # Row 3
    with st.container(border=True):
        r3_c1, r3_c2, r3_c3, r3_c4 = st.columns(4)
        with r3_c1:
            st.selectbox("exang (exercise angina 1/0)", options=[0, 1], key="exang")
        with r3_c2:
            st.number_input("oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1, key="oldpeak")
        with r3_c3:
            st.selectbox("slope (1..3)", options=[1, 2, 3], key="slope")
        with r3_c4:
            st.selectbox("ca (major vessels 0..3)", options=[0, 1, 2, 3], key="ca")
    
    # Row 4 (Full width)
    with st.container(border=True):
        st.selectbox("thal (3=normal, 6=fixed, 7=reversible)", options=[3, 6, 7], key="thal")
    
    # Example Patient selector and Predict trigger
    r5_c1, r5_c2 = st.columns(2)
    with r5_c1:
        with st.container(border=True):
            st.selectbox(
                "Select Example Patient", 
                options=list(examples.keys()), 
                key="example_select", 
                on_change=load_example
            )
    with r5_c2:
        predict_clicked = st.button("🔍 Predict", use_container_width=True)
            
# --- Right Column: Predictions Display ---
with right_col:
    # Prepare patient input dictionary
    raw_patient = {
        'age': st.session_state.age,
        'sex': st.session_state.sex,
        'cp': st.session_state.cp,
        'trestbps': st.session_state.trestbps,
        'chol': st.session_state.chol,
        'fbs': st.session_state.fbs,
        'restecg': st.session_state.restecg,
        'thalach': st.session_state.thalach,
        'exang': st.session_state.exang,
        'oldpeak': st.session_state.oldpeak,
        'slope': st.session_state.slope,
        'ca': st.session_state.ca,
        'thal': st.session_state.thal
    }
    
    # Scale patient input dynamically
    patient_scaled_df = model_backend.scale_raw_input(raw_patient)
    
    # Run inference across models
    predictions = {}
    for name, info in results.items():
        model = info['model']
        patient_input = patient_scaled_df[selected_features]
        pred = model.predict(patient_input)[0]
        prob_1 = model.predict_proba(patient_input)[0, 1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)
        
        predictions[name] = {
            'class': pred,
            'prob_1': prob_1
        }
        
    # Draw Plotly Bar Chart matching prototype
    models_order = ['Decision Tree', 'k-NN', 'Naive Bayes', 'Random Forest', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'Ensemble (Soft Voting)']
    
    x_bars = []
    y_bars = []
    colors = []
    texts = []
    hovers = []
    
    for name in models_order:
        pred_info = predictions[name]
        p1 = pred_info['prob_1']
        c = pred_info['class']
        
        if c == 1:
            conf = p1
            color = '#C5314B'  # Crimson-Pink Red
            text = '🫀 Heart Disease'
            hover = f"<b>{name}</b><br>Prediction: Heart Disease<br>Confidence: {conf:.1%}"
        else:
            conf = 1.0 - p1
            color = '#2D7D31'  # Grass Green
            text = '✅ No Heart Disease'
            hover = f"<b>{name}</b><br>Prediction: No Heart Disease<br>Confidence: {conf:.1%}"
            
        x_bars.append(name)
        y_bars.append(conf)
        colors.append(color)
        texts.append(text)
        hovers.append(hover)
        
    fig_pred = go.Figure()

    # Draw 7 base classifiers (no border)
    fig_pred.add_trace(go.Bar(
        x=x_bars[:-1],
        y=y_bars[:-1],
        text=texts[:-1],
        textposition='inside',
        insidetextanchor='end',
        marker=dict(
            color=colors[:-1],
            line=dict(width=0)
        ),
        hoverinfo='text',
        hovertext=hovers[:-1],
        textfont=dict(color='white', size=11, family='sans-serif'),
        textangle=90,
        showlegend=False
    ))

    # Draw Ensemble Classifier (no border)
    fig_pred.add_trace(go.Bar(
        x=[x_bars[-1]],
        y=[y_bars[-1]],
        text=[texts[-1]],
        textposition='inside',
        insidetextanchor='end',
        marker=dict(
            color=[colors[-1]],
            line=dict(width=0)
        ),
        hoverinfo='text',
        hovertext=[hovers[-1]],
        textfont=dict(color='white', size=11, family='sans-serif'),
        textangle=90,
        showlegend=False
    ))
    
    # Add text labels above bars
    annotations = []
    for idx, (name, val) in enumerate(zip(x_bars, y_bars)):
        annotations.append(dict(
            x=name,
            y=val + 0.01,
            text=f"<b>{int(round(val*100))}%</b>",
            showarrow=False,
            font=dict(color='black', size=12, family='sans-serif'),
            xanchor='center',
            yanchor='bottom'
        ))
        
    fig_pred.update_layout(
        title={
            'text': "Model Predictions",
            'y': 0.95,
            'x': 0.05,
            'xanchor': 'left',
            'yanchor': 'top',   
            'font': dict(size=18, color='black')
        },
        yaxis=dict(
            title=dict(text="Prediction Confidence", font=dict(color='black', size=13)),
            range=[0, 1.15],
            showgrid=False,
            showline=False,
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"],
            gridcolor='#E5E7EB',
            linecolor='black',
            linewidth=1.5,
            tickcolor='black',
            tickfont=dict(color='black', size=15)
        ),
        xaxis=dict(
            title=dict(text="Model", font=dict(color='black', size=13)),
            linecolor='black',
            linewidth=1.5,
            tickcolor='black',
            tickfont=dict(color='black', size=15),
            tickangle=35,
            showgrid=False,
            showline=False,
        ),
        template="plotly_white",
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=50, r=20, t=60, b=80),
        showlegend=False,
        annotations=annotations
    )    

    # Render results overview panel
    st.text("📝 Model Predictions Overview")

    st.plotly_chart(fig_pred, use_container_width=True)

    # Class names list
    model_classes_html = ""
    st.markdown(f"""
<div style="font-family: sans-serif; font-size: 17px; color: black; margin-top: 8px; font-weight: 500; line-height: 1.6; padding-left: 4px;">
{model_classes_html}
</div>
""", unsafe_allow_html=True)
