#!/usr/bin/env python3
"""
PPR501 - Machine Learning Homework 3
Core Machine Learning Backend Pipeline

This module provides data loading, preprocessing, model definition, training,
evaluation, and UI integration helpers for the Streamlit dashboard.
"""

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# Step 0: Data Loading
# ─────────────────────────────────────────────
def load_data():
    """
    Loads training, validation, and test datasets and splits them into X and y.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.join(BASE_DIR, "raw_train.csv")
    val_path   = os.path.join(BASE_DIR, "raw_val.csv")
    test_path  = os.path.join(BASE_DIR, "raw_test.csv")

    df_train = pd.read_csv(train_path).dropna()
    df_val   = pd.read_csv(val_path).dropna()
    df_test  = pd.read_csv(test_path).dropna()

    feature_cols = [col for col in df_train.columns if col != 'target']

    X_train = df_train[feature_cols];  y_train = df_train['target']
    X_val   = df_val[feature_cols];    y_val   = df_val['target']
    X_test  = df_test[feature_cols];   y_test  = df_test['target']

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_label_distribution(y):
    """
    Computes label counts and percentages for visualization.
    """
    counts      = y.value_counts()
    percentages = y.value_counts(normalize=True) * 100
    dist = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages})
    dist.index = dist.index.map({0: 'Healthy (0)', 1: 'Heart Disease (1)'})
    return dist


def compute_feature_importance(X_train, y_train):
    """
    Computes feature importances using a Random Forest classifier.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    return importances.sort_values(ascending=False)


# ─────────────────────────────────────────────
# Steps 1 - 4: Preprocessing clinical inputs
# ─────────────────────────────────────────────
def preprocess_raw_input(raw_dict):
    """
    Step 1: Pre-processing data
    Remove None values from raw user input dictionary.
    """
    return {k: v for k, v in raw_dict.items() if v is not None}


def feature_selection(data_dict):
    """
    Step 2: Feature Selection
    Select exactly the 13 clinical features in the correct order.
    """
    feature_order = [
        'age', 'sex', 'cp', 'trestbps', 'chol',
        'fbs', 'restecg', 'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal'
    ]
    return {feat: data_dict[feat] for feat in feature_order}


def feature_encoding(data_dict):
    """
    Step 3: Feature Encoding
    Scale categorical variables to [0.0, 1.0] range to match the
    Cleveland dataset preprocessing already applied to the CSV files.
    """
    enc = data_dict.copy()
    enc['sex']     = float(data_dict['sex'])
    enc['cp']      = float(data_dict['cp']    - 1) / 3.0
    enc['fbs']     = float(data_dict['fbs'])
    enc['restecg'] = float(data_dict['restecg']) / 2.0
    enc['exang']   = float(data_dict['exang'])
    enc['slope']   = float(data_dict['slope'] - 1) / 2.0
    enc['ca']      = float(data_dict['ca'])   / 3.0
    enc['thal']    = float(data_dict['thal']  - 3) / 4.0
    return enc


def feature_standardization(data_dict, scaler):
    """
    Step 4: Feature Standardization / Normalization
    Apply Z-score scaling to continuous columns using a fitted StandardScaler.
    """
    df = pd.DataFrame([data_dict])
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[continuous_cols] = scaler.transform(df[continuous_cols])

    feature_order = [
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak',
        'sex', 'cp', 'fbs', 'restecg', 'exang',
        'slope', 'ca', 'thal'
    ]
    return df[feature_order]


def scale_raw_input(raw_dict):
    """
    Runs preprocessing pipeline and returns one-row DataFrame for prediction.
    """
    X_train, _, _, _, _, _ = load_data()

    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    scaler = StandardScaler()

    # Nếu file train của bạn ĐÃ chuẩn hóa sẵn
    scaler.mean_ = np.array([54.549587, 130.958678, 249.838843, 149.962810, 0.999174])
    scaler.scale_ = np.array([8.978373, 17.586103, 52.737566, 22.639527, 1.120618])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_samples_seen_ = len(X_train)
    scaler.feature_names_in_ = np.array(continuous_cols)

    preprocessed = preprocess_raw_input(raw_dict)
    selected     = feature_selection(preprocessed)
    encoded      = feature_encoding(selected)
    standardized = feature_standardization(encoded, scaler)

    return standardized


# ─────────────────────────────────────────────
# Step 5: Model Training & Evaluation
# ─────────────────────────────────────────────
def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test, selected_features):
    """
    Step 5: Train 8 classifiers on selected features and evaluate on val + test.
    """
    X_train_sel = X_train[selected_features]
    X_val_sel   = X_val[selected_features]
    X_test_sel  = X_test[selected_features]

    # Define classifiers with exact matching hyperparameters
    dt  = DecisionTreeClassifier(max_depth=None, min_samples_leaf=15, criterion='entropy', random_state=0)
    knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='chebyshev')
    nb  = GaussianNB(var_smoothing=1e-12)
    rf  = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=1, random_state=0)
    
    try:
        ada = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=0, algorithm="SAMME")
    except TypeError:
        ada = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=0)
        
    gb  = GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.2, random_state=0)
    xgb = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.2, random_state=0, eval_metric='logloss')

    # Soft-voting ensemble over all 7 base learners
    estimators = [
        ('dt',  dt),  ('knn', knn), ('nb',  nb),
        ('rf',  rf),  ('ada', ada), ('gb',  gb), ('xgb', xgb)
    ]
    ensemble = VotingClassifier(estimators=estimators, voting='soft')

    models = {
        'Decision Tree':          dt,
        'k-NN':                   knn,
        'Naive Bayes':            nb,
        'Random Forest':          rf,
        'AdaBoost':               ada,
        'Gradient Boosting':      gb,
        'XGBoost':                xgb,
        'Ensemble (Soft Voting)': ensemble
    }

    results = {}

    for name, model in models.items():
        # Train
        model.fit(X_train_sel, y_train)

        # Validation metrics
        y_val_pred = model.predict(X_val_sel)
        y_val_prob = (
            model.predict_proba(X_val_sel)[:, 1]
            if hasattr(model, "predict_proba") else None
        )
        val_acc  = accuracy_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred, zero_division=0)
        val_rec  = recall_score(y_val, y_val_pred, zero_division=0)
        val_f1   = f1_score(y_val, y_val_pred, zero_division=0)
        val_auc  = roc_auc_score(y_val, y_val_prob) if y_val_prob is not None else 0.5
        val_cm   = confusion_matrix(y_val, y_val_pred)

        # Test metrics
        y_test_pred = model.predict(X_test_sel)
        y_test_prob = (
            model.predict_proba(X_test_sel)[:, 1]
            if hasattr(model, "predict_proba") else None
        )
        test_acc  = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, zero_division=0)
        test_rec  = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1   = f1_score(y_test, y_test_pred, zero_division=0)
        test_auc  = roc_auc_score(y_test, y_test_prob) if y_test_prob is not None else 0.5
        test_cm   = confusion_matrix(y_test, y_test_pred)

        results[name] = {
            'model': model,
            'val_metrics': {
                'Accuracy': val_acc, 'Precision': val_prec,
                'Recall': val_rec,   'F1-Score': val_f1,
                'ROC-AUC': val_auc,  'Confusion Matrix': val_cm
            },
            'test_metrics': {
                'Accuracy': test_acc, 'Precision': test_prec,
                'Recall': test_rec,   'F1-Score': test_f1,
                'ROC-AUC': test_auc,  'Confusion Matrix': test_cm
            }
        }

    return results


# ─────────────────────────────────────────────
# Example patients (raw clinical values)
# ─────────────────────────────────────────────
def get_example_patients():
    """Returns realistic example patients with raw clinical values."""
    return {
        "Example 1 (No Heart Disease)": {
            'age': 58, 'sex': 1, 'cp': 2, 'trestbps': 130, 'chol': 250,
            'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0,
            'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 3
        },
        "Example 2 (Heart Disease)": {
            'age': 67, 'sex': 1, 'cp': 4, 'trestbps': 160, 'chol': 286,
            'fbs': 0, 'restecg': 0, 'thalach': 108, 'exang': 1,
            'oldpeak': 1.5, 'slope': 2, 'ca': 3, 'thal': 3
        }
    }
