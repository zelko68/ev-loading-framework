#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptor-Guided Selection of Extracellular Vesicle Loading Strategies
for Small-Molecule Drug Delivery: A Mechanistically Interpretable
Decision-Support Framework

Authors: Romána Zelkó and Adrienn Kazsoki
University Pharmacy Department of Pharmacy Administration,
Semmelweis University, 1092 Budapest, Hungary

Corresponding author: zelko.romana@semmelweis.hu

Requirements:
    Python >= 3.11
    scikit-learn >= 1.3
    numpy >= 1.24
    pandas >= 2.0

Usage:
    python ev_loading_framework.py

    For prospective prediction of a new compound:
        from ev_loading_framework import predict_loading_strategy
        result = predict_loading_strategy(LogP=2.5, MW=400, Solubility=1.0,
                                          HBD=3, HBA=6, PSA=90, Charge=0)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import json

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
RANDOM_SEED = 42
DESCRIPTOR_COLS = ['LogP', 'MW', 'Solubility', 'HBD', 'HBA', 'PSA', 'Charge']
METHOD_COLS = ['Passive', 'Electroporation', 'Saponin', 'FreezeThaw', 'Sonication']
METHOD_LABELS = ['Passive Incubation', 'Electroporation', 'Saponin',
                 'Freeze-Thaw', 'Sonication']
EXTERNAL_DRUGS = ['Sildenafil', 'Caffeine', 'Ampicillin', 'Furosemide']
ALPHA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
LAMBDA_GRID = np.logspace(-4, 2, 50)
N_RANDOM_REPEATS = 50


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET (Table 1)
# ═══════════════════════════════════════════════════════════════════════════════
def load_dataset():
    """Load the 21-compound experimental dataset (Table 1)."""
    data = {
        'Drug': ['Doxorubicin', 'Paclitaxel', 'Cisplatin', 'Gemcitabine',
                 'Docetaxel', 'Methotrexate', 'Riboflavin', 'Ampicillin',
                 'Furosemide', 'Warfarin', 'Digoxin', 'Chloroquine',
                 'Irinotecan', '5-Fluorouracil', 'Caffeine', 'Quercetin',
                 'Resveratrol', 'Sildenafil', 'Curcumin', 'Pirarubicin',
                 'Tamoxifen'],
        'BCS_Class': ['NC (IV-only)', 'NC (no FDA oral IR)', 'NC (IV-only)',
                      'NC (IV-only)', 'NC (IV-only)', 'IV', 'III', 'III',
                      'IV', 'II', 'NC (conflicting II/III; NTI)', 'I',
                      'NC (primarily IV)', 'III/NC', 'I',
                      'NC (not approved)', 'NC (not approved)', 'II',
                      'NC (not approved)', 'NC (IV-only)', 'II'],
        'LogP':       [ 1.27, 3.97,-2.50,-1.20, 4.10,-1.85,-0.60, 0.87,
                        2.03, 2.92, 1.26, 3.81, 3.27,-0.89, 0.16, 1.83,
                        3.05, 2.71, 3.97, 1.34, 4.30],
        'MW':         [543.5, 853.9, 300.0, 263.2, 861.9, 454.4, 376.4, 349.4,
                       330.7, 308.3, 780.9, 319.9, 586.7, 130.1, 194.2, 302.2,
                       228.2, 474.6, 368.4, 557.5, 371.4],
        'Solubility': [  50, 0.3, 3.5, 100, 0.1, 0.3, 1.2, 1.0,
                        0.5, 0.14, 0.05, 0.7, 0.2, 12.2, 21.5, 0.003,
                        0.3, 3.5, 0.003, 45, 0.01],
        'HBD':        [4, 2, 0, 3, 2, 4, 5, 3, 2, 1, 5, 2, 2, 2, 0, 5, 3, 2, 2, 4, 1],
        'HBA':        [8,11, 2, 5,11, 9, 8, 8, 5, 4,12, 4, 9, 3, 3, 7, 3, 6, 4, 8, 2],
        'PSA':        [124, 97, 0, 95, 98,168,149,115, 99, 49,
                       206, 47,106, 66, 58,131, 60, 87, 93,126, 39],
        'Charge':     [1, 0, 0, 0, 0,-2, 0,-1,-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'Passive':    [72,75,65,68,77, 5, 8, 3,70,72,65,68,71,55,52,69,73,74,76,70,79],
        'Electroporation': [8, 2, 7, 9, 1.5, 52, 48, 55, 5, 3,
                           18, 8, 5, 10, 11, 4, 6, 4, 2, 9, 1],
        'Saponin':    [18,25,12,22,28,68,71,74,20,18,35,22,24,15,14,19,21,20,26,19,15],
        'FreezeThaw': [15,12,18,22,10,32,38,35, 8, 9,40,14,11,19,21,12,13,10, 9,16, 7],
        'Sonication': [ 4, 8, 6, 7, 6,45,48,51,15,11,38,18,13, 9,11,14,16,12, 7, 5, 5],
    }
    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
def optimize_hyperparameters(X_train, y_train):
    """Grid search for Elastic Net hyperparameters using LOOCV."""
    best_score = -np.inf
    best_alpha, best_lambda = None, None
    loo = LeaveOneOut()

    for alpha in ALPHA_GRID:
        for lam in LAMBDA_GRID:
            preds = np.zeros(len(y_train))
            for tr_idx, val_idx in loo.split(X_train):
                model = ElasticNet(alpha=lam, l1_ratio=alpha, max_iter=10000,
                                   random_state=RANDOM_SEED, fit_intercept=True)
                model.fit(X_train[tr_idx], y_train[tr_idx])
                preds[val_idx] = model.predict(X_train[val_idx])
            score = r2_score(y_train, preds)
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_lambda = lam

    return best_alpha, best_lambda, best_score


def train_models(df, training_drugs=None):
    """
    Train Elastic Net models for all five loading methods.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    training_drugs : list or None
        If None, uses all compounds except predefined external set.

    Returns
    -------
    models : dict
        Trained ElasticNet models keyed by method name.
    scaler : StandardScaler
        Fitted scaler for descriptor standardization.
    params : dict
        Best hyperparameters for each method.
    """
    if training_drugs is None:
        df_train = df[~df['Drug'].isin(EXTERNAL_DRUGS)].reset_index(drop=True)
    else:
        df_train = df[df['Drug'].isin(training_drugs)].reset_index(drop=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[DESCRIPTOR_COLS].values)

    models = {}
    params = {}

    for method in METHOD_COLS:
        y = df_train[method].values
        best_alpha, best_lambda, best_r2 = optimize_hyperparameters(X_train, y)
        model = ElasticNet(alpha=best_lambda, l1_ratio=best_alpha,
                           max_iter=10000, random_state=RANDOM_SEED, fit_intercept=True)
        model.fit(X_train, y)
        models[method] = model
        params[method] = {'l1_ratio': best_alpha, 'lambda': best_lambda, 'loocv_r2': best_r2}

    return models, scaler, params


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
def loocv_evaluation(df_train, scaler, params):
    """Perform LOOCV and return regression metrics + decision accuracy."""
    X = scaler.transform(df_train[DESCRIPTOR_COLS].values)
    loo = LeaveOneOut()
    results = {}
    pred_matrix = np.zeros((len(df_train), len(METHOD_COLS)))

    for m_idx, method in enumerate(METHOD_COLS):
        y = df_train[method].values
        p = params[method]
        preds = np.zeros(len(y))
        for tr_idx, val_idx in loo.split(X):
            model = ElasticNet(alpha=p['lambda'], l1_ratio=p['l1_ratio'],
                               max_iter=10000, random_state=RANDOM_SEED, fit_intercept=True)
            model.fit(X[tr_idx], y[tr_idx])
            preds[val_idx] = model.predict(X[val_idx])
        pred_matrix[:, m_idx] = preds
        results[method] = {
            'MAE': mean_absolute_error(y, preds),
            'RMSE': np.sqrt(mean_squared_error(y, preds)),
            'R2': r2_score(y, preds)
        }

    true_best = np.argmax(df_train[METHOD_COLS].values, axis=1)
    pred_best = np.argmax(pred_matrix, axis=1)
    decision_acc = np.mean(pred_best == true_best) * 100

    return results, decision_acc


def external_validation(models, scaler, df_ext):
    """Evaluate models on predefined external validation set."""
    X_ext = scaler.transform(df_ext[DESCRIPTOR_COLS].values)
    pred_matrix = np.column_stack([models[m].predict(X_ext) for m in METHOD_COLS])
    true_matrix = df_ext[METHOD_COLS].values
    pred_best = np.argmax(pred_matrix, axis=1)
    true_best = np.argmax(true_matrix, axis=1)

    results = []
    for i, drug in enumerate(df_ext['Drug']):
        results.append({
            'Drug': drug,
            'True_Optimal': METHOD_LABELS[true_best[i]],
            'Predicted_Optimal': METHOD_LABELS[pred_best[i]],
            'Correct': pred_best[i] == true_best[i]
        })
    accuracy = np.mean(pred_best == true_best) * 100
    return results, accuracy


def repeated_random_validation(df, params, n_repeats=N_RANDOM_REPEATS):
    """Perform repeated random train/test splits."""
    accuracies = []

    for rep in range(n_repeats):
        np.random.seed(rep)
        indices = np.random.permutation(len(df))
        ext_idx = indices[:4]
        train_idx = indices[4:]

        df_r_train = df.iloc[train_idx].reset_index(drop=True)
        df_r_ext = df.iloc[ext_idx].reset_index(drop=True)

        scaler_r = StandardScaler()
        X_r_train = scaler_r.fit_transform(df_r_train[DESCRIPTOR_COLS].values)
        X_r_ext = scaler_r.transform(df_r_ext[DESCRIPTOR_COLS].values)

        preds = []
        for method in METHOD_COLS:
            p = params[method]
            model = ElasticNet(alpha=p['lambda'], l1_ratio=p['l1_ratio'],
                               max_iter=10000, random_state=RANDOM_SEED, fit_intercept=True)
            model.fit(X_r_train, df_r_train[method].values)
            preds.append(model.predict(X_r_ext))

        pred_matrix = np.column_stack(preds)
        pred_best = np.argmax(pred_matrix, axis=1)
        true_best = np.argmax(df_r_ext[METHOD_COLS].values, axis=1)
        accuracies.append(np.mean(pred_best == true_best) * 100)

    return np.mean(accuracies), np.std(accuracies), accuracies


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICABILITY DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════
def applicability_domain(X_scaled, drug_names=None):
    """
    Calculate leverage values and applicability domain threshold.

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized descriptor matrix (n x p).
    drug_names : list or None
        Drug names for reporting.

    Returns
    -------
    leverages : np.ndarray
        Leverage value for each compound.
    h_star : float
        Applicability domain threshold.
    within_ad : np.ndarray
        Boolean array indicating whether each compound is within AD.
    """
    n, p = X_scaled.shape
    H = X_scaled @ np.linalg.inv(X_scaled.T @ X_scaled) @ X_scaled.T
    leverages = np.diag(H)
    h_star = 3 * (p + 1) / n
    within_ad = leverages < h_star
    return leverages, h_star, within_ad


# ═══════════════════════════════════════════════════════════════════════════════
# PROSPECTIVE PREDICTION (Main user-facing function)
# ═══════════════════════════════════════════════════════════════════════════════
_cached_models = None
_cached_scaler = None
_cached_X_full = None


def _ensure_models():
    """Lazy-load models on first prediction call."""
    global _cached_models, _cached_scaler, _cached_X_full
    if _cached_models is None:
        df = load_dataset()
        scaler_full = StandardScaler()
        X_full = scaler_full.fit_transform(df[DESCRIPTOR_COLS].values)

        # Use hyperparameters from training-set optimization
        _, _, params = train_models(df)

        models_full = {}
        for method in METHOD_COLS:
            p = params[method]
            model = ElasticNet(alpha=p['lambda'], l1_ratio=p['l1_ratio'],
                               max_iter=10000, random_state=RANDOM_SEED, fit_intercept=True)
            model.fit(X_full, df[method].values)
            models_full[method] = model

        _cached_models = models_full
        _cached_scaler = scaler_full
        _cached_X_full = X_full


def predict_loading_strategy(LogP, MW, Solubility, HBD, HBA, PSA, Charge):
    """
    Predict the optimal EV loading strategy for a new small-molecule compound.

    Parameters
    ----------
    LogP : float
        Octanol-water partition coefficient.
    MW : float
        Molecular weight (g/mol).
    Solubility : float
        Aqueous solubility (µg/mL).
    HBD : int
        Number of hydrogen bond donors.
    HBA : int
        Number of hydrogen bond acceptors.
    PSA : float
        Polar surface area (Å²).
    Charge : int
        Net formal charge at pH 7.4.

    Returns
    -------
    dict with keys:
        'recommended_method' : str
        'predicted_efficiencies' : dict (method → predicted LE%)
        'leverage' : float
        'within_applicability_domain' : bool
        'leverage_threshold' : float
    """
    _ensure_models()
    x = np.array([[LogP, MW, Solubility, HBD, HBA, PSA, Charge]])
    x_z = _cached_scaler.transform(x)

    predictions = {}
    for i, method in enumerate(METHOD_COLS):
        predictions[METHOD_LABELS[i]] = round(float(_cached_models[method].predict(x_z)[0]), 2)

    recommended = max(predictions, key=predictions.get)

    h_new = x_z @ np.linalg.inv(_cached_X_full.T @ _cached_X_full) @ x_z.T
    leverage = float(h_new[0, 0])
    h_star = 3 * (len(DESCRIPTOR_COLS) + 1) / 21

    return {
        'recommended_method': recommended,
        'predicted_efficiencies': dict(sorted(predictions.items(), key=lambda x: -x[1])),
        'leverage': round(leverage, 4),
        'within_applicability_domain': leverage < h_star,
        'leverage_threshold': round(h_star, 3)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL EQUATIONS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════
def print_final_equations(models_full, scaler_full):
    """Print the final regression equations (Equations 7-11)."""
    print("\n" + "="*70)
    print("FINAL REGRESSION EQUATIONS (fitted on full dataset, n=21)")
    print("="*70)
    for i, method in enumerate(METHOD_COLS):
        m = models_full[method]
        eq = f"  LE_{METHOD_LABELS[i]} = {m.intercept_:.2f}"
        for j, d in enumerate(DESCRIPTOR_COLS):
            c = m.coef_[j]
            if abs(c) > 0.001:
                sign = " + " if c > 0 else " - "
                eq += f"{sign}{abs(c):.2f}·z_{d}"
        print(eq)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(RANDOM_SEED)

    # 1. Load data
    df = load_dataset()
    df_train = df[~df['Drug'].isin(EXTERNAL_DRUGS)].reset_index(drop=True)
    df_ext = df[df['Drug'].isin(EXTERNAL_DRUGS)].reset_index(drop=True)

    print("="*70)
    print("EV Loading Strategy Decision-Support Framework")
    print("Descriptor-guided selection for small-molecule drug delivery")
    print("="*70)
    print(f"\nDataset: {len(df)} compounds, {len(DESCRIPTOR_COLS)} descriptors, "
          f"{len(METHOD_COLS)} loading methods")
    print(f"Training set: {len(df_train)} compounds")
    print(f"External validation set: {len(df_ext)} compounds "
          f"({', '.join(EXTERNAL_DRUGS)})")

    # 2. Train models
    print("\n--- Training Elastic Net models with LOOCV hyperparameter tuning ---")
    models, scaler, params = train_models(df)
    print("\nOptimal hyperparameters:")
    for method in METHOD_COLS:
        p = params[method]
        print(f"  {method:20s} | l1_ratio={p['l1_ratio']:.1f} | "
              f"lambda={p['lambda']:.6f} | LOOCV R²={p['loocv_r2']:.4f}")

    # 3. LOOCV evaluation
    print("\n--- Internal LOOCV Performance (Training Set, n=17) ---")
    loocv_res, loocv_dec_acc = loocv_evaluation(df_train, scaler, params)
    print(f"{'Method':25s} {'MAE (%)':>10s} {'RMSE (%)':>10s} {'R² (LOOCV)':>12s}")
    print("-"*60)
    for method, label in zip(METHOD_COLS, METHOD_LABELS):
        r = loocv_res[method]
        print(f"{label:25s} {r['MAE']:10.2f} {r['RMSE']:10.2f} {r['R2']:12.4f}")
    print(f"\nLOOCV Decision Accuracy: {loocv_dec_acc:.1f}%")

    # 4. External validation
    print("\n--- Predefined External Validation ---")
    ext_res, ext_acc = external_validation(models, scaler, df_ext)
    for r in ext_res:
        status = "✓" if r['Correct'] else "✗"
        print(f"  {r['Drug']:20s} True: {r['True_Optimal']:25s} "
              f"Pred: {r['Predicted_Optimal']:25s} {status}")
    print(f"  External Decision Accuracy: {ext_acc:.1f}%")

    # 5. Repeated random validation
    print("\n--- Repeated Random Validation (50 iterations) ---")
    mean_acc, std_acc, all_acc = repeated_random_validation(df, params)
    print(f"  Mean Decision Accuracy: {mean_acc:.1f}% ± {std_acc:.1f}%")
    print(f"  Range: {min(all_acc):.0f}% – {max(all_acc):.0f}%")

    # 6. Final equations (full dataset)
    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(df[DESCRIPTOR_COLS].values)
    models_full = {}
    for method in METHOD_COLS:
        p = params[method]
        m = ElasticNet(alpha=p['lambda'], l1_ratio=p['l1_ratio'],
                       max_iter=10000, random_state=RANDOM_SEED, fit_intercept=True)
        m.fit(X_full, df[method].values)
        models_full[method] = m
    print_final_equations(models_full, scaler_full)

    # 7. Applicability domain
    print("\n" + "="*70)
    print("APPLICABILITY DOMAIN ASSESSMENT")
    print("="*70)
    leverages, h_star, within_ad = applicability_domain(X_full, df['Drug'].tolist())
    print(f"  Leverage threshold h* = {h_star:.3f}")
    print(f"  Max leverage = {max(leverages):.3f}")
    print(f"  All compounds within AD: {all(within_ad)}")
    print("\n  Compound leverage values:")
    for i, drug in enumerate(df['Drug']):
        flag = " ⚠️ OUTSIDE AD" if not within_ad[i] else ""
        print(f"    {drug:20s}: h = {leverages[i]:.4f}{flag}")

    # 8. Prospective prediction example
    print("\n" + "="*70)
    print("PROSPECTIVE PREDICTION EXAMPLE")
    print("="*70)
    result = predict_loading_strategy(
        LogP=2.5, MW=400, Solubility=1.0, HBD=3, HBA=6, PSA=90, Charge=0
    )
    print(f"  Input: LogP=2.5, MW=400, Sol=1.0, HBD=3, HBA=6, PSA=90, Charge=0")
    print(f"  Recommended method: {result['recommended_method']}")
    print(f"  Predicted efficiencies:")
    for method, eff in result['predicted_efficiencies'].items():
        print(f"    {method:25s}: {eff:6.1f}%")
    print(f"  Leverage: {result['leverage']} "
          f"(threshold: {result['leverage_threshold']}) "
          f"→ {'WITHIN AD' if result['within_applicability_domain'] else 'OUTSIDE AD ⚠️'}")

    print("\n" + "="*70)
    print("Framework execution complete.")
    print("="*70)


if __name__ == "__main__":
    main()
