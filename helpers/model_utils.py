import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pickle

from IPython.display import Image, display
from typing import Dict, List, Tuple
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from statsmodels.stats.outliers_influence import variance_inflation_factor 

## Allgemeine Einstellungen
#pd.options.display.float_format = '{:.6f}'.format


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))


# Setze die Schriftart und Schriftgröße global
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12




from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    max_error
)


########## Allgemeine Funktionen ##########
class GenericModelWrapper:
    def __init__(self):
        self.model_pipeline = None
        self.used_names = []
        self.used_matrix_names = []

    def set_feature_names(self, used_names, used_matrix_names=None):
        self.used_names = used_names
        self.used_matrix_names = used_matrix_names if used_matrix_names else []

    def get_prediction(self, processed_features):
        X = self.preprocess_pipeline(processed_features)
        return self.model_pipeline.predict(X)

    def preprocess_pipeline(self, processed_features):
        return processed_features.get_features_mit_namen(self.used_names, self.used_matrix_names)

    def save_model(self, model_pipeline, save_path):
        self.model_pipeline = model_pipeline
        with open(save_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_and_prepare_data(filepath, target_column, irrelevant_columns):
    """
    Lädt den Datensatz, entfernt irrelevante Spalten und teilt die Daten in Features und Zielgröße.
    """
    df = pd.read_pickle(filepath)
    features = [col for col in df.columns if col not in irrelevant_columns and col != target_column]
    return df[features], df[target_column]

def direcetion_results (dataset_name, target_name, smote_on, Verkippung, Bauteil_Temp=True, model_name='default'):
    """
    Erstellt den Pfad für die Ergebnisse.
    """
    balance_suffix = "noSMOTE" if not smote_on else ""
    tilt_suffix = "noTilt" if not Verkippung else ""
    temp_suffix = "noTemp" if not Bauteil_Temp else ""
    
    # Erstelle die Pfade
    target_dir = f"results/{target_name}/{balance_suffix}{tilt_suffix}{temp_suffix}/{dataset_name}"
    feature_importance_path = f"{target_dir}/{model_name}/feature_importance_{model_name}.csv"
    feature_importance_path_best = f"results/best_models//{target_name}/{balance_suffix}{tilt_suffix}{temp_suffix}/feature_importance_{model_name}.csv"
    target_dir_best = f"results/best_models/{target_name}/{balance_suffix}{tilt_suffix}{temp_suffix}"
    
    return target_dir, feature_importance_path, balance_suffix, feature_importance_path_best, target_dir_best, tilt_suffix, temp_suffix



def save_scores_to_csv(
    results, 
    output_dir="results", 
    file_name="model_scores.csv", 
    Verkippung=True
):
    """
    Speichert die Modell-Scores aus einem strukturierten DataFrame in eine CSV-Datei.

    Args:
        results (dict or pd.DataFrame): Ergebnisse als Dictionary oder DataFrame.
        output_dir (str): Ordner, in dem die Datei gespeichert wird.
        file_name (str): Name der CSV-Datei.
        Verkippung (bool): Gibt an, ob Verkippung berücksichtigt wurde.
    """
    # Suffix für den Dateinamen
    tilt_suffix = "_no_tilt" if not Verkippung else ""

    # Sicherstellen, dass der Output-Ordner existiert
    os.makedirs(output_dir, exist_ok=True)

    # Ergebnisse in DataFrame umwandeln, falls sie ein Dictionary sind
    if isinstance(results, dict):
        results_df = pd.DataFrame.from_dict(results, orient="index")
    else:
        results_df = results.copy()

    # Konvertiere `best_params` in Strings für die Speicherung
    if "best_params" in results_df.index:
        results_df.loc["best_params"] = results_df.loc["best_params"].apply(
            lambda x: repr(x) if isinstance(x, dict) else str(x)
        )

    # Pfad zur Datei erstellen
    output_path = os.path.join(output_dir, f"{file_name.replace('.csv', '')}{tilt_suffix}.csv")

    # Ergebnisse speichern
    results_df.to_csv(output_path, index_label="Metric")
    print(f"Scores erfolgreich gespeichert unter: {output_path}")


#############

def display_top_models_for_target(target_dir, results_df, metric, top_n=3, verkippung=True):
    """
    Zeigt die besten Modelle basierend auf der Hauptmetrik an und visualisiert die zugehörigen SHAP-Diagramme.
    
    Args:
    - target_dir (str): Verzeichnis der Zielgröße, z. B. "results/Prozessqualität".
    - results_df (pd.DataFrame): DataFrame mit den Ergebnissen der Modelle.
    - metric (str): Die Hauptmetrik, anhand derer die Modelle sortiert werden sollen (z. B. 'f1').
    - top_n (int): Anzahl der Top-Modelle, die angezeigt werden sollen.
    - verkippung (bool): Ob die Verkippung berücksichtigt wurde.
    """
    # Suffix für den Plot-Typ basierend auf Verkippung
    tilt_suffix = "_no_tilt" if not verkippung else ""

    # Sicherstellen, dass die Hauptmetrik existiert
    if metric not in results_df.index:
        print(f"Metrik '{metric}' nicht im Ergebnis-DataFrame gefunden. Verfügbare Metriken: {list(results_df.index)}")
        return

    # Top-N Modelle basierend auf der Hauptmetrik
    print(f"\nTop-{top_n} Modelle basierend auf der Metrik '{metric}':")
    top_models = results_df.loc[metric].sort_values(ascending=False).head(top_n)
    print(top_models)

    # Gewichtete Metrik automatisch finden und anzeigen (alle Metriken mit 'weighted' im Namen)
    weighted_metrics = [col for col in results_df.index if 'weighted' in col]
    top_weighted_models = set()
    for wm in weighted_metrics:
        print(f"\nTop-{top_n} Modelle basierend auf der gewichteten Metrik '{wm}':")
        top_wm_models = results_df.loc[wm].sort_values(ascending=False).head(top_n)
        print(top_wm_models)
        top_weighted_models.update(top_wm_models.index)

    # Modelle, die in den Top-Listen vorkommen, kombinieren (nur eindeutige Modelle)
    all_top_models = set(top_models.index).union(top_weighted_models)

    print(f"\nAnzeigen der SHAP-Diagramme für Modelle in den Top-{top_n} (zusammengefasst): {', '.join(all_top_models)}")

    # Anzeigen der SHAP-Diagramme ohne Verkippung
    if not verkippung:
        for model_name in all_top_models:
            model_dir = os.path.join(target_dir, model_name)
            summary_path = os.path.join(model_dir, f"shap_summary_{model_name}.svg")

            if os.path.exists(summary_path):
                print(f"\nAnzeigen des SHAP-Diagramms für {model_name}...")
                display(Image(filename=summary_path))
            else:
                print(f"SHAP-Diagramm für {model_name} nicht gefunden unter {summary_path}.")
    else:
        # Anzeigen der SHAP-Diagramme
        for model_name in all_top_models:
            model_dir = os.path.join(target_dir, model_name)
            summary_path = os.path.join(model_dir, f"shap_summary_{model_name}.svg")

            if os.path.exists(summary_path):
                print(f"\nAnzeigen des SHAP-Diagramms für {model_name}...")
                display(Image(filename=summary_path))
            else:
                print(f"SHAP-Diagramm für {model_name} nicht gefunden unter {summary_path}.")



def get_estimator_scores_balanced_with_cm(
    pipeline: Pipeline, 
    best_params: Dict, 
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> Tuple[pd.DataFrame, pd.Series, List[np.ndarray]]:
    """
    Trainiert das Modell mit den besten Hyperparametern auf dem gesamten Trainingsset
    und berechnet ausschließlich die Testwerte und die Overfitting-Differenz.

    Args:
        pipeline (Pipeline): Die Pipeline des Modells.
        best_params (Dict): Beste Hyperparameter des Modells.
        X_train (np.ndarray): Trainingsfeatures.
        y_train (np.ndarray): Trainingslabels.
        X_test (np.ndarray): Testfeatures.
        y_test (np.ndarray): Testlabels.

    Returns:
        Tuple[pd.DataFrame, pd.Series, List[np.ndarray]]: 
            - DataFrame mit Test-Scores.
            - Durchschnitts-Score (Series).
            - Liste mit einer einzelnen finalen Confusion-Matrix.
    """
    pipeline.set_params(**best_params)

    # **Finales Training mit GANZEM Trainingsdatensatz**
    pipeline.fit(X_train, y_train)

    # **Vorhersagen für Training & Test**
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # **Falls Modell Wahrscheinlichkeiten ausgibt (ROC AUC)**
    y_train_probs = pipeline.predict_proba(X_train)[:, 1] if hasattr(pipeline, "predict_proba") else None
    y_test_probs = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    # **Confusion-Matrix für das Testset**
    cm = confusion_matrix(y_test, y_test_pred)

    # **Berechnung der Test-Scores**
    test_scores = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1": f1_score(y_test, y_test_pred),
        "f1_weighted": f1_score(y_test, y_test_pred, average='weighted'),
        "roc_auc": roc_auc_score(y_test, y_test_probs) if y_test_probs is not None else None
    }

    # **Berechnung der Training-Scores (nur für Overfitting-Analyse)**
    train_scores = {
        "accuracy": accuracy_score(y_train, y_train_pred),
        "precision": precision_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred),
        "f1": f1_score(y_train, y_train_pred),
        "f1_weighted": f1_score(y_train, y_train_pred, average='weighted'),
        "roc_auc": roc_auc_score(y_train, y_train_probs) if y_train_probs is not None else None
    }

    # **Overfitting-Messung: Prozentuale Differenz zwischen Train & Test**
    percent_diffs = {
        f"percent_diff_{key}": 100 * abs(train_scores[key] - test_scores[key]) / train_scores[key]
        for key in test_scores if train_scores[key] is not None and test_scores[key] is not None
    }

    # **Ergebnisse zusammenfassen**
    scores = {**test_scores, **percent_diffs}

    # **Ergebnisse in DataFrame speichern**
    results_df = pd.DataFrame([scores])  # Einzelnes Dictionary in DataFrame umwandeln
    
    # Durchschnitts-Score berechnen
    result_describe = results_df.mean(numeric_only=True)  # Nur numerische Werte verwenden

    return results_df, result_describe, [cm]  # Rückgabe inkl. finaler Confusion-Matrix




def mean_absolute_percentage_error(y_true, y_pred):
    """Einfache MAPE-Implementierung."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    nonzero_mask = y_true != 0
    return np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100

def get_estimator_scores_regression(
    pipeline, 
    best_params, 
    X_train, 
    y_train, 
    X_test, 
    y_test
):
    """
    Trainiert das Modell mit den besten Hyperparametern und berechnet Testwerte sowie die
    Differenz zwischen Trainings- und Test-Ergebnissen (Overfitting-Messung).
    """
    pipeline.set_params(**best_params)

    # Finales Training mit dem gesamten Trainingsdatensatz
    pipeline.fit(X_train, y_train)

    # Vorhersagen für Training & Test
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Test-Scores berechnen
    test_scores = {
        "Test MSE": mean_squared_error(y_test, y_test_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),
        "Test R²": r2_score(y_test, y_test_pred) if len(np.unique(y_test)) > 1 else None,
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        # "Test MaxError": max_error(y_test, y_test_pred),
        # "Test MAPE": mean_absolute_percentage_error(y_test, y_test_pred),
    }

    # Train-Scores berechnen (für Overfitting-Analyse)
    train_scores = {
        "Train MSE": mean_squared_error(y_train, y_train_pred),
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Train R²": r2_score(y_train, y_train_pred) if len(np.unique(y_train)) > 1 else None,
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        # "Train MaxError": max_error(y_train, y_train_pred),
        # "Train MAPE": mean_absolute_percentage_error(y_train, y_train_pred),
    }

    # Liste aller Metriken, die du vergleichen willst
    metrics = ["MSE", "MAE" , "R²", "RMSE"]#, "MaxError", "MAPE"]

    diffs = {}

    for metric in metrics:
        train_key = f"Train {metric}"
        test_key = f"Test {metric}"

        train_val = train_scores[train_key]
        test_val = test_scores[test_key]

        if train_val is not None and test_val is not None:
            # # 1) Absoluter Unterschied
            # diffs[f"abs_diff_{metric}"] = abs(train_val - test_val)

            # # 2) Verhältnis (Achtung bei 0 im Nenner)
            # if train_val != 0:
            #     diffs[f"ratio_{metric}"] = test_val / train_val
            # else:
            #     diffs[f"ratio_{metric}"] = np.inf if test_val != 0 else 1.0

            # 3) Symmetrische prozentuale Differenz
            denom = (abs(train_val) + abs(test_val)) / 2.0
            if denom != 0:
                diffs[f"percent_diff_{metric}"] = 100 * abs(train_val - test_val) / denom
            else:
                diffs[f"percent_diff_{metric}"] = None
        else:
            # Falls eine Metrik None ist (z.B. R² bei konstanten Labels)
            # diffs[f"abs_diff_{metric}"] = None
            # diffs[f"ratio_{metric}"] = None
            diffs[f"percent_diff_{metric}"] = None

    # Ergebnisse zusammenfassen
    scores = {**test_scores, **diffs}
    results_df = pd.DataFrame([scores]).round(6)
    result_describe = results_df.mean(numeric_only=True)
    result_describe["best_params"] = best_params

    return results_df, result_describe



def train_and_tune_models_balanced(
    pipelines, param_grids, X_train, y_train, cv, X_test=None, y_test=None
):
    """
    Trainiert Modelle mit Hyperparameter-Tuning, speichert Cross-Validation-Scores und 
    berechnet finale Test-Scores inkl. Overfitting-Messung.

    Args:
        pipelines (dict): Dictionary mit Modellpipelines.
        param_grids (dict): Dictionary mit Hyperparameter-Rastern.
        X_train (pd.DataFrame): Trainingsfeatures.
        y_train (pd.Series): Trainingslabels.
        cv (Cross-Validator): Cross-Validation Splitter.
        X_test (pd.DataFrame, optional): Finales Testset für die Bewertung.
        y_test (pd.Series, optional): Finales Test-Labels.

    Returns:
        Tuple[dict, dict, dict]: 
            - Bestangepasste Pipelines pro Modell.
            - Ergebnisse der finalen Testbewertung inkl. Overfitting-Analyse.
            - Confusion-Matrices für die finale Testbewertung.
    """
    results = {}
    best_pipelines = {}
    confusion_matrices = {}

    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        
        # **Hyperparameter-Suche mit Cross-Validation**
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            scoring="f1",
            cv=cv,
            verbose=1,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        
        # **Beste Parameter speichern und Pipeline aktualisieren**
        best_params = grid.best_params_
        pipeline.set_params(**grid.best_params_)
        best_pipelines[name] = pipeline

        # **Berechnung der finalen Testwerte mit Overfitting-Analyse**
        print(f"Berechnung der finalen Scores für {name}...")
        test_results_df, test_result_describe, final_cm = get_estimator_scores_balanced_with_cm(
            pipeline=pipeline,
            best_params=grid.best_params_,
            X_train=X_train,  # GANZES Trainingsset für das finale Training
            y_train=y_train,
            X_test=X_test,  # Testset für die Bewertung
            y_test=y_test,
        )

        # **Speicherung der Ergebnisse**
        results[name] = test_result_describe.to_dict()  # Test-Performance & Overfitting-Messung

        # **Hier die besten Hyperparameter als zusätzliche Information speichern**
        results[name]["best_params"] = best_params

        confusion_matrices[name] = final_cm  # Confusion-Matrix des finalen Testsets

    return best_pipelines, results, confusion_matrices



def train_and_tune_models_regression(
    pipelines, param_grids, X_train, y_train, cv, X_test=None, y_test=None
):
    """
    Trainiert Modelle mit Hyperparameter-Tuning für Regression und bewertet sie auf Basis von MSE.
    Zusätzlich werden finale Testwerte und Overfitting-Differenzen berechnet.

    Args:
        pipelines (dict): Dictionary mit Modellpipelines.
        param_grids (dict): Dictionary mit Hyperparameter-Rastern.
        X_train (pd.DataFrame): Trainingsfeatures.
        y_train (pd.Series): Zielwerte für Regression.
        cv (Cross-Validator): Cross-Validation Splitter.
        X_test (pd.DataFrame, optional): Finales Testset für die Bewertung.
        y_test (pd.Series, optional): Finales Test-Labels.

    Returns:
        Tuple[dict, dict]: 
            - Bestangepasste Pipelines pro Modell.
            - Ergebnisse der Modellbewertungen inkl. finaler Testwerte.
    """
    results = {}
    best_pipelines = {}

    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        
        # **Hyperparameter-Suche mit Cross-Validation**
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            scoring="neg_mean_squared_error",  # MSE als Scoring-Metrik
            cv=cv,
            verbose=1,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        # **Beste Parameter speichern und Pipeline aktualisieren**
        pipeline.set_params(**grid.best_params_)
        best_pipelines[name] = pipeline

        # **Berechnung der finalen Testwerte mit Overfitting-Analyse**
        print(f"Berechnung der finalen Scores für {name}...")
        test_results_df, test_result_describe = get_estimator_scores_regression(
            pipeline=pipeline,
            best_params=grid.best_params_,
            X_train=X_train.values,  # GANZES Trainingsset für das finale Training
            y_train=y_train.values,
            X_test=X_test.values,  # Testset für die Bewertung
            y_test=y_test.values,
        )

        # **Speicherung der Ergebnisse**
        results[name] = test_result_describe.to_dict()  # Test-Performance & Overfitting-Messung

    return best_pipelines, results

def compare_results_across_datasets(results_dfs, metric, top_n=None):
    """
    Vergleicht die Ergebnisse (Top-N) verschiedener Datasets für eine bestimmte Metrik und gibt die beste Kombination aus.
    
    Args:
        results_dfs (dict): Dictionary mit DataFrames der Ergebnisse (Key: Dataset-Name, Value: DataFrame).
        metric (str): Die Metrik, anhand derer verglichen werden soll (z. B. 'f1').
        top_n (int): Anzahl der Top-Modelle, die verglichen werden sollen.
    
    Returns:
        Tuple[pd.DataFrame, Tuple[str, str, float]]: 
            - Pivot-Tabelle mit den besten Ergebnissen pro Dataset und Metrik.
            - Beste Kombination aus Dataset, Modell und Score.
    """
    comparisons = []

    # Reihenfolge der Datasets festlegen
    column_order = ["Process Features", "Aggregated Features", "New Features"]

    # Set mit allen Modellnamen aus allen Datasets erstellen
    all_models = set()
    for results_df in results_dfs.values():
        if metric in results_df.index:
            all_models.update(results_df.columns)

    best_score = -float("inf")
    best_combination = (None, None, None)  # (Dataset, Model, Score)

    for dataset_name, results_df in results_dfs.items():
        if metric not in results_df.index:
            print(f"Metrik '{metric}' nicht im Dataset '{dataset_name}' gefunden.")
            continue
        
        # Dynamisches top_n, wenn es nicht gesetzt ist
        if top_n is None:
            top_n = len(results_df.columns)
        
        # Top-N Modelle für das aktuelle Dataset
        top_models = results_df.loc[metric].sort_values(ascending=False).head(top_n)
        for model in all_models:
            score = top_models.get(model, np.nan)  # Wert für das Modell, falls es unter den Top-N ist
            comparisons.append({
                "Dataset": dataset_name,
                "Model": model,
                "Score": score,
            })

            # Beste Kombination aktualisieren
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_combination = (dataset_name, model, score)

    # DataFrame aus den Vergleichsdaten erstellen
    comparison_df = pd.DataFrame(comparisons)

    # Pivot-Tabelle erstellen: Modelle in Zeilen, Datasets in Spalten
    comparison_pivot = comparison_df.pivot(index="Model", columns="Dataset", values="Score")

    # Spalten in der gewünschten Reihenfolge anordnen
    comparison_pivot = comparison_pivot[column_order]

    # Fehlende Werte durch NaN ersetzen und sicherstellen, dass alles numerisch bleibt
    comparison_pivot = comparison_pivot.apply(pd.to_numeric, errors="coerce").fillna(np.nan)

    # Tabelle sortieren nach der ersten Dataset-Kolonne
    if not comparison_pivot.empty:
        comparison_pivot = comparison_pivot.sort_values(by=column_order[0], ascending=False)

    return comparison_pivot, best_combination

def compare_results_regression(results_dfs, metric, top_n=None):
    """
    Vergleicht die Ergebnisse (Top-N) verschiedener Datasets für eine bestimmte Metrik (Regression)
    und gibt die beste Kombination basierend auf dem kleinsten Wert aus.

    Args:
        results_dfs (dict): Dictionary mit DataFrames der Ergebnisse (Key: Dataset-Name, Value: DataFrame).
        metric (str): Die Metrik, anhand derer verglichen werden soll (z. B. 'MAE', 'MSE').
        top_n (int): Anzahl der Top-Modelle, die verglichen werden sollen.

    Returns:
        Tuple[pd.DataFrame, Tuple[str, str, float]]: 
            - Pivot-Tabelle mit den besten Ergebnissen pro Dataset und Metrik.
            - Beste Kombination aus Dataset, Modell und Score.
    """
    comparisons = []

    # Reihenfolge der Datasets festlegen
    column_order = ["Process Features", "Aggregated Features", "New Features"]

    # Set mit allen Modellnamen aus allen Datasets erstellen
    all_models = set()
    for results_df in results_dfs.values():
        if metric in results_df.index:
            all_models.update(results_df.columns)

    best_score = float("inf")  # Da bei Regression kleinere Werte besser sind
    best_combination = (None, None, None)  # (Dataset, Model, Score)

    for dataset_name, results_df in results_dfs.items():
        if metric not in results_df.index:
            print(f"Metrik '{metric}' nicht im Dataset '{dataset_name}' gefunden.")
            continue
        
        # Dynamisches top_n, wenn es nicht gesetzt ist
        if top_n is None:
            top_n = len(results_df.columns)
        
        # Top-N Modelle für das aktuelle Dataset (niedrigste Werte zuerst)
        top_models = results_df.loc[metric].sort_values(ascending=True).head(top_n)
        for model in all_models:
            score = top_models.get(model, np.nan)  # Wert für das Modell, falls es unter den Top-N ist
            comparisons.append({
                "Dataset": dataset_name,
                "Model": model,
                "Score": score
            })

            # Beste Kombination aktualisieren
            if not np.isnan(score) and score < best_score:  # Nur den kleinsten Score berücksichtigen
                best_score = score
                best_combination = (dataset_name, model, score)

    # DataFrame aus den Vergleichsdaten erstellen
    comparison_df = pd.DataFrame(comparisons)

    # Pivot-Tabelle erstellen: Modelle in Zeilen, Datasets in Spalten
    comparison_pivot = comparison_df.pivot(index="Model", columns="Dataset", values="Score")

    # Spalten in der gewünschten Reihenfolge anordnen
    comparison_pivot = comparison_pivot[column_order]

    # Fehlende Werte durch NaN ersetzen und sicherstellen, dass alles numerisch bleibt
    comparison_pivot = comparison_pivot.apply(pd.to_numeric, errors="coerce").fillna(np.nan)

    # Tabelle sortieren nach der ersten Dataset-Kolonne (aufsteigend, da kleiner besser ist)
    if not comparison_pivot.empty:
        comparison_pivot = comparison_pivot.sort_values(by=column_order[0], ascending=True)

    return comparison_pivot, best_combination


def display_average_confusion_matrices(
    confusion_matrices, 
    label_mapping_path, 
    target_column, 
    target_dir, 
    save_plots=True, 
    normalize=False, 
    cmap=plt.cm.Blues, 
    Verkippung=True
):
    """
    Zeigt die durchschnittliche Confusion-Matrix für jedes Modell an und speichert sie optional.

    Args:
        confusion_matrices (dict): Dictionary mit Confusion-Matrices pro Modell.
        label_mapping_path (str): Pfad zur JSON-Datei mit den Label-Mappings.
        target_column (str): Zielspalte, für die die Confusion-Matrix generiert wird.
        target_dir (str): Hauptzielverzeichnis, um die Plots zu speichern.
        save_plots (bool): Ob die Plots gespeichert werden sollen. Default ist True.
        normalize (bool): Ob die Confusion-Matrix normalisiert werden soll. Default ist False.
        cmap: Colormap für die Anzeige. Default ist plt.cm.Blues.
        Verkippung (bool): Gibt an, ob die Verkippungsfeatures berücksichtigt werden. Default ist True.
    """
    # Label-Mapping laden
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    
    # Labels für die Zielspalte aus dem Mapping extrahieren
    if target_column in label_mapping:
        label_dict = label_mapping[target_column]
        # Labels für die binäre Klassifikation erstellen
        binary_labels = {0: "" + ", ".join([k for k, v in label_dict.items() if v == 0]),
                         1: ", ".join([k for k, v in label_dict.items() if v == 1])}
        display_labels = [binary_labels[0], binary_labels[1]]
    else:
        raise ValueError(f"Target column '{target_column}' nicht im Label-Mapping gefunden.")
    
    # Zusatz für den Dateinamen, wenn Verkippung deaktiviert ist
    tilt_suffix = "_no_tilt" if not Verkippung else ""

    for model_name, cms in confusion_matrices.items():
        # Durchschnittliche Confusion-Matrix berechnen
        avg_cm = np.mean(cms, axis=0)
        if normalize:
            avg_cm = avg_cm / avg_cm.sum(axis=1, keepdims=True)

        # Warnung, wenn die Anzahl der Klassen nicht mit den Labels übereinstimmt
        if avg_cm.shape[0] != len(display_labels):
            print(f"Warnung: Anzahl der Klassen ({avg_cm.shape[0]}) passt nicht zu den Labels ({len(display_labels)}).")
            display_labels = [f"Klasse {i}" for i in range(avg_cm.shape[0])]

        print(f"Durchschnittliche Confusion-Matrix für {model_name}:")
        disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm, display_labels=display_labels)
        disp.plot(cmap=cmap)
        plt.title(f"{model_name} - Durchschnittliche Confusion-Matrix")

        # Speichern der Confusion-Matrix
        if save_plots:
            # Struktur: {target_dir}/{model_name}/
            model_output_dir = os.path.join(target_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            # Datei speichern mit Suffix
            file_path = os.path.join(model_output_dir, f"average_confusion_matrix.svg")
            plt.savefig(file_path,format="svg", dpi=300, bbox_inches="tight")
            print(f"Confusion-Matrix für {model_name} gespeichert unter: {file_path}")
        
        plt.show()

def display_final_confusion_matrices(
    confusion_matrices, 
    label_mapping_path, 
    target_column, 
    target_dir, 
    save_plots=True, 
    normalize=True, 
    cmap=plt.cm.Blues, 
    Verkippung=True
):
    """
    Zeigt die finalen Confusion-Matrices für jedes Modell auf den Testdaten an und speichert sie optional.

    Args:
        confusion_matrices (dict): Dictionary mit den finalen Confusion-Matrices pro Modell.
        label_mapping_path (str): Pfad zur JSON-Datei mit den Label-Mappings.
        target_column (str): Zielspalte, für die die Confusion-Matrix generiert wird.
        target_dir (str): Hauptzielverzeichnis, um die Plots zu speichern.
        save_plots (bool): Ob die Plots gespeichert werden sollen (Default: True).
        normalize (bool): Ob die Confusion-Matrix normalisiert werden soll (Default: False).
        cmap: Colormap für die Anzeige (Default: plt.cm.Blues).
        Verkippung (bool): Gibt an, ob die Verkippungsfeatures berücksichtigt werden (Default: True).
    """
    # **Label-Mapping laden**
    with open(label_mapping_path, "r") as f:
        label_mapping = json.load(f)
    
    # **Labels für die Zielspalte extrahieren**
    if target_column in label_mapping:
        label_dict = label_mapping[target_column]
        binary_labels = {
            0: ", ".join([k for k, v in label_dict.items() if v == 0]),
            1: ", ".join([k for k, v in label_dict.items() if v == 1])
        }
        display_labels = [binary_labels[0], binary_labels[1]]
    else:
        raise ValueError(f"Target column '{target_column}' nicht im Label-Mapping gefunden.")

    # **Suffix für den Dateinamen setzen**
    tilt_suffix = "_no_tilt" if not Verkippung else ""

    for model_name, cm in confusion_matrices.items():
        cm = cm[0]  # Liste mit einer einzigen Matrix, daher `cm[0]` verwenden

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)  # Normierung der Matrix

        # **Anzeige der Confusion-Matrix**
        print(f"Confusion-Matrix für {model_name} (Testdaten):")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap=cmap)
        plt.title(f"{model_name} - Confusion-Matrix (Testdatensatz)")

        # **Speicherung der Confusion-Matrix als Bild**
        if save_plots:
            model_output_dir = os.path.join(target_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            file_path = os.path.join(model_output_dir, f"final_confusion_matrix.svg")
            try:
                plt.savefig(file_path, format="svg", dpi=300, bbox_inches="tight")
                print(f"Confusion-Matrix für {model_name} gespeichert unter: {file_path}")
            except Exception as e:
                print(f"Fehler beim Speichern der Confusion-Matrix für {model_name}: {e}")
        
        plt.show()


def analyze_model_performance(
    pipelines, 
    param_grids, 
    X, 
    y, 
    scoring, 
    cv, 
    save_plots=False, 
    output_dir=None, 
    Verkippung=True
):
    """
    Analysiert die Modellleistung basierend auf Validation und Learning Curves.
    
    Args:
        pipelines (dict): Modellpipelines.
        param_grids (dict): Hyperparameter-Bereiche.
        X (pd.DataFrame): Trainingsfeatures.
        y (pd.Series): Trainingslabels.
        scoring (str): Bewertungsmetrik.
        cv (Cross-Validator): Cross-Validation Splitter.
        save_plots (bool): Ob die Plots gespeichert werden sollen.
        output_dir (str): Pfad zum Speichern der Plots.
        Verkippung (bool): Gibt an, ob Verkippungsfeatures berücksichtigt werden.
    """
    # Suffix für Verkippung
    tilt_suffix = "_no_tilt" if not Verkippung else ""
    
    for model_name, pipeline in pipelines.items():
        print(f"Analysiere Modell: {model_name}...")

        # Dynamische Auswahl des ersten Hyperparameters aus param_grids
        if model_name in param_grids and param_grids[model_name]:
            param_name = list(param_grids[model_name].keys())[0]  # Erster Hyperparameter
            param_range = param_grids[model_name][param_name]
        else:
            print(f"Keine Hyperparameter für {model_name} gefunden.")
            continue

        # Überprüfen, ob param_range gültig ist
        if not param_range or any(p is None for p in param_range):
            print(f"Ungültiger Parameterbereich für {model_name}: {param_range}")
            continue

        # Validation Curve
        print(f"Erstelle Validation Curve für {model_name}...")
        train_scores, test_scores = validation_curve(
            pipeline, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring, n_jobs=-1
        )
        train_scores_mean = train_scores.mean(axis=1)
        test_scores_mean = test_scores.mean(axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(param_range, train_scores_mean, label="Trainingsscore", marker='o', linestyle='--')
        plt.plot(param_range, test_scores_mean, label="Testscores im Training", marker='s', linestyle='-')
        plt.title(f"Validation Curve: {model_name}")
        plt.xlabel(param_name.replace('clf__', '').replace('_', ' ').capitalize())
        plt.ylabel(f"{scoring}")
        plt.legend(loc="best")
        plt.grid()
        plt.tight_layout()

        if save_plots:
            # Zielverzeichnis erstellen
            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            plot_path = os.path.join(model_output_dir, f"validation_curve_{model_name}_{param_name}.svg")
            plt.savefig(plot_path,format="svg", dpi=300, bbox_inches="tight")
            print(f"Validation Curve gespeichert unter: {plot_path}")
        
        plt.show()

        # Learning Curve
        print(f"Erstelle Learning Curve für {model_name}...")
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline, X, y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
        )
        train_scores_mean = train_scores.mean(axis=1)
        test_scores_mean = test_scores.mean(axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_scores_mean, label="Trainingsscore", marker='o', linestyle='--')
        plt.plot(train_sizes, test_scores_mean, label="Testscores im Training", marker='s', linestyle='-')
        plt.title(f"Learning Curve: {model_name}")
        plt.xlabel("Trainingsgröße")
        plt.ylabel(f"{scoring}")
        plt.legend(loc="best")
        plt.grid()
        plt.tight_layout()

        if save_plots:
            # Zielverzeichnis erstellen
            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            plot_path = os.path.join(model_output_dir, f"learning_curve_{model_name}.svg")
            plt.savefig(plot_path,format="svg", dpi=300, bbox_inches="tight")
            print(f"Learning Curve gespeichert unter: {plot_path}")
        
        plt.show()


def plot_scores_and_percent_diff(
    results_dfs, 
    target_name, 
    metric="f1", 
    output_path=None,
    y_lim=(0, 1)
):
    """
    Erstellt ein Balkendiagramm für eine gewählte Metrik über verschiedene Feature-Sätze und ergänzt
    eine Liniengrafik für die prozentuale Abweichung (percent_diff). Optional wird das Diagramm gespeichert.
    
    Args:
        results_dfs (dict): Dictionary mit den DataFrames für verschiedene Feature-Sätze.
        target_name (str): Name des Zielwerts (Target), der im Diagrammtitel verwendet wird.
        metric (str): Name der Fehlermetrik (z. B. 'f1', 'f1_weighted', 'mse').
        output_path (str): Pfad, unter dem das Diagramm gespeichert werden soll (optional).
        y_lim (tuple): Y-Achsenbereich für die Metrik (z. B. (0, 1) für F1).
    """
    # Modelle aus einem der DataFrames extrahieren
    models = results_dfs["Process Features"].columns

    # Metrik-Werte abrufen
    metric_process = results_dfs["Process Features"].loc[metric].values
    metric_aggregated = results_dfs["Aggregated Features"].loc[metric].values
    metric_new = results_dfs["New Features"].loc[metric].values

    # Prozentuale Differenz
    percent_key = f"percent_diff_{metric}"
    if metric == "Test MSE":
        percent_key = "percent_diff_MSE"
    if metric == "Test MAE":
        percent_key = "percent_diff_MAE"
    percent_diff_process = results_dfs["Process Features"].loc[percent_key].values
    percent_diff_aggregated = results_dfs["Aggregated Features"].loc[percent_key].values
    percent_diff_new = results_dfs["New Features"].loc[percent_key].values

    # Breite der Balken und Positionen auf der X-Achse
    bar_width = 0.2
    x = np.arange(len(models))

    # Erstellen des Balkendiagramms für die gewählte Metrik
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(x - bar_width, metric_process, width=bar_width, label="Prozess Features", color='firebrick')
    ax1.bar(x, metric_aggregated, width=bar_width, label="Aggregierte Features", color='goldenrod')
    ax1.bar(x + bar_width, metric_new, width=bar_width, label="Neue Features", color='forestgreen')
    
    # Achsenbeschriftung und Titel
    ax1.set_xlabel("Modelle")
    ax1.set_ylabel(metric, color='black')
    ax1.set_title(f"Vergleich der Modellleistung für {target_name} mit unterschiedlichen Feature-Datensätzen")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend(loc="upper left")

    ax1.set_ylim(y_lim)  # Skalierung für Metrik

    # Zweite Y-Achse für percent_diff
    ax2 = ax1.twinx()
    ax2.plot(x, percent_diff_process, marker='o', linestyle='-', color='maroon', label="%-Diff (Prozess)")
    ax2.plot(x, percent_diff_aggregated, marker='s', linestyle='-', color='darkgoldenrod', label="%-Diff (Aggregiert)")
    ax2.plot(x, percent_diff_new, marker='^', linestyle='-', color='darkgreen', label="%-Diff (Neu)")

    max_percent_diff = max(
        np.max(percent_diff_process),
        np.max(percent_diff_aggregated),
        np.max(percent_diff_new)
    )
    ax2.set_ylim(0, max_percent_diff * 1.2)
    ax2.set_ylabel("%-Abweichung Train vs. Test", color='black')
    ax2.legend(loc="upper right")

    # Diagramm speichern
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path,format="svg", dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

def plot_f1_scores_and_percent_diff(results_dfs, target_name, output_path=None,y_lim=(0,1)):
    """
    Erstellt ein Balkendiagramm für die F1-Scores über verschiedene Feature-Sätze und ergänzt
    eine Liniengrafik für die prozentuale Abweichung (percent_diff). Optional wird das Diagramm gespeichert.
    
    Args:
        results_dfs (dict): Dictionary mit den DataFrames für verschiedene Feature-Sätze.
        target_name (str): Name des Zielwerts (Target), der im Diagrammtitel verwendet wird.
        output_path (str): Pfad, unter dem das Diagramm gespeichert werden soll (optional).
    """

    # Modelle aus einem der DataFrames extrahieren
    models = results_dfs["Process Features"].columns

    # F1-Scores für verschiedene Feature-Sätze abrufen
    f1_scores_process = results_dfs["Process Features"].loc["f1"].values
    f1_scores_aggregated = results_dfs["Aggregated Features"].loc["f1"].values
    f1_scores_new = results_dfs["New Features"].loc["f1"].values

    # Prozentuale Abweichung (percent_diff_f1) abrufen
    percent_diff_process = results_dfs["Process Features"].loc["percent_diff_f1"].values
    percent_diff_aggregated = results_dfs["Aggregated Features"].loc["percent_diff_f1"].values
    percent_diff_new = results_dfs["New Features"].loc["percent_diff_f1"].values

    # Breite der Balken und Positionen auf der X-Achse
    bar_width = 0.2
    x = np.arange(len(models))

    # Erstellen des Balkendiagramms für F1-Scores
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(x - bar_width, f1_scores_process, width=bar_width, label="Prozess Features", color='firebrick')
    ax1.bar(x, f1_scores_aggregated, width=bar_width, label="Aggregierte Features", color='goldenrod')
    ax1.bar(x + bar_width, f1_scores_new, width=bar_width, label="Neue Features", color='forestgreen')
    
    # Achsenbeschriftung und Titel
    ax1.set_xlabel("Modelle")
    ax1.set_ylabel("F1-Score", color='black')
    ax1.set_title("Vergleich der Modellleistung für die {} mit unterschiedlichen Feature-Datensätzen".format(target_name))
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend(loc="upper left")

    # Skalierung der linken Y-Achse anpassen
    ax1.set_ylim(y_lim)  # F1-Scores zwischen 0 und 1

    # Zweite Y-Achse für percent_diff
    ax2 = ax1.twinx()
    ax2.plot(x, percent_diff_process, marker='o', linestyle='-', color='maroon', label="Prozentuale Differenz (Prozess)")
    ax2.plot(x, percent_diff_aggregated, marker='s', linestyle='-', color='darkgoldenrod', label="Prozentuale Differenz (Aggregiert)")
    ax2.plot(x, percent_diff_new, marker='^', linestyle='-', color='darkgreen', label="Prozentuale Differenz (Neu)")

    # Skalierung der rechten Y-Achse anpassen
    max_percent_diff = max(
        np.max(percent_diff_process),
        np.max(percent_diff_aggregated),
        np.max(percent_diff_new)
    )
    ax2.set_ylim(0, max_percent_diff * 1.2)  # Dynamische Skalierung mit etwas Puffer

    ax2.set_ylabel("%-Abweichung zwischen Train- & Testdatensatz", color='black')
    ax2.legend(loc="upper right")

    # Diagramm speichern, falls ein Pfad angegeben ist
    if output_path:
        # Verzeichnis erstellen, falls es nicht existiert
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Speichert das Diagramm mit hoher Auflösung

    # Diagramm anzeigen
    plt.tight_layout()
    plt.show()

def calculate_vif(X: pd.DataFrame):
    """
    Berechnet den Variance Inflation Factor (VIF) für jedes Feature.
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def remove_high_vif_features(X: pd.DataFrame, vif_threshold: float):
    """
    Entfernt Features mit einem Variance Inflation Factor (VIF) über einem bestimmten Schwellenwert.
    """
    vif_data = calculate_vif(X)
    to_drop = set(vif_data[vif_data["VIF"] > vif_threshold]["Feature"])
    X_filtered = X.drop(columns=to_drop)

    return X_filtered, to_drop, vif_data

def save_feature_selection_summary(dataset_name, target_name, results_dict, num_features_dict, output_dir, is_regression=False):
    """
    Erstellt und speichert eine formatierte Tabelle mit den Ergebnissen der Feature Selection.
    """
    if not is_regression:
        summary_df = pd.DataFrame({
            "{}".format(dataset_name): [
                num_features_dict["original"],
                results_dict["original"]["f1"],
                results_dict["original"]["percent_diff_f1"]
            ],
            "SHAP Values 80% Selektion": [
                num_features_dict["shap"],
                results_dict["shap"]["f1"],
                results_dict["shap"]["percent_diff_f1"]
            ],
            "Korrelationsanalyse (Spearman)": [
                num_features_dict["correlation"],
                results_dict["correlation"]["f1"],
                results_dict["correlation"]["percent_diff_f1"]
            ],
            "Multikollinearitätsanalyse (VIF)": [
                num_features_dict["vif"],
                results_dict["vif"]["f1"],
                results_dict["vif"]["percent_diff_f1"]
            ]
        }, index=["Anzahl Feature", "F1-Score", "Prozentuale Abweichung Train-Test"])
    else:
        summary_df = pd.DataFrame({
            "{}".format(dataset_name): [
                num_features_dict["original"],
                results_dict["original"]["Test MSE"],
                results_dict["original"]["percent_diff_MSE"]
            ],
            "SHAP Values 80% Selektion": [
                num_features_dict["shap"],
                results_dict["shap"]["Test MSE"],
                results_dict["shap"]["percent_diff_MSE"]
            ],
            "Korrelationsanalyse (Spearman)": [
                num_features_dict["correlation"],
                results_dict["correlation"]["Test MSE"],
                results_dict["correlation"]["percent_diff_MSE"]
            ],
            "Multikollinearitätsanalyse (VIF)": [
                num_features_dict["vif"],
                results_dict["vif"]["Test MSE"],
                results_dict["vif"]["percent_diff_MSE"]
            ]
        }, index=["Anzahl Feature", "Test MSE", "Prozentuale Abweichung Train-Test"])

    result_file = os.path.join(output_dir, f"summary_feature_selection_{target_name}.csv")
    summary_df.to_csv(result_file, index=True, sep=";")

    print(f"Ergebnistabelle gespeichert unter: {result_file}")
    return summary_df

def save_selected_features_with_values(features_dict, output_dir, target_name):
    """
    Speichert die ausgewählten Features mit ihren zugehörigen Werten (SHAP, Korrelation, VIF).
    """
    features_df = pd.concat(features_dict, axis=1)

    feature_file = os.path.join(output_dir, f"selected_features_{target_name}.csv")
    features_df.to_csv(feature_file, index=True)

    print(f"Feature-Werte gespeichert unter: {feature_file}")
    return features_df

def save_feature_selection_results(dataset_name, target_name, results, num_features, output_dir, is_regression=False):
    """
    Speichert die Ergebnisse der Feature Selection in einer CSV-Datei.
    
    :param dataset_name: Name des verwendeten Datensatzes.
    :param target_name: Zielvariable.
    :param results: Ergebnisse aus dem Modelltraining.
    :param num_features: Anzahl der Features nach jedem Schritt der Selektion.
    :param output_dir: Verzeichnis für die Speicherung.
    """
    if is_regression:
        summary_data = {
        "Ausgewählter Datensatz": dataset_name,
        "SHAP 80% Selektion (Anzahl Features)": num_features["shap"],
        "Korrelationsanalyse (Spearman) (Anzahl Features)": num_features["correlation"],
        "Multikollinearitätsanalyse (VIF) (Anzahl Features)": num_features["vif"],
        "MSE": results.get("Test MSE", results.get("N/A")),
        "Prozentuale Abweichung Train-Test": results.get("percent_diff", results.get("Percent Difference", "N/A"))
    }
    else:
        summary_data = {
            "Ausgewählter Datensatz": dataset_name,
            "SHAP 80% Selektion (Anzahl Features)": num_features["shap"],
            "Korrelationsanalyse (Spearman) (Anzahl Features)": num_features["correlation"],
            "Multikollinearitätsanalyse (VIF) (Anzahl Features)": num_features["vif"],
            "F1-Score": results.get("f1", results.get("N/A")),
            "Prozentuale Abweichung Train-Test": results.get("percent_diff", results.get("Percent Difference", "N/A"))
        }

    # Erstellen des Ergebnis-DataFrames
    summary_df = pd.DataFrame([summary_data])

    # Speicherort
    result_file = os.path.join(output_dir, f"feature_selection_summary_{target_name}.csv")
    summary_df.to_csv(result_file, index=False)
    
    print(f"Ergebnistabelle gespeichert unter: {result_file}")
    return summary_df


def plot_actual_vs_predicted(y_true, y_pred, model_name="Modell"):
    """
    Erstellt einen Scatter-Plot für tatsächliche vs. vorhergesagte Werte.
    
    Args:
        y_true (array-like): Tatsächliche Werte.
        y_pred (array-like): Vorhergesagte Werte.
        model_name (str): Name des Modells für den Titel.
    """
    plt.figure(figsize=(8, 6))
    
    # Streudiagramm
    plt.scatter(y_pred, y_true, color='red', alpha=0.6, label="Datenpunkte")
    
    # Diagonale Linie (perfekte Vorhersage)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfekte Vorhersage")
    
    # Achsenbeschriftung
    plt.xlabel("Vorhergesagte Probenhöhe (mm)")
    plt.ylabel("Tatsächliche Probenhöhe (mm)")
    plt.title(f"Tatsächliche vs. vorhergesagte Probenhöhe ({model_name})")
    #plt.legend()
    plt.grid(True)
    
    plt.show()

def filter_and_retrain_with_vif(
    model_name: str,
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_importance_path: str,
    param_grids: dict,
    cv,
    scoring="f1",
    pareto_split=0.8, 
    correlation_threshold=0.9,  
    vif_threshold=20.0,  
    output_dir="results",
    dataset_name="default_dataset",
    target_name="default_target",
    verkippung=True,
    is_regression=False,
    label_mapping_path=None,    # Neuer Parameter für Confusion-Matrix
    target_column=None          # Neuer Parameter für Confusion-Matrix
):
    """
    Führt eine mehrstufige Feature Selection durch:
    1. SHAP-Werte nutzen (Pareto 80%)
    2. Korrelationsanalyse (Spearman)
    3. Multikollinearitätsanalyse (VIF)
    4. Trainieren mit reduzierten Features und direkte Speicherung der berechneten Scores
    """
    try:
        feature_importances = pd.read_csv(feature_importance_path, index_col=0)
        print(f"Feature-Wichtigkeiten aus {feature_importance_path} geladen.")
    except FileNotFoundError:
        print(f"Die Datei {feature_importance_path} wurde nicht gefunden.")
        return None, None

        ## **1. SHAP 80% SELEKTION**
    feature_importances = feature_importances.sort_values(by="SHAP Value", ascending=False)
    total_importance = feature_importances["SHAP Value"].sum()
    cumulative_importance = feature_importances["SHAP Value"].cumsum()
    selected_features_shap = feature_importances[cumulative_importance <= total_importance * pareto_split].index.tolist()
    if not selected_features_shap:
        print("⚠️ Kein Feature erfüllt den Pareto-Split, Auswahl auf erstes Feature beschränkt.")
        selected_features_shap = [X_train.columns[0]]
    # Nur vorhandene Features berücksichtigen
    selected_features_shap = [feat for feat in selected_features_shap if feat in X_train.columns]
    X_train_reduced = X_train[selected_features_shap]
    X_test_reduced = X_test[selected_features_shap]

    ## **2. KORRELATIONSANALYSE (Spearman)**
    corr_matrix = X_train_reduced.corr(method="spearman").abs()
    to_remove_corr = set()
    correlated_pairs = []
    for col in corr_matrix.columns:
        for row in corr_matrix.index:
            if col != row and corr_matrix.loc[row, col] > correlation_threshold:
                weaker_feature = row if feature_importances.loc[row, "SHAP Value"] < feature_importances.loc[col, "SHAP Value"] else col
                to_remove_corr.add(weaker_feature)
                correlated_pairs.append((row, col, corr_matrix.loc[row, col]))
    X_train_corr_filtered = X_train_reduced.drop(columns=to_remove_corr, errors="ignore")
    X_test_corr_filtered = X_test_reduced.drop(columns=to_remove_corr, errors="ignore")

    ## **3. MULTIKOLLINEARITÄTSANALYSE (VIF)**
    # Falls nach der Korrelationsanalyse keine Spalten übrig sind, überspringe diesen Schritt.
    if X_train_corr_filtered.shape[1] == 0:
        print("⚠️ Keine Features nach der Korrelationsanalyse übrig. VIF-Analyse übersprungen.")
        X_train_vif_filtered = X_train_corr_filtered.copy()
        X_test_vif_filtered = X_test_corr_filtered.copy()
        removed_vif_features = set()
        vif_data_before = pd.DataFrame()
    else:
        X_train_vif_filtered, removed_vif_features, vif_data_before = remove_high_vif_features(
            X_train_corr_filtered, vif_threshold
        )
        X_test_vif_filtered = X_test_corr_filtered.drop(columns=removed_vif_features, errors="ignore")

    # **AUSGABEN**
    print(f"Originale Trainingsdaten: {X_train.shape}, Testdaten: {X_test.shape}")
    print(f"Reduzierte Trainingsdaten nach SHAP: {X_train_reduced.shape}, Testdaten: {X_test_reduced.shape}")
    print(f"Reduzierte Trainingsdaten nach Korrelationsanalyse: {X_train_corr_filtered.shape}, Testdaten: {X_test_corr_filtered.shape}")
    print(f"Reduzierte Trainingsdaten nach VIF-Analyse: {X_train_vif_filtered.shape}, Testdaten: {X_test_vif_filtered.shape}")

    print("\nEntfernte hoch korrelierte Features:", to_remove_corr)
    print("\nEntfernte Features wegen hohem VIF (> {vif_threshold}):", removed_vif_features)

    print("\nKorrelierte Feature-Paare (Korrelation > {correlation_threshold}):")
    for f1, f2, corr in correlated_pairs:
        print(f"{f1} <--> {f2} (Korrelation: {corr:.2f})")

    print("\nVIF-Werte vor der Reduktion:")
    print(vif_data_before.sort_values(by="VIF", ascending=False))

    # **Visualisierungen**
    if not X_train_reduced.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(X_train_reduced.corr(), cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5)
        plt.title("Korrelationsmatrix der ausgewählten Features (vor Entfernung)")
        
        # Speicherpfad definieren
        plot_file = os.path.join(output_dir, f"correlation_plot_{target_name}.svg")
        plt.savefig(plot_file,format="svg", dpi=300, bbox_inches="tight")
        print(f"Korrelations-Plot gespeichert unter: {plot_file}")
        plt.show()

    if not vif_data_before.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(y=vif_data_before["Feature"], x=vif_data_before["VIF"])#, palette="coolwarm")

        # VIF-Schwellenwert als gestrichelte Linie einfügen
        plt.axvline(x=vif_threshold, color='r', linestyle='--', linewidth=2, label=f"VIF-Schwellenwert ({vif_threshold})")

        plt.xlabel("Variance Inflation Factor (VIF)")
        plt.ylabel("Feature")
        plt.title("VIF-Werte vor der Reduktion")
        plt.legend(loc="upper right")  # Legende an sinnvolle Stelle setzen
        plt.grid(True, linestyle=":", linewidth=0.5)  # Leichte Rasterung für bessere Lesbarkeit
        # Speicherpfad definieren
        plot_file = os.path.join(output_dir, f"vif_plot_{target_name}.svg")
        plt.savefig(plot_file,format="svg", dpi=300, bbox_inches="tight")
        print(f"VIF-Plot gespeichert unter: {plot_file}")
        plt.show()

    ## **Speicherung der Anzahl der Features**
    num_features = {
        "original": X_train.shape[1],
        "shap": len(X_train_reduced.columns),
        "correlation": len(X_train_corr_filtered.columns),
        "vif": len(X_train_vif_filtered.columns)
    }

    ## **Stufenweise Speicherung der Feature-Werte in einer CSV**
    # Original: Alle SHAP-Werte aus der geladenen Tabelle
    original_series = feature_importances["SHAP Value"]

    # SHAP: Nur die Werte der per Pareto-Split ausgewählten Features
    shap_series = feature_importances.loc[selected_features_shap, "SHAP Value"]

    # Korrelationsanalyse: Berechne für die SHAP-reduzierten Daten den maximalen (nicht-diagonalen) Korrelationswert pro Feature
    corr_matrix_shap = X_train_reduced.corr(method="spearman").abs()
    corr_series = corr_matrix_shap.apply(lambda x: x[x < 1].max(), axis=0)
    # Nur die Features, die nach der Korrelationsanalyse übrig geblieben sind
    corr_series_filtered = corr_series.loc[X_train_corr_filtered.columns]

    # VIF: VIF-Werte aus der Berechnung, beschränkt auf die finalen Features
    vif_series = vif_data_before.set_index("Feature")["VIF"].loc[X_train_vif_filtered.columns]

    # Da die Original-Spalte alle Features enthält, nutzen wir deren Index als Basis
    all_features = original_series.index
    original_series = original_series.reindex(all_features)
    shap_series = shap_series.reindex(all_features)
    corr_series_filtered = corr_series_filtered.reindex(all_features)
    vif_series = vif_series.reindex(all_features)

    features_dict = {
        "Original": original_series,
        f"SHAP – {pareto_split *100}%": shap_series,
        f"Korrelationsanalyse – {correlation_threshold}": corr_series_filtered,
        f"VIF – {vif_threshold}": vif_series
    }

    # Speichere die stufenweise Feature-Tabelle
    features_df = save_selected_features_with_values(features_dict, output_dir, target_name)

    ## **Speicherung der korrelierten Feature-Paare**
    if correlated_pairs:
        corr_pairs_df = pd.DataFrame(correlated_pairs, columns=["Feature 1", "Feature 2", "Korrelation"])
        corr_pairs_file = os.path.join(output_dir, f"correlated_feature_pairs_{target_name}.csv")
        corr_pairs_df.to_csv(corr_pairs_file, index=False)
        print(f"Korrelierte Feature-Paare gespeichert unter: {corr_pairs_file}")
    else:
        print("Keine korrelierten Feature-Paare gefunden.")

    ## **Speicherung der VIF-Werte vor der Reduktion**
    vif_before_file = os.path.join(output_dir, f"vif_before_reduction_{target_name}.csv")
    vif_data_before.to_csv(vif_before_file, index=False)
    print(f"VIF-Werte vor der Reduktion gespeichert unter: {vif_before_file}")


    ## **4. MODELLTRAINING & ERGEBNISSPEICHERUNG**
    results_dict = {}
    best_pipelines_all = {}
    confusion_matrices_all = {}

    for step_name, (X_train_step, X_test_step) in {
        "original": (X_train, X_test),
        "shap": (X_train_reduced, X_test_reduced),
        "correlation": (X_train_corr_filtered, X_test_corr_filtered),
        "vif": (X_train_vif_filtered, X_test_vif_filtered),
        }.items():
        try:
            if X_train_step is None or X_test_step is None:
                print(f"{step_name}-Daten sind `None`. Schritt übersprungen.")
                continue

            # Erstelle eine frische Pipeline-Instanz, um interne Zustände auszuschließen
            current_pipeline = clone(pipeline)
            pipelines = {model_name: current_pipeline}
            
            # Nutze Kopien der DataFrames, um unbeabsichtigte Seiteneffekte zu vermeiden
            X_train_local = X_train_step.copy()
            X_test_local = X_test_step.copy()
            y_train_local = y_train.copy()
            y_test_local = y_test.copy()
            
            if is_regression:
                best_pipelines, results = train_and_tune_models_regression(
                    pipelines=pipelines,
                    param_grids={model_name: param_grids[model_name]},
                    X_train=X_train_local,
                    y_train=y_train_local,
                    cv=cv,
                    X_test=X_test_local,
                    y_test=y_test_local
                )
            else:
                best_pipelines, results, confusion_matrices = train_and_tune_models_balanced(
                    pipelines=pipelines,
                    param_grids={model_name: param_grids[model_name]},
                    X_train=X_train_local,
                    y_train=y_train_local,
                    cv=cv,
                    X_test=X_test_local,
                    y_test=y_test_local
                )

            results_dict[step_name] = results[model_name]
            best_pipelines_all[step_name] = best_pipelines[model_name]
            if not is_regression:
                confusion_matrices_all[step_name] = confusion_matrices[model_name]

            # Falls es sich um ein Regressionsmodell handelt und wir im "shap"-Schritt sind, 
            # erstelle einen Scatter-Plot für tatsächliche vs. vorhergesagte Werte.
            if is_regression and step_name == "shap":
                final_model = best_pipelines[model_name]
                final_model.fit(X_train_local, y_train_local)
                y_pred_test = final_model.predict(X_test_local)
                plot_actual_vs_predicted(y_test_local, y_pred_test, model_name=model_name)

            # Falls Klassifikation und Schritt "shap": Confusion-Matrix normiert berechnen, anzeigen und speichern
            if not is_regression and step_name == "shap" and label_mapping_path is not None and target_column is not None:
                display_final_confusion_matrices({model_name: confusion_matrices[model_name]},
                                                 label_mapping_path, target_column, output_dir,
                                                 save_plots=True, normalize=True, cmap=plt.cm.Blues, Verkippung=verkippung)

            # --- Modell direkt speichern (nur im "shap"-Schritt) ---
            if step_name == "shap":
                # Hier wird das finale Modell (Pipeline) aus dem "shap"-Schritt genommen
                final_pipeline = best_pipelines[model_name]

                # Modell als Wrapper-Objekt speichern
                os.makedirs(output_dir, exist_ok=True)
                try:
                    model_save_path = os.path.join(output_dir, f"{target_name}Model.pkl")
                    model = GenericModelWrapper()
                    model.set_feature_names(X_train.columns.tolist())
                    model.save_model(final_pipeline, model_save_path)
                    print(f"Modell für {target_name} gespeichert unter: {model_save_path}")

                    # Speichere Feature-Namen
                    feature_names_path = os.path.join(output_dir, f"{target_name}Model_features.txt")
                    with open(feature_names_path, "w") as f:
                        f.write(f"Features für Modell '{target_name}' (shap-Schritt):\n")
                        f.write("\n".join(X_train_step.columns.tolist()))
                    print(f"Feature-Namen gespeichert unter: {feature_names_path}")

                    # Ergebnisse speichern
                    save_scores_to_csv(
                        results=pd.DataFrame.from_dict(results, orient='index').T,  # Direkt das DataFrame übergeben
                        output_dir=output_dir,    # Zielverzeichnis
                        file_name=f"model_scores_best_model_{target_name}.csv",  # Dateiname
                        Verkippung=verkippung    # Flag für Verkippung
                    )
                except Exception as speicher_error:
                    print(f"Fehler beim Speichern des Modells: {speicher_error}")

        except Exception as e:
            print(f"Fehler beim Trainieren mit {step_name}-Features: {e}")
            results_dict[step_name] = {"f1": "Error", "percent_diff": "Error"}

    ## **Speicherung der Ergebnistabelle**
    summary_df = save_feature_selection_summary(dataset_name, target_name, results_dict, {
        "original": X_train.shape[1],
        "shap": len(X_train_reduced.columns),
        "correlation": len(X_train_corr_filtered.columns),
        "vif": len(X_train_vif_filtered.columns)
    }, output_dir, is_regression=is_regression)

    return best_pipelines, results_dict

def compare_smote_effects(no_smote_path: str, smote_path: str, metric: str, 
                           index_col: str = "Metric", sep: str = ",", title: str = None,save_path: str = None) -> pd.DataFrame:
    """
    Liest zwei CSV-Dateien ein, die Modell-Scores enthalten – eine ohne SMOTE und eine mit SMOTE – 
    und visualisiert die Auswirkungen von SMOTE auf die gewählte Metrik in einem Balkendiagramm.
    
    Die CSV-Dateien haben folgende Struktur:
      - Eine Spalte "Metric"
      - Weitere Spalten für die Modelle (z. B. "XGBoost", "AdaBoost", etc.)
      
    Args:
        no_smote_path (str): Pfad zur CSV-Datei ohne SMOTE.
        smote_path (str): Pfad zur CSV-Datei mit SMOTE.
        metric (str): Die zu vergleichende Metrik (z. B. "f1_weighted", "f1" oder "MSE").
        index_col (str, optional): Name der Spalte, die als Index genutzt wird (Standard: "Metric").
        sep (str, optional): Trennzeichen der CSV-Datei (Standard: ",").
        title (str, optional): Titel des Diagramms. Falls None, wird ein Standardtitel verwendet.
        
    Returns:
        pd.DataFrame: Ein DataFrame, das die Metrikwerte für jedes Modell (SMOTE vs. No SMOTE) enthält.
    """
    # CSV-Dateien einlesen
    df_no_smote = pd.read_csv(no_smote_path, sep=sep, index_col=index_col)
    df_smote = pd.read_csv(smote_path, sep=sep, index_col=index_col)
    
    # Überprüfen, ob die gewünschte Metrik vorhanden ist
    if metric not in df_no_smote.index:
        raise KeyError(f"Spalte '{metric}' nicht gefunden in der Datei ohne SMOTE. Verfügbare Metriken: {df_no_smote.index.tolist()}")
    if metric not in df_smote.index:
        raise KeyError(f"Spalte '{metric}' nicht gefunden in der Datei mit SMOTE. Verfügbare Metriken: {df_smote.index.tolist()}")
    
    # Extrahiere die Zeile für die gewählte Metrik
    series_no_smote = df_no_smote.loc[metric]
    series_smote = df_smote.loc[metric]
    
    # Konvertiere die Werte in numerische Typen (falls als Strings gespeichert)
    series_no_smote = pd.to_numeric(series_no_smote, errors='coerce')
    series_smote = pd.to_numeric(series_smote, errors='coerce')
    
    # Erstelle ein Vergleichs-DataFrame (Transponieren ist hier nicht nötig, da die Modelle bereits als Index vorliegen)
    if metric == "Test MSE" or metric == "Test MAE":
        df_comparison = pd.DataFrame({
        "No SMOGN": series_no_smote,
        "SMOGN": series_smote
        })
        # Optional: Sortiere nach dem No-SMOTE-Wert
        df_comparison = df_comparison.sort_values(by="No SMOGN", ascending=False)
    else: 
        df_comparison = pd.DataFrame({
            "No SMOTE": series_no_smote,
            "SMOTE": series_smote
        })
        # Optional: Sortiere nach dem No-SMOTE-Wert
        df_comparison = df_comparison.sort_values(by="No SMOTE", ascending=False)
    
    
    
    # Visualisierung als Balkendiagramm
    plt.figure(figsize=(10, 6))
    ax = df_comparison.plot(kind="bar", width=0.8, rot=45, edgecolor="black")
    plt.ylabel(metric)
    if title is None:
        title = f"Vergleich der {metric}-Werte: SMOTE vs. No SMOTE"
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    # Plot speichern, bevor er angezeigt wird
    if save_path:
        plt.savefig(save_path, format= "svg", dpi=300, bbox_inches="tight")
        print(f"Plot gespeichert unter: {save_path}")

    plt.show()
    
    return df_comparison

