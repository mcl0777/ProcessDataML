import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel

from xgboost import XGBClassifier
from xgboost import XGBRegressor

# -------------------------------
# Helper Funktionen
# -------------------------------
def load_and_split_data(filepath, target_column, irrelevant_columns):
    """
    Lädt den Datensatz, entfernt irrelevante Spalten und teilt die Daten in Features und Zielgröße.
    """
    # Laden der Daten
    dataframe = pd.read_pickle(filepath)
    
    # Feature-Auswahl
    features = [col for col in dataframe.columns if col not in irrelevant_columns and col != target_column]
    X = dataframe[features]
    y = dataframe[target_column]

    return X, y

def define_pipelines_Prozessqualitaet():
    """
    Definiert Modellpipelines und die dazugehörigen Hyperparameter.
    """
    pipelines = {
        "XGBoost": Pipeline([
            ('clf', XGBClassifier(
                eval_metric='logloss',
                use_label_encoder=False,
                tree_method='hist',  # Schnellere Berechnung
            ))
        ]),
        "AdaBoost": Pipeline([
            ('clf', AdaBoostClassifier(
                algorithm='SAMME',
                random_state=42
            ))
        ]),
        "SVM": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für SVM
            ('clf', SVC(
                probability=True,
                random_state=42
            ))
        ]),
        "kNN": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für KNN
            ('clf', KNeighborsClassifier())
        ]),
        "Random Forest": Pipeline([
            ('clf', RandomForestClassifier(
                random_state=42
            ))
        ]),
        "Logistische Regression": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für die logistische Regression
            ('clf', LogisticRegression(
                random_state=42
            ))
        ])
    }

    param_grids = {
        'XGBoost': {
            'clf__n_estimators': [10, 25, 50],  # Erhöhung für stabilere Bäume
            'clf__max_depth': [3, 5, 7],  # Mittelwerte, um Overfitting zu vermeiden
            'clf__learning_rate': [0.01, 0.05, 0.1],  # Mäßige Lernraten für bessere Generalisierung
            'clf__gamma': [0, 0.1, 0.2],  # Minimaler Regularisierungseffekt
        },
        'AdaBoost': {
            'clf__n_estimators': [8, 13, 21],  # Mehr Stufen für bessere Stärkung schwacher Learner
            'clf__learning_rate': [0.05, 0.2, 0.5],  # Kleinere Raten für bessere Stabilität
            'clf__estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)]
        },
        'SVM': {'clf__C': [2,5,6,10], 'clf__kernel': ['rbf','linear']},
        'kNN': {
            'clf__n_neighbors': [3,5,8],  # Mehr Nachbarn für stabilere Entscheidungen
            'clf__weights': ['uniform'],  # Gleichmäßige oder gewichtete Nachbarn testen
            'clf__p': [ 1,2]  # Vergleich von Manhattan (p=1) und euklidischer Distanz (p=2)
        },
        'Random Forest': {
            'clf__n_estimators': [10, 25, 50],  # Bereich für effizientere Berechnung
            'clf__max_depth': [5, 10],  # Kleinere Werte für bessere Generalisierung
            'clf__min_samples_split': [5],  # Fixiert, da keine große Variation sichtbar ist
            'clf__min_samples_leaf': [2]  # Optimale Wahl aus bisherigen Tests
        },
        'Logistische Regression': {
            'clf__C': [0.5, 1, 5],  # Reguläre Werte für Stabilität
            'clf__penalty': ['l1', 'l2'],  # Vergleich von Lasso (L1) und Ridge (L2)
            'clf__solver': ['liblinear']  # Optimierung für kleinere Datensätze
        }
    }
    
    return pipelines, param_grids

def define_pipelines_Material():
    """
    Definiert Modellpipelines und die dazugehörigen allgemeiner gehaltenen Hyperparameter 
    für die Materialprognose im ersten Trainingslauf.
    """
    pipelines = {
        "XGBoost": Pipeline([
            ('clf', XGBClassifier(
                objective='binary:logistic',  # Binäre Klassifikation
                eval_metric='logloss',
                use_label_encoder=False,
                tree_method='hist',
                random_state=42
            ))
        ]),
        "AdaBoost": Pipeline([
            ('clf', AdaBoostClassifier(
                algorithm='SAMME',
                random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ('clf', RandomForestClassifier(
                random_state=42
            ))
        ]),
        "SVM": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für SVM
            ('clf', SVC(
                probability=True,
                random_state=42
            ))
        ]),
        "kNN": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für KNN
            ('clf', KNeighborsClassifier())
        ]),
        "Logistische Regression": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für die logistische Regression
            ('clf', LogisticRegression(
                random_state=42,
                max_iter=10000
            ))
        ])
    }

    param_grids = {
        'XGBoost': {
            'clf__n_estimators': [21, 30,70],  # Mehr Bäume testen für bessere Stabilität
            'clf__learning_rate': [0.01, 0.05, 0.1],  # Kleinere Lernrate testen
            'clf__max_depth': [3, 5,8],  # Mittelgroße Baumtiefen für bessere Generalisierung
            'clf__gamma': [0.05, 0.1, 0.15]  # Striktere Split-Kriterien zur Stabilisierung
        },
        'AdaBoost': {
            'clf__n_estimators': [8, 13, 21],  # Engerer Bereich für bessere Stabilität
            'clf__learning_rate': [0.15, 0.2, 0.25],  # Feinanpassung für mehr Balance
            'clf__estimator': [DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=4)]  # Alternative Max-Tiefen testen
        },
        'SVM': {
            'clf__C': [0.05, 0.1,0.5],  # Kleinere Variationen zur Stabilität
            'clf__kernel': ['rbf'],  # RBF bleibt als bester Kernel
            'clf__gamma': [0.001, 0.005, 0.01]  # Kleinere Werte für weniger Overfitting
        },
        'kNN': {
            'clf__n_neighbors': [3,5,8,13],  # Mehr Nachbarn testen, um Overfitting zu reduzieren
            'clf__weights': ['uniform']  # Unterschiedliche Gewichtungen testen
        },
        'Random Forest': {
            'clf__n_estimators': [15, 20,60],  # Engerer Bereich für mehr Stabilität
            'clf__max_depth': [5, 6, 7],  # Mittlere Tiefen für bessere Generalisierung
            'clf__min_samples_split': [3, 5, 7],  # Striktere Regeln für Split, um Overfitting zu vermeiden
            'clf__min_samples_leaf': [1, 2, 3]  # Kleinere Blattgrößen für Stabilität
        },
        'Logistische Regression': {
            'clf__C': [5, 15, 30,50],  # Weitere Regularisierungen testen
            'clf__solver': ['lbfgs', 'saga'],  # Stabile Solver für Multinomial-Klassifikation
            'clf__max_iter': [500, 1000]  # Mehr Iterationen für bessere Konvergenz
        }
    }
    
    return pipelines, param_grids

def define_pipelines_Probenposition():
    """
    Definiert Modellpipelines und die dazugehörigen allgemeiner gehaltenen Hyperparameter 
    für die Materialprognose im ersten Trainingslauf.
    """
    
    pipelines = {
        "XGBoost": Pipeline([
            ('clf', XGBClassifier(
                scale_pos_weight=1.3396,
                objective='binary:logistic',  # Binäre Klassifikation
                eval_metric='logloss',
                use_label_encoder=False,
                tree_method='hist',
                random_state=42
            ))
        ]),
        "AdaBoost": Pipeline([
            ('clf', AdaBoostClassifier(
                algorithm='SAMME',
                random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ('clf', RandomForestClassifier(
                random_state=42
            ))
        ]),
        "SVM": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für SVM
            ('clf', SVC(
                probability=True,
                random_state=42
            ))
        ]),
        "kNN": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für KNN
            ('clf', KNeighborsClassifier())
        ]),
        "Logistische Regression": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für die logistische Regression
            ('clf', LogisticRegression(
                random_state=42
            ))
        ])
    }

    param_grids = {
        "XGBoost": {
            # Schritt 1: Fixe oder kleine Variation der Lernrate + mehr Trees
            # Schritt 2: Weitere Parameter (max_depth etc.) inkrementell tunen
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__max_depth": [3, 5],
            "clf__min_child_weight": [1, 3],
            "clf__gamma": [0, 0.1],
            "clf__subsample": [0.8, 1.0]
        },
        "AdaBoost": {
            # Schritt 1: n_estimators und learning_rate variieren
            # Schritt 2: Base-Estimator (Tiefe des DecisionTrees)
            "clf__n_estimators": [50, 100],
            "clf__learning_rate": [0.1, 0.5],
            "clf__estimator": [
                DecisionTreeClassifier(max_depth=2),
                DecisionTreeClassifier(max_depth=3)
            ]
        },
        "SVM": {
            # Schritt 1: Variation von C, gamma
            # Schritt 2 (optional): Kernel-Typ prüfen
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["rbf"],
            "clf__gamma": ["scale", 0.01, 0.1]
        },
        "kNN": {
            # Schritt 1: Variation der Nachbarzahl, Distanzmetrik
            # Schritt 2: Gewichte oder Leaf-Size
            "clf__n_neighbors": [3, 5, 7],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2]
        },
        "Random Forest": {
            # Schritt 1: n_estimators und max_depth
            # Schritt 2: min_samples_split / min_samples_leaf
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [3, 6, None],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2]
            # Optional: "clf__max_features": ["sqrt", 0.5]
        },
        "Logistische Regression": {
            # Schritt 1: Variation der Regularisierung (C)
            # Schritt 2: penalty & solver
            "clf__C": [5,10,15,20],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear", "saga"],
            "clf__max_iter": [200,500]
        }
    }
    
    return pipelines, param_grids

def define_pipelines_Verbaut():
    """
    Definiert Modellpipelines und die dazugehörigen Hyperparameter.
    """
    pipelines = {
        "XGBoost": Pipeline([
            ('clf', XGBClassifier(
                eval_metric='logloss',
                use_label_encoder=False,
                tree_method='hist',  # Schnellere Berechnung
            ))
        ]),
        "AdaBoost": Pipeline([
            ('clf', AdaBoostClassifier(
                algorithm='SAMME',
                random_state=42
            ))
        ]),
        "SVM": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für SVM
            ('clf', SVC(
                probability=True,
                random_state=42
            ))
        ]),
        "kNN": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für KNN
            ('clf', KNeighborsClassifier())
        ]),
        "Random Forest": Pipeline([
            ('clf', RandomForestClassifier(
                random_state=42
            ))
        ]),
        "Logistische Regression": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für die logistische Regression
            ('clf', LogisticRegression(
                random_state=42
            ))
        ])
    }

    param_grids = {
        'XGBoost': {
            'clf__n_estimators': [21, 30,70],  # Mehr Bäume testen für bessere Stabilität
            'clf__learning_rate': [0.01, 0.05, 0.1],  # Kleinere Lernrate testen
            'clf__max_depth': [3, 5,8],  # Mittelgroße Baumtiefen für bessere Generalisierung
            'clf__gamma': [0.05, 0.1, 0.15]  # Striktere Split-Kriterien zur Stabilisierung
        },
        'AdaBoost': {
            'clf__n_estimators': [8, 13, 21],  # Engerer Bereich für bessere Stabilität
            'clf__learning_rate': [0.15, 0.2, 0.25],  # Feinanpassung für mehr Balance
            'clf__estimator': [DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=4)]  # Alternative Max-Tiefen testen
        },
        'SVM': {
            'clf__C': [0.05, 0.1,0.5],  # Kleinere Variationen zur Stabilität
            'clf__kernel': ['rbf'],  # RBF bleibt als bester Kernel
            'clf__gamma': [0.001, 0.005, 0.01]  # Kleinere Werte für weniger Overfitting
        },
        'kNN': {
            'clf__n_neighbors': [3,5,8,13],  # Mehr Nachbarn testen, um Overfitting zu reduzieren
            'clf__weights': ['uniform']  # Unterschiedliche Gewichtungen testen
        },
        'Random Forest': {
            'clf__n_estimators': [15, 20,60],  # Engerer Bereich für mehr Stabilität
            'clf__max_depth': [5, 6, 7],  # Mittlere Tiefen für bessere Generalisierung
            'clf__min_samples_split': [3, 5, 7],  # Striktere Regeln für Split, um Overfitting zu vermeiden
            'clf__min_samples_leaf': [1, 2, 3]  # Kleinere Blattgrößen für Stabilität
        },
        'Logistische Regression': {
            'clf__C': [5, 15, 30,50],  # Weitere Regularisierungen testen
            'clf__solver': ['lbfgs', 'saga'],  # Stabile Solver für Multinomial-Klassifikation
            'clf__max_iter': [500, 1000]  # Mehr Iterationen für bessere Konvergenz
        }
    }
    
    return pipelines, param_grids


###### Regression ######

def define_pipelines_Probenhoehe(input_dim=None):
    """
    Definiert Modellpipelines und die dazugehörigen Hyperparameter für die Prognose der Probenhöhe.
    """
    pipelines = {
        "XGBoost": Pipeline([
            ('clf', XGBRegressor(
                tree_method='hist',  # Schnellere Berechnung
                random_state=42
            ))
        ]),
        "AdaBoost": Pipeline([
            ('clf', AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=3),  # Korrektur von base_estimator zu estimator
                random_state=42
            ))
        ]),
        "SVR": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für SVM
            ('clf', SVR())
        ]),
        "kNN": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für KNN
            ('clf', KNeighborsRegressor())
        ]),
        "Random Forest": Pipeline([
            ('clf', RandomForestRegressor(
                random_state=42
            ))
        ]),
        "Lineare Regression": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung optional, für bessere Konvergenz
            ('clf', LinearRegression())
        ])
    }

    param_grids = {
        'XGBoost': {
            'clf__n_estimators': [144,233,377],  # Eingrenzung um den besten Bereich
            'clf__max_depth': [3, 4, 5],  # Mehr Flexibilität um mittlere Baumtiefen
            'clf__learning_rate': [0.01, 0.05, 0.1],  # Konzentration auf moderate Lernraten
            'clf__min_child_weight': [2, 3],  # Regularisierung auf stabile Werte
            'clf__subsample': [0.8, 0.9, 1.0],  # Diversität bei der Stichprobenanzahl
            'clf__colsample_bytree': [0.5, 0.6],  # Fokussierung auf moderate Diversität der Features
            'clf__gamma': [0.1, 0.15, 0.2]  # Stabilere Mindestverlustreduktion
            
        },
        "AdaBoost": {
            "clf__n_estimators": [50, 100, 150],  # Anzahl der Iterationen
            "clf__learning_rate": [0.01, 0.1, 0.5],  # Lernrate
            "clf__estimator": [DecisionTreeRegressor(max_depth=2), DecisionTreeRegressor(max_depth=3)]
        },
        'SVR': {
            'clf__C': [0.6,0.9],  # feinere Abstufung, eher niedrig halten
            'clf__gamma': ['scale'],  # zusätzlich zu 'scale' auch feste Werte testen
            'clf__kernel': ['rbf'],  # Fokus auf den bewährten Kernel
        },
        'kNN': {
            'clf__n_neighbors': [2],  # Eingrenzung auf stabile Nachbarwerte
            'clf__weights': ['uniform'],  # Fokus auf das bewährte Gewichtungsschema
            'clf__p': [1],  # Testen von Manhattan und Euclidean
            'clf__leaf_size': [15,25,35]  # Optimierung der Baumgröße
        },
        'Random Forest': {
            'clf__n_estimators': [15, 20,60],  # Engerer Bereich für mehr Stabilität
            'clf__max_depth': [5, 6, 7],  # Mittlere Tiefen für bessere Generalisierung
            'clf__min_samples_split': [3, 5, 7],  # Striktere Regeln für Split, um Overfitting zu vermeiden
            'clf__min_samples_leaf': [1, 2, 3]  # Kleinere Blattgrößen für Stabilität
        },
        'Lineare Regression': {
            # Keine Hyperparameter
        }
    }
    
    return pipelines, param_grids

def define_pipelines_BauteilTemp(input_dim=None):
    """
    Definiert Modellpipelines und die dazugehörigen Hyperparameter für die Prognose der Probenhöhe.
    """
    pipelines = {
        "XGBoost": Pipeline([
            ('feature_selection', SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))),
            ('clf', XGBRegressor(tree_method='hist', random_state=42))
        ]),
        "AdaBoost": Pipeline([
            ('clf', AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=3),  # Korrektur von base_estimator zu estimator
                random_state=42
            ))
        ]),
        "SVR": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung notwendig für SVM
            ('clf', SVR())
        ]),
        # "kNN": Pipeline([
        #     ('scaler', StandardScaler()),  # Skalierung notwendig für KNN
        #     ('clf', KNeighborsRegressor())
        # ]),
        "Random Forest": Pipeline([
            ('feature_selection', SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))),
            ('clf', RandomForestRegressor(random_state=42))
        ]),
        "Lineare Regression": Pipeline([
            ('scaler', StandardScaler()),  # Skalierung optional, für bessere Konvergenz
            ('clf', LinearRegression())
        ])
    }

    param_grids = {
        'XGBoost': {
            'clf__n_estimators': [450],          # Fokus auf bewährte hohe Werte
            'clf__max_depth': [4],                      # flache Bäume reduzieren Overfitting
            'clf__learning_rate': [ 0.1],       # stabile Lernraten
            'clf__min_child_weight': [3],            # stärkere Regularisierung
            'clf__subsample': [0.6, 0.8],                  # stärkere Diversität
            'clf__colsample_bytree': [0.4, 0.5],           # weniger Features pro Baum
            'clf__gamma': [ 0.4]                  # schärfere Splitting-Bedingung
        },
        "AdaBoost": {
            "clf__n_estimators": [144,233,377],  # Anzahl der Iterationen
            "clf__learning_rate": [0.01, 0.1, 0.5],  # Lernrate
            "clf__estimator": [DecisionTreeRegressor(max_depth=4), DecisionTreeRegressor(max_depth=3)]
        },
        'SVR': {
            'clf__C': [0.1, 0.5],
            'clf__kernel': ['linear'],  # einfacher Kernel
            'clf__gamma': ['scale']
        },
        # 'kNN': {
        #     'clf__n_neighbors': [5, 10, 15, 20],  # deutlich mehr Nachbarn!
        #     'clf__weights': ['uniform', 'distance'],
        #     'clf__p': [1, 2],
        #     'clf__leaf_size': [20, 30, 50]
        # },
        'Random Forest': {
            'clf__n_estimators': [200],           # größere Ensemble-Stabilität
            'clf__max_depth': [5],                   # flachere Tiefe gegen Overfitting
            'clf__min_samples_split': [3,5],          # striktere Trennung
            'clf__min_samples_leaf': [2],               # stärkere Glättung
            'clf__bootstrap': [True],                      # weiterhin Bootstrapping
            'clf__max_features': ['sqrt', 0.6]        # weniger Features pro Split
        },
        'Lineare Regression': {
            # Keine Hyperparameter
        }
    }
    
    return pipelines, param_grids


######### SHAP ANALYSE ###########

def shap_analysis(
    best_pipelines, 
    X_train, 
    target_name, 
    dataset_name, 
    output_dir="results", 
    plot_type="summary",  
    save_plots=True, 
    verkippung=True
):
    """
    Führt SHAP-Analysen durch, erstellt Plots und speichert Feature-Wichtigkeiten separat für jedes Modell.
    """
    tilt_suffix = "_no_tilt" if not verkippung else ""

    for name, pipeline in best_pipelines.items():
        print(f"Berechnung der SHAP-Werte für {name}...")

        # Speicherpfad
        model_output_dir = os.path.join(output_dir, name)
        os.makedirs(model_output_dir, exist_ok=True)

        # Transformation prüfen und anwenden
        if "scaler" in pipeline.named_steps:
            X_train_transformed = pipeline.named_steps["scaler"].transform(X_train)
            feature_names = X_train.columns
        else:
            X_train_transformed = X_train.values
            feature_names = X_train.columns

        try:
            if name == "AdaBoost":
                # AdaBoost mit TreeExplainer
                try:
                    base_estimators = pipeline.named_steps["clf"].estimators_
                    # Aggregierte SHAP-Werte über alle Estimatoren
                    shap_values = np.mean([
                        shap.TreeExplainer(est).shap_values(X_train_transformed) for est in base_estimators
                    ], axis=0)
                    if shap_values.ndim == 3:  # Multi-Class
                        shap_values = shap_values[..., 1]  # Wähle Klasse 1
                except Exception as e:
                    print(f"TreeExplainer für AdaBoost nicht unterstützt. Fallback auf PermutationExplainer: {e}")
                    explainer = shap.PermutationExplainer(pipeline.named_steps["clf"].predict, X_train_transformed)
                    shap_values = explainer.shap_values(X_train_transformed)
            elif name in ["Random Forest", "XGBoost"]:
                explainer = shap.TreeExplainer(pipeline.named_steps["clf"])
                shap_values = explainer.shap_values(X_train_transformed)
                if isinstance(shap_values, list):  # Falls mehrere Klassen vorliegen
                    shap_values = shap_values[1]  # Klasse 1 auswählen
                elif shap_values.ndim == 3:
                    shap_values = shap_values[..., 1]
            elif name in ["SVM", "kNN"]:
                explainer = shap.PermutationExplainer(pipeline.named_steps["clf"].predict, X_train_transformed)
                shap_values = explainer.shap_values(X_train_transformed)
            elif isinstance(pipeline.named_steps["clf"], LogisticRegression):
                explainer = shap.LinearExplainer(pipeline.named_steps["clf"], X_train_transformed)
                shap_values = explainer.shap_values(X_train_transformed)
            else:
                subset = pd.DataFrame(X_train_transformed, columns=feature_names).iloc[:100]
                explainer = shap.KernelExplainer(pipeline.named_steps["clf"].predict, subset)
                shap_values = explainer.shap_values(subset)
                X_train_transformed = subset
        except Exception as e:
            print(f"Fehler bei der SHAP-Berechnung für {name}: {e}")
            continue

        # Berechnung der durchschnittlichen SHAP-Werte
        try:
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            if len(mean_shap_values) != len(feature_names):
                raise ValueError(
                    f"Dimension mismatch für {name}: SHAP-Werte ({len(mean_shap_values)}) und Feature-Namen ({len(feature_names)}) stimmen nicht überein."
                )
            # Feature-Wichtigkeiten in DataFrame speichern
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": mean_shap_values
            }).sort_values(by="SHAP Value", ascending=False)

            # Speichern der Feature-Wichtigkeiten als CSV
            importance_path = os.path.join(model_output_dir, f"feature_importance_{name}{tilt_suffix}.csv")
            importance_df.to_csv(importance_path, index=False)
            print(f"Feature-Wichtigkeiten für {name} gespeichert unter {importance_path}.")
        except Exception as e:
            print(f"Fehler bei der Berechnung der Feature-Wichtigkeiten für {name}: {e}")

        # Plot erstellen
        try:
            plot_title = f"{plot_type.capitalize()} Plot for {name} ({target_name}){tilt_suffix}"
            if plot_type == "summary":
                shap.summary_plot(
                    shap_values,
                    X_train_transformed,
                    show=False,
                    feature_names=feature_names
                )
            elif plot_type == "bar":
                shap.summary_plot(
                    shap_values,
                    X_train_transformed,
                    plot_type="bar",
                    show=False,
                    feature_names=feature_names
                )
            elif plot_type == "interaction":
                shap.dependence_plot(
                    interaction_index=feature_names[0],
                    shap_values=shap_values,
                    features=X_train_transformed,
                    feature_names=feature_names,
                    show=False
                )
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")

            plt.title(plot_title, fontsize=10)
            plt.tight_layout()

            if save_plots:
                plot_path = os.path.join(model_output_dir, f"shap_{plot_type}_{name}.svg")
                plt.savefig(plot_path,format="svg", dpi=300, bbox_inches="tight")
                print(f"SHAP {plot_type}-Plot für {name} gespeichert unter {plot_path}.")
            plt.close()
        except Exception as e:
            print(f"Fehler beim Plotten für {name}: {e}")

        print(f"SHAP-Analyse für {name} abgeschlossen.")

def shap_analysis_single_model(
    model_name,
    pipeline, 
    X_train, 
    target_name, 
    dataset_name, 
    output_dir="results", 
    plot_type="summary",  
    save_plots=True, 
    verkippung=True
):
    """
    Führt SHAP-Analyse für ein einzelnes Modell durch, erstellt Plots und speichert Feature-Wichtigkeiten.
    """
    print(f"Berechnung der SHAP-Werte für {model_name}...")

    tilt_suffix = "_no_tilt" if not verkippung else ""
    model_output_dir = os.path.join(output_dir)#, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Transformation prüfen und anwenden
    if "scaler" in pipeline.named_steps:
        X_train_transformed = pipeline.named_steps["scaler"].transform(X_train)
        feature_names = X_train.columns
    else:
        X_train_transformed = X_train.values
        feature_names = X_train.columns

    try:
        if model_name == "AdaBoost":
            try:
                base_estimators = pipeline.named_steps["clf"].estimators_
                shap_values = np.mean([
                    shap.TreeExplainer(est).shap_values(X_train_transformed) for est in base_estimators
                ], axis=0)
                if shap_values.ndim == 3:  # Multi-Class
                    shap_values = shap_values[..., 1]  # Wähle Klasse 1
            except Exception as e:
                print(f"TreeExplainer für AdaBoost nicht unterstützt. Fallback auf PermutationExplainer: {e}")
                explainer = shap.PermutationExplainer(pipeline.named_steps["clf"].predict, X_train_transformed)
                shap_values = explainer.shap_values(X_train_transformed)

        elif model_name in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(pipeline.named_steps["clf"])
            shap_values = explainer.shap_values(X_train_transformed)
            if isinstance(shap_values, list):  
                shap_values = shap_values[1]  # Klasse 1 auswählen
            elif shap_values.ndim == 3:
                shap_values = shap_values[..., 1]

        elif model_name in ["SVM", "kNN"]:
            explainer = shap.PermutationExplainer(pipeline.named_steps["clf"].predict, X_train_transformed)
            shap_values = explainer.shap_values(X_train_transformed)

        elif isinstance(pipeline.named_steps["clf"], LogisticRegression):
            explainer = shap.LinearExplainer(pipeline.named_steps["clf"], X_train_transformed)
            shap_values = explainer.shap_values(X_train_transformed)

        else:
            subset = pd.DataFrame(X_train_transformed, columns=feature_names).iloc[:100]
            explainer = shap.KernelExplainer(pipeline.named_steps["clf"].predict, subset)
            shap_values = explainer.shap_values(subset)
            X_train_transformed = subset

    except Exception as e:
        print(f"Fehler bei der SHAP-Berechnung für {model_name}: {e}")
        return

    # Berechnung der durchschnittlichen SHAP-Werte
    try:
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        if len(mean_shap_values) != len(feature_names):
            raise ValueError(
                f"Dimension mismatch für {model_name}: SHAP-Werte ({len(mean_shap_values)}) und Feature-Namen ({len(feature_names)}) stimmen nicht überein."
            )

        # Feature-Wichtigkeiten in DataFrame speichern
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": mean_shap_values
        }).sort_values(by="SHAP Value", ascending=False)

        # Speichern der Feature-Wichtigkeiten als CSV
        importance_path = os.path.join(model_output_dir, f"feature_importance_{model_name}{tilt_suffix}.csv")
        importance_df.to_csv(importance_path, index=False)
        print(f"Feature-Wichtigkeiten für {model_name} gespeichert unter {importance_path}.")
    except Exception as e:
        print(f"Fehler bei der Berechnung der Feature-Wichtigkeiten für {model_name}: {e}")

    # Plot erstellen
    try:
        plot_title = f"{plot_type.capitalize()} Plot for {model_name} ({target_name}){tilt_suffix}"
        if plot_type == "summary":
            shap.summary_plot(
                shap_values,
                X_train_transformed,
                show=False,
                feature_names=feature_names
            )
        elif plot_type == "bar":
            shap.summary_plot(
                shap_values,
                X_train_transformed,
                plot_type="bar",
                show=False,
                feature_names=feature_names
            )
        elif plot_type == "interaction":
            shap.dependence_plot(
                interaction_index=feature_names[0],
                shap_values=shap_values,
                features=X_train_transformed,
                feature_names=feature_names,
                show=False
            )
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        plt.title(plot_title, fontsize=10)
        plt.tight_layout()

        if save_plots:
            plot_path = os.path.join(model_output_dir, f"shap_{plot_type}_{model_name}.svg")
            plt.savefig(plot_path,format="svg", dpi=300, bbox_inches="tight")
            print(f"SHAP {plot_type}-Plot für {model_name} gespeichert unter {plot_path}.")
        plt.close()
    except Exception as e:
        print(f"Fehler beim Plotten für {model_name}: {e}")

    print(f"SHAP-Analyse für {model_name} abgeschlossen.")