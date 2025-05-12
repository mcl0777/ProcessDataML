from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import dataframe_image as dfi


def get_estimator_scores(
    pipeline: Pipeline, best_params: Dict, kf, features: np.ndarray, labels: np.ndarray
) -> Tuple[pd.DataFrame, pd.Series]:
    pipeline.set_params(**best_params)
    scores: List[Dict] = []
    for train_index, test_index in kf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        scores.append(
            {
                "accuracy": accuracy_score(y_test, prediction),
                "precision": precision_score(y_test, prediction),
                "recall": recall_score(y_test, prediction),
                "f1": f1_score(y_test, prediction),
                "good_samples": y_test.sum(),
                "bad_samples": len(y_test) - y_test.sum(),
                "pred_good_samples": prediction.sum(),
            }
        )
    results = pd.DataFrame(scores)
    result_describe = results.mean()
    result_describe["best_params"] = best_params

    return (results, result_describe)


def export_dataframe_to_png(results: pd.DataFrame, name: str):
    dfi.export(
        results.drop(["best_params"]),
        f"C:/Uni/Bachelor/Bachelor-Arbeit/Bilder/Modelle/{name}.png",
    )
