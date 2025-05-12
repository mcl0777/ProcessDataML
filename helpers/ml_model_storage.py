from typing import List
from sklearn.preprocessing import (
    PolynomialFeatures,
    normalize,
    StandardScaler,
)  # type:ignore
from machine_learning.helpers.ml_model import ML_Model
from machine_learning.helpers.process_data import ProcessedData
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ErfolgModel(ML_Model):
    def preprocess_pipeline(self, processed_features: ProcessedData) -> np.ndarray:
        used_names = [
           "tilt_x_t0",
            "tilt_y_t0",
            "tilt_y_tiefster",
            "tilt_x_tiefster",
            "Umformzeit",
            "Geschwindigkeit_Ges_Min",
            "Presskraft_dyn_Std",
            "Werkzeug_Temp",
            "Geschwindigkeit_Ges_Median",
            "Geschwindigkeit_Ges_Mean",
            "max_druck",
            "Presskraft_dyn_Max",
            "Presskraft_dyn_Mean",
            "Presskraft_dyn_Median",
            "Berührzeit",
            "Verkippung_4_Median",
            "beruehrungsdauer",
            "Geschwindigkeit_Ges_Std",
            "Presskraft_dyn_Min",
            "Verkippung_1_Median",
            "Verkippung_3_Median",
            "Verkippung_2_Mean",
            "Bauteil_Temp",
            "Energie_Aufprall",
            "auftreffposition",
            "Stoesselhub_Median",
            "Höhe_Wegmessung_Aufprall",
            "Verkippung_2_Median",
            "Stoesselhub_Mean",
            "Wegmessung_Max",
            "Wegmessung_Median",
            "Verkippung_2_Min",
            "Verkippung_1_Mean",
            "arbeit_ab",
            "Verkippung_4_Mean",
            "auftreffzeitpunkt",
            "Energie_ab_Durchschnitt",
            "Motorstrom_Mean",
            "Tiefster_Punkt",
            "Verkippung_3_Min",
            "Verkippung_1_Min",
            "Verkippung_4_Min",
            "Motorstrom_Median",
        ]
        # Scaler anwenden
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(processed_features.get_features_mit_namen(used_names, []))
           
        return scaled_features


class MaterialModel(ML_Model):
    def preprocess_pipeline(self, processed_features: ProcessedData) -> np.ndarray:
        used_names = [
            "Bauteil_Temp",
            "Geschwindigkeit_Ges_Std",
            "Presskraft_dyn_Mean",
            "Geschwindigkeit_Ges_Min",
            "tilt_y_tiefster",
            "Motorstrom_Std",
            "Presskraft_dyn_Max",
            "Wegmessung_Median",
            "max_druck",
            "Tiefster_Punkt",
            "tilt_x_tiefster",
            "Umformzeit",
            "Energie_Aufprall",
            "Geschwindigkeit_Ges_Median",
            "Wegmessung_Min",
            "Motorstrom_Mean",
            "Presskraft_dyn_Median",
            "tilt_x_t0",
            "Verkippung_1_Max",
            "Verkippung_2_Max",
            "Geschwindigkeit_Ges_Max",
            "Verkippung_3_Max",
            "auftreffzeitpunkt",
            "beruehrungsdauer",
            "Energie_ab_Durchschnitt",
            "Berührzeit",
            "Verkippung_4_Max",
            "Presskraft_dyn_Min",
            "min_motorstrom",
            "Motorstrom_Max",
            "Stoesselhub_Min",
            "Verkippung_1_Median",
            "Verkippung_3_Median",
            "max_motorstrom",
        ]
        # Scaler anwenden
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(processed_features.get_features_mit_namen(used_names, []))
           
        return scaled_features


class PositionModel(ML_Model):
    def preprocess_pipeline(self, processed_features: ProcessedData) -> np.ndarray:
        used_names = [
            "Verkippung_2_Mean",
            "Verkippung_3_Mean",
            "Verkippung_2_Max",
            "Verkippung_3_Median",
            "Verkippung_2_Median",
            "Verkippung_1_Mean",
            "Verkippung_4_Max",
            "Verkippung_4_Mean",
            "Verkippung_1_Std",
            "Verkippung_1_Max",
            "Verkippung_3_Max",
            "Wegmessung_Min",
        ]
        
        # Scaler anwenden
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(processed_features.get_features_mit_namen(used_names, []))
           
        return scaled_features


class ProbenhoeheModel(ML_Model):
    def preprocess_pipeline(self, processed_features: ProcessedData) -> np.ndarray:
        try:
            used_names = [
                "höhe_wegmessung_aufprall",
                "Wegmessung_Max",
                "auftreffposition",
                "Verkippung_2_Min",
                "tilt_x_tiefster",
                "Bauteil_Temp",
                "tilt_y_tiefster",
            ]
            
            features = processed_features.get_features_mit_namen(used_names, [])
            if features.size == 0:
                logger.error("No features found for ProbenhoeheModel")
                return np.zeros((1, len(used_names)))
                
            return features
        except Exception as e:
            logger.error(f"Error in ProbenhoeheModel preprocessing: {e}")
            return np.zeros((1, len(used_names)))


class VerbautModel(ML_Model):
    def preprocess_pipeline(self, processed_features: ProcessedData) -> np.ndarray:
        used_names = [
            "Wegmessung_Min",
            "Verkippung_3_Std",
            "Verkippung_1_Median",
            "Verkippung_4_Median",
            "Verkippung_1_Mean",
            "Verkippung_2_Std",
            "Verkippung_4_Mean",
            "Verkippung_1_Std",
            "Geschwindigkeit_Ges_Mean",
            "Wegmessung_Std",
            "tilt_x_t0",
            "tilt_y_t0",
            "Stoesselhub_Min",
            "Wegmessung_Mean",
            "Verkippung_4_Std",
            "Verkippung_3_Mean",
            "Geschwindigkeit_Ges_Median",
            "Verkippung_2_Median",
            "Umformzeit",
            "Verkippung_4_Min",
            "Motorstrom_Mean",
            "Geschwindigkeit_Ges_Std",
            "max_druck",
            "Stoesselhub_Mean",
        ]
        # Scaler anwenden
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(processed_features.get_features_mit_namen(used_names, []))
           
        return scaled_features
    
class BauteiltemperaturModel(ML_Model):
    def preprocess_pipeline(self, processed_features: ProcessedData) -> np.ndarray:
        used_names = [
            "schwellwert_index",
            "umformzeit",
            "höhe_wegmessung_aufprall",
            "bauteil_temp",
            "press_geschwindigkeit",

        ]

        return processed_features.get_features_mit_namen(used_names, [])


all_models: List[ML_Model] = [
    ErfolgModel,
    MaterialModel,
    PositionModel,
    ProbenhoeheModel,
    VerbautModel,
    BauteiltemperaturModel,
]
