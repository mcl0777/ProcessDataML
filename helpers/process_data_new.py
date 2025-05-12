import math
from typing import Any, List
import numpy as np
import pandas as pd
import xarray as xr


# Konfiguration für Schwellenwerte und andere Parameter
DEFAULT_THRESHOLD = 9 # Schwellenwert für Berührungswerte (9V)
DEFAULT_THRESHOLD_LENGTH = 2000 # Vereinheitlichung der Versuchslänge
DEFAULT_TIME_STEP = 0.5
MEAN_DISTANCE_VERKIPPUNG_X = 745 # mm
MEAN_DISTANCE_VERKIPPUNG_Y = 950 # mm

# Liste relevanter Features
FEATURE_COLUMNS = [
    "Zeit",
    "Berührungswerte",
    "Wegmessung",
    "Verkippung_1",
    "Verkippung_2",  # Vorsicht: Kann fehlerhafte Werte enthalten. Wird in Hilfsfunktion bereinigt
    "Verkippung_3",
    "Verkippung_4",
    "Stoesselhub",
    "Presskraft",
    "Geschwindigkeit_UT",
    "Geschwindigkeit_Ges",
    "Energie_ab",
    "Presskraft_dyn",
    "Motorstrom",
    # "Kraft1", # Sensor wird nicht mehr berücksichtigt
    # "Kraft2", # Sensor wird nicht mehr berücksichtigt
    "Bauteil_temp_con",
    "Werkzeug_temp_con",
    # "Material_con",
    # "Position_con",
    # "Ergebnis_con",
    # "Probenhoehe",
]

# -----------------------------
# Hilfsfunktionen für Versuche und Features
# -----------------------------
def clean_data(da: xr.DataArray) -> xr.DataArray:
    da = da.loc[(da.loc[:, :, "Berührungswerte"] > DEFAULT_THRESHOLD).any(dim="row"), :, :]
    da = da.loc[
        da.loc[:, :, "Zeit"].groupby("versuch").count(dim="row") > DEFAULT_THRESHOLD_LENGTH,
        :,
        :,
    ]
    # indexes_to_drop = np.where(
    #     (
    #         (da.loc[:, :, "Wegmessung"].min(dim="row") > 30)
    #         & (da.loc[:, 0, "Ergebnis_con"] == "GUT")
    #     ),
    #     True,
    #     False,
    # )
    # da = da.sel(versuch=~indexes_to_drop)
    return da


def get_schwellwert_index(da: xr.DataArray, threshold=DEFAULT_THRESHOLD) -> np.ndarray:
    """
    Gibt den Index zurück, an dem der Schwellenwert erstmals überschritten wird.

    Args:
        da (xr.DataArray): Input-Daten.
        threshold (int): Schwellenwert.

    Returns:
        np.ndarray: Array mit den Indizes des Schwellenwerts.
    """
    return np.array((da.loc[:, :, "Berührungswerte"] > threshold).idxmax(dim="row"))

def get_berührzeit(da: xr.DataArray, threshold_indices: np.ndarray, threshold_down=DEFAULT_THRESHOLD) -> np.ndarray:
    """
    Berechnet die Berührungszeit, also die Dauer zwischen Überschreiten und Unterschreiten des Schwellenwerts.

    Args:
        da (xr.DataArray): Input-Daten.
        threshold_indices (np.ndarray): Indizes, an denen der Schwellenwert überschritten wird.
        threshold_down (int): Schwellenwert für den Abfall.

    Returns:
        np.ndarray: Array mit den Berührzeiten.
    """
    berührzeiten = []
    for idx, threshold_index in enumerate(threshold_indices):
        condition_down = da.loc[idx, threshold_index:, "Berührungswerte"] < threshold_down
        threshold_down_index = condition_down.idxmax(dim="row")
        if threshold_down_index == 0:
            threshold_down_index = len(da.loc[idx, :, "Berührungswerte"]) - threshold_index
        berührzeiten.append(threshold_down_index - threshold_index)
    return np.array(berührzeiten) * DEFAULT_TIME_STEP

# Wenn die Presse das erste Mal das Bauteil berührt, wie hoch liegt die Presse da?
def get_höhe_wegmessung_aufprall(
    da: xr.DataArray, threshhold_index_array: np.ndarray
) -> np.ndarray:
    heightList = []
    for index in range(len(da)):
        first_point = da.loc[index, threshhold_index_array[index], "Wegmessung"]
        heightList.append(first_point.item())
    # Return the height list as np.array
    return np.array(heightList)

def get_tiefster_punkt(da: xr.DataArray) -> np.ndarray:
    """
    Gibt den minimalen Wert der Wegmessung (tiefster Punkt) zurück. Auch als unterer Totpunkt bezeichnet.

    Args:
        da (xr.DataArray): Input-Daten.

    Returns:
        np.ndarray: Tiefste Punkte.
    """
    return np.array(da.loc[:, :, "Wegmessung"].min(dim="row"))

def get_energie_aufprall_p(da: xr.DataArray, threshold_indices: np.ndarray) -> np.ndarray:
    """
    Berechnet einen Wert proportional zur Energie beim Aufprall.

    Args:
        da (xr.DataArray): Input-Daten.
        threshold_indices (np.ndarray): Indizes des Schwellenwerts.

    Returns:
        np.ndarray: Energie-Werte.
    """
    velocity_list = []
    for idx, index in enumerate(threshold_indices):
        try:
            first_point = da.loc[idx, index - 100, "Wegmessung"]
            second_point = da.loc[idx, index, "Wegmessung"]
            velocity = (second_point.item() - first_point.item()) / 100
            velocity_list.append(velocity)
        except IndexError:
            velocity_list.append(0)
    return np.square(np.array(velocity_list))

## Neue Funktionen

def identifiziere_fehlversuche(df: pd.DataFrame, threshold_diff: float = 10.0) -> pd.DataFrame:
    """
    Identifiziert Versuche, bei denen Verkippung_2 schlagartig abfällt.

    Args:
        df (pd.DataFrame): Input-Daten mit 'Verkippung_2'.
        threshold_diff (float): Maximale erlaubte Differenz zwischen aufeinanderfolgenden Werten.

    Returns:
        pd.DataFrame: DataFrame mit einer zusätzlichen Spalte "Fehlerhaft", die anzeigt, ob ein Versuch fehlerhaft ist.
    """
    # Berechne die Differenz zwischen aufeinanderfolgenden Werten in 'Verkippung_2'
    df['Delta_Verkippung_2'] = df['Verkippung_2'].diff().abs()
    
    # Kennzeichne Versuche als fehlerhaft, wenn die Differenz den Schwellenwert überschreitet
    df['Fehlerhaft'] = df['Delta_Verkippung_2'] > threshold_diff
    
    # Entferne die Hilfsspalte, wenn sie nicht benötigt wird
    df.drop(columns=['Delta_Verkippung_2'], inplace=True)
    
    return df

def entferne_fehlversuche(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entfernt Versuche, die als fehlerhaft markiert wurden.

    Args:
        df (pd.DataFrame): Input-Daten mit einer Spalte "Fehlerhaft".

    Returns:
        pd.DataFrame: Bereinigte Daten ohne fehlerhafte Versuche.
    """
    return df[df['Fehlerhaft'] == False].drop(columns=['Fehlerhaft'])

def calculate_druckberuehrzeit(da: xr.DataArray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """
    Berechnet die Druckberührzeit t_B.

    Args:
        da (xr.DataArray): Zeitreihendaten des Versuchs.
        threshold (float): Schwellenwert der Berührungswerte.

    Returns:
        np.ndarray: Druckberührzeit in Sekunden.
    """
    t0_indices = (da.loc[:, :, "Berührungswerte"] > threshold).idxmax(dim="row")
    t_down_indices = (da.loc[:, :, "Berührungswerte"] < threshold).idxmax(dim="row")
    return (t_down_indices - t0_indices) * DEFAULT_TIME_STEP


def calculate_umformzeit(t0_index: int, t_b_index: int, time_per_index: float = DEFAULT_TIME_STEP) -> float:
    """
    Berechnet die Umformzeit t_U.

    Args:
        t0_index (int): Index des Auftreffzeitpunkts.
        t_b_index (int): Index des tiefsten Punkts.
        time_per_index (float): Zeit pro Index.

    Returns:
        float: Umformzeit in Sekunden.
    """
    return (t_b_index - t0_index) * time_per_index


def calculate_auftreffzeitpunkt(da: xr.DataArray, threshold: float = DEFAULT_THRESHOLD) -> float:
    """
    Berechnet den Auftreffzeitpunkt t_0.

    Args:
        da (xr.DataArray): Zeitreihendaten des Versuchs.
        threshold (float): Schwellenwert der Berührungswerte.

    Returns:
        float: Auftreffzeitpunkt in Sekunden.
    """
    t0_index = (da.loc[:, :, "Berührungswerte"] > threshold).idxmax(dim="row")
    return t0_index * DEFAULT_TIME_STEP


def calculate_auftreffposition(da: xr.DataArray, t0_index: int) -> float:
    """
    Berechnet die Auftreffposition der Presse.

    Args:
        da (xr.DataArray): Zeitreihendaten des Versuchs.
        t0_index (int): Index des Auftreffzeitpunkts.

    Returns:
        float: Auftreffposition in mm.
    """
    return da.loc[:, t0_index, "Wegmessung"].values


def calculate_umformgeschwindigkeit(s0: float, sb: float, t_u: float) -> float:
    """
    Berechnet die Umformgeschwindigkeit v_U.

    Args:
        s0 (float): Auftreffposition in mm.
        sb (float): Position am tiefsten Punkt in mm.
        t_u (float): Umformzeit in Sekunden.

    Returns:
        float: Umformgeschwindigkeit in mm/s.
    """
    return (s0 - sb) / t_u


def calculate_auftreffgeschwindigkeit(da: xr.DataArray, t0_index: int) -> float:
    """
    Berechnet die Auftreffgeschwindigkeit v_0.

    Args:
        da (xr.DataArray): Zeitreihendaten des Versuchs.
        t0_index (int): Index des Auftreffzeitpunkts.

    Returns:
        float: Auftreffgeschwindigkeit in mm/s.
    """
    displacement = da.loc[:, t0_index, "Wegmessung"].values - da.loc[:, t0_index - 1, "Wegmessung"].values
    return displacement / DEFAULT_TIME_STEP



def calculate_v_max(da: xr.DataArray) -> np.ndarray:
    """
    Berechnet die maximale Stößelgeschwindigkeit v_max.

    Args:
        da (xr.DataArray): Zeitreihendaten des Versuchs.

    Returns:
        np.ndarray: Maximale Geschwindigkeit in mm/s.
    """
    velocities = da.loc[:, :, "Wegmessung"].differentiate("row") / DEFAULT_TIME_STEP
    return velocities.max(dim="row")


def calculate_v_min(da: xr.DataArray) -> np.ndarray:
    """
    Berechnet die minimale Stößelgeschwindigkeit v_min.

    Args:
        da (xr.DataArray): Zeitreihendaten des Versuchs.

    Returns:
        np.ndarray: Minimale Geschwindigkeit in mm/s.
    """
    velocities = da.loc[:, :, "Wegmessung"].differentiate("row") / DEFAULT_TIME_STEP
    return velocities.min(dim="row")


def calculate_e_ab(da: xr.DataArray) -> float:
    """
    Berechnet die abgegebene Energie E_Ab.

    Args:
        da (xr.DataArray): Zeitreihendaten des Versuchs.

    Returns:
        float: Abgegebene Energie in J.
    """
    force = da.loc[:, :, "Presskraft"].values
    displacement = da.loc[:, :, "Wegmessung"].diff(dim="row").values
    energy = (force[:-1] * displacement).sum()
    return energy


def calculate_stoessel_verkippung(da: xr.DataArray, lm_x: float = MEAN_DISTANCE_VERKIPPUNG_X, lm_y: float = MEAN_DISTANCE_VERKIPPUNG_Y) -> tuple[np.ndarray, np.ndarray]:
    """
    Berechnet die Stößelkippung um die X- und Y-Achse basierend auf den Signalen der vier Verkippungssensoren.

    Args:
        da (xr.DataArray): Zeitreihendaten des Versuchs.
        lm_x (float): Abstand der Sensoren in X-Richtung (in mm).
        lm_y (float): Abstand der Sensoren in Y-Richtung (in mm).

    Returns:
        tuple: Kippung um die X-Achse und Y-Achse (in rad oder mm je nach Einheit der Messwerte).
    """
    # Verkippung um die X-Achse (k_A)
    x_kippung = ((da.loc[:, :, "Verkippung_3"] + da.loc[:, :, "Verkippung_1"]) -  # Sensor 3 + Sensor 1
                 (da.loc[:, :, "Verkippung_4"] + da.loc[:, :, "Verkippung_2"])) / (2 * lm_y)  # Sensor 4 + Sensor 2

    # Verkippung um die Y-Achse (k_B)
    y_kippung = ((da.loc[:, :, "Verkippung_2"] + da.loc[:, :, "Verkippung_1"]) -  # Sensor 2 + Sensor 1
                 (da.loc[:, :, "Verkippung_4"] + da.loc[:, :, "Verkippung_3"])) / (2 * lm_x)  # Sensor 4 + Sensor 3

    return x_kippung, y_kippung


def calculate_min_motorstrom(da: xr.DataArray) -> np.ndarray:
    """
    Berechnet den minimalen Motorstrom I_min.

    Args:
        da (xr.DataArray): Zeitreihendaten des Versuchs.

    Returns:
        np.ndarray: Minimaler Motorstrom in A.
    """
    return da.loc[:, :, "Motorstrom"].min(dim="row")


def calculate_mean_motorstrom(da: xr.DataArray) -> np.ndarray:
    """
    Berechnet den durchschnittlichen Motorstrom I_T.

    Args:
        da (xr.DataArray): Zeitreihendaten des Versuchs.

    Returns:
        np.ndarray: Durchschnittlicher Motorstrom in A.
    """
    return da.loc[:, :, "Motorstrom"].mean(dim="row")

# -----------------------
# Klasse zur Vorbereitung der Rohdaten 
# Nutzen: Feature-Berechnung, Datenorganisation
# -----------------------

class ProcessedData:
    """
    Wandelt Rohdaten aus einem xarray.DataArray in berechnete Features um.
    Optionale Matrizen können bei Bedarf initialisiert werden.
    """

    def __init__(self, da: xr.DataArray, init_matrix=True):
        """
        Initialisiert die Klasse mit den Rohdaten und berechnet Standardfeatures.

        Args:
            da (xr.DataArray): Input-Daten.
            init_matrix (bool): Ob Matrizen direkt initialisiert werden sollen.
        """
        self.raw_features = da.loc[:, :, FEATURE_COLUMNS]
        self.raw_features_fillna0 = self.raw_features.fillna(0)
        self.schwellwert_index = get_schwellwert_index(self.raw_features)
        self.berührzeit = get_berührzeit(self.raw_features, self.schwellwert_index)
        self.tiefster_punkt = get_tiefster_punkt(self.raw_features)
        self.energie_aufprall_p = get_energie_aufprall_p(self.raw_features, self.schwellwert_index)
        self.press_geschwindigkeit = None  # Wird später initialisiert.

        # Optional initialisierbare Matrizen
        self.daten_matrix: xr.DataArray | None = None
        self.daten_matrix_100: xr.DataArray | None = None
        self.daten_matrix_500: xr.DataArray | None = None

        if init_matrix:
            self.init_matrix_daten()

    def init_matrix_daten(self) -> None:
        """
        Initialisiert Matrizen, die für die Verarbeitung benötigt werden.
        """
        self.daten_matrix = self.raw_features[:, self.schwellwert_index:, :]
        self.daten_matrix_100 = self.daten_matrix[:, :100, :]
        self.daten_matrix_500 = self.daten_matrix[:, :500, :]

    def get_features_mit_namen(self, skalar_feature_namen: List[str], matrix_feature_namen: List[str]) -> np.ndarray:
        """
        Gibt die ausgewählten Features basierend auf Namen zurück.

        Args:
            skalar_feature_namen (List[str]): Namen der skalaren Features.
            matrix_feature_namen (List[str]): Namen der Matrizen-Features.

        Returns:
            np.ndarray: Array mit den ausgewählten Features.
        """
        features = [self.__dict__[name] for name in skalar_feature_namen]
        np_features = np.column_stack(features)

        for name in matrix_feature_namen:
            matrix = self.__dict__[name]
            if matrix is not None:
                np_features = np.hstack((np_features, matrix.reshape(len(matrix), -1)))

        return np_features

    def zeig_alle_namen_von_features(self) -> List[str]:
        """
        Gibt alle verfügbaren Feature-Namen in der Klasse zurück.

        Returns:
            List[str]: Namen der Features.
        """
        return list(self.__dict__.keys())
    

    ## Neue Funktionen
    
    def synchronisiere_verkippungssensoren(self, sensor_columns: list, threshold: float = 0.1):
        """
        Synchronisiert die Verkippungssensoren, indem die Zeitpunkte des Messbeginns für jeden Sensor
        erkannt und die Daten entsprechend verschoben werden.

        Args:
            sensor_columns (list): Liste der Spaltennamen der Verkippungssensoren.
            threshold (float): Schwellenwert, ab dem ein Sensor als aktiv gilt.
        """
        # Erstelle eine Kopie der Features für die Synchronisation
        synced_features = self.raw_features.copy()

        # Finde den Startzeitpunkt für jeden Sensor
        start_indices = {}
        for sensor in sensor_columns:
            start_index = synced_features.loc[:, :, sensor].gt(threshold).idxmax(dim="row")
            start_indices[sensor] = start_index

        # Bestimme den maximalen Startindex
        max_start_index = max(start_indices.values())

        # Verschiebe die Daten jedes Sensors
        for sensor in sensor_columns:
            shift_amount = max_start_index - start_indices[sensor]
            if shift_amount > 0:
                synced_features.loc[:, :, sensor] = synced_features.loc[:, :, sensor].shift(shift_amount, dim="row")
        
        # Entferne Zeilen mit NaN-Werten, die durch das Shiften entstanden sind
        self.raw_features = synced_features.dropna(dim="row")


    @staticmethod
    def turn_pandas_to_processed_data(pd_df: pd.DataFrame, init_matrix=False) -> "ProcessedData":
        """
        Wandelt einen Pandas-DataFrame in ein ProcessedData-Objekt um.

        Args:
            pd_df (pd.DataFrame): Input-Daten.
            init_matrix (bool): Ob Matrizen initialisiert werden sollen.

        Returns:
            ProcessedData: Instanz mit den verarbeiteten Daten.
        """
        xr_data = ProcessedData.turn_pandas_to_xArray(pd_df)
        return ProcessedData(xr_data, init_matrix=init_matrix)

    @staticmethod
    def turn_pandas_to_xArray(pd_df: pd.DataFrame) -> xr.DataArray:
        """
        Wandelt einen Pandas-DataFrame in ein xarray.DataArray um.

        Args:
            pd_df (pd.DataFrame): Input-Daten.

        Returns:
            xr.DataArray: Umgewandelte Daten.
        """
        return xr.DataArray(
            pd_df.values,
            dims=("row", "col"),
            coords={"row": pd_df.index, "col": pd_df.columns},
        ).expand_dims(dim={"versuch": 1}, axis=0)