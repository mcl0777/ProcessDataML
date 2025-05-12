import math
from typing import Any, List
import numpy as np
import pandas as pd
import xarray as xr


threshold = 9
threshold_length = 2000
time_each_index = 0.5


feature_columns = [
    "Zeit",
    "Berührungswerte",
    "Wegmessung",
    "Verkippung_1",
    "Verkippung_2",
    "Verkippung_3",
    "Verkippung_4",
    "Stoesselhub",
    "Presskraft",
    "Geschwindigkeit_UT",
    "Geschwindigkeit_Ges",
    "Energie_ab",
    "Presskraft_dyn",
    "Motorstrom",
    # "Kraft1",
    # "Kraft2",
    "Bauteil_temp_con",
    "Werkzeug_temp_con",
    # "Material_con",
    # "Position_con",
    # "Ergebnis_con",
    # "Probenhoehe",
]


def clean_data(da: xr.DataArray) -> xr.DataArray:
    da = da.loc[(da.loc[:, :, "Berührungswerte"] > threshold).any(dim="row"), :, :]
    da = da.loc[
        da.loc[:, :, "Zeit"].groupby("versuch").count(dim="row") > threshold_length,
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


# Der Index wo der Schwellwert zum ersten Mal überschritten wird
def get_schwellwert_index(da: xr.DataArray) -> np.ndarray:
    index_threshold_reached = (da.loc[:, :, "Berührungswerte"] > threshold).idxmax(
        dim="row"
    )

    return np.array(index_threshold_reached)


# Gibt die Druckberührzeit zurück
def get_berührzeit(
    da: xr.DataArray, threshhold_index_array: np.ndarray, thresholdDown=9
) -> np.ndarray:
    # Create a list of the Berührungswerte(This
    # is not an index but a normal value)
    b_werte_list = []
    for iteration, index_threshold in enumerate(threshhold_index_array):
        # Check where the value dips below the threshold down
        # value for the first time after the "up" threshold
        condition_down = (
            da.loc[iteration, index_threshold:, "Berührungswerte"] < thresholdDown
        )
        index_threshold_down_reached = condition_down.idxmax(dim="row")
        if index_threshold_down_reached == 0:
            index_threshold_down_reached = (
                len(da.loc[iteration, :, "Berührungswerte"]) - index_threshold
            )

        # Calculate the B_Wert. (Second threshold timepoint - first timepoint)
        b_werte_list.append(index_threshold_down_reached - index_threshold)

    return np.array(b_werte_list) * time_each_index


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


# Die Zeit zwischen, wo der Schwellwert zum ersten Mal überschritten wird und
# dem tiefsten Punkt
def get_umformzeit(da: xr.DataArray) -> np.ndarray:
    index_threshold_reached = (da.loc[:, :, "Berührungswerte"] > threshold).idxmax(
        dim="row"
    )
    lowest_point_index = da.loc[:, :, "Verkippung_1"].idxmax(dim="row")
    return np.array(lowest_point_index - index_threshold_reached) * time_each_index


# Gibt den tiefsten Wegmesspunkt zurück
#def get_tiefster_punkt(da: xr.DataArray) -> np.ndarray:
#    return np.array(da.loc[:, :, "Wegmessung"].min(dim="row"))

######## NEU ######
def get_tiefster_punkt(da: xr.DataArray) -> np.ndarray:
    """
    Berechnet die minimalen Werte der 'Wegmessung' für jeden Versuch,
    beschränkt auf die ersten 2000 Indizes.

    Args:
        da (xr.DataArray): Das DataArray mit den Daten.

    Returns:
        np.ndarray: Array mit den minimalen Werten.
    """
    try:
        # Begrenze die Daten auf die ersten 2000 Indizes
        restricted_da = da.loc[:, :threshold_length, "Wegmessung"]
        
        # Berechne das Minimum entlang der 'row'-Dimension
        min_values = restricted_da.min(dim="row", skipna=True).values
        return np.array(min_values)
    except Exception as e:
        print(f"Fehler bei der Berechnung des tiefsten Punkts: {e}")
        return np.full(da.sizes["versuch"], np.nan)
    

#############

# Gibt einen Wert zurück, der Proportional zur Energie ist
def get_energie_aufprall_p(
    da: xr.DataArray, threshhold_index_array: np.ndarray
) -> np.ndarray:
    velocityList = []
    for index in range(len(da)):
        first_point = da.loc[index, threshhold_index_array[index] - 100, "Wegmessung"]
        second_point = da.loc[index, threshhold_index_array[index], "Wegmessung"]
        # Get a value that is propotional to the velocity
        p_velocity = (second_point.item() - first_point.item()) / 100
        velocityList.append(p_velocity)
    # Return a value that is proportional to the energy
    return np.power(np.array(velocityList), 2)



# Berechnet ein Geschwindigkeitsdreieck vom Auftreffpunkt bis zum tiefsten Punkt
# geteilt durch die Zeit. Dieser Wert hat keine wirkliche Einheit
def get_press_geschwindigkeit(
    daten_matrix: xr.DataArray, tiefsten_punkt: np.ndarray, umformzeit: np.ndarray
) -> np.ndarray:
    weg = daten_matrix.loc[:, 0, "Wegmessung"].values - tiefsten_punkt[:]
    return weg / umformzeit[:]


def get_bauteil_temp(da: xr.DataArray) -> np.ndarray:
    return np.array(da.loc[:, 0, "Bauteil_temp_con"])


def get_werkzeug_temp(da: xr.DataArray) -> np.ndarray:
    return np.array(da.loc[:, 0, "Werkzeug_temp_con"])


# Die durschnittliche Höhe der Energie (nutzlos)
def get_energie_ab_durchschnitt(da: xr.DataArray) -> np.ndarray:
    return np.array(da.loc[:, :, "Energie_ab"].mean(dim="row"))


# Die durschnittliche Höhe vom Motorstrom (nutzlos)
def get_motorstrom_durchschnitt(da: xr.DataArray) -> np.ndarray:
    return np.array(da.loc[:, :, "Motorstrom"].mean(dim="row"))



class ProcessedData:
    """
    Turns the sensor data into features the model can work with.
    If you set init_matrix to False you have to call "init_matrix_daten()" afterwards.
    """
    def __init__(self, da: xr.DataArray, init_matrix=True):
        raw_features = da.loc[:, :, feature_columns]
        self.versuch_ids = da.sel(col="datei_name").isel(row=0).values # neu
        self.raw_features = raw_features
        self.raw_features_fillna0 = raw_features.fillna(0)
        self.schwellwert_index = get_schwellwert_index(raw_features)
        self.berührzeit = get_berührzeit(raw_features, self.schwellwert_index)
        self.umformzeit = get_umformzeit(raw_features)
        self.tiefster_punkt = get_tiefster_punkt(raw_features)
        self.energie_aufprall_p = get_energie_aufprall_p(
            raw_features, self.schwellwert_index
        )
        # self.B_WertMax = getB_WertMax(raw_features)
        self.höhe_wegmessung_aufprall = get_höhe_wegmessung_aufprall(
            raw_features, self.schwellwert_index
        )

        # Daten von der Presse direkt
        self.motorstrom_durchschnitt = get_motorstrom_durchschnitt(raw_features)
        self.energie_ab_durchschnitt = get_energie_ab_durchschnitt(raw_features)
        self.werkzeug_temp = get_werkzeug_temp(raw_features)
        self.bauteil_temp = get_bauteil_temp(raw_features)

        self.daten_matrix: None | xr.DataArray = None
        self.daten_matrix_100: None | xr.DataArray = None
        self.daten_matrix_500: None | xr.DataArray = None
        self.daten_matrix_1000: None | xr.DataArray = None
        self.daten_matrix_bevor_schwellwert_100: None | xr.DataArray = None

        self.verkippung_matrix: None | np.ndarray = None
        self.hochfahrweg_matrix: None | np.ndarray = None
        self.verkippung_differenz: None | np.ndarray = None

        self.press_geschwindigkeit = np.array([0])

        if init_matrix:
            self.init_matrix_daten()

    
    def get_features_mit_namen(
        self, skalar_feature_namen: List[str], matrix_feature_namen: List[str]
    ) -> np.ndarray:
        """
        This lets you choose which data should be passed to the model.

        Args:
            skalar_feature_namen (List[str]): The single scalar data you want to use.
            Available options can be viewed with zeig_alle_namen_von_features()

            matrix_feature_namen (List[str]): The dataArray data you want to use (2d-Matrix Data).

        Returns:
            np.ndarray: features
        """
        features = []
        for name in skalar_feature_namen:
            features.append(self.__dict__[name])

        np_features = np.array(features).transpose()

        for name_da in matrix_feature_namen:
            if len(np_features) > 0:
                np_features = np.hstack(
                    (
                        np_features,
                        np.array(self.__dict__[name_da]).reshape(
                            len(self.__dict__[name_da]), -1
                        ),
                    )
                )
            else:
                np_features = np.array(self.__dict__[name_da]).reshape(
                    len(self.__dict__[name_da]), -1
                )

        return np_features

    def combine_features(self, feature_arrays: List[np.ndarray]) -> np.ndarray:
        for iteration, new_feature in enumerate(feature_arrays):
            if iteration == 0:
                np_features = new_feature
            else:
                np_features = np.hstack(
                    (
                        np_features,
                        new_feature,
                    )
                )

        return np_features

    def zeig_alle_namen_von_features(self):
        return list(self.__dict__.keys())

    @staticmethod
    def turn_pandas_to_processed_data(
        pd_df: pd.DataFrame, init_matrix=False
    ) -> "ProcessedData":
        xr_data = ProcessedData.turn_pandas_to_xArray(pd_df)
        return ProcessedData(xr_data, init_matrix=init_matrix)

    @staticmethod
    def turn_pandas_to_xArray(pd_df: pd.DataFrame) -> xr.DataArray:
        da_2d = xr.DataArray(
            pd_df.values,
            dims=("row", "col"),
            coords={"row": pd_df.index, "col": pd_df.columns},
        )
        da_3d = da_2d.expand_dims(dim={"versuch": 1}, axis=0)

        # This step wont be needed in production since there will only be input
        # data and no labeled data.
        return da_3d
