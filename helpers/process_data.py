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


# def getB_WertMax(da):
#     return np.array(da.loc[:, :, "Berührungswerte"].max(dim="row"))


# Alle Daten als Matrix mit dem Schwellwert-index als Startpunkt
def get_daten_matrix(
    da: xr.DataArray, threshhold_index_array: np.ndarray, size=1
) -> xr.DataArray:
    result = []
    for iteration, th_index in enumerate(threshhold_index_array):
        end_index = th_index + size
        new_da = da[iteration, th_index:end_index, :]
        result.append(new_da.reset_index(dims_or_levels="row", drop=True))
    return xr.concat(result, dim="versuch")


# Alle Daten als Matrix mit dem Schwellwert-index als Endpunkt
def get_daten_matrix_bevor_schwellwert(
    da: xr.DataArray, threshhold_index_array: np.ndarray, size=1
) -> xr.DataArray:
    result = []
    for iteration, th_index in enumerate(threshhold_index_array):
        start_index = th_index - size
        new_da = da[iteration, start_index:th_index, :]
        result.append(new_da.reset_index(dims_or_levels="row", drop=True))
    return xr.concat(result, dim="versuch")


def get_hochfahrweg_matrix(da: xr.DataArray, size=1) -> np.ndarray:
    lowest_point_array = da.loc[:, :, "Wegmessung"].idxmin(dim="row").to_numpy()
    test = []
    for index, lowest_point in enumerate(lowest_point_array):
        # These are 500 values but xarray is inclusive it seems thats why we use 499
        test.append(
            da.loc[
                index, lowest_point : lowest_point + size - 1, "Wegmessung"
            ].to_numpy()
        )

    return np.array(test)


# Rechnet 100 Datenpunkte aus. Nimmt den Verkippungswert beim Schwellindex und zieht diesen Wert von allen späteren Schnitten ab.
# Pro Verkippungssensor 100 Schnitte.
def get_verkippung_matrix(
    da: xr.DataArray, threshhold_index_array: np.ndarray, druckzeit_list: np.ndarray
) -> np.ndarray:
    verkippung_array = ["Verkippung_1", "Verkippung_3", "Verkippung_4"]
    anzahl_datenstriche = 100 + 1
    verkippungs_list = []
    for index in range(len(da)):
        timeSpan = druckzeit_list[index] / anzahl_datenstriche
        firstTimePoint = threshhold_index_array[index]
        first_point = da.loc[index, firstTimePoint, verkippung_array].to_numpy()

        schnitt_indexes = (
            np.arange(1, anzahl_datenstriche) * timeSpan + firstTimePoint
        ).astype(int)

        second_point = da.loc[index, schnitt_indexes, verkippung_array].to_numpy()

        verkippungs_list.append(second_point - first_point)

    return np.array(verkippungs_list)


def get_verkippung_differenz(da: xr.DataArray, anzahl_werte=2500) -> np.ndarray:
    # verkippungs_list
    v_l = ["Verkippung_1", "Verkippung_2", "Verkippung_3", "Verkippung_4"]

    verkippungs_feature_list = []
    index_pairs = [(0, 2), (2, 3)]
    # Rechne die durchschnittliche Differenz zwischen dem 1 und 3 und zwischen dem 2 und 4 Sensor aus.
    for index_1, index_2 in index_pairs:
        value1 = da.loc[:, 500:anzahl_werte, v_l[index_1]].to_numpy()
        value2 = da.loc[:, 500:anzahl_werte, v_l[index_2]].to_numpy()
        value_result = (value1 - value2).mean(axis=1)
        verkippungs_feature_list.append(value_result)
    # Ein feature mit zwei Werten
    verkippungs_feature = np.stack(verkippungs_feature_list, axis=1)
    return verkippungs_feature


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

    def init_matrix_daten(self) -> None:
        """
        This has to be called to be able to work with the DataArray
        features and the pressGeschwindigkeit. This is seperated from the init method
        because errors can occur while initializing the matrix data.
        """
        self.daten_matrix = get_daten_matrix(self.raw_features, self.schwellwert_index)

        self.daten_matrix_1000 = get_daten_matrix(
            self.raw_features, self.schwellwert_index, 1000
        )

        self.daten_matrix = self.daten_matrix_1000[:, :1, :]
        self.daten_matrix_100 = self.daten_matrix_1000[:, :100, :]
        self.daten_matrix_500 = self.daten_matrix_1000[:, :500, :]

        self.daten_matrix_bevor_schwellwert_100 = get_daten_matrix_bevor_schwellwert(
            self.raw_features, self.schwellwert_index, 100
        )
        self.hochfahrweg_matrix = get_hochfahrweg_matrix(self.raw_features, size=500)

        self.verkippung_matrix = get_verkippung_matrix(
            self.raw_features, self.schwellwert_index, self.umformzeit
        )

        self.verkippung_differenz = get_verkippung_differenz(self.raw_features)

        self.press_geschwindigkeit = get_press_geschwindigkeit(
            self.daten_matrix, self.tiefster_punkt, self.umformzeit
        )

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
