import os, sys
from os.path import dirname
import pickle
from typing import List
import numpy as np
import xarray as xr  # type:ignore
import pandas as pd

from helpers.process_data import (
    ProcessedData,
    clean_data,
    feature_columns,
)  # type:ignore

# Der übergeordnete Ordner --> für Import von Modulen
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

DATA_PATH = f"{parent_dir}/training_data/csv"
DATA_PATH_EXCEL = f"{parent_dir}/training_data/excel"


label_columns = [
    "Material_con",
]


list_dir_verbaut = [
    ("Werkzeug falsch verbaut", False),
    ("Werkzeug richtig verbaut", True),
    ("Werkzeug richtig verbaut extra", True),
]


def get_features(da):
    return da.loc[:, :, feature_columns]


def get_labels_ergebnis(da):
    return np.where(da.loc[:, 0, "Ergebnis_con"] == "GUT", True, False)


def get_labels_verbaut(da):
    return np.where(da.loc[:, 0, "richtig_verbaut"], True, False)


def get_labels_material(da):
    return np.where(da.loc[:, 0, "Material_con"] == "C45", True, False)


def get_labels_position(da):
    return np.where(da.loc[:, 0, "Position_con"] == "Mitte", True, False)


def get_labels_probenhoehe(da):
    return np.array(da.loc[:, 0, "Probenhoehe"])

############################################
def get_labels_bauteil_temp(da):
    return np.array(da.loc[:, 0, "Bauteil_temp_con"])


def read_all_data(path=DATA_PATH, read_extra_data=False) -> xr.DataArray:
    print(f"Reading data from {path}")
    data_arrays = []
    for dir_name, b_richtig_verbaut in list_dir_verbaut[: (2 + read_extra_data)]:
        data_path_with_verbaut = f"{path}/{dir_name}"
        dir_list = [
            name
            for name in os.listdir(data_path_with_verbaut)
            if name.startswith("Versuchstag")
        ]
        for child_dir in dir_list:
            file_list = os.listdir(f"{path}/{dir_name}/{child_dir}")
            for file_name in file_list:
                pd_df = pd.read_csv(f"{path}/{dir_name}/{child_dir}/{file_name}")
                pd_df["richtig_verbaut"] = b_richtig_verbaut
                pd_df["datei_name"] = file_name
                pd_df["versuchstag"] = child_dir
                data_arrays.append(
                    xr.DataArray(
                        pd_df.values,
                        dims=("row", "col"),
                        coords={"row": pd_df.index, "col": pd_df.columns},
                    )
                )

    return xr.concat(data_arrays, dim="versuch")


# def read_all_data_excel(path=DATA_PATH_EXCEL, read_extra_data=False) -> xr.DataArray:
#     print(f"Reading data from {path}")
#     data_arrays = []
#     for name, richtig_verbaut in list_dir_verbaut[: (2 + read_extra_data)]:
#         data_path_with_verbaut = f"{path}/{name}/"
#         dir_list = [
#             name
#             for name in os.listdir(data_path_with_verbaut)
#             if name.startswith("Versuchstag")
#         ]
#         for child_dir in dir_list:
#             file_names = [
#                 name
#                 for name in os.listdir(f"{data_path_with_verbaut}{child_dir}")
#                 if name.startswith("Autopress_")
#             ]
#             for file_name in file_names:
#                 df = pd.read_excel(f"{data_path_with_verbaut}{child_dir}/{file_name}")
#                 df["richtig_verbaut"] = richtig_verbaut
#                 df["datei_name"] = file_name
#                 data_arrays.append(
#                     xr.DataArray(
#                         df.values,
#                         dims=("row", "col"),
#                         coords={"row": df.index, "col": df.columns},
#                     )
#                 )

#     return xr.concat(data_arrays, dim="versuch")
