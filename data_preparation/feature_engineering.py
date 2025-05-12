import xarray as xr
import numpy as np
import pandas as pd

from helpers.process_data import ProcessedData, get_schwellwert_index, get_berührzeit

# Konstante Werte
DEFAULT_THRESHOLD = 9
DEFAULT_TIME_STEP = 0.5 # in ms
MEAN_DISTANCE_VERKIPPUNG_X = 745.0  # in mm
MEAN_DISTANCE_VERKIPPUNG_Y = 950.0  # in mm




class FeatureEngineering:
    """
    Diese Klasse implementiert verschiedene Feature-Engineering-Methoden für ein xarray.DataArray.
    """

    def __init__(self, data_array: xr.DataArray, threshold: float = DEFAULT_THRESHOLD):
        """
        Initialisiert die FeatureEngineering-Klasse mit einem xarray.DataArray.

        Args:
            data_array (xr.DataArray): Die Eingabedaten für das Feature Engineering.
            threshold (float): Schwellenwert für Berührungswerte.
        """
        self.data_array = data_array
        self.threshold = threshold
        self.versuch_ids = data_array.sel(col="datei_name").isel(row=0).values

    # ----------------- Wegmessung -----------------
    def calculate_tiefster_punkt_idx(self) -> np.ndarray:
        """
        Berechnet die Indizes des tiefsten Punkts der Wegmessung.

        Returns:
            np.ndarray: Indizes des tiefsten Punkts.
        """
        indices = self.data_array.sel(col="Wegmessung").argmin(dim="row").values
        if np.isscalar(indices):  # Falls nur ein einzelner Wert zurückgegeben wird
            indices = np.array([indices])
        # Sicherstellen, dass Indizes ganzzahlig sind
        indices = indices.astype(int)
        return indices

    # ----------------- Zeitbezogene Features -----------------
    def calculate_druckberuehrzeit(self, time_step=0.5) -> np.ndarray:
        """
        Berechnet die Druckberührzeit t_B, d.h. die Zeitspanne von t0 bis zum Maximum der Presskraft_dyn.

        Args:
            time_step (float): Zeitintervall für die Berechnung.

        Returns:
            np.ndarray: Ein Array mit den Druckberührzeiten für jeden Versuch.
        """
        try:
            # Berechne die t0-Indizes
            t0_indices = self.calculate_t0_indices()  # Funktion zur Berechnung von t0

            # Extrahiere die "Presskraft_dyn" aus dem DataArray
            presskraft_dyn = self.data_array.sel(col="Presskraft_dyn").values  # Shape: (num_versuche, num_rows)
            num_versuche, num_rows = presskraft_dyn.shape

            # Initialisiere Array für Druckberührzeiten
            druckberuehrzeiten = np.full(num_versuche, np.nan)  # Standardmäßig NaN

            for i in range(num_versuche):
                # Skip if t0 index is invalid
                if t0_indices[i] == -1 or t0_indices[i] >= num_rows:
                    continue

                # Werte nach t0
                values_after_t0 = presskraft_dyn[i, t0_indices[i]:]

                # Finde den Index des Maximums von Presskraft_dyn nach t0
                max_index_relative = np.argmax(values_after_t0)  # Relativer Index zu t0
                max_index_absolute = t0_indices[i] + max_index_relative  # Absoluter Index

                # Berechne Druckberührzeit
                druckberuehrzeiten[i] = (max_index_absolute - t0_indices[i]) * time_step

                # Debugging
                print(f"Versuch {i}:")
                print(f"t0_index: {t0_indices[i]}, max_index: {max_index_absolute}")
                print(f"values_after_t0[:10]: {values_after_t0[:10]}")
                print(f"Druckberührzeit: {druckberuehrzeiten[i]} ms")

            return druckberuehrzeiten
        except Exception as e:
            print(f"Fehler bei der Berechnung der Druckberührzeit: {e}")
            return np.full(self.data_array.sizes["versuch"], np.nan)
    
    def calculate_auftreffzeitpunkt(self) -> np.ndarray:
        """Berechnet den Auftreffzeitpunkt t_0."""
        t0_indices = (self.data_array.loc[:, :, "Berührungswerte"] > self.threshold).idxmax(dim="row"                                                                                )
        return t0_indices * DEFAULT_TIME_STEP

    def calculate_umformzeit(self, t0_indices: np.ndarray, t_b_indices: np.ndarray) -> np.ndarray:
        """Berechnet die Umformzeit t_U."""
        return (t_b_indices - t0_indices) * DEFAULT_TIME_STEP

    # ----------------- Geschwindigkeitsfeatures -----------------
    def calculate_auftreffgeschwindigkeit(self, t0_indices: np.ndarray) -> np.ndarray:
        """Berechnet die Auftreffgeschwindigkeit v_0."""
        displacement = (
            self.data_array.loc[:, t0_indices, "Wegmessung"].values
            - self.data_array.loc[:, t0_indices - 1, "Wegmessung"].values
        )
        return displacement / DEFAULT_TIME_STEP

    def calculate_v_max(self) -> np.ndarray:
        """Berechnet die maximale Stößelgeschwindigkeit v_max."""
        velocities = self.data_array.loc[:, :, "Wegmessung"].differentiate("row") / DEFAULT_TIME_STEP
        return velocities.max(dim="row")

    def calculate_v_min(self) -> np.ndarray:
        """Berechnet die minimale Stößelgeschwindigkeit v_min."""
        velocities = self.data_array.loc[:, :, "Wegmessung"].differentiate("row") / DEFAULT_TIME_STEP
        return velocities.min(dim="row")

    # ----------------- Energie -----------------
    def calculate_e_ab(self) -> np.ndarray:
        """Berechnet die abgegebene Energie E_Ab. Eher Arbeit"""
        force = self.data_array.loc[:, :, "Presskraft"].values
        displacement = self.data_array.loc[:, :, "Wegmessung"].diff(dim="row").values
        energy = (force[:, :-1] * displacement).sum(axis=1)
        return energy

    # ----------------- Motorstrom -----------------
    def calculate_min_motorstrom(self) -> np.ndarray:
        """Berechnet den minimalen Motorstrom I_min."""
        return self.data_array.loc[:, :, "Motorstrom"].min(dim="row")
    
    def calculate_max_motorstrom(self) -> np.ndarray:
        """Berechnet den maximalen Motorstrom I_max."""
        return self.data_array.loc[:, :, "Motorstrom"].max(dim="row")

    def calculate_mean_motorstrom(self) -> np.ndarray:
        """Berechnet den durchschnittlichen Motorstrom I_T."""
        return self.data_array.loc[:, :, "Motorstrom"].mean(dim="row")

    # ----------------- Auftreffposition s_0 -----------------
    def calculate_auftreffposition(self) -> np.ndarray:
        """
        Berechnet die Auftreffposition s_0 beim Berührzeitpunkt t_0.

        Returns:
            np.ndarray: Auftreffposition der Presse.
        """
        t0_indices = (self.data_array.loc[:, :, "Berührungswerte"] > self.threshold).idxmax(dim="row")
        return self.data_array.loc[:, t0_indices, "Wegmessung"].values

    # ----------------- Umformgeschwindigkeit v_U -----------------
    def calculate_umformgeschwindigkeit(self, s0: np.ndarray, t_b: np.ndarray, t_u: np.ndarray) -> np.ndarray:
        """
        Berechnet die Umformgeschwindigkeit v_U.

        Args:
            s0 (np.ndarray): Auftreffposition.
            t_b (np.ndarray): Tiefster Punkt (unterer Totpunkt).
            t_u (np.ndarray): Umformzeit.

        Returns:
            np.ndarray: Umformgeschwindigkeit.
        """
        return (s0 - t_b) / t_u

    # ----------------- Auftreffgeschwindigkeit v_0 -----------------
    def calculate_auftreffgeschwindigkeit(self, t0_indices: np.ndarray) -> np.ndarray:
        """
        Berechnet die Auftreffgeschwindigkeit.

        Args:
            t0_indices (np.ndarray): Indizes des Auftreffzeitpunkts.

        Returns:
            np.ndarray: Auftreffgeschwindigkeiten.
        """
        velocities = []

        for versuch_idx, idx in enumerate(t0_indices):
            try:
                if 0 <= idx < self.data_array.sizes["row"]:  # Stelle sicher, dass der Index gültig ist
                    # Berechnung der Verschiebung
                    displacement = (
                        self.data_array.isel(versuch=versuch_idx, row=idx, col=self.data_array.coords["col"].to_index().get_loc("Wegmessung")).values -
                        self.data_array.isel(versuch=versuch_idx, row=idx - 1, col=self.data_array.coords["col"].to_index().get_loc("Wegmessung")).values
                    )
                    # Berechnung der Geschwindigkeit
                    velocity = displacement / DEFAULT_TIME_STEP
                    velocities.append(velocity)
                else:
                    velocities.append(np.nan)  # Ungültige Indizes mit NaN auffüllen
            except Exception as e:
                print(f"Fehler bei der Verarbeitung des Versuch {versuch_idx}, Index {idx}: {e}")
                velocities.append(np.nan)

        return np.array(velocities)

    # ----------------- Berührungsdauer -----------------
    def calculate_beruehrungsdauer(self, t0_indices: np.ndarray, t_down_indices: np.ndarray) -> np.ndarray:
        """
        Berechnet die Berührungsdauer bis zum tiefsten Punkt.

        Args:
            t0_indices (np.ndarray): Auftreffzeitpunkte (Schwellenüberschreitung).
            t_down_indices (np.ndarray): Zeitpunkte, an denen der Schwellenwert wieder unterschritten wird.

        Returns:
            np.ndarray: Berührungsdauer in Sekunden.
        """
        try:
            # Gültige Maskierung sicherstellen
            valid_mask = (t0_indices >= 0) & (t_down_indices >= 0) & (t_down_indices > t0_indices)
            
            # Berechnung der Berührungsdauer nur für gültige Indizes
            beruehrungsdauern = np.full(t0_indices.shape, np.nan, dtype=float)
            beruehrungsdauern[valid_mask] = (t_down_indices[valid_mask] - t0_indices[valid_mask]) * DEFAULT_TIME_STEP
            
            return beruehrungsdauern
        except Exception as e:
            print(f"Fehler bei der Berechnung der Berührungsdauer: {e}")
            return np.full(t0_indices.shape, np.nan, dtype=float)


    # ----------------- Maximaler Druck und Zeitpunkt -----------------
    def calculate_maximaler_druck(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Berechnet den maximalen Druck (Presskraft_dyn) und dessen Zeitpunkt während der Berührungszeit.

        Returns:
            tuple[np.ndarray, np.ndarray]: Maximale Druckwerte und Zeitpunkte.
        """
        try:
            # Zugriff auf "Presskraft_dyn"
            if "Presskraft_dyn" not in self.data_array.coords["col"].values:
                raise KeyError("Spalte 'Presskraft_dyn' nicht gefunden.")

            presskraft_dyn = self.data_array.sel(col="Presskraft_dyn")
            beruehrungswerte = self.data_array.sel(col="Berührungswerte")

            max_druck = []
            max_druck_time = []

            # Iteration über jeden Versuch
            for idx in range(len(self.versuch_ids)):
                try:
                    # Maske basierend auf Berührungswerten
                    mask = beruehrungswerte.isel(versuch=idx) > self.threshold
                    filtered_presskraft = presskraft_dyn.isel(versuch=idx).where(mask, drop=True)

                    # Maximalen Druck und Zeitpunkt berechnen
                    max_value = filtered_presskraft.max(dim="row", skipna=True).values
                    max_time = filtered_presskraft.argmax(dim="row", skipna=True).values * DEFAULT_TIME_STEP

                    max_druck.append(max_value)
                    max_druck_time.append(max_time)
                except Exception as e:
                    print(f"Fehler bei Versuch {idx}: {e}")
                    max_druck.append(np.nan)
                    max_druck_time.append(np.nan)

            return np.array(max_druck), np.array(max_druck_time)
        except Exception as e:
            print(f"Fehler bei 'calculate_maximaler_druck': {e}")
            raise

    # ----------------- Energieeffizienz -----------------
    def calculate_e_ab(self) -> np.ndarray:
        """
        Berechnet die abgegebene Energie E_Ab als Summe der Produktwerte von Presskraft und Verschiebung. Es handelt sich eher um Arbeit als um Energie
        """
        try:
            # Extrahiere Kraft- und Verschiebungsdaten
            force = self.data_array.sel(col="Presskraft_dyn", drop=True).values
            displacement = self.data_array.sel(col="Wegmessung", drop=True).diff(dim="row").values

            # Umwandlung in Float, falls notwendig
            force = force.astype(float)
            displacement = displacement.astype(float)

            # Debugging: Shapes prüfen
            print(f"Shape von force: {force.shape}")
            print(f"Shape von displacement: {displacement.shape}")

            # Shape anpassen: Schneiden der letzten Spalte von force
            force = force[:, :-1]

            # Ignoriere NaN-Werte
            valid_mask = ~np.isnan(displacement)
            force = np.where(valid_mask, force, 0)  # NaN-Werte in force durch 0 ersetzen
            displacement = np.where(valid_mask, displacement, 0)  # NaN-Werte in displacement durch 0 ersetzen

            # Berechnung der Energie (Summe der Produkte pro Zeile)
            energy = np.sum(force * displacement, axis=1)
            return energy
        except Exception as e:
            print(f"Fehler bei der Berechnung von Energie: {e}")
            return np.full(self.data_array.sizes["versuch"], np.nan)
    
    # ----------------- Verkippung -----------------
    # def calculate_stoessel_verkippung(self, index_array) -> tuple[np.ndarray, np.ndarray]:
    #     """
    #     Berechnet die Stößelkippung um die X- und Y-Achse für die angegebenen Indizes.

    #     Args:
    #         index_array (np.ndarray): Array von Indizes.

    #     Returns:
    #         tuple: X- und Y-Kippungen.
    #     """
    #     if np.isscalar(index_array):  # Konvertiere einen einzelnen Index in ein Array
    #         index_array = np.array([index_array])

    #     def compute_tilt_at_index(idx, axis):
    #         try:
    #             if axis == "x":
    #                 tilt = (
    #                     (self.data_array.sel(row=idx, col="Verkippung_3").values +
    #                     self.data_array.sel(row=idx, col="Verkippung_1").values) -
    #                     (self.data_array.sel(row=idx, col="Verkippung_4").values +
    #                     self.data_array.sel(row=idx, col="Verkippung_2").values)
    #                 ) / (2 * MEAN_DISTANCE_VERKIPPUNG_Y)
    #             elif axis == "y":
    #                 tilt = (
    #                     (self.data_array.sel(row=idx, col="Verkippung_2").values +
    #                     self.data_array.sel(row=idx, col="Verkippung_1").values) -
    #                     (self.data_array.sel(row=idx, col="Verkippung_4").values +
    #                     self.data_array.sel(row=idx, col="Verkippung_3").values)
    #                 ) / (2 * MEAN_DISTANCE_VERKIPPUNG_X)
    #             return tilt
    #         except Exception as e:
    #             print(f"Fehler bei der Verarbeitung des Index {idx} auf Achse {axis}: {e}")
    #             return np.nan

    #     tilt_x = [compute_tilt_at_index(idx, "x") for idx in index_array]
    #     tilt_y = [compute_tilt_at_index(idx, "y") for idx in index_array]

    #     return np.array(tilt_x), np.array(tilt_y)

    def calculate_stoessel_verkippung(self, index_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Berechnet die Stößelkippung um die X- und Y-Achse für die angegebenen Indizes.

        Args:
            index_array (np.ndarray): Array von Indizes (Punkte) pro Versuch.

        Returns:
            tuple[np.ndarray, np.ndarray]: Arrays für X- und Y-Kippungen pro Versuch.
        """
        tilt_x = []
        tilt_y = []

        for idx, row_idx in enumerate(index_array):
            try:
                if 0 <= row_idx < self.data_array.sizes["row"]:  # Gültigkeit des Index prüfen
                    # Verkippung um die X-Achse
                    tilt_x_value = (
                        (self.data_array.isel(versuch=idx, row=row_idx, col=self.data_array.coords["col"].to_index().get_loc("Verkippung_3")).values +
                        self.data_array.isel(versuch=idx, row=row_idx, col=self.data_array.coords["col"].to_index().get_loc("Verkippung_1")).values) -
                        (self.data_array.isel(versuch=idx, row=row_idx, col=self.data_array.coords["col"].to_index().get_loc("Verkippung_4")).values +
                        self.data_array.isel(versuch=idx, row=row_idx, col=self.data_array.coords["col"].to_index().get_loc("Verkippung_2")).values)
                    ) / (2 * MEAN_DISTANCE_VERKIPPUNG_Y)

                    # Verkippung um die Y-Achse
                    tilt_y_value = (
                        (self.data_array.isel(versuch=idx, row=row_idx, col=self.data_array.coords["col"].to_index().get_loc("Verkippung_2")).values +
                        self.data_array.isel(versuch=idx, row=row_idx, col=self.data_array.coords["col"].to_index().get_loc("Verkippung_1")).values) -
                        (self.data_array.isel(versuch=idx, row=row_idx, col=self.data_array.coords["col"].to_index().get_loc("Verkippung_4")).values +
                        self.data_array.isel(versuch=idx, row=row_idx, col=self.data_array.coords["col"].to_index().get_loc("Verkippung_3")).values)
                    ) / (2 * MEAN_DISTANCE_VERKIPPUNG_X)

                    tilt_x.append(tilt_x_value)
                    tilt_y.append(tilt_y_value)
                else:
                    # Wenn der Index ungültig ist, NaN hinzufügen
                    tilt_x.append(np.nan)
                    tilt_y.append(np.nan)
            except Exception as e:
                print(f"Fehler bei der Verarbeitung von Versuch {idx}, Index {row_idx}: {e}")
                tilt_x.append(np.nan)
                tilt_y.append(np.nan)

        return np.array(tilt_x), np.array(tilt_y)
    
    def calculate_stoessel_verkippung_vectorized(self, valid_indices, valid_t0, valid_t_down):
        """
        Berechnet die Verkippung (X- und Y-Richtung) für mehrere gültige Zeitbereiche.

        Args:
            valid_indices (np.ndarray): Indizes der gültigen Versuche.
            valid_t0 (np.ndarray): Startzeitpunkte der gültigen Bereiche.
            valid_t_down (np.ndarray): Endzeitpunkte der gültigen Bereiche.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays mit Verkippungswerten in X- und Y-Richtung.
        """
        try:
            # Extrahiere die relevanten Daten
            relevant_tilt_x = []
            relevant_tilt_y = []

            for i, idx in enumerate(valid_indices):
                start_idx = valid_t0[i]
                end_idx = valid_t_down[i] + 1  # Bereich inklusive des Endzeitpunkts

                # Verkippungsdaten berechnen (angepasst an die konkrete Berechnung in calculate_stoessel_verkippung)
                tilt_x, tilt_y = self.calculate_stoessel_verkippung(np.arange(start_idx, end_idx))
                relevant_tilt_x.append(tilt_x)
                relevant_tilt_y.append(tilt_y)

            # Stapeln der Ergebnisse für vektorisierte Berechnung
            tilt_x_array = np.array([np.nanmax(tilt_x) if tilt_x.size > 0 else np.nan for tilt_x in relevant_tilt_x])
            tilt_y_array = np.array([np.nanmax(tilt_y) if tilt_y.size > 0 else np.nan for tilt_y in relevant_tilt_y])

            return tilt_x_array, tilt_y_array
        except Exception as e:
            print(f"Fehler bei der Berechnung der Stoessel-Verkippung (vektorisiert): {e}")
            raise
    
    # ----------------- Maximale Verkippung -----------------
    # def calculate_maximale_verkippung(self, t0_indices: np.ndarray, t_down_indices: np.ndarray) -> np.ndarray:
    #     """
    #     Berechnet die maximale Verkippung während der Berührungszeit effizienter.

    #     Args:
    #         t0_indices (np.ndarray): Indizes der Auftreffzeitpunkte.
    #         t_down_indices (np.ndarray): Indizes der Zeitpunkte, an denen der Berührungswert unter den Schwellenwert fällt.

    #     Returns:
    #         np.ndarray: Maximalwerte der Verkippungssensoren für jeden Versuch.
    #     """
    #     try:
    #         # Extrahiere die Verkippungsdaten (z. B. aus der Datenstruktur)
    #         tilt_x_data = self.data_array.sel(col="Verkippung_X").values
    #         tilt_y_data = self.data_array.sel(col="Verkippung_Y").values

    #         max_verkippungen = np.full(t0_indices.shape, np.nan)  # Initialisiere mit NaN

    #         # Iteriere nur über gültige Indizes
    #         valid_mask = (t0_indices >= 0) & (t_down_indices > t0_indices)

    #         for versuch_idx in np.where(valid_mask)[0]:
    #             try:
    #                 # Bereich der relevanten Indizes
    #                 start_idx = t0_indices[versuch_idx]
    #                 end_idx = t_down_indices[versuch_idx]

    #                 # Extrahiere Daten im relevanten Bereich
    #                 tilt_x = tilt_x_data[versuch_idx, start_idx:end_idx + 1]
    #                 tilt_y = tilt_y_data[versuch_idx, start_idx:end_idx + 1]

    #                 # Berechne den maximalen Verkippungswert
    #                 max_verkippungen[versuch_idx] = max(np.nanmax(tilt_x), np.nanmax(tilt_y))
    #             except Exception as e:
    #                 print(f"Fehler bei Versuch {versuch_idx}: {e}")

    #         return max_verkippungen
    #     except Exception as e:
    #         print(f"Fehler bei der Berechnung der maximalen Verkippung: {e}")
    #         raise

    def calculate_maximale_verkippung(self, t0_indices: np.ndarray, t_down_indices: np.ndarray) -> np.ndarray:
        """
        Berechnet die maximale Verkippung während der Berührungszeit.

        Args:
            t0_indices (np.ndarray): Indizes der Auftreffzeitpunkte.
            t_down_indices (np.ndarray): Indizes der Zeitpunkte, an denen der Berührungswert unter den Schwellenwert fällt.

        Returns:
            np.ndarray: Maximalwerte der Verkippungssensoren für jeden Versuch.
        """
        try:
            # Initialisiere die Ergebnisse mit NaN
            max_verkippungen = np.full(t0_indices.shape, np.nan)

            # Maske für gültige Werte (prüfe, ob Indizes innerhalb der Dimension liegen)
            valid_mask = (
                (t0_indices >= 0) & (t_down_indices > t0_indices) & 
                (t_down_indices < self.data_array.shape[1])  # Sicherstellen, dass Indizes nicht außerhalb liegen
            )

            # Filtern von gültigen Indizes
            valid_t0 = t0_indices[valid_mask]
            valid_t_down = t_down_indices[valid_mask]
            valid_indices = np.where(valid_mask)[0]

            # Wenn keine gültigen Werte vorhanden sind, direkt zurückgeben
            if valid_indices.size == 0:
                return max_verkippungen

            # Berechne Verkippung für die gesamten Datenbereiche
            tilt_x_data, tilt_y_data = self.calculate_stoessel_verkippung_vectorized(valid_t0, valid_t_down)

            # Maximalwerte berechnen und speichern
            max_verkippungen[valid_indices] = np.nanmax(np.stack([tilt_x_data, tilt_y_data], axis=-1), axis=1)

            return max_verkippungen
        except Exception as e:
            print(f"Fehler bei der Berechnung der maximalen Verkippung: {e}")
            raise

    # def calculate_t0_indices(self) -> np.ndarray:
    #     """
    #     Berechnet die Indizes, an denen der Schwellwert für 'Berührungswerte' überschritten wird.
    #     """
    #     beruehrungswerte = self.data_array.sel(col="Berührungswerte")
    #     indices = (beruehrungswerte > self.threshold).idxmax(dim="row").values
    #     # Sicherstellen, dass Indizes ganzzahlig sind
    #     indices = indices.astype(int)
    #     return indices

    # def calculate_t_down_indices(self) -> np.ndarray:
    #     """
    #     Berechnet die Indizes, an denen der Berührungswert wieder unter den Schwellenwert fällt.

    #     Returns:
    #         np.ndarray: Indizes des Zeitpunkts, an dem der Berührungswert unter den Schwellenwert fällt.
    #     """
    #     t_down_indices = []
    #     beruehrungswerte = self.data_array.sel(col="Berührungswerte")

    #     for versuch_idx in range(len(self.versuch_ids)):
    #         try:
    #             # Überprüfen, ob es Werte unter dem Threshold gibt
    #             values_below_threshold = (beruehrungswerte.isel(versuch=versuch_idx) < self.threshold).values
    #             if not np.any(values_below_threshold):  # Kein Wert unter dem Threshold
    #                 t_down_indices.append(np.nan)  # Füge NaN hinzu
    #             else:
    #                 # Berechne den ersten Index, der die Bedingung erfüllt
    #                 t_down_index = np.argmax(values_below_threshold)
    #                 t_down_indices.append(t_down_index)
    #         except Exception as e:
    #             print(f"Fehler bei Versuch {versuch_idx}: {e}")
    #             t_down_indices.append(np.nan)  # Füge NaN hinzu, wenn ein Fehler auftritt

    #     return np.array(t_down_indices)

    def calculate_t0_indices(self):
        """
        Berechnet die Indizes \( t_0 \), an denen der Schwellenwert für "Berührungswerte" 
        zum ersten Mal überschritten wird.

        Returns:
            np.ndarray: Ein Array mit den \( t_0 \)-Indizes für jeden Versuch.
        """
        try:
            # Extrahiere die Werte der Spalte "Berührungswerte"
            beruehrungswerte = self.data_array.sel(col="Berührungswerte").values  # Shape: (num_versuche, num_rows)

            # Berechnung von t0: Erstes Überschreiten des Schwellenwerts
            above_threshold = beruehrungswerte > self.threshold
            t0_indices = np.argmax(above_threshold, axis=1)  # Erster True-Wert pro Versuch

            # Markiere Versuche, bei denen kein Wert über dem Threshold liegt
            no_above_threshold = ~np.any(above_threshold, axis=1)
            t0_indices[no_above_threshold] = -1  # Markiere als ungültig

            # Konvertiere alle Werte zu Integern (einschließlich -1)
            t0_indices = t0_indices.astype(int)

            # Debugging: Ausgabe der berechneten Werte
            #print("Berechnete t0-Indizes (validiert):", t0_indices)

            # Rückgabe der \( t_0 \)-Indizes
            return t0_indices

        except Exception as e:
            print(f"Fehler bei der Berechnung der t0-Indizes: {e}")
            return np.full(self.data_array.shape[0], -1, dtype=int)  # Rückgabe von -1 für alle Versuche bei Fehler

    def calculate_t_down_indices(self, t0_indices: np.ndarray) -> np.ndarray:
        """
        Berechnet die Indizes, an denen der Berührungswert wieder unter den Schwellenwert fällt.

        Args:
            t0_indices (np.ndarray): Indizes, an denen der Schwellenwert überschritten wurde.

        Returns:
            np.ndarray: Indizes von t_down (Zeitpunkte, an denen der Schwellenwert unterschritten wird).
        """
        try:
            # Extrahiere die Werte der Spalte "Berührungswerte"
            beruehrungswerte = self.data_array.sel(col="Berührungswerte").values  # Shape: (num_versuche, num_rows)
            num_versuche, num_cols = beruehrungswerte.shape

            # Initialisiere t_down-Indizes mit -1 als Platzhalter für ungültige Werte
            t_down_indices = np.full(num_versuche, -1, dtype=int)

            # Gültige t0-Indizes identifizieren
            valid_t0_mask = (t0_indices >= 0) & (t0_indices < num_cols)  # Gültige Indizes in der Spalte
            valid_rows = np.where(valid_t0_mask & (np.arange(len(t0_indices)) < num_versuche))[0]  # Gültige Zeilen
            valid_t0_indices = t0_indices[valid_rows].astype(int)

            # Debugging: Ausgabe der Dimensionen und gültigen Werte
            # print(f"Shape von beruehrungswerte: {beruehrungswerte.shape}")
            # print("Gültige Zeilen:", valid_rows)
            # print("Validierte t0-Indizes:", valid_t0_indices)

            # **Verarbeitung der gültigen Zeilen**
            for row, t0 in zip(valid_rows, valid_t0_indices):
                try:
                    # Werte nach t0
                    values_after_t0 = beruehrungswerte[row, t0:]
                    below_threshold = values_after_t0 < self.threshold

                    # Erster Zeitpunkt unterhalb des Schwellenwertes
                    if np.any(below_threshold):
                        t_down_indices[row] = t0 + np.argmax(below_threshold)
                    else:
                        t_down_indices[row] = -1

                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von Versuch {row}, Index {t0}: {e}")

            # Debugging: Ausgabe der berechneten Werte
            #print("Berechnete t_down-Indizes (validiert):", t_down_indices)

            return t_down_indices

        except Exception as e:
            print(f"Fehler bei der Berechnung von t_down-Indizes: {e}")
            return np.full(len(t0_indices), -1, dtype=int)

    # def calculate_t_down_indices(self):
    #     """
    #     Berechnet die Indizes \( t_{\text{down}} \), an denen der Schwellenwert 
    #     für "Berührungswerte" von hinten betrachtet unterschritten wird.

    #     Returns:
    #         np.ndarray: Ein Array mit den \( t_{\text{down}} \)-Indizes für jeden Versuch.
    #     """
    #     try:
    #         # Extrahiere die Werte der Spalte "Berührungswerte"
    #         beruehrungswerte = self.data_array.sel(col="Berührungswerte").values  # Shape: (num_versuche, num_rows)

    #         # Berechnung von t_down: Werte von hinten betrachten, erstes Unterschreiten des Schwellenwerts
    #         below_threshold = beruehrungswerte[::-1] < self.threshold  # Werte umdrehen und Schwellenwertprüfung
    #         reversed_t_down_indices = np.argmax(below_threshold, axis=1)  # Erster True-Wert von hinten

    #         # Konvertiere den Index zurück zur ursprünglichen Reihenfolge
    #         t_down_indices = beruehrungswerte.shape[1] - reversed_t_down_indices - 1

    #         # Markiere Versuche, bei denen kein Wert unter dem Threshold liegt
    #         no_below_threshold = ~np.any(below_threshold, axis=1)
    #         t_down_indices[no_below_threshold] = -1  # Markiere als ungültig

    #         # Konvertiere alle Werte zu Integern (einschließlich -1)
    #         t_down_indices = t_down_indices.astype(int)

    #         # Debugging: Ausgabe der berechneten Werte
    #         print("Berechnete t_down-Indizes (validiert):", t_down_indices)

    #         return t_down_indices

    #     except Exception as e:
    #         print(f"Fehler bei der Berechnung der t_down-Indizes: {e}")
    #         return np.full(self.data_array.shape[0], -1, dtype=int)  # Rückgabe von -1 für alle Versuche bei Fehler

    # ----------------- Features in DataFrame hinzufügen -----------------
    def add_features(self) -> pd.DataFrame:
        """
        Berechnet alle Features und gibt sie als DataFrame zurück, 
        wobei jeder Versuch eine Zeile ist.
        """
        try:
            # Berechnung der relevanten Features
            t0_indices = self.calculate_t0_indices()
            t_down_indices = self.calculate_t_down_indices(t0_indices)
            tiefster_punkt_idx = self.calculate_tiefster_punkt_idx()

            # Berechnung der Verkippung für unterschiedliche Punkte
            tiefster_punkt_idx = self.calculate_tiefster_punkt_idx()
            tilt_x_tiefster, tilt_y_tiefster = self.calculate_stoessel_verkippung(tiefster_punkt_idx)

            tilt_x_t0, tilt_y_t0 = self.calculate_stoessel_verkippung(t0_indices)

            s0 = self.calculate_auftreffposition()
            v0 = self.calculate_auftreffgeschwindigkeit(t0_indices)
            beruehrungsdauer = self.calculate_beruehrungsdauer(t0_indices, t_down_indices)
            #max_verkippung = self.calculate_maximale_verkippung(t0_indices, t_down_indices)
            
            # Verarbeitung der max_druck-Ausgabe 
            max_druck, max_druck_time = self.calculate_maximaler_druck()
            max_druck = np.array(max_druck).flatten()
            max_druck_time = np.array(max_druck_time).flatten()

            arbeit_ab = self.calculate_e_ab()

            # Debugging: Längen aller Features ausgeben
            for key, value in {
                "versuch": self.versuch_ids,
                "druckberuehrzeit": self.calculate_druckberuehrzeit(),
                "auftreffzeitpunkt": self.calculate_auftreffzeitpunkt(),
                "umformzeit": self.calculate_umformzeit(t0_indices, tiefster_punkt_idx),
                "v_max": self.calculate_v_max(),
                "v_min": self.calculate_v_min(),
                "arbeit_ab": arbeit_ab,
                "tilt_x_tiefster": tilt_x_tiefster,
                "tilt_y_tiefster": tilt_y_tiefster,
                "tilt_x_t0": tilt_x_t0,
                "tilt_y_t0": tilt_y_t0,
                "min_motorstrom": self.calculate_min_motorstrom(),
                "max_motorstrom": self.calculate_max_motorstrom(),
                "mean_motorstrom": self.calculate_mean_motorstrom(),
                "auftreffposition": s0,
                "auftreffgeschwindigkeit": v0,
                #"maximale_verkippung": max_verkippung,
                "beruehrungsdauer": beruehrungsdauer,
                "max_druck": max_druck,
                "max_druck_time": max_druck_time,
            }.items():
                print(f"Feature '{key}' hat Länge: {len(value)}")

            # Erstellen des DataFrames
            features_df = pd.DataFrame({
                "VersuchID": self.versuch_ids,
                "druckberuehrzeit": self.calculate_druckberuehrzeit(),
                "auftreffzeitpunkt": self.calculate_auftreffzeitpunkt(),
                "umformzeit": self.calculate_umformzeit(t0_indices, tiefster_punkt_idx),
                "v_max": self.calculate_v_max(),
                "v_min": self.calculate_v_min(),
                "arbeit_ab": arbeit_ab,
                "tilt_x_tiefster": tilt_x_tiefster, 
                "tilt_y_tiefster": tilt_y_tiefster,
                "tilt_x_t0": tilt_x_t0,
                "tilt_y_t0": tilt_y_t0, 
                "min_motorstrom": self.calculate_min_motorstrom(),
                "max_motorstrom": self.calculate_max_motorstrom(),
                "mean_motorstrom": self.calculate_mean_motorstrom(),
                "auftreffposition": s0,
                "auftreffgeschwindigkeit": v0, 
                "beruehrungsdauer": beruehrungsdauer, 
                #"maximale_verkippung": max_verkippung, 
                "max_druck": max_druck,
                #"max_druck_time": max_druck_time,
            })

            return features_df
        except Exception as e:
            print(f"Fehler beim Hinzufügen der Features: {e}")
            raise

def calculate_t0_indices(data_array: xr.DataArray, threshold: float = 9) -> np.ndarray:
    """
    Berechnet die Indizes \( t_0 \), an denen der Schwellenwert überschritten wird.
    """
    beruehrungswerte = data_array.sel(col="Berührungswerte").values  # Shape: (num_versuche, num_rows)
    above_threshold = beruehrungswerte > threshold
    t0_indices = np.argmax(above_threshold, axis=1)
    no_above_threshold = ~np.any(above_threshold, axis=1)
    t0_indices[no_above_threshold] = -1
    return t0_indices.astype(int)

def calculate_t_down_indices(data_array: xr.DataArray, t0_indices: np.ndarray, threshold: float = 9) -> np.ndarray:
    """
    Berechnet die Indizes, an denen der Berührungswert wieder unter den Schwellenwert fällt.

    Args:
        data_array (xr.DataArray): Das Input-DataArray.
        t0_indices (np.ndarray): Indizes, an denen der Schwellenwert überschritten wurde.
        threshold (float): Schwellenwert für Berührungswerte.

    Returns:
        np.ndarray: Indizes von t_down (Zeitpunkte, an denen der Schwellenwert unterschritten wird).
    """
    beruehrungswerte = data_array.sel(col="Berührungswerte").values
    num_versuche, num_cols = beruehrungswerte.shape

    t_down_indices = np.full(num_versuche, -1, dtype=int)
    valid_t0_mask = (t0_indices >= 0) & (t0_indices < num_cols)
    valid_rows = np.where(valid_t0_mask)[0]
    valid_t0_indices = t0_indices[valid_rows]

    for row, t0 in zip(valid_rows, valid_t0_indices):
        values_after_t0 = beruehrungswerte[row, t0:]
        below_threshold = values_after_t0 < threshold

        if np.any(below_threshold):
            t_down_indices[row] = t0 + np.argmax(below_threshold)
        else:
            t_down_indices[row] = -1

    return t_down_indices

def calculate_t_down_indices_debug(data_array: xr.DataArray, t0_indices: np.ndarray, threshold: float = 9) -> np.ndarray:
    """
    Debugging-Version: Berechnet die Indizes, an denen der Berührungswert wieder unter den Schwellenwert fällt,
    und gibt zusätzliche Informationen zu möglichen Problemen aus.
    """
    beruehrungswerte = data_array.sel(col="Berührungswerte").values
    num_versuche, num_cols = beruehrungswerte.shape

    t_down_indices = np.full(num_versuche, -1, dtype=int)
    valid_t0_mask = (t0_indices >= 0) & (t0_indices < num_cols)
    valid_rows = np.where(valid_t0_mask)[0]
    valid_t0_indices = t0_indices[valid_rows]

    for row, t0 in zip(valid_rows, valid_t0_indices):
        values_after_t0 = beruehrungswerte[row, t0:]
        below_threshold = values_after_t0 < threshold

        # Debugging: Überprüfe die Werte nach t0
        print(f"Versuch {row}: t0 = {t0}, Werte nach t0 = {values_after_t0[:10]}")
        print(f"Versuch {row}: below_threshold = {below_threshold[:10]}")

        if np.any(below_threshold):
            t_down_indices[row] = t0 + np.argmax(below_threshold)
            print(f"Versuch {row}: t_down gefunden bei Index = {t_down_indices[row]}")
        else:
            t_down_indices[row] = -1
            print(f"Versuch {row}: Kein t_down gefunden. Alle Werte bleiben über dem Schwellenwert.")

    return t_down_indices

def finde_betroffene_versuche(dataArray: xr.DataArray, threshold: float = 9) -> list:
    """
    Bestimmt die Versuche, bei denen kein t_down-Index gefunden wird, d.h. t_down = -1.

    Args:
        dataArray (xr.DataArray): Das Input-DataArray.
        threshold (float): Schwellenwert für die Berechnung von t0 und t_down.

    Returns:
        list: Liste der betroffenen VersuchIDs.
    """
    # Berechnung der Indizes t0 und t_down
    t0_indices = calculate_t0_indices(dataArray, threshold=threshold)
    t_down_indices = calculate_t_down_indices(dataArray, t0_indices, threshold=threshold)
    
    # Finden der Versuche mit t_down = -1
    betroffene_versuche_indices = np.where(t_down_indices == -1)[0]
    
    # Extrahiere die entsprechenden VersuchIDs
    versuch_ids = dataArray.sel(col="datei_name").isel(row=0).values
    betroffene_versuch_ids = versuch_ids[betroffene_versuche_indices]
    
    # Debugging-Informationen
    print(f"t0-Indizes für alle Versuche: {t0_indices}")
    print(f"t_down-Indizes für alle Versuche: {t_down_indices}")
    print(f"Betroffene Versuche (t_down = -1): {betroffene_versuch_ids}")
    
    return betroffene_versuch_ids.tolist()

def berechne_aggregierte_features(dataArray: xr.DataArray) -> pd.DataFrame:
    """
    Berechnet aggregierte Werte (Min, Max, Mittelwert, Median, Std) für ausgewählte Features
    sowie aggregierte Werte der Berührungswerte während der Berührzeit.

    Args:
        dataArray (xr.DataArray): Das Input-DataArray mit den Versuchen und Daten.

    Returns:
        pd.DataFrame: Ein DataFrame mit aggregierten Features und VersuchID.
    """
    time_each_index = 0.5
    feature_list = [
        'Wegmessung', 'Verkippung_1', 'Verkippung_2', 'Verkippung_3', 
        'Verkippung_4', 'Stoesselhub', 'Geschwindigkeit_Ges', 
        'Presskraft_dyn', 'Motorstrom'
    ]

    # Berechnung von t0 und t_down
    t0_indices = calculate_t0_indices(dataArray, threshold=9)
    t_down_indices = calculate_t_down_indices(dataArray, t0_indices, threshold=9)

    aggregierte_features = {"VersuchID": dataArray.sel(col="datei_name").isel(row=0).values}
    
    for feature in feature_list:
        feature_data = dataArray.sel(col=feature).values
        
        min_vals, max_vals, mean_vals, median_vals, std_vals = [], [], [], [], []

        for i, (start_idx, end_idx) in enumerate(zip(t0_indices, t_down_indices)):
            print(f"Versuch {i}: Feature = {feature}, t0 = {start_idx}, t_down = {end_idx}")
            if start_idx < 0 or end_idx <= start_idx:
                print(f"Versuch {i}: Ungültige Indizes für t0 und t_down. Werte werden NaN.")
                min_vals.append(np.nan)
                max_vals.append(np.nan)
                mean_vals.append(np.nan)
                median_vals.append(np.nan)
                std_vals.append(np.nan)
                continue

            relevant_values = feature_data[i, start_idx:end_idx + 1]

            # Sicherstellen, dass die Werte numerisch sind
            try:
                relevant_values = relevant_values.astype(float)
            except ValueError as e:
                print(f"Versuch {i}: Konvertierung zu float fehlgeschlagen für {feature}. Fehler: {e}")
                min_vals.append(np.nan)
                max_vals.append(np.nan)
                mean_vals.append(np.nan)
                median_vals.append(np.nan)
                std_vals.append(np.nan)
                continue

            # Debugging: Prüfen auf NaN-Werte oder leere Arrays
            if relevant_values.size == 0:
                print(f"Versuch {i}: Keine Werte im Bereich t0 bis t_down für {feature}.")
                min_vals.append(np.nan)
                max_vals.append(np.nan)
                mean_vals.append(np.nan)
                median_vals.append(np.nan)
                std_vals.append(np.nan)
                continue

            if np.isnan(relevant_values).all():
                print(f"Versuch {i}: Alle Werte NaN im Bereich t0 bis t_down für {feature}.")
                min_vals.append(np.nan)
                max_vals.append(np.nan)
                mean_vals.append(np.nan)
                median_vals.append(np.nan)
                std_vals.append(np.nan)
                continue

            # Berechnungen durchführen
            try:
                min_vals.append(np.nanmin(relevant_values))
                max_vals.append(np.nanmax(relevant_values))
                mean_vals.append(np.nanmean(relevant_values))
                median_vals.append(np.nanmedian(relevant_values))
                std_vals.append(np.nanstd(relevant_values))
            except Exception as e:
                print(f"Fehler bei Berechnung für Versuch {i}, Feature {feature}: {e}")
                min_vals.append(np.nan)
                max_vals.append(np.nan)
                mean_vals.append(np.nan)
                median_vals.append(np.nan)
                std_vals.append(np.nan)

        aggregierte_features[f"{feature}_Min"] = min_vals
        aggregierte_features[f"{feature}_Max"] = max_vals
        aggregierte_features[f"{feature}_Mean"] = mean_vals
        aggregierte_features[f"{feature}_Median"] = median_vals
        aggregierte_features[f"{feature}_Std"] = std_vals

    return pd.DataFrame(aggregierte_features)

def berechne_process_features(dataArray: xr.DataArray) -> pd.DataFrame:
    """
    Berechnet alle skalarbasierten Features aus dem DataArray und speichert sie in einem DataFrame,
    einschließlich der Labels für jeden Versuch.

    Args:
        dataArray (xr.DataArray): Input-DataArray, das die Rohdaten enthält.

    Returns:
        pd.DataFrame: Ein DataFrame mit den berechneten Features und Labels für jeden Versuch.
    """
    # Initialisiere ProcessedData
    processed_data = ProcessedData(dataArray, init_matrix=False)
    
    # Einzelne Features berechnen
    features = {
        "VersuchID": processed_data.versuch_ids,
        "Berührzeit": processed_data.berührzeit,
        "Höhe_Wegmessung_Aufprall": processed_data.höhe_wegmessung_aufprall,
        "Umformzeit": processed_data.umformzeit,
        "Tiefster_Punkt": processed_data.tiefster_punkt,
        "Energie_Aufprall": processed_data.energie_aufprall_p,
        "Motorstrom_Durchschnitt": processed_data.motorstrom_durchschnitt,
        "Energie_ab_Durchschnitt": processed_data.energie_ab_durchschnitt,
        "Bauteil_Temp": processed_data.bauteil_temp,
        "Werkzeug_Temp": processed_data.werkzeug_temp,
    }

    # Labels extrahieren
    labels = {
        "Material_con": dataArray.loc[:, 0, "Material_con"].values,
        "Position_con": dataArray.loc[:, 0, "Position_con"].values,
        "Ergebnis_con": dataArray.loc[:, 0, "Ergebnis_con"].values,
        "Probenhoehe": dataArray.loc[:, 0, "Probenhoehe"].values,
        "richtig_verbaut": dataArray.loc[:, 0, "richtig_verbaut"].values,
    }
    
    # Debugging: Längen aller Features anzeigen
    for feature_name, feature_values in features.items():
        print(f"{feature_name}: Länge = {len(feature_values)}")
    
    for label_name, label_values in labels.items():
        print(f"{label_name}: Länge = {len(label_values)}")
    
    # Erstelle einen DataFrame aus den berechneten Features und Labels
    df_features = pd.DataFrame(features)
    df_labels = pd.DataFrame(labels)
    
    # Kombiniere Features und Labels
    df_combined = pd.concat([df_features, df_labels], axis=1)

    return df_combined