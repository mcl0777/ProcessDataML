import pickle
from typing import Callable, List
from loguru import logger
import numpy as np
import os

from sklearn import pipeline
from helpers.process_data import ProcessedData

from helpers.process_data import ProcessedData

# model_path = p_settings_instance.get_models_dir()


class ML_Model:
    """
    This should only be used as a base class.
    Create a child class (example: "class Example_Model(ML_Model)")
    and override the preprocess_pipeline method. To be able to load it
    into the program later on the class has to be defined in ml_model_storage.py and add the
    class to the all_models variable (at the bottom of the file).
    """

    def __init__(
        self,
    ):
        self.model_pipeline = None

    def preprocess_pipeline(self, processed_features: ProcessedData) -> np.ndarray:
        raise NotImplementedError("Please Implement this method")

    # This returns a 2d numpy array of height one (1,:) if you pass in only one versuch
    def get_prediction(self, processed_features: ProcessedData) -> np.ndarray:
        if self.model_pipeline is None:
            raise Exception("Machine learning model wasnt added in building step.")
        input_data = self.preprocess_pipeline(processed_features)
        return self.model_pipeline.predict(input_data)

    def save_model(self, model_pipeline: pipeline.Pipeline):
        """This stores the object to disk."""

        file_name = f"{model_path}/{self.get_name()}.pkl"
        self.model_pipeline = model_pipeline

        with open(file_name, "wb") as handle:
            pickle.dump(
                self,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load_model(cls) -> "ML_Model":
        try:
            file_name = f"{model_path}/{cls.get_name()}.pkl"
            if not os.path.exists(file_name):
                logger.error(f"Model file not found: {file_name}")
                return cls()
                
            with open(file_name, "rb") as handle:
                try:
                    model = pickle.load(handle)
                    if model is None:
                        logger.error(f"Failed to load model: {cls.get_name()}")
                        return cls()
                    return model
                except pickle.UnpicklingError as e:
                    logger.error(f"Error unpickling model {cls.get_name()}: {e}")
                    return cls()
        except Exception as e:
            logger.error(f"Error loading model {cls.get_name()}: {e}")
            return cls()

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__
