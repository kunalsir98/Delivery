# Basic Import
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def validate_and_clean_data(self, array):
        """
        Validate and clean the data to ensure all target values are numeric.
        """
        df = pd.DataFrame(array)
        # Assuming the last column is the target column
        target_column = df.columns[-1]
        
        # Check for non-numeric values in the target column
        non_numeric = df[~df[target_column].apply(lambda x: self.is_float(x))]
        if not non_numeric.empty:
            logging.info(f'Found non-numeric values in the target column: {non_numeric[target_column].unique()}')
            raise ValueError(f'Non-numeric values found in the target column: {non_numeric[target_column].unique()}')

        # Convert target column to float
        df[target_column] = df[target_column].astype(float)

        return df.to_numpy()

    def is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Validating and cleaning the training and test data')
            train_array = self.validate_and_clean_data(train_array)
            test_array = self.validate_and_clean_data(test_array)

            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info('Training the CatBoost Regressor')
            model = CatBoostRegressor(verbose=0)
            model.fit(X_train, y_train)

            logging.info('Evaluating the model on the test data')
            y_pred = model.predict(X_test)
            r2_score = model.score(X_test, y_test)

            logging.info(f'CatBoost Regressor R2 Score: {r2_score}')
            print(f'CatBoost Regressor R2 Score: {r2_score}')
            print('\n====================================================================================\n')

            logging.info('Saving the trained model')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

        except Exception as e:
            logging.info('Exception occurred during model training')
            raise CustomException(e, sys)

# Example usage
# Make sure to replace train_arr and test_arr with your actual data
# model_trainer = ModelTrainer()
# model_trainer.initiate_model_training(train_arr, test_arr)
