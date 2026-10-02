import os,sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from mlops_e2e.constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from mlops_e2e.entity.config_entity import DataTransformationConfig
from mlops_e2e.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from mlops_e2e.exception.execption import NetworkSecurityException
from mlops_e2e.logging.logger import logging   
from mlops_e2e.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_transformation_config:DataTransformationConfig, data_validation_artifact:DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info("Starting data transformation process")
            # Load the valid train and test datasets
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            # Separate features and target variable
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN].replace(-1,0)
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN].replace(-1,0)

            # Create a KNN imputer with specified parameters
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)

            # Pipline for data transformation
            transformation_pipeline = Pipeline(steps=[
                ("imputer", imputer)
            ]) 

            # Fit the pipeline on the training data and transform both train and test datasets
            X_train_imputed = transformation_pipeline.fit_transform(X_train)
            X_test_imputed = transformation_pipeline.transform(X_test)

            # Save the transformed datasets
            transformed_train_file_path = self.data_transformation_config.transformed_train_file_path
            transformed_test_file_path = self.data_transformation_config.transformed_test_file_path

            save_numpy_array_data(transformed_train_file_path, np.c_[X_train_imputed, y_train.to_numpy()])
            save_numpy_array_data(transformed_test_file_path, np.c_[X_test_imputed, y_test.to_numpy()])

            # Save the imputer object for future use
            transformed_object_file_path = self.data_transformation_config.transformed_object_file_path
            save_object(transformed_object_file_path, imputer)

            logging.info("Data transformation process completed successfully")

            return DataTransformationArtifact(
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                transformed_object_file_path=transformed_object_file_path
            )
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)