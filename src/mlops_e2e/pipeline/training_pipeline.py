import os , sys

from mlops_e2e.exception.execption import NetworkSecurityException
from mlops_e2e.logging.logger import logging

from mlops_e2e.components.data_ingestion import DataIngestion
from mlops_e2e.components.data_validation import DataValidation
from mlops_e2e.components.data_transformation import DataTransformation
from mlops_e2e.components.model_trainer import ModelTrainer

from mlops_e2e.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from mlops_e2e.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Started data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed and artifacts: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Started data validation")
            data_validation = DataValidation(data_validation_config=self.data_validation_config,data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data Validation completed and artifacts: {data_validation_artifact}")
            return data_validation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def start_data_transformation(self,data_validation_artifact: DataValidationArtifact):
        try:
            self.data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Started data transformation")
            data_transformation = DataTransformation(data_transformation_config=self.data_transformation_config, data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformation completed and artifacts; {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            return NetworkSecurityException(e,sys)

    def start_model_training(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Started with Model Training")
            model_trainer = ModelTrainer(model_trainer_config=self.model_trainer_config,data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifacts  = model_trainer.initiate_model_trainer()
            logging.info(f"Model Training completed and artifacts: {model_trainer_artifacts}")
            return model_trainer_artifacts
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def run_pipeline(self):
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            data_validation_artifacts = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifacts)
            data_transformation_artifacts = self.start_data_transformation(data_validation_artifact=data_validation_artifacts)
            model_training_artifacts = self.start_model_training(data_transformation_artifact=data_transformation_artifacts)
            return model_training_artifacts
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)