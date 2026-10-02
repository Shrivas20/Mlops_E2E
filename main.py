from src.mlops_e2e.components.data_ingestion import DataIngestion
from src.mlops_e2e.components.data_validation import DataValidation
from src.mlops_e2e.components.data_transformation import DataTransformation
from src.mlops_e2e.components.model_trainer import ModelTrainer
from src.mlops_e2e.exception.execption import NetworkSecurityException
from  src.mlops_e2e.logging.logger import logging
from src.mlops_e2e.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig

if __name__ == "__main__":
    try:

        # Data ingestion process
        trainingpipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(trainingpipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Starting data ingestion process...")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed successfully. Train file path: {dataingestionartifact.train_file_path}, Test file path: {dataingestionartifact.test_file_path}")

        # Start data validation process
        data_validation_config = DataValidationConfig(trainingpipeline_config)
        data_validation = DataValidation(data_validation_config, dataingestionartifact)
        logging.info("Starting data validation process...")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed successfully.")

        # Start data transformation process
        data_transformation_config = DataTransformationConfig(trainingpipeline_config)
        data_transformation = DataTransformation(data_transformation_config, data_validation_artifact)
        logging.info("Starting data transformation process...")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed successfully.")

        # Start model training process
        model_trainer_config = ModelTrainerConfig(trainingpipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        logging.info("Starting model training process...")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training completed successfully.")

    except NetworkSecurityException as e:
        logging.error(f"Network security exception occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")