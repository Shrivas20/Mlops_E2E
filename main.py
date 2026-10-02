from src.mlops_e2e.components.data_ingestion import DataIngestion
from src.mlops_e2e.components.data_validation import DataValidation
from src.mlops_e2e.exception.execption import NetworkSecurityException
from  src.mlops_e2e.logging.logger import logging
from src.mlops_e2e.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.mlops_e2e.entity.config_entity import TrainingPipelineConfig

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

    except NetworkSecurityException as e:
        logging.error(f"Network security exception occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")