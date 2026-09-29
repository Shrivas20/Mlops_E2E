from datetime import datetime
import os 
from mlops_e2e.constants import training_pipeline

class TrainingPipelineConfig:
    def __init__(self,timestamp =datetime.now()):
        try:
            self.pipeline_name = training_pipeline.PIPELINE_NAME
            self.artifact_dir = training_pipeline.ARTIFACT_DIR

            # Create a timestamped directory for artifacts
            timestamp = timestamp.strftime("%Y%m%d%H%M%S")
            self.timestamped_artifact_dir = os.path.join(self.artifact_dir, timestamp)
            os.makedirs(self.timestamped_artifact_dir, exist_ok=True)

        except Exception as e:
            raise e

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
        
            # Create a directory for data ingestion artifacts
            self.data_ingestion_dir = os.path.join(
                training_pipeline_config.timestamped_artifact_dir,
                training_pipeline.DATA_INGESTION_DIR_NAME
            )

            self.feature_store_dir = os.path.join(
                self.data_ingestion_dir,
                training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR_NAME,
                training_pipeline.FILE_NAME
            )

            self.feature_store_file_name = training_pipeline.FILE_NAME

            self.training_file_path = os.path.join(
                self.data_ingestion_dir,
                training_pipeline.DATA_INGESTION_DIR_NAME,
                training_pipeline.TRAIN_FILE_NAME
            )

            self.testing_file_path = os.path.join(
                self.data_ingestion_dir,
                training_pipeline.DATA_INGESTION_DIR_NAME,
                training_pipeline.TEST_FILE_NAME
            )   

            self.train_test_split_ratio = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
            self.collection_name = training_pipeline.DATA_INGESTION_COLLECTION_NAME
            self.database_name = training_pipeline.DATA_INGESTION_DATABASE_NAME

        except Exception as e:
            raise e