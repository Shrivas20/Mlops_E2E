import sys, os
import mlflow

from mlops_e2e.exception.execption import NetworkSecurityException
from mlops_e2e.logging.logger import logging
from mlops_e2e.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact, ClassificationMetricArtifact
from mlops_e2e.entity.config_entity import ModelTrainerConfig

from mlops_e2e.utils.main_utils.utils import load_object, save_object,load_numpy_array_data, save_numpy_array_data
from mlops_e2e.utils.ml_utils.metric.classification_metric import get_classification_metrics

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from mlops_e2e.utils.ml_utils.model.estimator import NetworkSecurityModel 


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

        

    def track_mlflow(self, best_model, classification_metric_artifact:ClassificationMetricArtifact):
        with mlflow.start_run():
            mlflow.sklearn.log_model(best_model, name = "model",skops_trusted_types=["sklearn.tree._tree.Tree"])
            mlflow.log_metric("f1_score", classification_metric_artifact.f1_score)
            mlflow.log_metric("precision_score", classification_metric_artifact.precision_score)
            mlflow.log_metric("recall_score", classification_metric_artifact.recall_score)

        

    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "RandomForestClassifier": RandomForestClassifier(verbose=1),
            "GradientBoostingClassifier": GradientBoostingClassifier(verbose=1),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "LogisticRegression": LogisticRegression(verbose=1),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "KNeighborsClassifier": KNeighborsClassifier()
        }

        params = {
            "RandomForestClassifier": {
                "n_estimators": [10, 50, 100],
                # "max_depth": [None, 5, 10],
                # "min_samples_split": [2, 5, 10]
            },
            "GradientBoostingClassifier": {
                "n_estimators": [50, 100, 200],
                # "learning_rate": [0.01, 0.1, 0.2],
                # "max_depth": [3, 5, 7],
                # "subsample": [0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
            },
            "AdaBoostClassifier": {
                "n_estimators": [50, 100, 200],
                # "learning_rate": [0.01, 0.1, 0.2],
            },
            "LogisticRegression": {
                "C": [0.01, 0.1, 1, 10],
                # "penalty": ["l1", "l2"],
                # "solver": ["liblinear", "saga"]
            },
            "DecisionTreeClassifier": {
                "max_depth": [None, 5, 10],
                # "min_samples_split": [2, 5, 10],
            },
            "KNeighborsClassifier": {
                "n_neighbors": [3, 5, 7],
                # "weights": ["uniform", "distance"],
                # "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
            }   
        }

        model_report: dict = self.evaluate_model(X_train, y_train, X_test, y_test, models, params)

        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name] ## Sine its Model object and best params are already set in evaluate_model function

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        classification_train_metric_artifact = get_classification_metrics(
                            y_true=y_train,
                            y_pred=y_train_pred
                        )
        
        classification_test_metric_artifact = get_classification_metrics(
                            y_true=y_test,
                            y_pred=y_test_pred
                        )

        self.track_mlflow(best_model = best_model, classification_metric_artifact=classification_train_metric_artifact)
        self.track_mlflow(best_model = best_model, classification_metric_artifact=classification_test_metric_artifact)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = self.model_trainer_config.model_trainer_dir
        os.makedirs(model_dir_path, exist_ok=True)

        Network_Model = NetworkSecurityModel(preprocessor=preprocessor, model=best_model)
        save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=Network_Model)

        save_object("final_model/model.pkl", best_model)

        return ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric_artifact,
            test_metric_artifact=classification_test_metric_artifact
        )
        

    def evaluate_model(self, X_train, y_train, X_test, y_test, models, params):
        try:
            model_report: dict = {}
            for model_name, model in models.items():
                param = params[model_name]
                gs = GridSearchCV(model, param, cv=5, n_jobs=-1)
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                model_report[model_name] = test_model_score
            
            return model_report
        
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e



    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array_data(file_path=transformed_train_file_path)

            logging.info(f"Loading transformed testing dataset")
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_array = load_numpy_array_data(file_path=transformed_test_file_path)

            logging.info(f"Splitting training and testing input and target feature")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            Model_Trainer_artifact = self.train_model(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
            logging.info(f"Model Trainer Artifact: {Model_Trainer_artifact}")
            return Model_Trainer_artifact
            
        
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e