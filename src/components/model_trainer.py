import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
       

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            mse = mean_squared_error(y_test, predicted)
            rmse = float(mse ** 0.5)
            mae = float(mean_absolute_error(y_test, predicted))

            importances = []
            try:
                import dill
                preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
                with open(preprocessor_path, 'rb') as f:
                    preprocessor = dill.load(f)
                raw_names = preprocessor.get_feature_names_out()
                
                clean_names = []
                for n in raw_names:
                    if n.startswith('num_pipeline__'): 
                        clean_names.append(n.replace('num_pipeline__', ''))
                    elif n.startswith('cat_pipelines__'): 
                        clean_names.append(n.replace('cat_pipelines__', ''))
                    else: 
                        clean_names.append(n)
                
                if hasattr(best_model, 'feature_importances_'):
                    fi = best_model.feature_importances_
                    for name, imp in zip(clean_names, fi):
                        importances.append({"feature": name, "importance": float(imp)})
                    importances = sorted(importances, key=lambda x: x['importance'], reverse=True)
                elif hasattr(best_model, 'coef_'):
                    import numpy as np
                    fi = np.abs(best_model.coef_)
                    total = np.sum(fi)
                    if total > 0:
                        fi = fi / total
                    for name, imp in zip(clean_names, fi):
                        importances.append({"feature": name, "importance": float(imp)})
                    importances = sorted(importances, key=lambda x: x['importance'], reverse=True)
            except Exception as e:
                logging.warning(f"Could not extract feature importances: {e}")

            return {
                "r2": float(r2_square),
                "rmse": rmse,
                "mae": mae,
                "best_model_name": best_model_name,
                "importances": importances
            }
        
            
        except Exception as e:
            raise CustomException(e,sys)