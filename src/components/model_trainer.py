
import os,sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.utils import evaluate_model
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
      trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
     def __init__(self):
           self.model_trainer_config=ModelTrainerConfig()
        
     def initiate_model_training(self,train_array,test_array):
            try:
                  logging.info('Splitting  dependent and independent data from train and test')
                  X_train,X_test,y_train,y_test =(
                         train_array[:,:-1],
                         test_array[:,:-1], 
                         train_array[:,-1], 
                         test_array[:,-1])
                  models={
                        'linear regression':LinearRegression(),
                        'decision tree rgression':DecisionTreeRegressor(),
                        'lasso regression':Lasso(),
                        'elasticnet':ElasticNet(),
                        'ridge regression':Ridge()
                  }
                  model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
                  print(model_report)
                  print("="*35)
                  logging.info(f'model report is: {model_report}')
                  best_model_score = max(sorted(model_report.values()))
                  best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
                  best_model=models[best_model_name]
                  print('='*35)
                  print(f'best model name is:{best_model} and r2 score is:{best_model_score}')
                  

                  save_object(
                        file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model
                  
                  )

               
            except Exception as e:
                  raise CustomException(e,sys)