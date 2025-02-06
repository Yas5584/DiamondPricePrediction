
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
import sys,os
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from  dataclasses import dataclass
from src.utils import  save_object


@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Process started')
            numerical_col=['carat','depth','table','x','y','z']
            categorical_col=['cut','color','clarity']
            
            cut_categories=['Fair','Good','Very Good','Premium','Ideal']
            color_categories=['D','E','F','G','H','I','J']
            clarity_categories=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline concept started')
            numerical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            categorical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OrdinalEncoder()),
                    ('scaler',StandardScaler())

                ]
            )
            preprocessor=ColumnTransformer(
                transformers=[
                    ('numerical_pipeline',numerical_pipeline,numerical_col),
                    ('categorical_pipeline',categorical_pipeline,categorical_col)
                ]
            )
            return preprocessor
            logging.info('Pipeline completed')
        except Exception as e:
            raise CustomException(e, "Failed to get data transformation object") from e
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info('read train and test are completed')
            logging.info(f'Train DataFrame Path: \n{train_df.head().to_string()} ')
            logging.info(f'Test DataFrame Path: \n{test_df.head().to_string()} ')
            logging.info('Obtainig Preproessor object')

            preprocessor_obj=self.get_data_transformation_object()
            target_column_name='price'
            drop_column_name=['id',target_column_name]

            input_feature_train_df=train_df.drop(columns=drop_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=drop_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]


            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and testing data')
        
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            
            save_object(
              file_path=self.data_transformation_config.preprocessor_obj_file_path,
              obj= preprocessor_obj
            )
            
            logging.info('pickle file created and completed')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )
        except Exception as e:
            raise CustomException(e,sys) 