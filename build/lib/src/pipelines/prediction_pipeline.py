import sys,os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e,sys)         
        
class CustomData:
    def __init__(self,
                 carat=float,
                 depth=float,
                 table=float,
                 x=float,
                 y=float,
                 z=float,
                 cut=float,
                 color=str,
                 clarity=str):
        carat=self.carat
        depth=self.depth
        table=self.table
        x=self.x
        y=self.y
        z=self.z
        cut=self.cut
        color=self.color
        clarity=self.clarity
    def get_data_as_dataframe(self):
        try:
               CustomData={
                   'carat':[self.carat],
                   'depth':[self.depth],
                   'table':[self.table],
                   'x':[self.x],
                   'y':[self.y],
                   'z':[self.z],
                   'cut':[self.cut],
                   'color':[self.color],
                   'clarity':[self.clarity]
                   }
               df=pd.DataFrame(CustomData)
               logging.info('DataFrame Gathered')
               return df
        except Exception as e:
            logging.info('dataframe not gathered due to error in prediction pipeline')
            raise CustomException(e ,sys)
        

        
