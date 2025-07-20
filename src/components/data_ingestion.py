# here we read the data from the data source
# after this we need to do data tarnsformation


import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd  
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainingConfig
from src.components.model_trainer import ModelTrainer
# this class is for what we want out of the file
@dataclass
class DataIngestionConfig:
    train_data_path=str=os.path.join('artifacts','train.csv')
    test_data_path=str=os.path.join('artifacts','test.csv')
    raw_data_path=str=os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered in data ingestion method or component')
        try:
            df=pd.read_csv('notebook/data/stud.csv')  #here we are reading it from the data source, it can be from db or any other source as well
            logging.info('Read the datasets as dataframe')

            # here we are making the directries if they dont exist..... 
            # even if we skip the lines, it will create anyways
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)  #here we are storing the raw data
            logging.info('Train rest split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            #storing the train and test data
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) 
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('train test split completed')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)



# here is for initiating the full process-
if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path= obj.initiate_data_ingestion()
   
    data_transformation=DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_path=train_data_path,test_path=test_data_path)
    
    modelTrainer=ModelTrainer()
    r2= modelTrainer.initiate_model_trainer(train_array=train_arr,test_array=test_arr)
    print(f'score is {r2}')