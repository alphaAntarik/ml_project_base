
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd  
import numpy as np
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os
from src.utils import save_object

# this class is for what we want out of the file
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_tarnsformer_object(self,df_path,target):
       try:
         
           df=pd.read_csv(df_path)

        #    print(f'df_path {df.head()}')
           numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O' ]
           cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']
           numeric_features.remove(target)
         
        #    imputer is a tool used to handle missing data in datasets
           num_pipeline=Pipeline(
               steps=[
                  ( 'imputer',SimpleImputer(strategy='median')),  #as we have outliers and it is applied on numerical features
                  ('scaler',StandardScaler())
               ]
           )
           cat_pipeline=Pipeline(
               steps=[
                   ( 'imputer',SimpleImputer(strategy='most_frequent')),  #as we have outliers, and it is applied on categorical features, so replacing with mode
                   ('one_hot_encoder',OneHotEncoder()),
                  ("scaler",StandardScaler(with_mean=False))
               ]
           )
           logging.info('numerical columns standard scaler completed')
           logging.info('categorical columns one hot encoding completed')

           preprocessor=ColumnTransformer(transformers=[
               ('num_pipeline',num_pipeline,numeric_features),
               ('cat_pipeline',cat_pipeline,cat_features),
           ])
           return preprocessor
       except Exception as e:
            raise CustomException(e,sys)
       
    def initiate_data_transformation(self, train_path, test_path):
       try:
           train_df=pd.read_csv(train_path)
           test_df=pd.read_csv(test_path)

           logging.info('readin train and test data is done')
           logging.info('obtaining preprocessing object')
           print(f'obtaining preprocessing object {train_path}')
           target_col_name='math_score'
           preprocessing_obj=self.get_tarnsformer_object( df_path=train_path,target=target_col_name)
      
           input_train_features=train_df.drop(columns=[target_col_name],axis=1)
           target_train_feature=train_df[target_col_name]
           input_test_features=test_df.drop(columns=[target_col_name],axis=1)
           target_test_feature=test_df[target_col_name]

           logging.info('Applying preprocessing object on training and testing')
           input_features_train_transformed= preprocessing_obj.fit_transform(input_train_features)
           input_features_test_transformed= preprocessing_obj.transform(input_test_features)

            # what is this c_ , its a vvi important interview qs
            # The numpy.c_ object is a simple way to concatenate arrays along the second axis (columns). 
            # It is particularly useful for creating column vectors from 1-D arrays.
           train_arr=np.c_[input_features_train_transformed,np.array(target_train_feature)]
           test_arr=np.c_[input_features_test_transformed,np.array(target_test_feature)]

           logging.info(f"Saving preprocessing object.")
           save_object(
               file_path=self.data_transformation_config.preprocessor_obj_file_path,
               obj=preprocessing_obj
           )

           return(
               train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
           )


       except Exception as e:
        raise CustomException(e, sys)