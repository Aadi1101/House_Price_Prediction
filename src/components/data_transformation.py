import os,sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from keras.utils import normalize 

from src.exception import CustomException
from src.logger import logging

class DataTransformation():
    def initiate_data_transformation(self,trainset_path,testset_path):
        try:
            train_df = pd.read_csv(trainset_path)
            test_df = pd.read_csv(testset_path)
            target_column_name = 'Price'

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df['Price']

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on both training and testing object.")

            input_feature_train_arr = normalize(input_feature_train_df)
            input_feature_test_arr = normalize(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            logging.info("Data Transformation complete now saving the preprocessing object")
            return(
                train_arr,test_arr
            )
        except Exception as e:
            raise CustomException(e,sys)
