import os,sys
import numpy as np
from dataclasses import dataclass
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping
from src.exception import CustomException
from src.logger import logging
from src.utils import save_model, save_json_object

@dataclass
class ModelTrainerConfig():
    model_path = os.path.join('artifacts','model.pkl')
    model_report_path = os.path.join('artifacts','models_report.json')
class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            x_train,y_train,x_test,y_test = (train_array[:,:-1],train_array[:,-1],
                                             test_array[:,:-1],test_array[:,-1])
            n_cols = x_train.shape[1]


            model = keras.Sequential()

            model.add(keras.layers.Dense(150, activation=tf.nn.relu,input_shape=(n_cols,)))
            model.add(keras.layers.Dense(150, activation=tf.nn.relu))
            model.add(keras.layers.Dense(150, activation=tf.nn.relu))
            model.add(keras.layers.Dense(150, activation=tf.nn.relu))
            model.add(keras.layers.Dense(150, activation=tf.nn.relu))
            model.add(keras.layers.Dense(1))


            model.compile(loss='mse', optimizer='adam', metrics=['mae']) # use metric as mean absolute error
            early_stop = EarlyStopping(monitor='val_loss', patience=15) 
            history = model.fit(x_train, y_train, epochs=300,validation_split=0.2, verbose=1, callbacks=[early_stop])
            model.summary()
            score = model.evaluate(x_test, y_test, verbose=1)
            logging.info('loss value: ', score[0])
            logging.info('Mean absolute error: ', score[1])
            save_model(self.model_trainer_config.model_path,model)
            save_json_object(self.model_trainer_config.model_report_path,{"Simple Neural Network":score})
            test_predictions = model.predict(x_test).flatten()
            print(test_predictions)
            logging.info("Model Training Completed.")
        except Exception as e:
            raise CustomException(e,sys)
