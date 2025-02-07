import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTransformationConfig()

    def get_data_transform_object(self):
        try:
            numerical_columns = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
            categorical_columns = ["Gender"]

            # Pipeline for numerical columns
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler()),
            ])

            # Pipeline for categorical columns
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            logging.info("Numerical columns scaling and categorical encoding completed")

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initial_data_transform(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")

            # Reading data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded train and test data successfully")

            preprocessing_obj = self.get_data_transform_object()
            
            target_col_name = "Calories"
            input_feature_train_df = train_df.drop(columns=[target_col_name], axis=1)
            input_target_train_df = train_df[target_col_name]
            input_feature_test_df = test_df.drop(columns=[target_col_name], axis=1)
            input_target_test_df = test_df[target_col_name]

            logging.info("Applying transformations on train and test datasets")

            # Transform features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combining transformed features with target variable
            train_arr = np.c_[input_feature_train_arr, np.array(input_target_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(input_target_test_df)]

            logging.info("Data transformation completed successfully")

            # Save preprocessor object
            save_object(
                file_path=self.data_transform_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transform_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
