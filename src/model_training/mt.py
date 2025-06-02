import os
from Mylib import myfuncs, fit_incremental_sl_model
import time
from src.utils import funcs
from sklearn.base import clone
import gc
from sklearn.model_selection import ParameterSampler
import numpy as np


def load_data_for_model_training(data_transformation_path):
    train_feature_data = myfuncs.load_python_object(
        f"{data_transformation_path}/train_features.pkl"
    )
    train_target_data = myfuncs.load_python_object(
        f"{data_transformation_path}/train_target.pkl"
    )
    val_feature_data = myfuncs.load_python_object(
        f"{data_transformation_path}/val_features.pkl"
    )
    val_target_data = myfuncs.load_python_object(
        f"{data_transformation_path}/val_target.pkl"
    )

    return train_feature_data, train_target_data, val_feature_data, val_target_data
