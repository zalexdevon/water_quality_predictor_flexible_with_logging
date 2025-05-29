import os
from Mylib import myfuncs, fit_incremental_sl_model
import time
from src.utils import funcs
from sklearn.base import clone
import gc
from sklearn.model_selection import ParameterSampler
import numpy as np

SCORINGS_PREFER_MININUM = ["log_loss", "mse", "mae"]
SCORINGS_PREFER_MAXIMUM = ["accuracy"]


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


def load_data_for_batch_model_training(data_transformation_path, batches_folder_path):
    num_batch = myfuncs.load_python_object(f"{batches_folder_path}/num_batch.pkl")
    val_feature_data = myfuncs.load_python_object(
        f"{data_transformation_path}/val_features.pkl"
    )
    val_target_data = myfuncs.load_python_object(
        f"{data_transformation_path}/val_target.pkl"
    )

    return num_batch, val_feature_data, val_target_data


def is_val_scoring_better_than_target_scoring(target_scoring, val_scoring, scoring):
    if scoring in funcs.SCORINGS_PREFER_MAXIMUM:
        return val_scoring > target_scoring
    if scoring in funcs.SCORINGS_PREFER_MININUM:
        return val_scoring < target_scoring

    raise ValueError(f"Chưa định nghĩa cho {scoring}")


def save_model_after_training(
    model_saving_val_scoring_limit, val_scoring, scoring, model_folder_path, model
):
    do_allow_to_save_model = is_val_scoring_better_than_target_scoring(
        model_saving_val_scoring_limit, val_scoring, scoring
    )

    # Tốt hơn thì mới lưu lại model
    if do_allow_to_save_model:
        myfuncs.save_python_object(f"{model_folder_path}/fitted_model.pkl", model)


def get_list_param(param_dict, num_models):
    param_list = list(ParameterSampler(param_dict, n_iter=num_models, random_state=42))
    return param_list


def create_model(base_model, param):
    model = clone(base_model)
    model.set_params(**param)
    return model


def get_sign_for_val_scoring_find_best_model(scoring):
    if scoring in SCORINGS_PREFER_MININUM:
        return -1

    if scoring in SCORINGS_PREFER_MAXIMUM:
        return 1

    raise ValueError(f"Chưa định nghĩa cho {scoring}")


def train_models(
    model_training_path,
    num_models,
    base_model,
    param_dict,
    train_feature,
    train_target,
    val_feature,
    val_target,
    scoring,
    model_saving_val_scoring_limit,
):
    log_message = ""

    list_param = get_list_param(param_dict, num_models)
    best_val_scoring = -np.inf
    sign_for_val_scoring_find_best_model = get_sign_for_val_scoring_find_best_model(
        scoring
    )
    model_saving_val_scoring_limit = (
        model_saving_val_scoring_limit * sign_for_val_scoring_find_best_model
    )

    for i, param in enumerate(list_param):
        # Tạo model
        model = create_model(base_model, param)

        # Train model
        print(f"Train model {i} / {num_models}")
        start_time = time.time()
        model.fit(train_feature, train_target)
        training_time = time.time() - start_time

        train_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            model,
            train_feature,
            train_target,
            scoring,
        )
        val_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            model,
            val_feature,
            val_target,
            scoring,
        )

        # In kết quả
        training_result_text = f"{param}\n -> Val {scoring}: {val_scoring}, Train {scoring}: {train_scoring}, Time: {training_time} (s)\n"
        print(training_result_text)

        # Logging
        log_message += training_result_text

        # Cập nhật best model và lưu lại
        val_scoring_find_best_model = val_scoring * sign_for_val_scoring_find_best_model

        if best_val_scoring < val_scoring_find_best_model:
            best_val_scoring = val_scoring_find_best_model

            # Lưu model
            if best_val_scoring > model_saving_val_scoring_limit:
                myfuncs.save_python_object(f"{model_training_path}/model.pkl", model)

            # Lưu kết quả
            myfuncs.save_python_object(
                f"{model_training_path}/result.pkl",
                (param, val_scoring, train_scoring, training_time),
            )

        # Giải phóng bộ nhớ
        del model
        gc.collect()

    return log_message


def train_on_batches(model, batches_folder_path, num_batch, scoring):
    list_train_scoring = []  # Cần biến này vì có thể sau này lấy min, max, ... tùy ý

    # Fit batch đầu tiên
    first_feature_batch = myfuncs.load_python_object(
        f"{batches_folder_path}/train_features_0.pkl"
    )
    first_target_batch = myfuncs.load_python_object(
        f"{batches_folder_path}/train_target_0.pkl"
    )

    # Lần đầu nên fit bình thường
    print("Train batch thứ 0")
    model.fit(first_feature_batch, first_target_batch)

    first_train_scoring = myfuncs.evaluate_model_on_one_scoring_17(
        model,
        first_feature_batch,
        first_target_batch,
        scoring,
    )

    list_train_scoring.append(first_train_scoring)

    # Fit batch thứ 1 trở đi
    for i in range(1, num_batch - 1 + 1):
        feature_batch = myfuncs.load_python_object(
            f"{batches_folder_path}/train_features_{i}.pkl"
        )
        target_batch = myfuncs.load_python_object(
            f"{batches_folder_path}/train_target_{i}.pkl"
        )

        # Lần thứ 1 trở đi thì fit theo kiểu incremental
        print(f"Train batch thứ {i}")
        fit_incremental_sl_model.fit_model_incremental_learning(
            model, feature_batch, target_batch
        )

        train_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            model,
            feature_batch,
            target_batch,
            scoring,
        )

        list_train_scoring.append(train_scoring)

    return list_train_scoring[-1]  # Lấy kết quả trên batch cuối cùng


def train_models_batch(
    batches_folder_path,
    model_training_path,
    num_models,
    base_model,
    param_dict,
    val_feature,
    val_target,
    scoring,
    num_batch,
    model_saving_val_scoring_limit,
):

    log_message = ""
    list_param = get_list_param(param_dict, num_models)
    best_val_scoring = -np.inf
    sign_for_val_scoring_find_best_model = get_sign_for_val_scoring_find_best_model(
        scoring
    )
    model_saving_val_scoring_limit = (
        model_saving_val_scoring_limit * sign_for_val_scoring_find_best_model
    )

    for i, param in enumerate(list_param):
        # Tạo model
        model = create_model(base_model, param)

        # Train model
        print(f"Train model {i} / {num_models}")
        start_time = time.time()
        train_scoring = train_on_batches(model, batches_folder_path, num_batch, scoring)
        training_time = time.time() - start_time

        val_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            model,
            val_feature,
            val_target,
            scoring,
        )

        # In kết quả
        training_result_text = f"{param}\n -> Val {scoring}: {val_scoring}, Train {scoring}: {train_scoring}, Time: {training_time} (s)\n"
        print(training_result_text)

        # Logging
        log_message += training_result_text

        # Cập nhật best model và lưu lại
        val_scoring_find_best_model = val_scoring * sign_for_val_scoring_find_best_model

        if best_val_scoring < val_scoring_find_best_model:
            best_val_scoring = val_scoring_find_best_model

            # Lưu model
            if best_val_scoring > model_saving_val_scoring_limit:
                myfuncs.save_python_object(f"{model_training_path}/model.pkl", model)

            # Lưu kết quả
            myfuncs.save_python_object(
                f"{model_training_path}/result.pkl",
                (param, val_scoring, train_scoring, training_time),
            )

        # Giải phóng bộ nhớ
        del model
        gc.collect()

    return log_message
