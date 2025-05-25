import os
from Mylib import myfuncs, fit_incremental_sl_model
import time
from src.utils import funcs


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


def save_models_before_training(model_training_path, model_indices, models):
    for model_index, model in zip(model_indices, models):
        myfuncs.create_directories([f"{model_training_path}/{model_index}"])

        myfuncs.save_python_object(
            f"{model_training_path}/{model_index}/model.pkl", model
        )


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


def train_and_save_models(
    model_training_path,
    model_name,
    model_indices,
    train_feature_data,
    train_target_data,
    val_feature_data,
    val_target_data,
    scoring,
    model_saving_val_scoring_limit,
):
    log_message = ""
    for model_index in model_indices:
        model_folder_path = f"{model_training_path}/{model_index}"

        # Load model để train
        model = myfuncs.load_python_object(f"{model_folder_path}/model.pkl")
        os.remove(
            f"{model_folder_path}/model.pkl"
        )  # Load xong thì file này ko cần thiết nữa

        full_model_index = f"{model_name}_{model_index}"

        print(f"Train model {full_model_index}")
        start_time = time.time()
        model.fit(train_feature_data, train_target_data)
        training_time = time.time() - start_time

        train_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            model,
            train_feature_data,
            train_target_data,
            scoring,
        )
        val_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            model,
            val_feature_data,
            val_target_data,
            scoring,
        )

        # In kết quả
        training_result_text = f"Model {full_model_index}\n -> Train {scoring}: {train_scoring}, Val {scoring}: {val_scoring}, Time: {training_time} (s)\n"
        print(training_result_text)

        # Logging
        log_message += training_result_text

        # Lưu model sau khi trained và kết quả
        save_model_after_training(
            model_saving_val_scoring_limit,
            val_scoring,
            scoring,
            model_folder_path,
            model,
        )
        myfuncs.save_python_object(
            f"{model_folder_path}/result.pkl",
            (full_model_index, train_scoring, val_scoring, training_time),
        )

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


def train_and_save_models_on_batch_training_data(
    batches_folder_path,
    model_name,
    model_training_path,
    val_feature_data,
    val_target_data,
    model_indices,
    num_batch,
    scoring,
    model_saving_val_scoring_limit,
):

    log_message = ""
    for model_index in model_indices:
        model_folder_path = f"{model_training_path}/{model_index}"
        # Load model để train
        model = myfuncs.load_python_object(f"{model_folder_path}/model.pkl")
        os.remove(
            f"{model_folder_path}/model.pkl"
        )  # Load xong thì file này ko cần thiết nữa

        full_model_index = f"{model_name}_{model_index}"

        print(f"Train model {full_model_index}")
        start_time = time.time()
        train_scoring = train_on_batches(model, batches_folder_path, num_batch, scoring)
        training_time = time.time() - start_time

        val_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            model,
            val_feature_data,
            val_target_data,
            scoring,
        )

        # In kết quả
        training_result_text = f"Model {full_model_index}\n -> Train {scoring}: {train_scoring}, Val {scoring}: {val_scoring}, Time: {training_time} (s)\n"
        print(training_result_text)

        # Logging
        log_message += training_result_text

        # Lưu model sau khi trained và kết quả
        save_model_after_training(
            model_saving_val_scoring_limit,
            val_scoring,
            scoring,
            model_folder_path,
            model,
        )
        myfuncs.save_python_object(
            f"{model_folder_path}/result.pkl",
            (full_model_index, train_scoring, val_scoring, training_time),
        )

    return log_message
