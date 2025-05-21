import os
from Mylib import myfuncs
import time
import re
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from Mylib.myclasses import CustomStackingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def load_data_for_model_training(data_transformation_path):
    # Load các training data
    train_feature_data = myfuncs.load_python_object(
        os.path.join(data_transformation_path, "train_features.pkl")
    )
    train_target_data = myfuncs.load_python_object(
        os.path.join(data_transformation_path, "train_target.pkl")
    )
    val_feature_data = myfuncs.load_python_object(
        os.path.join(data_transformation_path, "val_features.pkl")
    )
    val_target_data = myfuncs.load_python_object(
        os.path.join(data_transformation_path, "val_target.pkl")
    )

    return train_feature_data, train_target_data, val_feature_data, val_target_data


def save_models_before_training(model_training_path, model_indices, models):
    for model_index, model in zip(model_indices, models):
        myfuncs.save_python_object(
            os.path.join(model_training_path, f"{model_index}.pkl"), model
        )


def load_data_for_batch_model_training(data_transformation_path, batches_folder):
    num_batch = myfuncs.load_python_object(
        os.path.join(batches_folder, "num_batch.pkl")
    )
    val_feature_data = myfuncs.load_python_object(
        os.path.join(data_transformation_path, "val_features.pkl")
    )
    val_target_data = myfuncs.load_python_object(
        os.path.join(data_transformation_path, "val_target.pkl")
    )

    return num_batch, val_feature_data, val_target_data


def train_on_batches(model, data_transformation_path, num_batch, scoring):
    list_train_scoring = []  # Cần biến này vì có thể sau này lấy min, max, ...

    # Fit batch đầu tiên
    first_feature_batch = myfuncs.load_python_object(
        os.path.join(data_transformation_path, f"train_features_0.pkl")
    )
    first_target_batch = myfuncs.load_python_object(
        os.path.join(data_transformation_path, f"train_target_0.pkl")
    )

    # Lần đầu nên fit bình thường
    print("Bắt đầu train batch thứ 0")
    model.fit(first_feature_batch, first_target_batch)
    print("Kết thúc train batch thứ 0")

    first_train_scoring = myfuncs.evaluate_model_on_one_scoring_17(
        model,
        first_feature_batch,
        first_target_batch,
        scoring,
    )

    list_train_scoring.append(first_train_scoring)

    # Fit batch thứ 1 trở đi
    for i in range(1, num_batch):
        feature_batch = myfuncs.load_python_object(
            os.path.join(data_transformation_path, f"train_features_{i}.pkl")
        )
        target_batch = myfuncs.load_python_object(
            os.path.join(data_transformation_path, f"train_target_{i}.pkl")
        )

        # Lần thứ 1 trở đi thì fit theo kiểu incremental
        print(f"Bắt đầu train batch thứ {i}")
        myfuncs.fit_model_incremental_learning(model, feature_batch, target_batch)
        print(f"Kết thúc train batch thứ {i}")

        train_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            model,
            feature_batch,
            target_batch,
            scoring,
        )

        list_train_scoring.append(train_scoring)

    return list_train_scoring[-1]  # Lấy kết quả trên batch cuối cùng


def train_and_save_models_on_batch_training_data(
    data_transformation_path,
    model_name,
    model_training_path,
    val_feature_data,
    val_target_data,
    model_indices,
    num_batch,
    num_models,
    scoring,
    plot_dir,
):
    print(
        f"\n========Bắt đầu train {num_models} models với số batch = {num_batch}!!!!!!================\n"
    )

    start_time = time.time()  # Bắt đầu tính thời gian train model
    for model_index in model_indices:
        # Load model để train
        model = myfuncs.load_python_object(
            os.path.join(model_training_path, f"{model_index}.pkl")
        )

        print(f"Bắt đầu train  model {model_name} - {model_index}")
        train_scoring = train_on_batches(
            model, data_transformation_path, num_batch, scoring
        )
        print(f"Kết thúc train model {model_name} - {model_index}")

        val_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            model,
            val_feature_data,
            val_target_data,
            scoring,
        )

        # In kết quả
        print("Kết quả của model")
        print(
            f"Model index {model_name} - {model_index}\n -> Train {scoring}: {train_scoring}, Val {scoring}: {val_scoring}\n"
        )

        # Lưu model sau khi trained
        myfuncs.save_python_object(
            os.path.join(model_training_path, f"{model_index}.pkl"), model
        )

        # Lưu dữ liệu để vẽ biểu đồ
        model_name_in_plot = f"{model_name}_{model_index}"

        myfuncs.save_python_object(
            os.path.join(plot_dir, f"{model_name_in_plot}.pkl"),
            (model_name_in_plot, train_scoring, val_scoring),
        )

    all_model_end_time = time.time()  # Kết thúc tính thời gian train model
    true_all_models_train_time = (all_model_end_time - start_time) / 60

    print(f"Thời gian chạy tất cả: {true_all_models_train_time} (mins)")
    print(
        f"\n========Kết thúc train {num_models} models với số batch = {num_batch}!!!!!!================\n"
    )
