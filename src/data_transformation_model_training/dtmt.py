from Mylib import myfuncs
import os
import time
import re
from src.utils import funcs, classes
from sklearn.pipeline import Pipeline


def load_data_for_data_transformation_model_training(data_transformation_path):
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


def load_data_for_batch_data_transformation_model_training(data_transformation_path):
    val_feature_data = myfuncs.load_python_object(
        os.path.join(data_transformation_path, "val_features.pkl")
    )
    val_target_data = myfuncs.load_python_object(
        os.path.join(data_transformation_path, "val_target.pkl")
    )

    return val_feature_data, val_target_data


def create_and_save_models_before_training(model_training_path, model_indices, models):
    for model_index, model in zip(model_indices, models):
        model = myfuncs.convert_list_estimator_into_pipeline_59(model)
        myfuncs.save_python_object(
            os.path.join(model_training_path, f"{model_index}.pkl"), model
        )


def convert_list_estimator_into_pipeline_for_batch_dtmt(
    list_estimator, batch_size, batches_folder, scoring
):
    if list_estimator == []:
        return Pipeline(steps=[("passthrough", "passthrough")])

    transformers = list_estimator[:-1]
    model = list_estimator[-1]

    return Pipeline(
        steps=[
            (str(index), transformer) for index, transformer in enumerate(transformers)
        ]
        + [
            (
                str(len(transformers)),
                classes.ConvertTrainingDataToBatchesTransformer(
                    batch_size, batches_folder
                ),
            ),
            (
                str(len(transformers) + 1),
                classes.CustomClassifierForBatchDataTransformationModelTraining(
                    model, batches_folder, scoring
                ),
            ),
        ]
    )


def create_and_save_models_before_training_for_batch(
    model_training_path, model_indices, models, batch_size, batches_folder, scoring
):
    for model_index, model in zip(model_indices, models):
        model = convert_list_estimator_into_pipeline_for_batch_dtmt(
            model, batch_size, batches_folder, scoring
        )
        myfuncs.save_python_object(
            os.path.join(model_training_path, f"{model_index}.pkl"), model
        )


def train_and_save_models_for_batch(
    data_transformation_path,
    model_name,
    model_training_path,
    df_val_features,
    df_val_target,
    model_indices,
    num_models,
    scoring,
    plot_dir,
):
    print(
        f"\n========Bắt đầu train {num_models} models với chế độ chia từng batch================\n"
    )

    start_time = time.time()  # Bắt đầu tính thời gian train model
    for model_index in model_indices:
        # Load model để train
        model = myfuncs.load_python_object(
            os.path.join(model_training_path, f"{model_index}.pkl")
        )

        # Load train feature và train target lên
        df_train_features = myfuncs.load_python_object(
            os.path.join(data_transformation_path, "train_features.pkl")
        )
        df_train_target = myfuncs.load_python_object(
            os.path.join(data_transformation_path, "train_target.pkl")
        )

        # Train model
        print(f"Bắt đầu train  model {model_name} - {model_index}")
        model.fit(df_train_features, df_train_target)
        train_scoring = model.steps[-1][1].train_scoring
        print(f"Kết thúc train model {model_name} - {model_index}")

        val_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            model,
            df_val_features,
            df_val_target,
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
        f"\n========Kết thúc train {num_models} models với chế độ chia từng batch================\n"
    )
