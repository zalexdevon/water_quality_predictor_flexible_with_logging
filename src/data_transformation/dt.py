from Mylib import myfuncs, stringToObjectConverter, myclasses
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


def load_data_for_data_transformation(data_correction_path, class_names_path):
    # Load df train đã corrected
    df_train = myfuncs.load_python_object(
        os.path.join(data_correction_path, "data.pkl")
    )

    # Load dict để biến đổi các biến ordinal
    feature_ordinal_dict = myfuncs.load_python_object(
        os.path.join(
            data_correction_path,
            "feature_ordinal_dict.pkl",
        )
    )

    # Load transformer của data correction để sau này transform val_data
    correction_transformer = myfuncs.load_python_object(
        os.path.join(data_correction_path, "transformer.pkl")
    )

    # Load val data đã corrected
    val_data_path = "artifacts/data_ingestion/val_data.pkl"
    df_val = myfuncs.load_python_object(val_data_path)

    # Các cột feature và cột target
    feature_cols, target_col = myfuncs.get_feature_cols_and_target_col_from_df_27(
        df_train
    )

    # Get class names
    class_names = myfuncs.load_python_object(class_names_path)

    return (
        df_train,
        feature_ordinal_dict,
        correction_transformer,
        df_val,
        feature_cols,
        target_col,
        class_names,
    )


def create_data_transformation_transformer(
    list_after_feature_transformer,
    feature_ordinal_dict,
    feature_cols,
    target_col,
    class_names,
):
    after_feature_pipeline = myfuncs.convert_list_estimator_into_pipeline_59(
        list_after_feature_transformer
    )

    feature_pipeline = Pipeline(
        steps=[
            (
                "during",
                myclasses.DuringFeatureTransformer(feature_ordinal_dict),
            ),
            ("after", after_feature_pipeline),
            ("final_scale", MinMaxScaler()),
        ]
    )
    feature_transformer = myclasses.NamedColumnTransformer(
        ColumnTransformer(transformers=[("1", feature_pipeline, feature_cols)])
    )

    target_transformer = myclasses.NamedColumnTransformer(
        ColumnTransformer(
            transformers=[("1", OrdinalEncoder(categories=[class_names]), [target_col])]
        )
    )

    return feature_transformer, target_transformer


def do_transform_data_in_data_transformation(
    feature_transformer,
    target_transformer,
    df_train,
    df_val,
    correction_transformer,
):
    df_train_feature = feature_transformer.fit_transform(df_train).astype("float32")
    df_train_target = (
        target_transformer.fit_transform(df_train).values.reshape(-1).astype("int8")
    )

    df_val_corrected = correction_transformer.transform(df_val)
    df_val_feature = feature_transformer.transform(df_val_corrected).astype("float32")
    df_val_target = (
        target_transformer.transform(df_val_corrected).values.reshape(-1).astype("int8")
    )

    return df_train_feature, df_train_target, df_val_feature, df_val_target


def save_training_batches(
    df_train_feature,
    df_train_target,
    batch_size,
    training_batches_folder_path,
):
    num_train_samples = len(df_train_feature)
    start_indices = range(
        0, num_train_samples, batch_size
    )  # List các start_index của các batch
    num_batch = len(start_indices)

    # Save số lượng batch
    myfuncs.save_python_object(
        os.path.join(training_batches_folder_path, "num_batch.pkl"), num_batch
    )

    # Save training batch
    for batch_index, start_index in enumerate(start_indices):
        feature = df_train_feature.iloc[start_index : start_index + batch_size, :]
        target = df_train_target.iloc[start_index : start_index + batch_size]

        # Save feature và target
        myfuncs.save_python_object(
            os.path.join(
                training_batches_folder_path,
                f"train_features_{batch_index}.pkl",
            ),
            feature,
        )
        myfuncs.save_python_object(
            os.path.join(
                training_batches_folder_path,
                f"train_target_{batch_index}.pkl",
            ),
            target,
        )


def save_data_for_data_transformation(
    data_transformation_path,
    feature_transformer,
    target_transformer,
    df_train_feature,
    df_train_target,
    df_val_feature,
    df_val_target,
):
    myfuncs.save_python_object(
        f"{data_transformation_path}/feature_transformer.pkl", feature_transformer
    )
    myfuncs.save_python_object(
        f"{data_transformation_path}/target_transformer.pkl", target_transformer
    )
    myfuncs.save_python_object(
        f"{data_transformation_path}/train_features.pkl", df_train_feature
    )
    myfuncs.save_python_object(
        f"{data_transformation_path}/train_target.pkl", df_train_target
    )
    myfuncs.save_python_object(
        f"{data_transformation_path}/val_features.pkl", df_val_feature
    )
    myfuncs.save_python_object(
        f"{data_transformation_path}/val_target.pkl", df_val_target
    )
