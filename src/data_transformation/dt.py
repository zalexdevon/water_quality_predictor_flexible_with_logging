from Mylib import myfuncs, stringToObjectConverter, myclasses
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OrdinalEncoder,
)


def load_data_for_data_transformation(data_correction_path):
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

    # Số lượng training
    num_train_sample = len(df_train)

    # Các cột feature và cột target
    feature_cols, target_col = myfuncs.get_feature_cols_and_target_col_from_df_27(
        df_train
    )

    return (
        df_train,
        feature_ordinal_dict,
        correction_transformer,
        df_val,
        num_train_sample,
        feature_cols,
        target_col,
    )


def create_data_transformation_transformer(
    list_after_feature_transformer, feature_ordinal_dict, feature_cols, target_col
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
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("feature", feature_pipeline, feature_cols),
            ("target", OrdinalEncoder(), [target_col]),
        ]
    )

    transformation_transformer = myclasses.NamedColumnTransformer(column_transformer)

    return transformation_transformer


def do_transform_data_in_data_transformation(
    transformation_transformer,
    df_train,
    df_val,
    target_col,
    correction_transformer,
):
    df_train_transformed = transformation_transformer.fit_transform(df_train)
    df_train_feature = df_train_transformed.drop(columns=[target_col]).astype("float32")
    df_train_target = df_train_transformed[target_col].astype("int8")

    df_val_corrected = correction_transformer.transform(df_val)
    df_val_transformed = transformation_transformer.transform(df_val_corrected)
    df_val_feature = df_val_transformed.drop(columns=[target_col]).astype("float32")
    df_val_target = df_val_transformed[target_col].astype("int8")

    class_names = list(
        transformation_transformer.column_transformer.named_transformers_[
            "target"
        ].categories_[0]
    )

    return df_train_feature, df_train_target, df_val_feature, df_val_target, class_names


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
    transformation_transformer,
    df_train_feature,
    df_train_target,
    df_val_feature,
    df_val_target,
    class_names,
):
    myfuncs.save_python_object(
        os.path.join(data_transformation_path, "transformer.pkl"),
        transformation_transformer,
    )

    myfuncs.save_python_object(
        os.path.join(data_transformation_path, "train_features.pkl"),
        df_train_feature,
    )
    myfuncs.save_python_object(
        os.path.join(data_transformation_path, "train_target.pkl"),
        df_train_target,
    )
    myfuncs.save_python_object(
        os.path.join(data_transformation_path, "val_features.pkl"),
        df_val_feature,
    )
    myfuncs.save_python_object(
        os.path.join(data_transformation_path, "val_target.pkl"),
        df_val_target,
    )
    myfuncs.save_python_object(
        os.path.join(data_transformation_path, "class_names.pkl"),
        class_names,
    )


def create_weight_data_transformation_transformer(
    weights,
    list_after_feature_transformer,
    feature_ordinal_dict,
    feature_cols,
    target_col,
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
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("feature", feature_pipeline, feature_cols),
            ("target", OrdinalEncoder(), [target_col]),
        ]
    )

    column_transformer = Pipeline(
        steps=[
            ("1", column_transformer),
            (
                "2",
                myclasses.MultiplyWeightsTransformer(weights),
            ),  # Transformer cho weight
        ]
    )

    transformation_transformer = myclasses.NamedColumnTransformer(column_transformer)

    return transformation_transformer
