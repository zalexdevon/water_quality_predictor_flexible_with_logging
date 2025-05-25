from Mylib import myfuncs, myclasses
import os


def load_data_for_model_evaluation_on_test(
    test_data_path,
    correction_transformer_path,
    transformation_transformer_path,
    class_names_path,
    model_path,
):
    test_data = myfuncs.load_python_object(test_data_path)
    correction_transformer = myfuncs.load_python_object(
        os.path.join(correction_transformer_path)
    )
    transformation_transformer = myfuncs.load_python_object(
        os.path.join(transformation_transformer_path)
    )
    model = myfuncs.load_python_object(model_path)
    class_names = myfuncs.load_python_object(os.path.join(class_names_path))

    return (
        test_data,
        correction_transformer,
        transformation_transformer,
        model,
        class_names,
    )


def transform_test_data(test_data, correction_transformer, transformation_transformer):
    test_data_corrected = correction_transformer.transform(test_data)
    test_data_transformed = transformation_transformer.transform(test_data_corrected)
    target_col = myfuncs.get_target_col_from_df_26(test_data_transformed)
    test_features = test_data_transformed.drop(columns=[target_col])
    test_target = test_data_transformed[target_col]

    return test_features, test_target


def evaluate_model_on_test(
    test_features,
    test_target,
    model,
    class_names,
    root_dir,
):

    final_model_results_text = "===========Kết quả đánh giá model ================\n"

    model_results_text, test_confusion_matrix = myclasses.ClassifierEvaluator(
        model=model,
        class_names=class_names,
        train_feature_data=test_features,
        train_target_data=test_target,
    ).evaluate()
    final_model_results_text += model_results_text

    # Lưu confusion matrix cho tập test
    test_confusion_matrix_path = os.path.join(root_dir, "test_confusion_matrix.png")
    test_confusion_matrix.savefig(
        test_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
    )

    # Lưu vào file results.txt
    with open(os.path.join(root_dir, "result.txt"), mode="w") as file:
        file.write(final_model_results_text)
