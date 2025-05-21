from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from Mylib import myfuncs
import os


class ConvertTrainingDataToBatchesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer phục vụ cho data transformation_model training trong trường hợp chia nhiều batch ra để fit model <br>
    Tức là pipeline (n estimators) có từ 1 -> (n-1) là transformers, còn cái cuối là model <br>
    Vì vậy sau n - 1 transformers, cần chia ra các batch trước khi fit vào model, mục tiêu là khỏi bị tràn RAM

    Args:
        batch_size (_type_): kích cỡ batch
        training_batches_folder_path (_type_): thư mục lưu các batches
    """

    def __init__(self, batch_size, training_batches_folder_path) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.training_batches_folder_path = training_batches_folder_path

    def fit(self, X, y=None):
        # TODO: d
        print("Tiến hành chia batch cho tập train nè !!!!!")
        print("X: ", X)
        print("y: ", y)
        # d

        num_train_samples = len(X)

        start_indices = range(
            0, num_train_samples, self.batch_size
        )  # List các start_index của các batch
        num_batch = len(start_indices)

        # Save số lượng batch
        myfuncs.save_python_object(
            os.path.join(self.training_batches_folder_path, "num_batch.pkl"), num_batch
        )

        # Save training batch
        for batch_index, start_index in enumerate(start_indices):
            feature = X[start_index : start_index + self.batch_size, :]
            target = y[start_index : start_index + self.batch_size]

            # Save feature và target
            myfuncs.save_python_object(
                os.path.join(
                    self.training_batches_folder_path,
                    f"train_features_{batch_index}.pkl",
                ),
                feature,
            )
            myfuncs.save_python_object(
                os.path.join(
                    self.training_batches_folder_path,
                    f"train_target_{batch_index}.pkl",
                ),
                target,
            )

    def transform(self, X, y=None):
        # TODO: d
        print("Tập val thì không làm gì cả nhé !!!!!")
        # d

        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class CustomClassifierForBatchDataTransformationModelTraining(
    BaseEstimator, ClassifierMixin
):
    def __init__(self, model, data_transformation_path, scoring):
        """Phục vụ cho step model trong pipeline sử dụng trong data_transformation_model_training có batch <br>
        Đáp ứng cho việc train trên tập test và train trên tập train (có batch)

        Args:
            model (_type_): model được gói vào, vd: LogisticRegression(), XGBClassifier()
            data_transformation_path (_type_): đường dẫn đến các thư mục chứa các training batches
            num_batch (_type_): số lượng batch
            scoring (_type_): chỉ số
        """
        self.model = model
        self.data_transformation_path = data_transformation_path
        self.scoring = scoring

    def fit(self, X, y):
        # TODO: d
        print("Fit cho tập training nè")
        # d

        # Get num_batch
        num_batch = myfuncs.load_python_object(
            os.path.join(self.data_transformation_path, "num_batch.pkl")
        )

        list_train_scoring = []  # Cần biến này vì có thể sau này lấy min, max, ...

        # Fit batch đầu tiên
        first_feature_batch = myfuncs.load_python_object(
            os.path.join(self.data_transformation_path, f"train_features_0.pkl")
        )
        first_target_batch = myfuncs.load_python_object(
            os.path.join(self.data_transformation_path, f"train_target_0.pkl")
        )

        # Lần đầu nên fit bình thường
        print("Bắt đầu train batch thứ 0")
        self.model.fit(first_feature_batch, first_target_batch)
        print("Kết thúc train batch thứ 0")

        first_train_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            self.model,
            first_feature_batch,
            first_target_batch,
            self.scoring,
        )

        list_train_scoring.append(first_train_scoring)

        # Fit batch thứ 1 trở đi
        for i in range(1, num_batch):
            feature_batch = myfuncs.load_python_object(
                os.path.join(self.data_transformation_path, f"train_features_{i}.pkl")
            )
            target_batch = myfuncs.load_python_object(
                os.path.join(self.data_transformation_path, f"train_target_{i}.pkl")
            )

            # Lần thứ 1 trở đi thì fit theo kiểu incremental
            print(f"Bắt đầu train batch thứ {i}")
            myfuncs.fit_model_incremental_learning(
                self.model, feature_batch, target_batch
            )
            print(f"Kết thúc train batch thứ {i}")

            train_scoring = myfuncs.evaluate_model_on_one_scoring_17(
                self.model,
                feature_batch,
                target_batch,
                self.scoring,
            )

            list_train_scoring.append(train_scoring)

        self.train_scoring = list_train_scoring[-1]  # Lấy đại diện là phần tử cuối cùng

        print(f"Train scoring ở batch dtmt nè !!!!!!  = {self.train_scoring}")

        return self

    def predict(self, X):
        # TODO: d
        print(f"Chỗ này đi đánh giá nè")
        # d

        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
