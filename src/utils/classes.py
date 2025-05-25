from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from Mylib import myfuncs
import os
from datetime import datetime
import plotly.express as px
from src.utils import funcs
import pandas as pd


class LoggingDisplayer:
    DATE_FORMAT = "%d-%m-%Y-%H-%M-%S"
    READ_FOLDER_NAME = "artifacts/logs"
    WRITE_FOLDER_NAME = "artifacts/gather_logs"

    # Tạo thư mục
    os.makedirs(WRITE_FOLDER_NAME, exist_ok=True)

    def __init__(self, mode, file_name=None, start_time=None, end_time=None):
        self.mode = mode
        self.file_name = file_name
        self.start_time = start_time
        self.end_time = end_time

        if self.file_name is None:
            self.file_name = f"{datetime.now().strftime(self.DATE_FORMAT)}.log"

    def print_and_save(self):
        file_path = f"{self.WRITE_FOLDER_NAME}/{self.file_name}"

        if self.mode == "all":
            result = self.gather_all_logging_result()
        else:
            result = self.gather_logging_result_from_start_to_end_time()

        print(result)
        print(f"Lưu result tại {file_path}")
        myfuncs.write_content_to_file(result, file_path)

    def gather_all_logging_result(self):
        logs_filenames = self.get_logs_filenames()

        return self.read_from_logs_filenames(logs_filenames)

    def gather_logging_result_from_start_to_end_time(self):
        logs_filenames = pd.Series(self.get_logs_filenames())
        logs_filenames = logs_filenames[
            (logs_filenames > self.start_time) & (logs_filenames < self.end_time)
        ].tolist()

        return self.read_from_logs_filenames(logs_filenames)

    def read_from_logs_filenames(self, logs_filenames):
        result = ""
        for logs_filename in logs_filenames:
            logs_filepath = f"{self.READ_FOLDER_NAME}/{logs_filename}.log"
            content = myfuncs.read_content_from_file_60(logs_filepath)
            result += f"{content}\n\n"

        return result

    def get_logs_filenames(self):
        logs_filenames = os.listdir(self.READ_FOLDER_NAME)
        date_format_in_filename = f"{self.DATE_FORMAT}.log"
        logs_filenames = [
            datetime.strptime(item, date_format_in_filename) for item in logs_filenames
        ]
        logs_filenames = sorted(logs_filenames)  # Sắp xếp theo thời gian tăng dần
        return logs_filenames


class ModelTrainingResultPlotter:
    def __init__(self, max_val_value, target_val_value):
        self.max_val_value = max_val_value
        self.target_val_value = target_val_value

    def plot(self):
        components = funcs.gather_result_from_model_training()
        fig = self.plot_from_components(components)

        fig.show()

    def plot_from_components(self, components):
        model_names, train_scores, val_scores, _ = zip(*components)

        for i in range(len(train_scores)):
            if train_scores[i] > self.max_val_value:
                train_scores[i] = self.max_val_value

            if val_scores[i] > self.max_val_value:
                val_scores[i] = self.max_val_value

        # Vẽ biểu đồ
        df = pd.DataFrame(
            {
                "x": model_names,
                "train": train_scores,
                "val": val_scores,
            }
        )

        df_long = df.melt(
            id_vars=["x"],
            value_vars=["train", "val"],
            var_name="Category",
            value_name="y",
        )

        fig = px.line(
            df_long,
            x="x",
            y="y",
            color="Category",
            markers=True,
            color_discrete_map={
                "train": "gray",
                "val": "blue",
            },
            hover_data={"x": False, "y": True, "Category": False},
        )

        fig.add_hline(
            y=self.max_val_value,
            line_dash="solid",
            line_color="black",
            line_width=2,
        )

        fig.add_hline(
            y=self.target_val_value,
            line_dash="dash",
            line_color="green",
            line_width=2,
        )

        fig.update_layout(
            autosize=False,
            width=100 * (len(model_names) + 2) + 30,
            height=400,
            margin=dict(l=30, r=10, t=10, b=0),
            xaxis=dict(
                title="",
                range=[
                    0,
                    len(model_names),
                ],
                tickmode="linear",
            ),
            yaxis=dict(
                title="",
                range=[0, self.max_val_value],
            ),
            showlegend=False,
        )

        return fig
