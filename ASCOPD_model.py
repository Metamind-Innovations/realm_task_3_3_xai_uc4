import subprocess
import pandas as pd
from pathlib import Path
from typing import Literal
from utils import load_csv, store_df_to_csv
import uuid


class DockerModelWrapper:
    def __init__(
        self,
        target: Literal["VenDep", "ARF", "Mortality"],
        in_mount: str = "temp_mount",
        out_mount: str = "temp_mount",
        docker_image: str = "forth_copd",
    ):
        """
        Args:
            target (Literal["VenDep", "ARF", "Mortality"]): Target column name to search
                                                            in Docker model output.
            in_mount (str): Local directory path to mount as /app/in in the container.
                            Defaults to `"temp_mount"`.
            out_mount (str): Local directory path to mount as /app/out in the container.
                             Defaults to `"temp_mount"`.
            docker_image (str): Name of the docker image (e.g., "forth_copd").
        """
        self.docker_image = docker_image
        self.in_mount = Path(in_mount).resolve()
        self.out_mount = Path(out_mount).resolve()
        self.target = target

    def fit(self, X=None, y=None):
        """
        Fit method to satisfy scikit-learn interface requirements.
        This method does nothing because the Dockerized model is already trained
        and cannot be retrained from Python. It exists solely to allow usage with
        scikit-learn utilities like permutation_importance that expect a fit method.

        Args:
            X: Input features. Defaults to `None`.
            y: Target labels. Defaults to `None`.

        Returns:
            self: Returns the class instance.
        """
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Run the dockerized model on the given input DataFrame.

        This method saves the input features to a temporary CSV file inside
        the mounted input directory, executes the Docker container with the
        model, and then reads back the predictions from the mounted output
        directory. Temporary input/output files are deleted after use.
        The temp files have a unique identifier.

        Args:
            X (pd.DataFrame): Input features as a pandas DataFrame.

        Returns:
            pd.DataFrame: Predictions loaded from the model's output CSV.
        """

        self.in_mount.mkdir(parents=True, exist_ok=True)
        self.out_mount.mkdir(parents=True, exist_ok=True)
        u_id = uuid.uuid4().hex

        # Save input data to a temp CSV inside input mount
        tmp_in_filename = f"temp_input_{u_id}.csv"
        input_file = self.in_mount / tmp_in_filename
        store_df_to_csv(df=X, path=input_file)

        # Run docker container
        tmp_out_filename = f"temp_output_{u_id}.csv"
        output_file = self.out_mount / tmp_out_filename

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self.in_mount}:/app/in",
            "-v",
            f"{self.out_mount}:/app/out",
            self.docker_image,
            "--input_data",
            f"/app/in/{tmp_in_filename}",
            "--output",
            f"/app/out/{tmp_out_filename}",
        ]
        subprocess.run(cmd, check=True)

        # Read predictions
        preds = load_csv(output_file)[self.target]

        # Cleanup
        try:
            input_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)
        except Exception as e:
            print(f"Warning: could not delete temp files: {e}")

        return preds
