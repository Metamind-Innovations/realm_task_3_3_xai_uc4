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
        in_docker_run: bool = False,
    ):
        """
        Wrapper for a Dockerized model to allow prediction from Python.

        Args:
            target (Literal["VenDep", "ARF", "Mortality"]): Name of the target column
                in the Docker model output CSV.
            in_mount (str, optional): Local directory to mount as `/app/in` inside
                the container. Defaults to `"temp_mount"`.
            out_mount (str, optional): Local directory to mount as `/app/out` inside
                the container. Defaults to `"temp_mount"`.
            docker_image (str, optional): Name of the Docker image containing the model.
                Defaults to `"forth_copd"`.
            in_docker_run (str, optional): Indicates if the class runs inside the container.
                Defaults to `False`.
        """
        self.docker_image = docker_image
        self.in_mount = Path(in_mount).resolve()
        self.out_mount = Path(out_mount).resolve()
        self.target = target
        self.in_docker_run = in_docker_run

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

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Run the Dockerized model on the given input and return predictions.
        This method saves the input to a temporary CSV, runs the Docker container
        and reads back the output CSV. Checks if the process is executed inside
        the model container and runs the appropriate command.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.Series: Predictions corresponding to the target column.
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

        if not self.in_docker_run:
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
        else:
            cmd = [
                "/app/main",
                "--input_data",
                str(input_file),
                "--output",
                str(output_file),
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

    def predict_proba(self, X):
        """
        Method to satisfy scikit-learn backend interface
        requirements for dice library.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.Series: Model predictions.
        """
        return self.predict(X)
