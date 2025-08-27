import subprocess
import pandas as pd
from pathlib import Path
from typing import Literal
from utils import load_csv, store_df_to_csv


class DockerModelWrapper:
    def __init__(
        self,
        in_mount: str,
        out_mount: str,
        target: Literal["VenDep", "ARF", "Mortality"],
        docker_image: str = "forth_copd",
    ):
        """
        Args:
            docker_image (str): Name of the docker image (e.g., "forth_copd").
            in_mount (str): Local directory path to mount as /app/in in the container.
            target (Literal["VenDep", "ARF", "Mortality"]): Target column name to search
                                                            in Docker model output.
            out_mount (str): Local directory path to mount as /app/out in the container.
        """
        self.docker_image = docker_image
        self.in_mount = Path(in_mount)
        self.out_mount = Path(out_mount)
        self.target = target

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Run the dockerized model on the given input DataFrame.

        This method saves the input features to a temporary CSV file inside
        the mounted input directory, executes the Docker container with the
        model, and then reads back the predictions from the mounted output
        directory. Temporary input/output files are deleted after use.

        Args:
            X (pd.DataFrame): Input features as a pandas DataFrame.

        Returns:
            pd.DataFrame: Predictions loaded from the model's output CSV.
        """

        self.in_mount.mkdir(parents=True, exist_ok=True)
        self.out_mount.mkdir(parents=True, exist_ok=True)

        # Save input data to a temp CSV inside input mount
        input_file = self.in_mount / "temp_input.csv"
        store_df_to_csv(df=X, path=input_file)

        # Run docker container
        output_file = self.out_mount / "temp_output.csv"

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
            "/app/in/temp_input.csv",
            "--output",
            "/app/out/temp_output.csv",
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
