from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model


# -----------------------
# Step 1: Download Repo
# -----------------------
@dsl.component(base_image="python:3.13-slim")
def download_repo(
    github_repo_url: str,
    project_files: Output[Model],
    data: Output[Dataset],
    branch: str = "main",
) -> None:
    """Download specific scripts and data from a GitHub repository.
    This component clones a GitHub repository, copies selected Python scripts
    into the `project_files` output, and the `data` folder into the `data` output.

    Args:
        github_repo_url (str): URL of the GitHub repository to clone.
        project_files (Output[Model]): Output path for project scripts.
        data (Output[Dataset]): Output path for data folder.
        branch (str): Branch name to pull from (defaults to 'main').
    """
    import shutil
    from pathlib import Path
    import subprocess

    repo_dir = Path("/tmp/repo")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    print("Installing git...")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "git"], check=True)

    print(f"Cloning repo {github_repo_url} (branch: {branch})...")
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            branch,
            "--single-branch",
            github_repo_url,
            str(repo_dir),
        ],
        check=True,
    )

    # List of files to copy
    files_to_copy = [
        "ASCOPD_model.py",
        "explainer.py",
        "fairness_bias_analysis.py",
        "utils.py",
    ]

    # Copy specific scripts
    proj_path = Path(project_files.path)
    proj_path.mkdir(parents=True, exist_ok=True)
    for filename in files_to_copy:
        src_file = repo_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, proj_path / src_file.name)
            print(f"Copied {filename} to project_files")
        else:
            print(f"Warning: {filename} not found in repo")

    # Copy data folder
    data_path = Path(data.path)
    data_path.mkdir(parents=True, exist_ok=True)
    src_data_path = repo_dir / "data"
    if (src_data_path).exists():
        shutil.copytree(src_data_path, data_path, dirs_exist_ok=True)
        print("Copied data folder")


# -----------------------
# Step 2: Fairness Analysis
# -----------------------
@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=["pandas==2.3.2", "scikit-learn==1.7.1"],
)
def fairness_analysis(
    project_files: Input[Model],
    data: Input[Dataset],
    fairness_results: Output[Dataset],
    target_col: str,
) -> None:
    """Run fairness and bias analysis using the ASCOPD model.
    This component installs required Python packages, executes the
    `fairness_bias_analysis.py` script from the project repository, and writes
    results to the `fairness_results` output.

    Args:
        project_files (Input[Model]): Input path containing project scripts.
        data (Input[Dataset]): Input path containing `X.csv`, `y.csv`, and `output.csv`.
        fairness_results (Output[Dataset]): Output path for fairness analysis results (JSON).
        target_col (str): Target column for the analysis.
    """
    from pathlib import Path
    import subprocess

    # Prepare paths
    proj_path = Path(project_files.path)
    data_path = Path(data.path)
    results_path = Path(fairness_results.path)
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare script and arguments
    script = proj_path / "fairness_bias_analysis.py"
    if not script.exists():
        raise FileNotFoundError(f"Fairness analyzer script not found at {script}")

    print(f"Running fairness analysis with {script}")

    cmd = [
        "python",
        str(script),
        "--tabular_data",
        str(data_path / "X.csv"),
        "--actual_target",
        str(data_path / "y.csv"),
        "--pred_target",
        str(data_path / "output.csv"),
        "--target_col",
        target_col,
        "--output",
        str(results_path / "fairness_analysis.json"),
    ]
    subprocess.run(cmd, check=True)

    print(f"Fairness analysis finished. Results saved to {results_path}")


# -----------------------
# Step 3: Explainer Analysis
# -----------------------
@dsl.component(
    base_image="docker.io/username/reponame:latest",
    packages_to_install=[
        "pandas==2.3.2",
        "scikit-learn==1.7.1",
        "dice_ml==0.12",
    ],
)
def explainer_analysis(
    project_files: Input[Model],
    data: Input[Dataset],
    explainer_results: Output[Dataset],
    sensitivity: float,
    target_col: str,
) -> None:
    """Run explainer analysis on the ASCOPD model using Docker.
    This component installs Docker and required Python packages, pulls the specified
    Docker image, executes the `explainer.py` script from the project repository,
    and writes the results to the `explainer_results` output.

    Args:
        project_files (Input[Model]): Input path containing project scripts.
        data (Input[Dataset]): Input path containing `X.csv` and `y.csv`.
        explainer_results (Output[Dataset]): Output path for explainer results.
        sensitivity (float): Sensitivity parameter for the explainer script.
        target_col (str): Target column for the analysis.
    """
    from pathlib import Path
    import subprocess

    # Prepare paths
    proj_path = Path(project_files.path)
    data_path = Path(data.path)
    results_path = Path(explainer_results.path)
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare script and arguments
    script = proj_path / "explainer.py"
    if not script.exists():
        raise FileNotFoundError(f"Explainer script not found at {script}")

    print(f"Running explainer analysis with {script}")

    cmd = [
        "python",
        str(script),
        "--tabular_data",
        str(data_path / "X.csv"),
        "--actual_target",
        str(data_path / "y.csv"),
        "--target_col",
        target_col,
        "--sensitivity",
        str(sensitivity),
        "--output",
        str(results_path),
        "--in_docker",
        "True",
    ]
    subprocess.run(cmd, check=True)

    print(f"Explainer analysis finished. Results saved to {results_path}")


# -----------------------
# -----------------------
# Define Pipeline
# -----------------------
# -----------------------
@dsl.pipeline(
    name="ASCOPD Model Fairness-Bias and Explainer Pipeline",
    description="Runs fairness-bias and explainer analyses.",
)
def ascopd_pipeline(
    github_repo_url: str,
    target_col: str,
    branch: str = "main",
    sensitivity: float = 0.7,
):
    """Pipeline to run ASCOPD model fairness/bias and explainer analyses.

    Args:
        github_repo_url (str): URL of the GitHub repository containing the ASCOPD code and data.
        target_col (str): Target column for the analyses.
        branch (str): Branch name to pull from (defaults to 'main').
        sensitivity (float): Sensitivity parameter for the explainer analysis. Defaults to 0.7.
    """

    if target_col not in ["VenDep", "ARF", "Mortality"]:
        raise ValueError(f"Invalid target_col: {target_col}")

    repo_task = download_repo(github_repo_url=github_repo_url, branch=branch)
    repo_task.set_caching_options(False)
    repo_task.set_cpu_request("4000m")
    repo_task.set_cpu_limit("6000m")
    repo_task.set_memory_request("8Gi")
    repo_task.set_memory_limit("10Gi")

    fairness_task = fairness_analysis(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
        target_col=target_col,
    )
    fairness_task.set_caching_options(False)
    fairness_task.set_cpu_request("6000m")
    fairness_task.set_cpu_limit("6000m")
    fairness_task.set_memory_request("12Gi")
    fairness_task.set_memory_limit("14Gi")

    explainer_task = explainer_analysis(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
        sensitivity=sensitivity,
        target_col=target_col,
    )
    explainer_task.after(fairness_task)
    explainer_task.set_caching_options(False)
    explainer_task.set_cpu_request("6000m")
    explainer_task.set_cpu_limit("6000m")
    explainer_task.set_memory_request("12Gi")


if __name__ == "__main__":
    compiler = compiler.Compiler()
    compiler.compile(pipeline_func=ascopd_pipeline, package_path="ascopd_pipeline.yaml")
