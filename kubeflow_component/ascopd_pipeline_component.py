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
        "explainer_visualization.py",
        "fairness_bias_visualization.py",
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
    # Insert your dockerhub image below (e.g. base_image="docker.io/<username>/<image_name>:<tag>")
    base_image="<your_dockerhub_image>",
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


@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=["pandas==2.3.2", "matplotlib==3.10.5"],
)
def explainer_visualization(
        project_files: Input[Model],
        explainer_results: Input[Dataset],
        visualizations: Output[Dataset],
) -> None:
    """Generate visualizations for explainer analysis results.
    This component executes the `explainer_visualization.py` script to create
    visual representations of feature importance from explainer results.

    :param project_files: Input path containing project scripts.
    :param explainer_results: Input path containing explainer analysis JSON results.
    :param visualizations: Output path for generated visualization images.
    """
    from pathlib import Path
    import subprocess
    import os

    proj_path = Path(project_files.path)
    results_path = Path(explainer_results.path)
    viz_path = Path(visualizations.path)
    viz_path.mkdir(parents=True, exist_ok=True)

    script = proj_path / "explainer_visualization.py"
    if not script.exists():
        raise FileNotFoundError(f"Visualization script not found at {script}")

    json_files = list(results_path.glob("*_analysis.json"))
    if not json_files:
        raise FileNotFoundError(f"No analysis JSON files found in {results_path}")

    analysis_file = json_files[0]
    print(f"Found explainer results: {analysis_file}")

    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'

    output_dir = viz_path / "explainer"

    cmd = [
        "python",
        str(script),
        "--analysis_results",
        str(analysis_file),
        "--output",
        str(output_dir),
    ]

    print(f"Running explainer visualization with {script}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)

    if result.stdout:
        print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")

    print(f"Explainer visualization finished. Results saved to {output_dir}")


@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=["pandas==2.3.2", "matplotlib==3.10.5", "seaborn==0.13.2"],
)
def fairness_bias_visualization(
        project_files: Input[Model],
        fairness_results: Input[Dataset],
        visualizations: Output[Dataset],
) -> None:
    """Generate visualizations for fairness and bias analysis results.
    This component executes the `fairness_bias_visualization.py` script to create
    visual representations of fairness metrics.

    :param project_files: Input path containing project scripts.
    :param fairness_results: Input path containing fairness analysis JSON results.
    :param visualizations: Output path for generated visualization images.
    """
    from pathlib import Path
    import subprocess
    import os

    proj_path = Path(project_files.path)
    results_path = Path(fairness_results.path)
    viz_path = Path(visualizations.path)
    fairness_viz_path = viz_path / "fairness_bias"
    fairness_viz_path.mkdir(parents=True, exist_ok=True)

    script = proj_path / "fairness_bias_visualization.py"
    if not script.exists():
        raise FileNotFoundError(f"Fairness visualization script not found at {script}")

    analysis_results_file = results_path / "fairness_analysis.json"
    if not analysis_results_file.exists():
        json_files = list(results_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {results_path}")
        analysis_results_file = json_files[0]

    print(f"Found fairness results: {analysis_results_file}")

    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'

    cmd = [
        "python",
        str(script),
        "--analysis_results",
        str(analysis_results_file),
        "--output",
        str(fairness_viz_path),
    ]

    print(f"Running fairness visualization with {script}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)

    if result.stdout:
        print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")

    png_files = list(fairness_viz_path.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No visualization PNG files found in {fairness_viz_path}")

    print(f"Fairness visualization finished. {len(png_files)} images saved to {fairness_viz_path}")


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
    explainer_task.set_memory_limit("14Gi")

    explainer_viz_task = explainer_visualization(
        project_files=repo_task.outputs["project_files"],
        explainer_results=explainer_task.outputs["explainer_results"],
    )
    explainer_viz_task.set_caching_options(False)
    explainer_viz_task.set_cpu_request("2000m")
    explainer_viz_task.set_cpu_limit("4000m")
    explainer_viz_task.set_memory_request("4Gi")
    explainer_viz_task.set_memory_limit("6Gi")

    fairness_viz_task = fairness_bias_visualization(
        project_files=repo_task.outputs["project_files"],
        fairness_results=fairness_task.outputs["fairness_results"],
    )
    fairness_viz_task.set_caching_options(False)
    fairness_viz_task.set_cpu_request("2000m")
    fairness_viz_task.set_cpu_limit("4000m")
    fairness_viz_task.set_memory_request("4Gi")
    fairness_viz_task.set_memory_limit("6Gi")


if __name__ == "__main__":
    compiler = compiler.Compiler()
    compiler.compile(pipeline_func=ascopd_pipeline, package_path="ascopd_pipeline.yaml")
