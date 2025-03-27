"""
This script is adapted from the mm-health-bench project:
https://github.com/konst-int-i/mm-health-bench

Original copyright belongs to the respective authors.
Modifications have been made for use in this project.
"""

from invoke import task
from typing import *
import os
from pathlib import Path
from mmhb.loader.chestx import preprocess_chestx


@task
def download(
    c,
    dataset: str,
    site: Optional[List[str]] = "brca",
    data_dir: Optional[str] = None,
    samples: Optional[int] = None,
):
    print(site)
    if data_dir is None:
        # set default
        data_dir = Path(f"data/{dataset}/")

    valid_datasets = ["tcga", "mimic", "chestx"]
    assert (
        dataset in valid_datasets
    ), f"Invalid dataset, specify one of {valid_datasets}"

    if dataset == "chestx":
        download_chestx(c, data_dir)


@task
def download_chestx(c, data_dir: Path):
    """

    Args:
        c:
        data_dir:

    Returns:

    """

    raw_dir = data_dir.joinpath("raw")
    proc_dir = data_dir.joinpath("proc")
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading chestx dataset to {raw_dir}...")

    # download PNG images
    if "NLMCXR_png.tgz" not in os.listdir(raw_dir):
        print("Downloading chestx images...")
        c.run(
            f"curl -0 https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz -o {str(raw_dir.joinpath('NLMCXR_png.tgz'))}"
        )
    # download reports
    if "NLMCXR_reports.tgz" not in os.listdir(raw_dir):
        print("Downloading chestx reports...")
        c.run(
            f"curl -0 https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz -o {str(raw_dir.joinpath('NLMCXR_reports.tgz'))}"
        )
    # download term mapping
    if "radiology_vocabulary_final.xlsx" not in os.listdir(raw_dir):
        print("Downloading radiology vocabulary...")
        c.run(
            f"curl -0 https://openi.nlm.nih.gov/imgs/collections/radiology_vocabulary_final.xlsx -o {str(raw_dir.joinpath('radiology_vocabulary_final.xlsx'))}"
        )

    if "TransChex_openi.zip" not in os.listdir(raw_dir):
        print(f"Downloading indeces...")
        c.run(  
            f"curl -0 https://developer.download.nvidia.com/assets/Clara/monai/tutorials/TransChex_openi.zip -o {str(raw_dir.joinpath('TransChex_openi.zip'))}"
        )

    # unzip
    if "NLMCXR_png" not in os.listdir(raw_dir):
        print("Extracting images...")
        raw_dir.joinpath("NLMCXR_png").mkdir(exist_ok=True)
        c.run(
            f"tar -xvzf {raw_dir.joinpath('NLMCXR_png.tgz')} -C {raw_dir.joinpath('NLMCXR_png')}"
        )
    if "NLMCXR_reports" not in os.listdir(raw_dir):
        print("Extracting reports...")
        raw_dir.joinpath("NLMCXR_reports").mkdir(exist_ok=True)
        c.run(
            f"tar -xvzf {raw_dir.joinpath('NLMCXR_reports.tgz')} -C {raw_dir.joinpath('NLMCXR_reports')}"
        )
    if "TransChex_openi" not in os.listdir(raw_dir):
        print("Extracting indeces...")
        c.run(f"unzip {raw_dir.joinpath('TransChex_openi.zip')} -d {raw_dir}")

    print("ChestX dataset downloaded successfully.")

    print("Preprocessing chestx dataset...")
    preprocess_chestx(raw_dir, proc_dir)
