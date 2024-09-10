from typing import Any, Dict
from pathlib import Path
import json

import pandas as pd
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import Dataset

from crystal_gnn.datamodules.base_datamodule import BaseDataModule
from crystal_gnn.datasets.custom_dataset import CustomDataset


class CustomDatamodule(BaseDataModule):
    def __init__(self, _config: Dict[str, Any]) -> None:
        super().__init__(_config)

        self.data_dir = _config["data_dir"]
        self.source = _config["source"]
        self.target = _config["target"]
        self.classification_threshold = _config["classification_threshold"]
        self.split_seed = _config["split_seed"]
        self.train_ratio = _config["train_ratio"]
        self.val_ratio = _config["val_ratio"]
        self.test_ratio = _config["test_ratio"]
        self.mean = _config["mean"]
        self.std = _config["std"]
        self.database_name = None  # this is not used for Matbench

    def prepare_data(self) -> None:
        """Prepare data for custom dataset.

        The custom data should be prepared in the following format:
        - {data_dir}/{source}/data.csv
        The `data.csv` should have the following columns:
        - "id": unique id for each data
        - "structure": string format of Pymatgen Structure (e.g., structure.to(fmt="cif"))
        - "{target}": target value for the task

        It will save the torch_geometric graph data for train, val, test
        in the `{data_dir}/{source}/{target}` with the following names:
        - train-{target}.pt
        - val-{target}.pt
        - test-{target}.pt
        - {target}.json (info files)
        """
        # check if custom data exists
        path_data = Path(self.data_dir, self.source, "data.csv")
        if not path_data.exists():
            raise NotImplementedError(
                f"The custom data {path_data} does not exist. "
                "Please prepare the data with the csv format."
            )
        else:
            data = pd.read_csv(path_data)
            print(f"load data from {path_data}")

        # check the columns of the data
        if "id" not in data.columns:
            raise ValueError("The data should have 'id' column.")
        if "structure" not in data.columns:
            raise ValueError("The data should have 'structure' column.")
        if self.target not in data.columns:
            raise ValueError(f"The data should have '{self.target}' column.")

        # check if the prepare_data has been done
        path_target = Path(self.data_dir, self.source, self.target)
        if not path_target.exists():
            path_target.mkdir(parents=True, exist_ok=True)
        path_train = Path(path_target, f"train-{self.target}.pt")
        path_val = Path(path_target, f"val-{self.target}.pt")
        path_test = Path(path_target, f"test-{self.target}.pt")
        path_info = Path(path_target, f"{self.target}.json")
        if (
            path_train.exists()
            and path_val.exists()
            and path_test.exists()
            and path_info.exists()
        ):
            print(f"load graph data from {path_target} directory")
            info = json.load(open(path_info, "r"))
            if self.mean is None:
                self.mean = info["train_mean"]
            if self.std is None:
                self.std = info["train_std"]
            return
        # split data
        data_train, data_test = train_test_split(
            data, test_size=self.test_ratio, random_state=self.split_seed
        )
        data_train, data_val = train_test_split(
            data_train, test_size=self.val_ratio, random_state=self.split_seed
        )
        print(
            f"split total {len(data)} data into train: {len(data_train)}, "
            f"val: {len(data_val)}, test: {len(data_test)}"
        )

        # save graph data for train, val, test
        for split, data_split in zip(
            ["train", "val", "test"], [data_train, data_val, data_test]
        ):
            # convert Structure to ase Atoms
            st_list = [
                Structure.from_str(st, fmt="cif") for st in data_split["structure"]
            ]
            atoms_list = [st.to_ase_atoms() for st in st_list]

            # calcuate mean and std for train data
            if split == "train":
                train_mean = data_split[self.target].mean()
                train_std = data_split[self.target].std()

            # make graph data
            graphs = self._make_graph_data(
                atoms_list,
                target=data_split[self.target].values,
                name=data_split["id"].values,
                train_mean=train_mean,
                train_std=train_std,
            )
            # save graph
            path_split = Path(path_target, f"{split}-{self.target}.pt")
            torch.save(graphs, path_split)
            print(f"DONE: saved data to {path_split}")

        # save info
        info = {
            "total": len(data),
            "train": len(data_train),
            "val": len(data_val),
            "test": len(data_test),
            "train_mean": train_mean,
            "train_std": train_std,
        }
        json.dump(info, open(path_info, "w"))
        print(info)
        print(f"DONE: saved data to {path_target}")
        if self.mean is None:
            self.mean = train_mean
        if self.std is None:
            self.std = train_std

    @property
    def dataset_cls(self) -> Dataset:
        return CustomDataset

    @property
    def dataset_name(self) -> str:
        return "custom"
