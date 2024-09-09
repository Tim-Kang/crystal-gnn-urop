from crystal_gnn.datamodules.jarvis_datamodule import JarvisDataModule
from crystal_gnn.datamodules.matbench_datamodule import MatbenchDataModule
from crystal_gnn.datamodules.custom_datamodule import CustomDatamodule

_datamodules = {
    "jarvis": JarvisDataModule,
    "matbench": MatbenchDataModule,
    "custom": CustomDatamodule,
}
