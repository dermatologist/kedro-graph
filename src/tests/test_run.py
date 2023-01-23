"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path

import pytest

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager

from kedro_graph.extras.datasets.dgl_dataset import DglDataset
import torch as th


@pytest.fixture
def config_loader():
    return ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))


@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="kedro_graph",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality
class TestProjectContext:
    # def test_project_path(self, project_context):
    #     assert project_context.project_path == Path.cwd()
    def test_dgl_dataset(self, project_context):
        path = "data/01_raw/graph-dataset"
        dgl_dataset = DglDataset(path)
        import dgl
        g1 = dgl.graph(([0, 1, 2], [1, 2, 3]))
        g2 = dgl.graph(([0, 2], [2, 3]))
        g2.edata["e"] = th.ones(2, 4)
        labels = th.tensor([0, 1])
        data = ([g1,g2], labels, 3)
        dgl_dataset.save(data)
