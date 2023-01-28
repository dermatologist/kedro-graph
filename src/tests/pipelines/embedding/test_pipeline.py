
from pathlib import Path

import pytest

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.extras.datasets.pandas import CSVDataSet
from kedro.extras.datasets.tensorflow import TensorFlowModelDataset

from kedro_tf_text.pipelines.tabular.nodes import tabular_model
from kedro_tf_image.extras.datasets.tf_model_weights import TfModelWeights
from kedro_tf_utils.pipelines.embedding.nodes import create_embedding

from kedro_tf_utils.pipelines.fusion.nodes import fusion
from kedro.io import PartitionedDataSet
from kedro.extras.datasets.pickle import PickleDataSet
from kedro_tf_utils.pipelines.train.nodes import train_multimodal

from kedro_graph.pipelines.embedding.nodes import create_knn, create_graph


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


class TestEmbeddingPipeline:

    def test_create_knn(self, project_context):
        image_embedding = PickleDataSet(
            filepath="data/04_feature/image-embedding.pkl")
        tabular_embedding = PickleDataSet(
            filepath="data/04_feature/tabular-embedding.pkl")

        _image_embedding = image_embedding.load()
        _tabular_embedding = tabular_embedding.load()

        conf_params = project_context.config_loader.get('**/embedding.yml')

        knn = create_knn(_image_embedding, _tabular_embedding, conf_params['embedding'])
        assert knn is not None

    def test_create_graph(self, project_context):
        image_embedding = PickleDataSet(
            filepath="data/04_feature/image-embedding.pkl")
        tabular_embedding = PickleDataSet(
            filepath="data/04_feature/tabular-embedding.pkl")

        _image_embedding = image_embedding.load()
        _tabular_embedding = tabular_embedding.load()

        conf_params = project_context.config_loader.get('**/embedding.yml')

        graph = create_graph(_image_embedding, _tabular_embedding, conf_params['embedding'])
        assert graph is not None
