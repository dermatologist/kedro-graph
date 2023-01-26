"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from kedro_tf_utils.pipelines.embedding.pipeline import create_embedding_pipeline
from kedro_tf_text.pipelines.tabular.pipeline import create_tabular_model_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    pipelines["embedding"] = create_tabular_model_pipeline() + create_embedding_pipeline()
    return pipelines
