"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from kedro_tf_utils.pipelines.embedding.pipeline import create_embedding_pipeline
from kedro_tf_text.pipelines.tabular.pipeline import create_tabular_model_pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    tabular_inputs = {"parameters": "params:embedding", "model": "tabular_model",
              "tabular_data": "tabular_data", "outputs": "tabular_embedding"}
    image_inputs = {"parameters": "params:embedding", "model": "image_model",
                      "image_data": "image_data", "outputs": "image_embedding"}
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    pipelines["tabular"] = create_tabular_model_pipeline() + create_embedding_pipeline(**tabular_inputs)
    pipelines["image"] = create_embedding_pipeline(**image_inputs)
    return pipelines
