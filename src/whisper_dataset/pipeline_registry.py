"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from whisper_dataset.pipelines.pipeline import create_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipeline = create_pipeline()
    return {"__default__": pipeline}
