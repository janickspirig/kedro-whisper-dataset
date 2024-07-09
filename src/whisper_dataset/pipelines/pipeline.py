from kedro.pipeline import Pipeline, node

from whisper_dataset.nodes.feature.sentiment_analysis import fea_sentiment_analysis


def create_pipeline(**kwargs):
    return Pipeline(
        nodes=[
            node(
                fea_sentiment_analysis,
                inputs=["raw_audio_files", "params:openai_args"],
                outputs="fea_sentiment_scores",
                name="sentiment_analysis",
            )
        ],
    )
