from typing import Callable, Dict, Union

import instructor
from dotenv import load_dotenv
from openai import OpenAI

from whisper_dataset.entities import SentimentScores

load_dotenv()


def fea_sentiment_analysis(
    audio_partitions: Dict[str, Callable], openai_args: Dict[str, Union[str, Dict]]
) -> Dict[str, Dict]:
    """Performs sentiment analysis on call transcriptions.

    Args:
        audio_partitions: dictionary containing the file names as keys and the load
            function callables as values
        openai_args: dictionary containing OpenAI system message and model kwargs

    Returns:
         Dictionary with the original audio file names as keys and the sentiment scores
            as values
    """
    sentiment_scores = {}

    for f_name, load_func in audio_partitions.items():
        audio = load_func()
        client = instructor.from_openai(OpenAI())
        score = client.chat.completions.create(
            response_model=SentimentScores,
            messages=[
                {
                    "role": "system",
                    "content": openai_args["system_message"],
                },
                {"role": "user", "content": audio.transcript},
            ],
            **openai_args["model_kwargs"],
        )
        audio.sentiment_scores = score
        sentiment_scores[f_name] = audio.model_dump()

    return sentiment_scores
