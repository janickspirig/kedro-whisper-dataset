from typing import Optional

from pydantic import BaseModel, Field


class SentimentScores(BaseModel):
    happiness: float = Field(
        ...,
        description="Score between 0 and 1 indicating "
        "the level of happiness during the "
        "conversation.",
    )
    angriness: float = Field(
        ..., description="Score between 0 and 1 indicating the level of angriness"
    )
    sadness: float = Field(
        ..., description="Score between 0 and 1 indicating the level of sadness"
    )


class AudioData(BaseModel):
    f_name: str
    transcript: str
    sentiment_scores: Optional[SentimentScores] = None
