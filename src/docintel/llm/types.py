"""Provider-neutral data types for the LLM layer."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["user", "assistant"]


class Message(BaseModel):
    """One turn of a conversation. The system prompt is passed separately."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class Usage(BaseModel):
    """Token counts reported by the provider for one call."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class RawGeneration(BaseModel):
    """What a provider adapter returns before timing and cost are added."""

    model_config = ConfigDict(frozen=True)

    text: str
    usage: Usage


class Generation(BaseModel):
    """A finished generation with everything needed for tracing."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    usage: Usage
    latency_ms: float = Field(ge=0)
    cost_usd: float | None = Field(
        default=None,
        description="Estimated cost at list price; None if the model has no price.",
    )
