from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=262_144)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(default=None, max_length=256)
    messages: list[ChatMessage] = Field(min_length=1, max_length=256)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    stop: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 128 for item in value):
            raise ValueError("stop strings must contain 1 to 128 characters")
        return value


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: Literal["stop", "length", "cancelled", "error"]


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage
