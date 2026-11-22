from typing import Annotated, Literal

from pydantic import BaseModel, Field


class VlmTextPart(BaseModel):
    type: Literal["text"]
    text: str


class VlmImageUrl(BaseModel):
    url: str


class VlmImageUrlPart(BaseModel):
    type: Literal["image_url"]
    image_url: VlmImageUrl


class VlmFileImagePart(BaseModel):
    type: Literal["image"]
    image: str


VlmContentPart = Annotated[
    VlmTextPart | VlmImageUrlPart | VlmFileImagePart, Field(discriminator="type")
]


class VlmMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[VlmContentPart]


class VlmCompletionRequest(BaseModel):
    model: str
    messages: list[VlmMessage] = Field(min_length=1)
    stream: Literal[False] = False
