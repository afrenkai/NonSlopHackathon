from pydantic import BaseModel, Field


class PhotoMetadata(BaseModel):
    filename: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    captured_at: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    focal_length_mm: float | None = Field(default=None, gt=0)
    camera_model: str | None = None
