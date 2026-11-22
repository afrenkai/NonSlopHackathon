from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class WebSettings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    api_key: SecretStr = Field(min_length=16)
    password: SecretStr = Field(min_length=8)
    session_secret: SecretStr = Field(min_length=32)
    data_root: Path = Path("/data")
    output_root: Path = Path("/outputs")
    serving_root: Path = Path("/tmp/nonslop-videos")
    world_root: Path = Path("/worlds")
    world_serving_root: Path = Path("/tmp/nonslop-worlds")
    assets_root: Path = Path("/opt/nonslop-web")
    upload_limit: int = 10 * 1024 * 1024
    total_upload_limit: int = 40 * 1024 * 1024
    reference_limit: int = 8
    text_limit: int = 20000
    secure_cookies: bool = True
    references_only: bool = False


class ReferenceBundle(BaseModel):
    images: list[bytes] = Field(default_factory=list)
    filenames: list[str] = Field(default_factory=list)
