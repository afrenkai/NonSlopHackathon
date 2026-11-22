from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


def prepare_reference_images(images: list[bytes], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(images[0])) as opening:
        size = opening.size
    paths: list[Path] = []
    for index, content in enumerate(images):
        with Image.open(BytesIO(content)) as image:
            prepared = ImageOps.fit(
                image.convert("RGB"), size, Image.Resampling.LANCZOS
            )
            path = directory / f"{index:04d}.png"
            prepared.save(path)
            paths.append(path)
    return paths
