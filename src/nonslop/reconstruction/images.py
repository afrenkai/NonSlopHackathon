from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


def prepare_review_image(content: bytes, max_edge: int) -> bytes:
    with Image.open(BytesIO(content)) as source:
        image = source.convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


def prepare_reference_images(
    images: list[bytes], directory: Path, target_size: int
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(images[0])) as opening:
        size = (
            max(1, round(opening.width * target_size / max(opening.size))),
            max(1, round(opening.height * target_size / max(opening.size))),
        )
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
