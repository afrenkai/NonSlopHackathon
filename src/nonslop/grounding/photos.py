import logging
from datetime import datetime
from io import BytesIO
from math import isfinite
from numbers import Real

from PIL import ExifTags, Image

from nonslop.grounding.photo_schemas import PhotoMetadata


def metadata_number(value: object) -> float | None:
    if not isinstance(value, Real):
        return None
    result = float(value)
    return result if isfinite(result) else None


def gps_coordinate(value: object, hemisphere: object, limit: int) -> float | None:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    degrees, minutes, seconds = [metadata_number(part) for part in value]
    if degrees is None or minutes is None or seconds is None:
        return None
    if not (0 <= degrees <= limit and 0 <= minutes < 60 and 0 <= seconds < 60):
        return None
    if hemisphere not in (("N", "S") if limit == 90 else ("E", "W")):
        return None
    coordinate = degrees + minutes / 60 + seconds / 3600
    if coordinate > limit:
        return None
    return -coordinate if hemisphere in ("S", "W") else coordinate


def photo_metadata(payload: bytes, filename: str) -> PhotoMetadata:
    with Image.open(BytesIO(payload)) as image:
        metadata = PhotoMetadata(
            filename=filename, width=image.width, height=image.height
        )
        try:
            exif = image.getexif()
            camera_model = exif.get(ExifTags.Base.Model)
            if isinstance(camera_model, str):
                metadata.camera_model = camera_model.strip("\x00 ")[:200] or None
            details = exif.get_ifd(ExifTags.IFD.Exif)
            captured_at = details.get(ExifTags.Base.DateTimeOriginal)
            if isinstance(captured_at, str):
                try:
                    metadata.captured_at = datetime.strptime(
                        captured_at.strip("\x00 "), "%Y:%m:%d %H:%M:%S"
                    ).isoformat()
                except ValueError:
                    pass
            focal_length = metadata_number(details.get(ExifTags.Base.FocalLength))
            if focal_length is not None and focal_length > 0:
                metadata.focal_length_mm = focal_length
            gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
            latitude = gps_coordinate(
                gps.get(ExifTags.GPS.GPSLatitude),
                gps.get(ExifTags.GPS.GPSLatitudeRef),
                90,
            )
            longitude = gps_coordinate(
                gps.get(ExifTags.GPS.GPSLongitude),
                gps.get(ExifTags.GPS.GPSLongitudeRef),
                180,
            )
            if latitude is not None and longitude is not None:
                metadata.latitude = latitude
                metadata.longitude = longitude
        except (KeyError, TypeError, ValueError, OSError, ZeroDivisionError):
            logging.getLogger(__name__).info("Unreadable optional photo metadata")
        return metadata


def describe_photo_metadata(metadata: list[PhotoMetadata]) -> str:
    descriptions: list[str] = []
    for photo in metadata:
        details: list[str] = []
        if photo.captured_at:
            details.append(f"recorded capture time {photo.captured_at}")
        if photo.latitude is not None and photo.longitude is not None:
            details.append(f"GPS {photo.latitude:.6f}, {photo.longitude:.6f}")
        if photo.camera_model:
            details.append(f"camera {photo.camera_model}")
        if photo.focal_length_mm is not None:
            details.append(f"focal length {photo.focal_length_mm:g} mm")
        if details:
            descriptions.append(f"{photo.filename}: {', '.join(details)}.")
    return " ".join(descriptions)
