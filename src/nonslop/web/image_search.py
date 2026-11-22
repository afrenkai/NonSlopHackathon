import asyncio
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import zip_longest

import httpx
import requests
from anyio import to_thread
from fastapi import HTTPException

from nonslop.web.image_relevance import image_subject, nous_title_scores, title_score
from nonslop.web.uploads import normalize_image

IMAGE_REFERENCE_LIMIT = 4
GALLERY_IMAGE_LIMIT = 16
MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
logger = logging.getLogger(__name__)


class ImageSearchError(Exception):
    pass


@dataclass(frozen=True)
class ImageReference:
    title: str
    content: bytes
    collection: str = ""

    @property
    def caption(self) -> str:
        return (f"{self.collection} — {self.title}" if self.collection else self.title)[
            :250
        ]


def image_query(description: str, location: str = "", year: int | None = None) -> str:
    return " ".join(f"{location} {year or ''} {description}".split()[:32])[:512]


def thumbnail_url(value: object) -> httpx.URL | None:
    if not isinstance(value, str):
        return None
    try:
        url = httpx.URL(value)
    except httpx.InvalidURL:
        return None
    if url.scheme != "https" or url.userinfo or url.port is not None:
        return None
    if url.host in {
        "thumb.wikimedia.org",
        "upload.wikimedia.org",
    } and url.path.startswith("/wikipedia/commons/thumb/"):
        return url.copy_with(host="upload.wikimedia.org", query=None)
    return None


async def download_thumbnail(client: httpx.AsyncClient, url: httpx.URL) -> bytes:
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        if not response.headers.get("Content-Type", "").startswith("image/"):
            raise ValueError("The thumbnail response was not an image.")
        content = bytearray()
        async for chunk in response.aiter_bytes():
            if len(content) + len(chunk) > MAX_THUMBNAIL_BYTES:
                raise ValueError("The thumbnail exceeded the size limit.")
            content.extend(chunk)
    return await to_thread.run_sync(normalize_image, bytes(content), 768)


async def search_commons(
    query: str, limit: int, subject: tuple[str, ...]
) -> list[ImageReference]:
    if not query or limit <= 0:
        return []
    candidate_limit = limit * 2
    async with httpx.AsyncClient(
        timeout=8,
        follow_redirects=False,
        headers={"User-Agent": "AgainHere/0.1 (historical reference image search)"},
    ) as client:
        try:
            response = await client.get(
                COMMONS_API,
                params={
                    "action": "query",
                    "format": "json",
                    "formatversion": "2",
                    "generator": "search",
                    "gsrnamespace": "6",
                    "gsrsearch": f"{query} filetype:bitmap",
                    "gsrlimit": candidate_limit,
                    "prop": "imageinfo",
                    "iiprop": "url",
                    "iiurlwidth": "500",
                },
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or data.get("error"):
                raise ValueError("Invalid image search response.")
            search = data.get("query", {})
            if not isinstance(search, dict) or not isinstance(
                search.get("pages", []), list
            ):
                raise TypeError("Invalid image search results.")
            rows = search.get("pages", [])
        except (httpx.HTTPError, TypeError, ValueError) as error:
            logger.warning("Wikimedia Commons search failed (%s)", type(error).__name__)
            raise ImageSearchError(
                "Wikimedia Commons search is unavailable. Try again shortly."
            ) from None

        references = []
        seen = set()
        ranked = sorted(
            (
                row
                for row in rows
                if isinstance(row, dict) and isinstance(row.get("index"), int)
            ),
            key=lambda row: row["index"],
        )
        for row in ranked[:candidate_limit]:
            title = str(row.get("title") or "").removeprefix("File:")
            if not title_score(title, subject):
                continue
            image_info = row.get("imageinfo")
            if (
                not image_info
                or not isinstance(image_info, list)
                or not isinstance(image_info[0], dict)
            ):
                continue
            url = thumbnail_url(image_info[0].get("thumburl"))
            if url is None or url in seen:
                continue
            seen.add(url)
            try:
                content = await download_thumbnail(client, url)
            except (httpx.HTTPError, HTTPException, ValueError):
                continue
            references.append(ImageReference(title=title[:250], content=content))
            if len(references) == limit:
                break
        return references


def library_results(query: str, limit: int) -> list[dict]:
    response = requests.get(
        "https://www.loc.gov/pictures/search/",
        params={"q": query, "fo": "json", "c": limit * 2},
        headers={"User-Agent": "AgainHere/0.1 (historical reference image search)"},
        timeout=6,
    )
    response.raise_for_status()
    return response.json().get("results", [])


async def search_library(
    query: str, limit: int, subject: tuple[str, ...]
) -> list[ImageReference]:
    try:
        rows = await to_thread.run_sync(library_results, query, limit)
    except (requests.RequestException, ValueError) as error:
        logger.warning("Library of Congress search failed (%s)", type(error).__name__)
        return []
    references = []
    seen = set()
    async with httpx.AsyncClient(timeout=6, follow_redirects=False) as client:
        for row in rows[: limit * 2]:
            title = str(row.get("title") or "")
            if not title_score(title, subject):
                continue
            image = row.get("image") or {}
            try:
                url = httpx.URL(image.get("thumb") or "")
            except httpx.InvalidURL:
                continue
            if (
                url.scheme != "https"
                or url.host != "tile.loc.gov"
                or url.port is not None
                or url.userinfo
                or not url.path.startswith("/storage-services/service/pnp/")
                or url in seen
            ):
                continue
            seen.add(url)
            try:
                content = await download_thumbnail(client, url)
            except (httpx.HTTPError, HTTPException, ValueError):
                continue
            references.append(
                ImageReference(
                    title=title[:250], content=content, collection="Library of Congress"
                )
            )
            if len(references) == limit:
                break
    return references


def archive_results(query: str, limit: int) -> list[tuple[str, str]]:
    response = requests.get(
        "https://archive.org/advancedsearch.php",
        params={
            "q": (
                "mediatype:image AND collection:(arkivkopia OR flickr-ows) "
                f"AND title:({query})"
            ),
            "fl[]": ["identifier", "title"],
            "rows": limit * 2,
            "output": "json",
        },
        timeout=6,
    )
    response.raise_for_status()
    candidates = []
    for item in response.json()["response"]["docs"]:
        identifier = item.get("identifier")
        if isinstance(identifier, str) and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier
        ):
            candidates.append(
                (
                    str(item.get("title") or ""),
                    f"https://archive.org/services/img/{identifier}",
                )
            )
    return candidates


def museum_results(query: str, limit: int) -> list[tuple[str, str]]:
    response = requests.get(
        "https://openaccess-api.clevelandart.org/api/artworks",
        params={
            "q": query,
            "type": "Photograph",
            "cc0": "1",
            "has_image": "1",
            "fields": "title,type,images",
            "limit": limit * 2,
        },
        timeout=6,
    )
    response.raise_for_status()
    return [
        (str(item.get("title") or ""), str(url))
        for item in response.json()["data"]
        if item.get("type") == "Photograph"
        and (url := ((item.get("images") or {}).get("web") or {}).get("url"))
    ]


async def search_photo_collection(
    search: Callable[[str, int], list[tuple[str, str]]],
    collection: str,
    host: str,
    query: str,
    limit: int,
    subject: tuple[str, ...],
) -> list[ImageReference]:
    try:
        candidates = await to_thread.run_sync(search, query, limit)
    except (requests.RequestException, ValueError, TypeError, KeyError) as error:
        logger.warning("%s search failed (%s)", collection, type(error).__name__)
        return []

    references = []
    seen = set()
    async with httpx.AsyncClient(timeout=6, follow_redirects=False) as client:
        for title, thumbnail in candidates:
            if not title_score(title, subject):
                continue
            try:
                url = httpx.URL(thumbnail)
            except httpx.InvalidURL:
                continue
            if (
                url.host != host
                or url.scheme != "https"
                or url.port is not None
                or url.userinfo
                or url in seen
            ):
                continue
            seen.add(url)
            try:
                content = await download_thumbnail(client, url)
            except (httpx.HTTPError, HTTPException, ValueError):
                continue
            references.append(ImageReference(title[:250], content, collection))
            if len(references) == limit:
                break
    return references


async def search_images(
    query: str, limit: int, *, location: str = ""
) -> list[ImageReference]:
    if not query or limit <= 0:
        return []
    subject = image_subject(query, location)
    if not subject:
        return []
    extra_limit = min(6, limit)
    commons, library, archive, museum = await asyncio.gather(
        search_commons(query, limit, subject),
        search_library(query, limit, subject),
        search_photo_collection(
            archive_results,
            "Internet Archive",
            "archive.org",
            " AND ".join(f'"{word}"' for word in subject),
            extra_limit,
            subject,
        ),
        search_photo_collection(
            museum_results,
            "Cleveland Museum of Art",
            "openaccess-cdn.clevelandart.org",
            query,
            extra_limit,
            subject,
        ),
        return_exceptions=True,
    )
    for result in (library, archive, museum):
        if isinstance(result, BaseException):
            raise result
    if isinstance(commons, ImageSearchError) and (library or archive or museum):
        commons = []
    if isinstance(commons, BaseException):
        raise commons
    unique = {}
    for pair in zip_longest(commons, library, archive, museum):
        for image in pair:
            if image is not None:
                unique.setdefault(image.content, image)
    results = list(unique.values())
    results.sort(
        key=lambda image: title_score(image.title, subject),
        reverse=True,
    )
    scores = await nous_title_scores(query, [image.title for image in results])
    if scores is not None:
        ranked = sorted(
            zip(scores, results, strict=True), key=lambda item: item[0], reverse=True
        )
        results = [image for score, image in ranked if score >= 3]
    return results[:limit]
