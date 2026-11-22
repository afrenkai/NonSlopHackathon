import logging
from html.parser import HTMLParser
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from nonslop.generation.schemas import SceneRequest, SourceEvidence
from nonslop.grounding.schemas import ArchiveResponse, GroundingResult


class PlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def plain_text(value: str | list[str]) -> str:
    parser = PlainTextParser()
    parser.feed(value if isinstance(value, str) else " ".join(value))
    return " ".join(" ".join(parser.parts).split())


def street_excerpt(text: str, street: str) -> str:
    position = text.casefold().find(street.casefold())
    start = max(0, position - 200)
    return text[start : start + 1000]


async def archive_search(
    client: httpx.AsyncClient, street: str, year_query: str
) -> list[SourceEvidence]:
    quoted_street = street.replace("\\", "\\\\").replace('"', '\\"')
    response = await client.get(
        "https://archive.org/advancedsearch.php",
        params=[
            ("q", f'"{quoted_street}" AND {year_query}'),
            ("output", "json"),
            ("rows", "3"),
            ("fl[]", "identifier"),
            ("fl[]", "title"),
            ("fl[]", "description"),
            ("fl[]", "date"),
        ],
    )
    response.raise_for_status()
    payload = ArchiveResponse.model_validate(response.json())
    return [
        SourceEvidence(
            title=plain_text(item.title),
            url=f"https://archive.org/details/{quote(item.identifier, safe='')}",
            excerpt=street_excerpt(plain_text(item.description), street),
            kind="Archive period evidence; review relevance",
            date=item.date,
        )
        for item in payload.response.docs
    ]


async def retrieve_sources(scene: SceneRequest) -> GroundingResult:
    if not scene.use_web:
        return GroundingResult()
    grounding = GroundingResult()
    street = scene.location.split(",", maxsplit=1)[0].strip()
    try:
        async with httpx.AsyncClient(
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "AgainHere/0.1 (historical scene research)"},
        ) as client:
            grounding.sources = await archive_search(
                client, street, f"year:{scene.year}"
            )
            if not grounding.sources:
                grounding.sources = await archive_search(
                    client, street, f"year:[{scene.year - 10} TO {scene.year + 10}]"
                )
                if grounding.sources:
                    grounding.notices.append(
                        "No exact-year match was found. These sources are from within ten years; check their dates."
                    )
    except (httpx.HTTPError, ValidationError, ValueError) as error:
        logging.getLogger(__name__).warning("Archive lookup failed: %s", error)
        grounding.notices.append("Archive search was unavailable for this request.")
    if not grounding.sources:
        grounding.notices.append(
            "No web evidence was retrieved. The scene uses your description and uploads."
        )
    grounding.notices.append(
        "Archive matches provide context. The generated scene may include details not established by the sources."
    )
    return grounding
