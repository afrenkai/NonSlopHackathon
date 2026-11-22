import logging
import re
from asyncio import gather
from html.parser import HTMLParser
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from nonslop.generation.schemas import SceneRequest, SourceEvidence
from nonslop.grounding.schemas import (
    ArchiveResponse,
    EntityResponse,
    GroundingResult,
    PlacePage,
    PlaceResponse,
)


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


def location_excerpt(text: str, location: str) -> str:
    position = text.casefold().find(
        location.split(",", maxsplit=1)[0].strip().casefold()
    )
    start = max(0, position - 200)
    return text[start : start + 1000]


def search_words(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.casefold()))


async def resolve_place(client: httpx.AsyncClient, location: str) -> PlacePage | None:
    response = await client.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": location,
            "gsrlimit": "5",
            "prop": "extracts|coordinates|pageprops|info",
            "exintro": "1",
            "explaintext": "1",
            "exchars": "1400",
            "inprop": "url",
        },
    )
    response.raise_for_status()
    pages = PlaceResponse.model_validate(response.json()).query.pages
    parts = [part.strip() for part in location.split(",") if part.strip()]
    name = search_words(parts[0])
    candidates = [
        page
        for page in pages
        if page.coordinates
        and "disambiguation" not in page.pageprops
        and search_words(page.title.split("(", maxsplit=1)[0]) == name
        and all(
            search_words(qualifier) <= search_words(f"{page.title} {page.extract}")
            for qualifier in parts[1:]
        )
    ]
    return candidates[0] if len(candidates) == 1 else None


async def geographic_names(client: httpx.AsyncClient, place: PlacePage) -> list[str]:
    identifier = place.pageprops.get("wikibase_item")
    if not identifier:
        return []
    identifiers = {identifier}
    visited: set[str] = set()
    names: list[str] = []
    for level in range(4):
        if not identifiers:
            break
        response = await client.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "ids": "|".join(sorted(identifiers)),
                "props": "labels|aliases|claims",
                "languages": "en",
                "format": "json",
            },
        )
        response.raise_for_status()
        entities = EntityResponse.model_validate(response.json()).entities
        visited.update(identifiers)
        if level == 0:
            names.extend(
                alias.value
                for entity in entities.values()
                for alias in entity.aliases.get("en", [])
                if len(search_words(alias.value) - {"the", "a", "of", "and"}) >= 2
                and not search_words(alias.value) <= search_words(place.title)
            )
        if level > 0:
            names.extend(
                entity.labels["en"].value
                for entity in entities.values()
                if "en" in entity.labels
            )
            names.extend(
                alias.value
                for entity in entities.values()
                for alias in entity.aliases.get("en", [])
            )
        identifiers = {
            statement.mainsnak.datavalue.value.id
            for entity in entities.values()
            for statement in entity.claims.locations
            if statement.mainsnak.datavalue is not None
            and statement.rank != "deprecated"
        } - visited
    return names


def archive_matches_place(item_text: str, names: list[str]) -> bool:
    text = " " + " ".join(re.findall(r"[^\W_]+", item_text.casefold())) + " "
    return any(
        " " + " ".join(re.findall(r"[^\W_]+", name.casefold())) + " " in text
        for name in names
    )


async def archive_search(
    client: httpx.AsyncClient, location: str, year_query: str, names: list[str]
) -> list[SourceEvidence]:
    if not names:
        return []
    quoted_location = location.replace("\\", "\\\\").replace('"', '\\"')
    geography_query = " OR ".join(
        '"' + name.replace("\\", "\\\\").replace('"', '\\"') + '"' for name in names
    )
    response = await client.get(
        "https://archive.org/advancedsearch.php",
        params=[
            (
                "q",
                f'"{quoted_location}" AND ({geography_query}) AND {year_query} AND (mediatype:image OR mediatype:texts)',
            ),
            ("output", "json"),
            ("rows", "30"),
            ("fl[]", "identifier"),
            ("fl[]", "title"),
            ("fl[]", "description"),
            ("fl[]", "date"),
            ("fl[]", "subject"),
            ("fl[]", "coverage"),
        ],
    )
    response.raise_for_status()
    payload = ArchiveResponse.model_validate(response.json())
    return [
        SourceEvidence(
            title=plain_text(item.title),
            url=f"https://archive.org/details/{quote(item.identifier, safe='')}",
            excerpt=location_excerpt(
                " ".join(
                    plain_text(value)
                    for value in (item.description, item.subject, item.coverage)
                ),
                location,
            ),
            kind="Archive match for this place; publication date is not the scene date",
            date=item.date,
        )
        for item in payload.response.docs
        if archive_matches_place(
            " ".join(
                plain_text(value)
                for value in (
                    item.title,
                    location_excerpt(plain_text(item.description), location),
                    item.subject,
                    item.coverage,
                )
            ),
            names,
        )
    ][:3]


async def retrieve_sources(scene: SceneRequest) -> GroundingResult:
    if not scene.use_web:
        return GroundingResult()
    location_parts = scene.location.split(",")
    route = location_parts[0].strip()
    if route.casefold().startswith("between "):
        endpoints = re.split(r"\s+and\s+", route[8:], maxsplit=1, flags=re.IGNORECASE)
        if len(endpoints) == 2:
            results = await gather(
                *(
                    retrieve_sources(
                        scene.model_copy(
                            update={
                                "location": ", ".join([endpoint, *location_parts[1:]])
                            }
                        )
                    )
                    for endpoint in endpoints
                )
            )
            identity_sources = [
                result.sources[0] for result in results if result.sources
            ]
            archive_sources = [
                source for result in results for source in result.sources[1:]
            ]
            sources = {
                str(source.url): source for source in identity_sources + archive_sources
            }
            notices = list(
                dict.fromkeys(notice for result in results for notice in result.notices)
            )
            return GroundingResult(
                sources=list(sources.values()),
                notices=notices
                + [
                    "Route endpoint sources provide regional context, not an exact camera location."
                ],
            )
    grounding = GroundingResult()
    try:
        async with httpx.AsyncClient(
            timeout=15,
            follow_redirects=True,
            headers={
                "User-Agent": "MemoryWorlds/0.1 (https://afrenkai--nonslop-streets-web.us-east.modal.run; historical place research)"
            },
        ) as client:
            place = await resolve_place(client, scene.location)
            if place is None:
                grounding.notices.append(
                    "The place name could not be resolved unambiguously. Add a town, region, or country for better search results. Unverified name matches were excluded."
                )
            else:
                grounding.sources.append(
                    SourceEvidence(
                        title=place.title,
                        url=place.fullurl,
                        excerpt=place.extract[:1400],
                        kind="Place identity and general history; not evidence of the requested year",
                    )
                )
                location = scene.location.split(",", maxsplit=1)[0].strip()
                names = (
                    await geographic_names(client, place)
                    if scene.year is not None
                    else []
                )
                period_sources = (
                    await archive_search(client, location, f"year:{scene.year}", names)
                    if scene.year is not None
                    else []
                )
                if scene.year is None:
                    grounding.notices.append(
                        "The capture year is unspecified. Search provides place identity and general history; period archive material was not assigned to this memory."
                    )
                if not period_sources and scene.year is not None:
                    period_sources = await archive_search(
                        client,
                        location,
                        f"year:[{scene.year - 10} TO {scene.year + 10}]",
                        names,
                    )
                    if period_sources:
                        grounding.notices.append(
                            "No exact-year match was found. These sources are from within ten years; check their dates."
                        )
                grounding.sources.extend(period_sources)
                if not period_sources and scene.year is not None:
                    grounding.notices.append(
                        "No relevant period archive material was found for this place. General history does not establish its appearance in the requested year."
                    )
    except (httpx.HTTPError, ValidationError, ValueError) as error:
        logging.getLogger(__name__).warning("Place lookup failed: %s", error)
        grounding.notices.append(
            "Part of the place search was unavailable for this request."
        )
    if not grounding.sources:
        grounding.notices.append(
            "No web evidence was retrieved. The scene uses your description and uploads."
        )
    grounding.notices.append(
        "Retrieved sources provide context. The generated scene may include details not established by the sources."
    )
    return grounding
