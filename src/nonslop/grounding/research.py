import json

from pydantic import ValidationError

from nonslop.generation.schemas import DeerFlowObservation, SourceEvidence


def deerflow_observations(journal: str) -> tuple[DeerFlowObservation, ...]:
    observations = []
    for line in journal.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("tool") == "inspect_document" and record.get("ok") is True:
            observations.append(DeerFlowObservation.model_validate(record))
    return tuple(observations)


def evidence_date(item: dict) -> str | None:
    start, end = item.get("capture_from"), item.get("capture_to")
    if isinstance(start, int) and isinstance(end, int):
        label = str(start) if start == end else f"{start}–{end}"
        basis = item.get("date_basis")
        if isinstance(basis, str) and basis.strip():
            return f"{label} ({basis.strip()[:120]})"
        return label
    date = item.get("date")
    return date.strip()[:120] if isinstance(date, str) and date.strip() else None


def item_source(item: dict, max_excerpt: int) -> SourceEvidence | None:
    title, url = item.get("title"), item.get("record_url") or item.get("url")
    if not (
        isinstance(title, str)
        and title.strip()
        and isinstance(url, str)
        and url.strip()
    ):
        return None
    kind_parts = [
        str(part).strip()
        for part in (item.get("provider"), item.get("kind"), item.get("date_status"))
        if part
    ]
    excerpt = item.get("excerpt")
    try:
        return SourceEvidence(
            title=title.strip()[:200],
            url=url.strip(),
            excerpt=(excerpt.strip()[:max_excerpt] if isinstance(excerpt, str) else ""),
            kind=" · ".join(kind_parts)[:200] or "Saved research",
            date=evidence_date(item),
        )
    except ValidationError:
        return None


def research_packet_sources(
    payload: bytes, *, max_items: int = 24, max_excerpt: int = 1000
) -> list[SourceEvidence]:
    try:
        packet = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("The research packet must be a UTF-8 JSON file.") from error
    items = packet.get("items") if isinstance(packet, dict) else packet
    if not isinstance(items, list):
        raise TypeError("The research packet has no items list.")
    sources = [
        source
        for item in items[:max_items]
        if isinstance(item, dict)
        and (source := item_source(item, max_excerpt)) is not None
    ]
    if not sources:
        raise ValueError("The research packet has no usable evidence items.")
    return sources
