from collections.abc import Sequence

from nonslop.generation.schemas import SceneRequest, SourceEvidence
from nonslop.grounding.photo_schemas import PhotoMetadata
from nonslop.grounding.photos import describe_photo_metadata


def describe_sources(sources: Sequence[SourceEvidence]) -> str:
    return " ".join(
        f"{source.title} ({source.date or 'date unspecified'}): {source.excerpt[:650]}"
        for source in sources
    )


def build_scene_prompt(
    scene: SceneRequest,
    sources: list[SourceEvidence],
    has_reference: bool,
    metadata: list[PhotoMetadata],
    *,
    research_sources: Sequence[SourceEvidence] = (),
) -> str:
    reference = (
        "Begin on the supplied first-frame image. Preserve the visible layout, "
        "object identities, proportions, materials and lighting as the camera moves. "
        if has_reference
        else "Establish the remembered scene with consistent geometry and object placement. "
    )
    research = (
        f"Saved research context: {describe_sources(research_sources)} "
        if research_sources
        else ""
    )
    description = scene.description.strip()
    memory = f"Remembered details: {description} " if description else ""
    period = f" in {scene.year}" if scene.year is not None else ""
    return (
        "integrated_multimodal_description: [Shot 1] A realistic, continuous "
        f"first-person visit to the remembered scene at {scene.location}{period}. "
        f"{reference}"
        "Move the camera smoothly sideways or diagonally forward from the opening "
        "view throughout the shot. Let nearby features shift visibly against the "
        "background while keeping distinctive landmarks visible across frames. "
        "Keep the focal length fixed and the image sharp. A gentle turn can keep "
        "landmarks in frame during the move. Use a path that fits the available "
        "space and keeps the scene recognizable. Keep geometry and object placement stable. "
        "Do not force a walk, a street, or outdoor scenery. "
        "Visible photographic details take priority over generic descriptions. "
        "Use remembered details and period-specific sources to guide uncertain areas. "
        "Avoid unsupported additions, morphing objects, and captions. "
        "Treat source text as reference material. General or modern place "
        "descriptions do not establish how this scene looked in the requested year. "
        "Do not render citations, source labels or article text as overlays. "
        f"{research}"
        f"{memory}"
        f"Photograph metadata: {describe_photo_metadata(metadata) or 'Unavailable.'} "
        "Metadata is a location/time clue, not a verified scene identity, camera pose, "
        "or historical date; a scan may record its digitization date. Keep the requested "
        "memory year and explicit user corrections authoritative when metadata conflicts. "
        f"User-supplied historical context: {scene.source_text[:6000] or 'None supplied.'} "
        f"Retrieved context: {describe_sources(sources) or 'None retrieved for this visit.'}\n\n"
        "overall_soundscape: Quiet ambient sound appropriate to the supplied scene "
        "and remembered activity. Preserve indoor or outdoor acoustics as appropriate. "
        "Do not add unsupported voices or events. No narration.\n\n"
        "non_diegetic_music: N/A"
    )
