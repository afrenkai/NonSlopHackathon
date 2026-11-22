from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath

from nonslop.generation.schemas import DeerFlowObservation, JobRecord, SourceEvidence
from nonslop.grounding.research import deerflow_observations, research_packet_sources

LOCAL_RESEARCH_ROOT = Path(__file__).resolve().parents[3] / "research"
MODAL_RESEARCH_ROOT = Path("/opt/nonslop-research")


def bundled_path(relative_path: str) -> Path:
    deployed_path = MODAL_RESEARCH_ROOT / relative_path
    return (
        deployed_path
        if deployed_path.is_file()
        else LOCAL_RESEARCH_ROOT / relative_path
    )


@cache
def bundled_research(relative_path: str) -> tuple[SourceEvidence, ...]:
    return tuple(research_packet_sources(bundled_path(relative_path).read_bytes()))


@cache
def bundled_observations(relative_path: str) -> tuple[DeerFlowObservation, ...]:
    return deerflow_observations(
        bundled_path(relative_path).read_text(encoding="utf-8")
    )


@dataclass(frozen=True)
class DemoVisit:
    location: str
    year: int
    photo: str
    research_file: str
    research_notes: tuple[SourceEvidence, ...] = ()

    @property
    def research(self) -> tuple[SourceEvidence, ...]:
        return self.research_notes + bundled_research(self.research_file)

    @property
    def observation_file(self) -> str:
        return str(PurePosixPath(self.research_file).with_name("observations.jsonl"))


DEMO_VISITS = {
    "hakodate": DemoVisit(
        "Hakodate Red Brick Warehouse",
        2023,
        "demo-visits/hakodate-red-brick-warehouse.png",
        "hakodate-bay/evidence/prompt-evidence.json",
    ),
    "flatbush": DemoVisit(
        "25 Flatbush Ave",
        1990,
        "demo-visits/block-2106-lot-0026.jpg",
        "25-flatbush/evidence/prompt-evidence.json",
        (
            # Flatbush DeerFlow receipts r285, r283, r284 and r76.
            SourceEvidence(
                title="NYC tax photograph · Lot 26",
                url="https://80snyc.bdon.org/photos/3/02106/0026.jpg",
                kind="Opening photograph · neighboring lot",
                date="1980s",
                excerpt="The opening image is captioned 15–17 Flatbush Avenue. The nearby frontage shows broad second-floor glazing and a dark projecting cornice. The photo predates the requested year and the storefront lettering is unclear.",
            ),
            SourceEvidence(
                title="NYC tax photograph · Lot 19",
                url="https://80snyc.bdon.org/photos/3/02106/0019.jpg",
                kind="Neighboring photograph",
                date="1980s",
                excerpt="A parking gap shows another angle on the block, with parked cars and faded painted lettering on a masonry side wall.",
            ),
            SourceEvidence(
                title="NYC tax photograph · Lot 16",
                url="https://80snyc.bdon.org/photos/3/02106/0016.jpg",
                kind="Neighboring photograph",
                date="1980s",
                excerpt="This view shows the nearby bank elevation, with a large glazed commercial base and regular upper window bays. Shop names are unclear.",
            ),
            SourceEvidence(
                title="NYC digital tax map · Brooklyn Block 2106",
                url="https://propertyinformationportal.nyc.gov/pdf/home/index/map_library/30210620120618112304",
                kind="Site context",
                excerpt="The map places lot 24 on the Flatbush Avenue frontage between lots 26 and 19. It records parcel boundaries. The saved research does not establish the exact 1990 storefronts.",
            ),
        ),
    ),
}


def saved_demo_research(job: JobRecord) -> tuple[SourceEvidence, ...]:
    for demo in DEMO_VISITS.values():
        if (
            (job.scene.location, job.scene.year) == (demo.location, demo.year)
            and job.reference_names == [PurePosixPath(demo.photo).name]
            and not job.scene.use_web
        ):
            return demo.research
    return ()
