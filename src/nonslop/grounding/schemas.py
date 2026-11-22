from pydantic import BaseModel, Field

from nonslop.generation.schemas import SourceEvidence


class ArchiveItem(BaseModel):
    identifier: str
    title: str | list[str]
    description: str | list[str] = ""
    date: str | None = None
    subject: str | list[str] = ""
    coverage: str | list[str] = ""


class PlaceCoordinates(BaseModel):
    lat: float
    lon: float


class PlacePage(BaseModel):
    title: str
    extract: str = ""
    fullurl: str = ""
    coordinates: list[PlaceCoordinates] = Field(default_factory=list)
    pageprops: dict[str, str] = Field(default_factory=dict)


class PlaceQuery(BaseModel):
    pages: list[PlacePage] = Field(default_factory=list)


class PlaceResponse(BaseModel):
    query: PlaceQuery = Field(default_factory=PlaceQuery)


class EntityReference(BaseModel):
    id: str


class EntityValue(BaseModel):
    value: EntityReference


class EntitySnak(BaseModel):
    datavalue: EntityValue | None = None


class EntityStatement(BaseModel):
    mainsnak: EntitySnak
    rank: str = "normal"


class AdministrativeClaims(BaseModel):
    locations: list[EntityStatement] = Field(default_factory=list, alias="P131")


class EntityLabel(BaseModel):
    value: str


class GeographicEntity(BaseModel):
    labels: dict[str, EntityLabel] = Field(default_factory=dict)
    aliases: dict[str, list[EntityLabel]] = Field(default_factory=dict)
    claims: AdministrativeClaims = Field(default_factory=AdministrativeClaims)


class EntityResponse(BaseModel):
    entities: dict[str, GeographicEntity] = Field(default_factory=dict)


class ArchiveResults(BaseModel):
    numFound: int = 0
    docs: list[ArchiveItem] = Field(default_factory=list)


class ArchiveResponse(BaseModel):
    response: ArchiveResults


class GroundingResult(BaseModel):
    sources: list[SourceEvidence] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
