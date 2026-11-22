# 25 Flatbush Avenue | DeerFlow visual research

One workflow and its evidence: DeerFlow looked for public records of the
area around 25 Flatbush Avenue, Brooklyn, circa 1990. The run timed out
before writing a final report; it did save inspections of ten tax photos
and two document pages.

## Run the workflow

Run `./research/25-flatbush/run.sh /tmp/flatbush-deerflow-new` from the
NonSlop root, using a new or empty output directory. The wrapper runs only
`deer-visual` and uses the `scene_scout` adapter in the sibling
`nonslophackathon` checkout. Set `SCOUT_ROOT` if that checkout is elsewhere.
Its existing `.scenery-harness/deer-venv` must have DeerFlow installed at
`1dafd8a3b45fbb9b98ca13ab3291c8f432bc81ed` (see the setup in that checkout).
The local model gateway must serve `kimi-k3-1-pro-high` at
`http://127.0.0.1:3001/v1`. The wrapper sets a 1,000-step limit; the adapter
stops after 900 seconds. No benchmark reference material is fed to the run.

The agent uses web and image search to find candidate records, browser tools
to navigate archives, and `inspect_document` to read the original pages or
photographs. The evidence journal ties each observation to its source URL
and a tool receipt. Gateway or upstream changes can affect reruns.

## Evidence

- The sibling reconstruction's [1989 certificate of occupancy](evidence/1989-occupancy.pdf)
  records the uses at 19–27 Flatbush following the alteration. It lists a
  45-foot height, while a later footprint survey gives 69.87 feet.
- The [April 1996 aerial mosaic](evidence/1996-survey.jpg) provides a later
  view of the block and its street alignment. The mosaic postdates the
  requested scene by six years.
- The sibling project supplied the [reconstruction evidence export](evidence/evidence.js),
  [catalog](evidence/catalog.json), and [derived site plan](evidence/plan.svg).
  The [prompt evidence packet](evidence/prompt-evidence.json) uses the
  export's building descriptions and limitations alongside notes on the
  certificate and aerial mosaic. New Flatbush demo visits load it automatically.
- [Lot 26 (r285)](evidence/photos/block-2106-lot-0026.jpg) is captioned
  15-17 Flatbush. The five-storey building beside that frontage is the
  25 Flatbush anchor. The broad second-floor glazing and projecting dark
  cornice are visible.
- [Lot 19 (r283)](evidence/photos/block-2106-lot-0019.jpg) shows the anchor
  from the parking gap, with cars, faded side-wall advertising, and the bank
  beyond. [Lot 16 (r284)](evidence/photos/block-2106-lot-0016.jpg) shows the
  bank elevation more directly.
- [All ten tax photographs](evidence/photos/) are local copies of the Public
  80s.NYC mirrors of NYC Department of Finance photos. The published lot
  captions identify the record; they do not identify every building in view.
- [NYC digital tax map (r76)](https://propertyinformationportal.nyc.gov/pdf/home/index/map_library/30210620120618112304)
  and [Downtown Brooklyn planning document (r73)](https://www.nyc.gov/assets/planning/download/pdf/plans/downtown-brooklyn2/dwnbklyn2.pdf)
  are contextual document inspections, not photographs of 1990.
- [Observations](evidence/observations.jsonl), [matching receipts](evidence/receipts.jsonl),
  and [run status](evidence/run.json) preserve the saved evidence. The raw
  visual observations contain imperfect OCR and attributions; use the
  original images to check them.

The tax photos predate 1990. The run did not inspect a usable photograph
from the target lot itself, and the neighboring views do not establish
exact 1990 tenant names. Check archival reuse terms before redistributing
the photographs.
