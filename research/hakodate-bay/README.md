# Hakodate Bay, December 2023 | DeerFlow visual research

One workflow and its evidence: DeerFlow looked for public records of the
scenery within ~250 metres of 41.767, 140.717 — the Hakodate Bay waterfront
in front of the Kanemori Red Brick Warehouses, Suehiro-chō, Hakodate — during
December 2023. The run was stopped manually (~22 min of a 30-minute bound)
before writing a final report; it saved 42 successful document/photo
inspections, including December-2023 shots of the floating Christmas Fantasy
tree and the illuminated warehouse district.

## Run the workflow

Run `./research/hakodate-bay/run.sh /tmp/hakodate-deerflow-new` from the
NonSlop root, using a new or empty output directory. The wrapper runs only
`deer-visual` and uses the `scene_scout` adapter in the sibling
`nonslophackathon` checkout. Set `SCOUT_ROOT` if that checkout is elsewhere.
Its existing `.scenery-harness/deer-venv` must have DeerFlow installed at
`1dafd8a3b45fbb9b98ca13ab3291c8f432bc81ed` (see the setup in that checkout).
The local model gateway must serve `kimi-k3-1-pro-high` at
`http://127.0.0.1:3001/v1`. The wrapper sets a 1,000-step limit and patches
`scenery_protocol.RUN_SECONDS` to 1800 before invoking the trial (the
protocol's pinned default is 900; the sibling checkout is left unmodified).
No benchmark reference material is fed to the run.

The protocol only takes `(location, year)`; "December" rides inside the
location string, so the task reads "...of <location> in 2023".

The agent uses web and image search to find candidate records, browser tools
to navigate archives, and `inspect_document` to read the original pages or
photographs. The evidence journal ties each observation to its source URL
and a tool receipt. Gateway or upstream changes can affect reruns.

## Evidence

- The centerpiece found is the 2023 **はこだてクリスマスファンタジー**
  (Hakodate Christmas Fantasy): a ~20 m tree on a barge floating in the bay
  in front of the warehouses, lit nightly with fireworks at 18:00. The agent
  inspected the [official 2023 event poster (r194)](evidence/christmas-fantasy-2023-poster.pdf)
  ("海に浮かぶツリー" — a tree floating on the sea).
- [christmas-fantasy-tree-red.jpg](evidence/photos/christmas-fantasy-tree-red.jpg)
  and [christmas-fantasy-tree-white.jpg](evidence/photos/christmas-fantasy-tree-white.jpg)
  are Wikimedia Commons copies of the floating tree's colour variants;
  [christmas-fantasy-tree-daytime.jpg](evidence/photos/christmas-fantasy-tree-daytime.jpg)
  shows it unlit. The white-lit variant matches the tree in the reference
  photo that prompted this run.
- [kanemori-20231223a–d.jpg](evidence/photos/) and
  [kanemori-winter-20231224.jpg](evidence/photos/kanemori-winter-20231224.jpg)
  are Commons photographs of the Kanemori warehouses dated 23–24 December
  2023 (per filename and Commons description);
  [nishi-hatoba-20231223.jpg](evidence/photos/nishi-hatoba-20231223.jpg) shows
  the Nishi Hatoba wharf the same weekend, and
  [kanemori-and-mt-hakodate-winter.jpg](evidence/photos/kanemori-and-mt-hakodate-winter.jpg)
  frames the warehouses against Mount Hakodate.
- [hakoshin-20231211-a.jpg](evidence/photos/hakoshin-20231211-a.jpg)/[b](evidence/photos/hakoshin-20231211-b.jpg)
  (hakoshin.jp, 11 Dec 2023) and
  [mainichi-20231202.jpg](evidence/photos/mainichi-20231202.jpg) (Mainichi,
  2 Dec 2023) are news photographs the agent inspected: warehouse gables
  traced with warm-white lights, and the tree lit for the event opening.
- [hakotamu-*.jpg](evidence/photos/) are December-2023 "函館イルミネーション
  2023" streetscape uploads from hakotamu.com the agent inspected: the tree
  over the water, light-wrapped huts and promenade, snow-dusted brick gables.
- [hakodate-event-2312e33.jpg](evidence/photos/hakodate-event-2312e33.jpg)/[e34](evidence/photos/hakodate-event-2312e34.jpg)
  are hakodate-event.com's December 2023 event coverage;
  [christmas-crowds.jpg](evidence/photos/christmas-crowds.jpg) and
  [hakodate-bay-christmas.jpg](evidence/photos/hakodate-bay-christmas.jpg)
  show the venue crowds and bay illumination.
- [Observations](evidence/observations.jsonl), [matching receipts](evidence/receipts.jsonl),
  and [run status](evidence/run.json) preserve the saved evidence. The raw
  visual observations contain imperfect OCR and attributions; use the
  original images to check them.
- The [source file index](evidence/source-files.json) maps successful
  inspections to local files by receipt ID and SHA-256. It covers all 42
  successful inspections. The poster and 32 photographs were added during
  recovery. Three Instagram photographs were recovered through their
  public post embeds using the matching image filenames.
- The [prompt evidence packet](evidence/prompt-evidence.json) uses seven
  successful, relevant inspections with source dates and receipt IDs.
  New Hakodate demo visits load it automatically. The December display,
  timed color changes, and fireworks apply only to the relevant scenes.

The agent identified but could not inspect the Wikimedia Commons files —
every wikimedia.org fetch timed out from the run environment (69 failed
inspections vs 42 successful). The Commons copies in `evidence/photos/` were
downloaded afterward for verification; their December-2023 dating rests on
filenames and Commons descriptions, not inspected content. Check archival
reuse terms before redistributing the photographs.
