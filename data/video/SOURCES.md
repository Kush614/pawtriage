# Video clip sources

Put clips (5–15 s, mp4/mov) in `data/video/clips/`. One row per clip in `labels.csv`:
- `label_a`, `label_b`: two humans' independent `vid_*` scenario ids (see `data/rubric.csv`). Rows where they disagree are **dropped**.
- `source`: `team-recorded` or the URL; `license`: must be permissive (CC0/CC-BY/own footage).

| clip | source | license |
|---|---|---|
