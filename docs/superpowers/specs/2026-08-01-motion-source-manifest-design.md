# Motion Source Manifest Design

## Goal

Record the relationship between every exported robot motion and its source
SMPL-X file so `scripts/vis_gmr_debug.py` can load both from only `--motion`.
Keep the viewer's existing explicit inputs as optional overrides for old data
and recovery from stale paths.

## Directory and manifest format

Each robot export directory contains one shared manifest:

```text
retarget_data/<robot>/
├── joints.json
├── manifest.json
├── motions/<name>.pkl
├── datasets/<name>.npz
└── beyondmimic/<name>.csv
```

The version 1 schema is:

```json
{
  "format_version": 1,
  "robot": "dr02",
  "motions": {
    "motions/walk.pkl": {
      "source": {
        "path": "motion_data/AMASS/walk.npz",
        "base": "repository"
      },
      "dataset": "datasets/walk.npz",
      "beyondmimic": "beyondmimic/walk.csv"
    }
  }
}
```

Motion, dataset, and BeyondMimic paths are POSIX paths relative to the robot
export directory. A source inside the repository uses a POSIX path relative to
the repository root and `base: "repository"`. A source outside the repository
uses its normalized absolute path and `base: "absolute"`. The exporter resolves
relative source arguments against its process working directory before deciding
which representation to store.

The dictionary key is the exact relative PKL path, not only a basename. This
prevents accidental matches against a similarly named file in another
directory.

## Export behavior

`export_retarget_motion` keeps writing the existing four artifacts. After all
artifact files are successfully replaced, it atomically creates or updates
`manifest.json`.

Creating a manifest records its format version and robot. Updating one requires
the existing version and robot to match exactly. The exporter replaces the
entry for the current relative motion path, preserves all other entries, sorts
entries by path for stable diffs, writes a temporary sibling, and atomically
replaces the manifest. A repeated export is therefore idempotent; exporting the
same motion from a different source intentionally updates its relationship.

An artifact serialization failure leaves the existing manifest untouched. A
manifest update failure may leave newly written artifacts that are not yet
listed, but cannot publish an entry pointing to incomplete artifact writes.

## Viewer resolution

The simplified command is:

```bash
PYTHONPATH=. uv run python scripts/vis_gmr_debug.py \
  --motion retarget_data/dr02/motions/walk.pkl
```

Given a motion path, the viewer expects its robot export directory to be the
motion directory's parent directory and locates `manifest.json` there. It
normalizes the selected motion path relative to that directory and requires an
exact entry. It then:

1. obtains the robot name from the manifest;
2. resolves the source reference according to `source.base`;
3. selects `ROBOT_XML_DICT[robot]` as the MJCF;
4. selects `IK_CONFIG_DICT["smplx"][robot]` as the IK configuration;
5. runs the existing robot-plus-reference visualization pipeline.

`--reference`, `--mjcf`, and `--ik-config` remain accepted but become optional.
Each explicit value overrides only its corresponding inferred value. If all
three are given, the existing fully explicit workflow does not require a
manifest. When any value still needs inference, a valid manifest and robot
entry are required.

The manifest robot is authoritative for inferred project configuration. No
separate `--robot` argument is introduced.

## Errors

The viewer reports a focused error for each of these cases:

- missing or malformed manifest;
- unsupported manifest version;
- motion located outside the manifest's robot directory;
- exact motion entry missing;
- unsupported source base;
- resolved source file missing;
- manifest robot absent from `ROBOT_XML_DICT` or the SMPL-X IK config map;
- inferred or explicit MJCF/IK configuration missing on disk.

An explicit `--reference` bypasses only manifest source resolution. Explicit
MJCF and IK configuration values similarly bypass only their inference. This
makes stale external source paths recoverable without weakening validation of
the remaining inferred values.

## Components

Manifest path normalization, validation, atomic update, and lookup live in the
retarget export module rather than the CLI scripts. The exporter calls the
update API after artifact serialization. The viewer calls the lookup API and
then applies its CLI overrides. This keeps JSON details out of both scripts and
provides one tested contract.

Existing untracked debug-viewer files in the working tree are user-owned. The
implementation must preserve their current content and make targeted edits
only after establishing an isolated worktree from a committed or otherwise
safe base.

## Tests

Unit tests cover repository-relative and external absolute source paths,
creation, multi-motion preservation, same-motion replacement, stable ordering,
and version/robot mismatch rejection. Lookup tests cover exact path matching
and all malformed or missing manifest states.

CLI tests cover the one-argument viewer flow, each explicit override, the
fully explicit legacy flow, missing source files, and missing robot
configuration. Existing visualization math and rendering tests remain
unchanged. The complete project test suite runs before integration.
