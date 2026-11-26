# anim-pixelizer

## Description
`anim-pixelizer` is a command-line interface (CLI) utility and multi-stage processing workflow for pixel-art animation assets. It coordinates raw sprite generation (`llm`), sheet slicing and background cleanup (`preprocess`), pixel grid and palette alignment (`fixer`), deterministic pixel cleanup and contour styling (`postprocess`), interactive frame inspection (`review`), and full-chain execution (`run`).

## Setup
1. Navigate to the project root directory:
   ```bash
   cd anim_pixelizer
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install project dependencies:
   ```bash
   pip install -U pip
   pip install -e .
   ```
4. Create or verify `config.json` in the project root directory to configure stage defaults, working directory path, palette hex codes, target dimensions, and external fixer settings:
   ```bash
   cp config.example.json config.json
   ```
5. Verify CLI availability and help output:
   ```bash
   anim-pixelizer --help
   ```

## Usage

### Global CLI Options
- `-c`, `--config`: Path to configuration JSON file (default: `config.json`).
- `-v`, `--verbose`: Enable debug-level logging across pipeline stages.
- `-w`, `--workdir`: Override working directory for intermediate artifacts.

### `llm`
Generates raw animation frames or sprite sheets via an LLM backend.
- `-r`, `--ref`: Path to reference sprite image.
- `-p`, `--prompt`: Required text prompt describing the animation.
- `-o`, `--output`: Output directory for generated artifacts.
- `-n`, `--frames`: Number of animation frames to generate.

Sample output:
```text
not ready yet
```

### `preprocess`
Validates, cleans backgrounds and divider lines, extracts sprite frames, and checks frame counts.
- `input`: Required input sprite sheet file or directory.
- `-o`, `--output`: Output directory for extracted frames.
- `--pad`: Padding in pixels around each extracted frame.
- `--min-size`: Minimum width and height in pixels for extracted frames.
- `--fuzz`: Background flood-fill tolerance percentage.
- `--bg-color`: Explicit background color in `#RRGGBB` hex format.
- `--no-dividers`: Disable divider line detection and erasure.
- `--divider-min-coverage`: Minimum coverage ratio (`0.0`–`1.0`) required to classify a line as a divider.
- `--divider-max-luma`: Maximum luminance threshold for dark divider lines.
- `--divider-thickness`: Erasure thickness in pixels for detected divider lines.
- `--no-clean-pockets`: Disable cleanup of isolated background pockets.
- `--pocket-fuzz`: Color tolerance percentage for isolated background pocket cleanup.
- `--no-merge-shards`: Disable grouping of nearby sprite components into a single frame box.
- `--shard-merge-ratio`: Multiplier applied to average nearest-neighbor distance for component grouping.
- `--max-shard-gap`: Maximum pixel gap allowed when grouping nearby sprite components.
- `--expected-frames`: Expected frame count for validation.
- `--no-strict-frames`: Do not raise an error when extracted frame count differs from `--expected-frames`.
- `--bg-mode`: Output background mode (`transparent` or `solid`).
- `--solid-bg-color`: Solid background color in `#RRGGBB` hex format when `--bg-mode solid` is used.

Sample output:
```text
[INFO] preprocess.orchestrator: Processing sheet 'workdir/preprocess/sheet.png' (prefix='sheet', out='workdir/preprocess')
[INFO] preprocess.orchestrator: Loaded 'workdir/preprocess/sheet.png' (1024x1024)
[INFO] preprocess.clean: Divider pass: erased 24 horizontal and 16 vertical lines (40576 px, min_coverage=0.85)
[INFO] preprocess.clean: Flood-fill background pass: cleared 637437 pixels (ref_rgb=(236, 216, 179), fuzz=15.0%)
[preprocess] workdir/preprocess/sheet.png -> workdir/preprocess/sheet_00.png
```

### `fixer`
Runs an external pixel-art fixer tool and optional post-command or nearest-neighbor resize on frames.
- `input`: Required input image file or directory.
- `-o`, `--output`: Output file or directory path.
- `--size`: Target width and height (`W H`).
- `--fixer-arg`: Additional argument passed to the fixer command (repeatable).
- `--palette-image`: Path to palette image for color remapping.
- `--post-cmd`: Enable execution of configured post-command.
- `--no-post-cmd`: Disable execution of configured post-command.
- `--resize-to-target`: Force nearest-neighbor resize to target dimensions after fixer execution.

Sample output:
```text
[fixer] workdir/preprocess/sheet_00.png -> workdir/fixer/sheet_00_fixed.png
```

### `postprocess`
Applies alpha binarization, isolated pixel removal, local noise cluster cleanup, palette snapping, and contour painting.
- `input`: Required input image file or directory.
- `-o`, `--output`: Output file or directory path.
- `--no-outline`: Disable contour painting around opaque pixels.
- `--no-palette`: Disable snapping colors to the configured palette.

Sample output:
```text
[postprocess] workdir/fixer/sheet_00_fixed.png -> workdir/postprocess/sheet_00_fixed_clean.png
```

### `review`
Launches an interactive loop to inspect processed images and optionally trigger rework steps.
- `input`: Required input image file or directory.

Sample output:
```text
Approved frames: 4/4
```

### `run`
Executes a sequence of stages starting from a chosen stage and optionally runs interactive review.
- `input`: Required input file or directory.
- `--from`: Starting stage (`preprocess`, `fixer`, or `postprocess`).
- `--no-review`: Skip the interactive review stage at the end of the run.
- Plus all preprocess override options (`--pad`, `--min-size`, `--fuzz`, `--bg-color`, `--no-dividers`, `--divider-min-coverage`, `--divider-max-luma`, `--divider-thickness`, `--no-clean-pockets`, `--pocket-fuzz`, `--no-merge-shards`, `--shard-merge-ratio`, `--max-shard-gap`, `--expected-frames`, `--no-strict-frames`, `--bg-mode`, `--solid-bg-color`).
- Plus all fixer override options (`--size`, `--fixer-arg`, `--palette-image`, `--post-cmd`, `--no-post-cmd`, `--resize-to-target`).

Sample output:
```text
[preprocess] workdir/preprocess/sheet.png -> workdir/preprocess/sheet_00.png
[fixer] workdir/preprocess/sheet_00.png -> workdir/fixer/sheet_00_fixed.png
[postprocess] workdir/fixer/sheet_00_fixed.png -> workdir/postprocess/sheet_00_fixed_clean.png

Done. Approved frames: 1/1
```