# StepMania .sm File Parser

This project contains a Python script to parse StepMania (.sm) files and convert them to JSON format for ML feature extraction.

## Overview

The parser extracts:
- All top-level metadata keys (TITLE, ARTIST, BPM, etc.)
- NOTES sections with difficulty and stepchart data
- Stepchart data as arrays of 4-character strings representing dance steps

## Files

- `src/feature_processing/process.py` - Main parser script
- `test_process.py` - Test script to run on Euphoria.sm and process all files
- `output/test/Euphoria.json` - Sample output from Euphoria.sm
- `output/processed/` - Directory containing all processed JSON files

## Usage

### Process a single file
```python
from src.feature_processing.process import parse_sm_file

# Parse a single .sm file
parsed_data = parse_sm_file("path/to/file.sm")
```

### Process all files in a directory
```python
from src.feature_processing.process import process_sm_files

# Process all .sm files in a directory
process_sm_files("input_directory", "output_directory")
```

### Command line usage
```bash
python src/feature_processing/process.py input_directory output_directory
```

### Run the test script
```bash
python test_process.py
```

## Output Format

The JSON output contains:

```json
{
  "TITLE": "Song Title",
  "ARTIST": "Artist Name",
  "BPM": "140.000",
  "NOTES": [
    {
      "stepartist": "dance-single",
      "difficulty": "t0ni",
      "level": "Challenge",
      "groovemeter": "12",
      "stepchart": [
        "0000",
        "1001",
        "0001",
        "0010"
      ]
    }
  ]
}
```

## Stepchart Format

Each stepchart line is a 4-character string representing:
- `0` = No step
- `1` = Step
- `2` = Hold start
- `3` = Hold end
- `M` = Mine

The 4 characters represent: Left, Down, Up, Right

## Example

The test script processes Euphoria.sm and shows:
- 21 top-level keys extracted
- 5 NOTES sections (different difficulties)
- Stepchart lines: 3012 (Challenge), 2788 (Hard), 2660 (Medium), 2612 (Easy), 860 (Beginner)

## Requirements

- Python 3.6+
- No external dependencies (uses only standard library) 