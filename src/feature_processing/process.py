#!/usr/bin/env python3
"""
StepMania (.sm) file parser for ML feature extraction.

This script parses .sm files and converts them to JSON format, extracting:
- All top-level metadata keys
- NOTES sections with difficulty and stepchart data

The output is saved in a flat structure (all JSON files in one directory)
with automatic conflict resolution for duplicate filenames.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


def parse_sm_file(file_path: str) -> Dict[str, Any]:
    """
    Parse a StepMania .sm file and convert it to JSON format.
    
    Args:
        file_path: Path to the .sm file
        
    Returns:
        Dictionary containing parsed data with top-level keys and NOTES array
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Initialize result dictionary
    result = {}
    
    # Parse top-level keys (lines starting with #)
    # These follow the pattern: #KEY:value;
    top_level_pattern = r'^#([^:]+):([^;]*);'
    
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('#') and ':' in line and line.endswith(';'):
            match = re.match(top_level_pattern, line)
            if match:
                key = match.group(1)
                value = match.group(2)
                result[key] = value
    
    # Parse NOTES sections
    notes_sections = []
    
    # Find all NOTES sections - look for #NOTES: followed by content until the next #NOTES: or end of file
    notes_pattern = r'#NOTES:\s*\n(.*?)(?=\n#NOTES:|$)'
    notes_matches = re.findall(notes_pattern, content, re.DOTALL)
    
    for notes_content in notes_matches:
        notes_data = parse_notes_section(notes_content)
        if notes_data:
            notes_sections.append(notes_data)
    
    result['NOTES'] = notes_sections
    
    return result


def parse_notes_section(notes_content: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single NOTES section to extract difficulty and stepchart.
    
    Args:
        notes_content: The content of a NOTES section
        
    Returns:
        Dictionary with stepartist, difficulty, level, groovemeter, and stepchart
    """
    lines = [line.strip() for line in notes_content.split('\n') if line.strip()]
    
    if len(lines) < 6:
        return None
    
    # Extract the first 6 lines (stepartist, difficulty, level, groovemeter, stepchart)
    # Remove trailing colons and clean up
    level = lines[3].rstrip(':')
    
    # The stepchart data starts from line 5 onwards
    # It's a series of 4-character lines separated by commas and newlines
    stepchart_lines = []
    
    # Process all lines from line 5 onwards
    for line in lines[5:]:
        # Split by comma to preserve measure boundaries
        parts = line.split(',')
        for i, part in enumerate(parts):
            part = part.strip()
            # Check if this is a valid 4-character stepchart line
            if len(part) == 4 and all(c in '0123456789M' for c in part):
                stepchart_lines.append(part)
            # Add comma only if this is an empty part (measure boundary)
            elif part == '' and i < len(parts) - 1:  # Don't add comma for the last empty part
                stepchart_lines.append(',')
    
    return {
        'level': level,
        'stepchart': stepchart_lines
    }


def process_sm_files(input_dir: str, output_dir: str) -> None:
    """
    Process all .sm files in a directory and save them as JSON in a flat structure.
    
    Args:
        input_dir: Directory containing .sm files
        output_dir: Directory to save JSON files (flat structure)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .sm files
    sm_files = list(input_path.rglob("*.sm"))
    
    print(f"Found {len(sm_files)} .sm files to process")
    
    # Track used filenames to handle conflicts
    used_filenames = set()
    
    for sm_file in sm_files:
        try:
            # Parse the .sm file
            parsed_data = parse_sm_file(str(sm_file))
            
            # Create flat output filename (just the song name)
            song_name = sm_file.stem  # Remove .sm extension
            
            # Handle filename conflicts
            output_filename = f"{song_name}.json"
            counter = 1
            
            # If filename already exists, add parent folder name or counter
            while output_filename in used_filenames:
                parent_folder = sm_file.parent.name
                if counter == 1:
                    output_filename = f"{parent_folder}_{song_name}.json"
                else:
                    output_filename = f"{song_name}_{counter}.json"
                counter += 1
            
            used_filenames.add(output_filename)
            output_file = output_path / output_filename
            
            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            
            print(f"Processed: {sm_file.name} -> {output_filename}")
            
        except Exception as e:
            print(f"Error processing {sm_file}: {e}")


def main():
    """Main function to run the processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse StepMania .sm files to JSON')
    parser.add_argument('input_dir', help='Directory containing .sm files')
    parser.add_argument('output_dir', help='Directory to save JSON files')
    
    args = parser.parse_args()
    
    process_sm_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main() 

    # python src/feature_processing/process.py stepfile_assets/TranceMania3 feature_data/processed/all_songs_flat