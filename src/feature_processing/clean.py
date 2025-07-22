#!/usr/bin/env python3
"""
StepMania stepchart cleaning and processing functions.

This module provides functions to clean and restructure stepchart data
from processed JSON files, creating standardized measures and cleaned stepcharts.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def simplify_line(stepchart_line: str) -> str:
    """
    Simplify a stepchart line by converting special characters to basic steps/gaps.
    
    Args:
        stepchart_line: A 4-character string representing a stepchart line
        
    Returns:
        Simplified 4-character string with only 0s and 1s
        
    Raises:
        ValueError: If invalid character is found
    """
    if len(stepchart_line) != 4:
        raise ValueError(f"Stepchart line must be 4 characters, got: {len(stepchart_line)}")
    
    cleaned_line = ''
    for char in stepchart_line:
        if char in ('2', '4'):  # holds and drills to normal steps
            cleaned_line += '1'
        elif char in ('1', '0'):  # no change for steps and gaps
            cleaned_line += char
        elif char in ('M', '3'):  # mines and hold releases to gaps
            cleaned_line += '0'
        else:
            raise ValueError(f"Invalid character: {char}")
    
    return cleaned_line


def create_measure_arrays(stepchart: List[str]) -> List[List[str]]:
    """
    Convert a stepchart array into an array of measure arrays.
    
    Args:
        stepchart: List of stepchart strings with commas as measure boundaries
        
    Returns:
        List of measure arrays, where each measure is a list of 4-character strings
    """
    measures = []
    current_measure = []
    
    for line in stepchart:
        if line == ",":
            # End of measure, add current measure to measures
            if current_measure:  # Only add non-empty measures
                measures.append(current_measure)
            current_measure = []
        else:
            # Valid stepchart line, add to current measure
            current_measure.append(line)
    
    # Add the last measure if it exists
    if current_measure:
        measures.append(current_measure)
    
    return measures


def clean_measure(measure: List[str]) -> List[str]:
    """
    Clean a measure by simplifying all stepchart lines.
    
    Args:
        measure: List of 4-character stepchart strings
        
    Returns:
        List of cleaned 4-character stepchart strings
    """
    return [simplify_line(line) for line in measure]


def find_lowest_representation(measure: List[str]) -> int:
    """
    Find the lowest quantization level that can represent this measure.
    
    Args:
        measure: List of stepchart strings representing a measure
        
    Returns:
        The lowest quantization level (4, 8, 12, 16, 24) or original length if no simplification possible
    """
    measure_length = len(measure)
    quantization_attempts = [4, 8, 12, 16, 24]
    
    for quantization in quantization_attempts:
        # Check if measure length is divisible by quantization
        if measure_length % quantization != 0:
            continue
        
        # Calculate step size
        step_size = measure_length // quantization
        
        # Check if all non-quantized positions are "0000"
        success = True
        for i in range(measure_length):
            if i % step_size != 0:  # Not a quantization point
                if measure[i] != "0000":
                    success = False
                    break
        
        if success:
            return quantization
    
    # No simplification possible, return original length
    return measure_length


def represent_lower(measure: List[str], quantization: int) -> List[str]:
    """
    Reduce a measure to its lowest representation based on quantization.
    
    Args:
        measure: List of stepchart strings
        quantization: Target quantization level
        
    Returns:
        Reduced measure with only quantization points
    """
    if len(measure) == quantization:
        return measure
    
    if quantization < 4:
        raise ValueError(f"Quantization is too low: {quantization}")
    
    if len(measure) % quantization != 0:
        raise ValueError(f"Measure is not divisible by quantization: {len(measure)} % {quantization} != 0")
    
    # Extract only the quantization points
    step_size = len(measure) // quantization
    reduced_measure = []
    for i in range(0, len(measure), step_size):
        reduced_measure.append(measure[i])
    
    return reduced_measure


def represent_higher(measure: List[str], quantization: int) -> List[str]:
    """
    Expand a measure to a higher quantization level.
    
    Args:
        measure: List of stepchart strings
        quantization: Target quantization level
        
    Returns:
        Expanded measure with "0000" fillers
    """
    if len(measure) == quantization:
        return measure
    
    if quantization < 4:
        return measure
    
    if quantization % len(measure) != 0:
        return measure
    
    # Expand measure with "0000" fillers
    step_size = quantization // len(measure)
    expanded_measure = []
    for i in range(quantization):
        if i % step_size == 0:
            expanded_measure.append(measure[i // step_size])
        else:
            expanded_measure.append("0000")
    
    return expanded_measure


def standardize_measure(measure: List[str], target_quantization: int = 16) -> List[str]:
    """
    Standardize a measure to a target quantization level.
    
    Args:
        measure: List of stepchart strings
        target_quantization: Target quantization level (default: 16)
        
    Returns:
        Standardized measure with target quantization
    """
    # Find the lowest representation
    lowest_quantization = find_lowest_representation(measure)
    
    # Reduce to lowest representation
    reduced_measure = represent_lower(measure, lowest_quantization)
    
    # Expand to target quantization
    standardized_measure = represent_higher(reduced_measure, target_quantization)
    
    return standardized_measure


def clean_notes_section(notes_section: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean a single NOTES section by adding cleaned and standardized stepcharts.
    
    Args:
        notes_section: Dictionary containing a NOTES section
        
    Returns:
        Updated notes_section with new stepchart data
    """
    if 'stepchart' not in notes_section:
        return notes_section
    
    try:
        # Create measure arrays from stepchart
        measure_arrays = create_measure_arrays(notes_section['stepchart'])
        
        # Clean each measure
        cleaned_measures = [clean_measure(measure) for measure in measure_arrays]
        
        # Standardize each measure to 16 quantization
        standardized_measures = [standardize_measure(measure, 16) for measure in cleaned_measures]
        
        # Add new data to notes section
        notes_section['cleaned_stepchart'] = cleaned_measures
        notes_section['standardized_stepchart'] = standardized_measures
        
    except Exception as e:
        print(f"Error cleaning notes section: {e}")
        # Add empty arrays if cleaning fails
        notes_section['cleaned_stepchart'] = []
        notes_section['standardized_stepchart'] = []
    
    return notes_section


def process_song_json(song_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a complete song JSON by cleaning all NOTES sections.
    
    Args:
        song_data: Complete song JSON data
        
    Returns:
        Updated song data with CLEANED_NOTES
    """
    if 'NOTES' not in song_data:
        return song_data
    
    # Process each NOTES section
    cleaned_notes = []
    for notes_section in song_data['NOTES']:
        cleaned_section = clean_notes_section(notes_section)
        cleaned_notes.append(cleaned_section)
    
    # Add CLEANED_NOTES to the song data
    song_data['CLEANED_NOTES'] = cleaned_notes
    
    return song_data


def process_json_file(input_path: str, output_path: str) -> None:
    """
    Process a single JSON file and save the cleaned version.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to save cleaned JSON file
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            song_data = json.load(f)
        
        # Process the song data
        cleaned_data = process_song_json(song_data)
        
        # Save the cleaned data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        
        print(f"Processed: {input_path} -> {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def process_directory(input_dir: str, output_dir: str) -> None:
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir: Directory containing JSON files to process
        output_dir: Directory to save cleaned JSON files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = list(input_path.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files to process")
    
    for json_file in json_files:
        output_file = output_path / json_file.name
        process_json_file(str(json_file), str(output_file))
    
    print(f"Processing complete. Check {output_dir} for cleaned files.")


if __name__ == "__main__":
    # Example usage
    input_directory = "feature_data/processed/"
    output_directory = "feature_data/cleaned/"
    
    if os.path.exists(input_directory):
        process_directory(input_directory, output_directory)
    else:
        print(f"Input directory {input_directory} not found!")


    # need to find out what's the highest granularity in our data set 

    out_dir = 'feature_data/cleaned/' 
    measure_lengths = {}   
    offending_tracks = []
    chart_ct = 0 
    for file in os.listdir(out_dir): 
        file_path = os.path.join(out_dir, file)
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            notes = data['NOTES']

            for chart in notes: 
                chart_ct += 1
                chart_arr = chart['standardized_stepchart']
                for measure in chart_arr: 
                    ct = len(measure)
                    if ct not in measure_lengths: 
                        measure_lengths[ct] = 1
                    else: 
                        measure_lengths[ct] +=1

                    # if ct in [32, 64]:
                    #     offending_tracks.append((data['TITLE'], chart['level']))
                    #     print(data['TITLE'], chart['level'])
                    #     print(measure)

                    if ct != 16:
                        offending_tracks.append((file, chart['level']))


    print(measure_lengths)
    print(set(offending_tracks))
    print(len(set(offending_tracks)))
    print(chart_ct)


    offending_tracks =  [('Blue Sunshine (Original Mix).json', '11'), ('Magic Waters.json', '11'), ('Good For Me (Above & Beyond Club Mix).json', '10'), ('Good For Me (Above & Beyond Club Mix).json', '12'), ('A Life So Changed.json', '12'), ("Who's afraid of 138.json", '16'), ('Euphoria.json', '11'), ('M.O.N.I. (Barcoda Project Remix).json', '13'), ('Daydream.json', '11'), ('Anahera.json', '11'), ('No Inbetween.json', '11'), ('On A Good Day (Daniel Kandi Remix).json', '9'), ("Everything (John O'Callaghan Remix).json", '10'), ('Blue Sunshine (Original Mix).json', '13'), ('Magic Waters.json', '13'), ("Everything (John O'Callaghan Remix).json", '12'), ('The Human Spirit.json', '12'), ('Good For Me (Above & Beyond Club Mix).json', '11'), ('A Life So Changed.json', '11'), ('On A Good Day (Daniel Kandi Remix).json', '10'), ('On A Good Day (Daniel Kandi Remix).json', '12'), ('Listen Feel Enjoy.json', '10'), ('Blue Sunshine (Original Mix).json', '10'), ('Listen Feel Enjoy.json', '12'), ('Magic Waters.json', '10'), ('Anahera.json', '9'), ('Blue Sunshine (Original Mix).json', '12'), ('Magic Waters.json', '12'), ('The Human Spirit.json', '11'), ("Everything (John O'Callaghan Remix).json", '11'), ('No Inbetween.json', '9'), ('Viola 2005.json', '13'), ('Euphoria.json', '12'), ('Anahera.json', '10'), ('Good For Me (Above & Beyond Club Mix).json', '9'), ('Daydream.json', '12'), ('Anahera.json', '12'), ('Stargazer.json', '13'), ('On A Good Day (Daniel Kandi Remix).json', '11'), ('No Inbetween.json', '10'), ('No Inbetween.json', '12'), ('Listen Feel Enjoy.json', '11')]
    
    print("\n" + "="*60)
    print("CREATING FINAL STEPCHART FOLDER WITH EXCLUSIONS")
    print("="*60)
    
    # Create final stepchart folder by copying from cleaned folder with exclusions
    def create_final_stepchart_folder(input_dir: str, output_dir: str, exclusion_list: List[Tuple[str, str]]) -> None:
        """
        Create final stepchart folder by copying from cleaned folder while excluding specific song-difficulty combinations.
        
        Args:
            input_dir: Directory containing cleaned JSON files
            output_dir: Directory to save final JSON files
            exclusion_list: List of (filename, level) tuples to exclude
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert exclusion list to set for faster lookup
        exclusion_set = set(exclusion_list)
        
        # Find all JSON files
        json_files = list(input_path.glob("*.json"))
        
        print(f"Found {len(json_files)} JSON files to process")
        print(f"Excluding {len(exclusion_set)} song-difficulty combinations")
        
        total_charts = 0
        excluded_charts = 0
        processed_files = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    song_data = json.load(f)
                
                # Get the filename for exclusion checking
                filename = json_file.name
                
                # Process NOTES section, excluding specified combinations
                if 'NOTES' in song_data:
                    original_notes = song_data['NOTES']
                    filtered_notes = []
                    
                    for chart in original_notes:
                        total_charts += 1
                        chart_level = chart.get('level', '')
                        
                        # Check if this chart should be excluded
                        exclusion_key = (filename, chart_level)
                        if exclusion_key in exclusion_set:
                            excluded_charts += 1
                            print(f"Excluding: {filename} - Level {chart_level}")
                        else:
                            filtered_notes.append(chart)
                    
                    # Update the song data with filtered notes
                    song_data['NOTES'] = filtered_notes
                
                # Save the filtered data
                output_file = output_path / json_file.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(song_data, f, indent=2, ensure_ascii=False)
                
                processed_files += 1
                print(f"Processed: {filename} ({len(filtered_notes)} charts kept)")
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        print(f"\nFinal Summary:")
        print(f"- Files processed: {processed_files}")
        print(f"- Total charts: {total_charts}")
        print(f"- Charts excluded: {excluded_charts}")
        print(f"- Charts kept: {total_charts - excluded_charts}")
        print(f"- Output directory: {output_dir}")
    
    # Create the final stepchart folder
    cleaned_dir = "feature_data/cleaned/"
    final_dir = "feature_data/stepchart_final/"
    
    if os.path.exists(cleaned_dir):
        create_final_stepchart_folder(cleaned_dir, final_dir, offending_tracks)
    else:
        print(f"Cleaned directory {cleaned_dir} not found!")
