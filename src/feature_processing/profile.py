#!/usr/bin/env python3
"""
StepMania stepchart profiling helper functions.

This script provides utilities to analyze stepchart data and extract
statistical information about character usage, patterns, and distributions.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Counter
from collections import defaultdict


def profile_stepchart_characters(stepchart_array: List[str]) -> Dict[str, int]:
    """
    Analyze a stepchart array and count the usage of each character.
    
    Args:
        stepchart_array: List of stepchart strings (e.g., ["0000", "1001", "M000"])
        
    Returns:
        Dictionary with character counts (e.g., {"0": 1000, "1": 500, "M": 50})
    """
    char_counts = defaultdict(int)
    
    for stepchart_line in stepchart_array:
        # Skip commas (measure boundaries)
        if stepchart_line == ',':
            continue
            
        # Split each 4-character line into individual characters
        for char in stepchart_line:
            char_counts[char] += 1
    
    return dict(char_counts)


def profile_all_files_character_usage(json_directory: str) -> Dict[str, Any]:
    """
    Profile character usage across all processed JSON files.
    
    Args:
        json_directory: Directory containing processed JSON files
        
    Returns:
        Dictionary with overall statistics and per-file breakdowns
    """
    json_path = Path(json_directory)
    
    if not json_path.exists():
        raise FileNotFoundError(f"Directory {json_directory} not found")
    
    # Find all JSON files
    json_files = list(json_path.rglob("*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {json_directory}")
    
    # Overall character counts across all files
    overall_char_counts = defaultdict(int)
    
    # Per-file statistics
    file_stats = {}
    
    # Per-difficulty statistics
    difficulty_stats = defaultdict(lambda: defaultdict(int))
    
    print(f"Profiling {len(json_files)} JSON files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_char_counts = defaultdict(int)
            file_name = json_file.stem
            
            # Process each NOTES section
            if 'NOTES' in data:
                for i, notes_section in enumerate(data['NOTES']):
                    if 'stepchart' in notes_section:
                        stepchart = notes_section['stepchart']
                        difficulty = notes_section.get('level', f'Unknown_{i}')
                        
                        # Get character counts for this stepchart
                        section_counts = profile_stepchart_characters(stepchart)
                        
                        # Add to overall counts
                        for char, count in section_counts.items():
                            overall_char_counts[char] += count
                            file_char_counts[char] += count
                        
                        # Add to difficulty-specific counts
                        for char, count in section_counts.items():
                            difficulty_stats[difficulty][char] += count
            
            # Convert defaultdict to regular dict for JSON serialization
            file_stats[file_name] = {
                'character_counts': dict(file_char_counts),
                'total_steps': sum(file_char_counts.values()),
                'file_path': str(json_file)
            }
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Convert defaultdicts to regular dicts
    overall_stats = {
        'total_files': len(json_files),
        'overall_character_counts': dict(overall_char_counts),
        'total_steps': sum(overall_char_counts.values()),
        'character_percentages': {}
    }
    
    # Calculate percentages
    total_steps = overall_stats['total_steps']
    if total_steps > 0:
        for char, count in overall_char_counts.items():
            overall_stats['character_percentages'][char] = (count / total_steps) * 100
    
    # Convert difficulty stats
    difficulty_stats_dict = {}
    for difficulty, char_counts in difficulty_stats.items():
        difficulty_stats_dict[difficulty] = {
            'character_counts': dict(char_counts),
            'total_steps': sum(char_counts.values())
        }
    
    return {
        'overall_stats': overall_stats,
        'file_stats': file_stats,
        'difficulty_stats': difficulty_stats_dict
    }


def print_character_profile_summary(profile_data: Dict[str, Any]) -> None:
    """
    Print a summary of the character profiling results.
    
    Args:
        profile_data: Output from profile_all_files_character_usage
    """
    overall = profile_data['overall_stats']
    
    print("\n" + "="*60)
    print("STEPCHART CHARACTER USAGE PROFILE")
    print("="*60)
    
    print(f"\nTotal files analyzed: {overall['total_files']}")
    print(f"Total steps analyzed: {overall['total_steps']:,}")
    
    print(f"\nOverall character counts:")
    print("-" * 40)
    for char, count in sorted(overall['overall_character_counts'].items()):
        percentage = overall['character_percentages'].get(char, 0)
        print(f"  '{char}': {count:,} ({percentage:.2f}%)")
    
    print(f"\nDifficulty breakdown:")
    print("-" * 40)
    for difficulty, stats in profile_data['difficulty_stats'].items():
        total = stats['total_steps']
        print(f"\n  {difficulty}:")
        for char, count in sorted(stats['character_counts'].items()):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"    '{char}': {count:,} ({percentage:.2f}%)")


def save_profile_results(profile_data: Dict[str, Any], output_file: str) -> None:
    """
    Save profiling results to a JSON file.
    
    Args:
        profile_data: Output from profile_all_files_character_usage
        output_file: Path to save the results
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(profile_data, f, indent=2, ensure_ascii=False)
    print(f"\nProfile results saved to: {output_file}")


def main():
    """Main function to run the profiler."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Profile StepMania stepchart character usage')
    parser.add_argument('json_directory', help='Directory containing processed JSON files')
    parser.add_argument('--output', '-o', help='Output file for results (default: profile_results.json)')
    
    args = parser.parse_args()
    
    # Set default output file
    output_file = args.output or 'profile_results.json'
    
    try:
        # Run the profiler
        profile_data = profile_all_files_character_usage(args.json_directory)
        
        # Print summary
        print_character_profile_summary(profile_data)
        
        # Save results
        save_profile_results(profile_data, output_file)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 

    # python src/feature_processing/profile.py feature_data/processed/all_songs_flat