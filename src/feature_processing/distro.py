#!/usr/bin/env python3
"""
StepMania difficulty distribution analysis.

This script analyzes the difficulty distribution of stepcharts
after excluding a predefined ban list of song-difficulty combinations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict, Counter


# Ban list of (song_name, difficulty_level) tuples
# Using exact TITLE values from JSON files
BAN_LIST = {
    ('Anahera', '9'), ('Euphoria', '12'), ('On A Good Day', '9'), ('Blue Sunshine', '13'), 
    ('Viola 2005', '13'), ('Good For Me', '9'), ('A Life So Changed', '11'), ('Daydream', '11'), 
    ('Daydream', '12'), ('Everything', '11'), ('No Inbetween', '9'), ('A Life So Changed', '12'), 
    ('Everything', '12'), ('Everything', '10'), ('Magic Waters', '11'), ('Magic Waters', '12'), 
    ('Magic Waters', '10'), ('Anahera', '11'), ('Listen Feel Enjoy', '11'), ('Anahera', '10'), 
    ('On A Good Day', '11'), ('Listen Feel Enjoy', '10'), ('Anahera', '12'), ("Who's Afraid Of 138?!", '16'), 
    ('Listen Feel Enjoy', '12'), ('On A Good Day', '12'), ('On A Good Day', '10'), ('The Human Spirit', '11'), 
    ('Magic Waters', '13'), ('Stargazer', '13'), ('Blue Sunshine', '11'), ('M.O.N.I.', '13'), 
    ('Blue Sunshine', '12'), ('Blue Sunshine', '10'), ('The Human Spirit', '12'), ('Good For Me', '12'), 
    ('Good For Me', '11'), ('Good For Me', '10'), ('No Inbetween', '11'), ('No Inbetween', '12'), 
    ('Euphoria', '11'), ('No Inbetween', '10')
}


def is_banned(song_name: str, difficulty: str) -> bool:
    """
    Check if a song-difficulty combination is in the ban list.
    
    Args:
        song_name: Song name from JSON (TITLE field)
        difficulty: Difficulty level
        
    Returns:
        True if the combination is banned, False otherwise
    """
    return (song_name, difficulty) in BAN_LIST


def analyze_difficulty_distribution(json_directory: str) -> Dict[str, Any]:
    """
    Analyze difficulty distribution after excluding banned charts.
    
    Args:
        json_directory: Directory containing processed JSON files
        
    Returns:
        Dictionary with analysis results
    """
    json_path = Path(json_directory)
    
    if not json_path.exists():
        raise FileNotFoundError(f"Directory {json_directory} not found")
    
    # Find all JSON files
    json_files = list(json_path.rglob("*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {json_directory}")
    
    print(f"Analyzing {len(json_files)} JSON files...")
    
    # Statistics tracking
    total_charts = 0
    banned_charts = 0
    included_charts = 0
    difficulty_counts = Counter()
    song_difficulty_pairs = []
    banned_pairs = []
    
    # Process each file
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            song_name = data.get('TITLE', json_file.stem)
            
            # Process each NOTES section
            if 'NOTES' in data:
                for notes_section in data['NOTES']:
                    difficulty = notes_section.get('level', 'Unknown')
                    total_charts += 1
                    
                    # Check if this chart is banned
                    if is_banned(song_name, difficulty):
                        banned_charts += 1
                        banned_pairs.append((song_name, difficulty))
                    else:
                        included_charts += 1
                        difficulty_counts[difficulty] += 1
                        song_difficulty_pairs.append((song_name, difficulty))
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Calculate statistics
    difficulty_distribution = dict(difficulty_counts)
    total_included = sum(difficulty_distribution.values())
    
    # Calculate percentages
    difficulty_percentages = {}
    for difficulty, count in difficulty_distribution.items():
        if total_included > 0:
            difficulty_percentages[difficulty] = (count / total_included) * 100
        else:
            difficulty_percentages[difficulty] = 0
    
    return {
        'total_charts': total_charts,
        'banned_charts': banned_charts,
        'included_charts': included_charts,
        'ban_rate': (banned_charts / total_charts * 100) if total_charts > 0 else 0,
        'difficulty_distribution': difficulty_distribution,
        'difficulty_percentages': difficulty_percentages,
        'included_song_difficulty_pairs': song_difficulty_pairs,
        'banned_song_difficulty_pairs': banned_pairs,
        'ban_list_size': len(BAN_LIST)
    }


def print_distribution_summary(analysis_data: Dict[str, Any]) -> None:
    """
    Print a summary of the difficulty distribution analysis.
    
    Args:
        analysis_data: Output from analyze_difficulty_distribution
    """
    print("\n" + "="*60)
    print("DIFFICULTY DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total charts analyzed: {analysis_data['total_charts']}")
    print(f"   Charts excluded (banned): {analysis_data['banned_charts']}")
    print(f"   Charts included: {analysis_data['included_charts']}")
    print(f"   Ban rate: {analysis_data['ban_rate']:.2f}%")
    print(f"   Ban list size: {analysis_data['ban_list_size']}")
    
    print(f"\n📈 Difficulty Distribution (Included Charts):")
    print("-" * 50)
    
    # Sort difficulties numerically
    sorted_difficulties = sorted(
        analysis_data['difficulty_distribution'].keys(),
        key=lambda x: int(x) if x.isdigit() else float('inf')
    )
    
    for difficulty in sorted_difficulties:
        count = analysis_data['difficulty_distribution'][difficulty]
        percentage = analysis_data['difficulty_percentages'][difficulty]
        print(f"   Level {difficulty}: {count} charts ({percentage:.2f}%)")
    
    print(f"\n🎯 Key Insights:")
    print("-" * 30)
    
    if analysis_data['difficulty_distribution']:
        most_common = max(analysis_data['difficulty_distribution'].items(), key=lambda x: x[1])
        least_common = min(analysis_data['difficulty_distribution'].items(), key=lambda x: x[1])
        
        print(f"   Most common difficulty: Level {most_common[0]} ({most_common[1]} charts)")
        print(f"   Least common difficulty: Level {least_common[0]} ({least_common[1]} charts)")
        
        # Find difficulty range
        difficulties = [int(d) for d in analysis_data['difficulty_distribution'].keys() if d.isdigit()]
        if difficulties:
            print(f"   Difficulty range: {min(difficulties)} - {max(difficulties)}")
    
    print(f"\n🚫 Banned Charts (Sample):")
    print("-" * 30)
    banned_pairs = analysis_data['banned_song_difficulty_pairs']
    for i, (song, difficulty) in enumerate(banned_pairs[:10]):  # Show first 10
        print(f"   {song} (Level {difficulty})")
    if len(banned_pairs) > 10:
        print(f"   ... and {len(banned_pairs) - 10} more")


def save_analysis_results(analysis_data: Dict[str, Any], output_file: str) -> None:
    """
    Save analysis results to a JSON file.
    
    Args:
        analysis_data: Output from analyze_difficulty_distribution
        output_file: Path to save the results
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Analysis results saved to: {output_file}")


def get_included_charts_list(analysis_data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Get a list of all included song-difficulty pairs.
    
    Args:
        analysis_data: Output from analyze_difficulty_distribution
        
    Returns:
        List of (song_name, difficulty) tuples for included charts
    """
    return analysis_data['included_song_difficulty_pairs']


def main():
    """Main function to run the distribution analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze difficulty distribution after excluding banned charts')
    parser.add_argument('json_directory', help='Directory containing processed JSON files')
    parser.add_argument('--output', '-o', help='Output file for results (default: difficulty_analysis.json)')
    
    args = parser.parse_args()
    
    # Set default output file
    output_file = args.output or 'difficulty_analysis.json'
    
    try:
        # Run the analysis
        analysis_data = analyze_difficulty_distribution(args.json_directory)
        
        # Print summary
        print_distribution_summary(analysis_data)
        
        # Save results
        save_analysis_results(analysis_data, output_file)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 

    # python src/feature_processing/distro.py output/processed_16
