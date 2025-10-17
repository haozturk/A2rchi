#!/usr/bin/env python3
"""
Quick benchmark results analyzer.

Usage:
    python analyze_benchmark.py bench_out/results.json
    python analyze_benchmark.py bench_out/results.json --plot
    python analyze_benchmark.py bench_out/*.json --compare
"""

import json
import sys
import argparse
from pathlib import Path
import pandas as pd


def load_results(filepath):
    """Load benchmark results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_config_results(data):
    """Extract configuration results from the full output"""
    configs = {}
    for key, value in data.items():
        if 'benchmarking' in key or '-' in key:
            if isinstance(value, dict) and 'single question results' in value:
                configs[key] = value
    return configs


def print_summary(filepath, data):
    """Print summary statistics for benchmark results"""
    print(f"\n{'='*80}")
    print(f"Results from: {filepath}")
    print(f"{'='*80}")
    
    # Print timestamp if available
    if 'time' in data:
        print(f"Timestamp: {data['time']}")
    
    configs = extract_config_results(data)
    
    for config_name, config_data in configs.items():
        print(f"\n{'─'*80}")
        print(f"Configuration: {config_name}")
        print(f"{'─'*80}")
        
        # Get total results
        totals = config_data.get('total_results', {})
        
        # Print RAGAS metrics if available
        ragas_metrics = ['aggregate_answer_relevancy', 'aggregate_faithfulness', 
                        'aggregate_context_precision', 'aggregate_context_recall']
        
        has_ragas = any(m in totals for m in ragas_metrics)
        if has_ragas:
            print("\n📊 RAGAS Metrics (Averages):")
            if 'aggregate_answer_relevancy' in totals:
                print(f"  Answer Relevancy:   {totals['aggregate_answer_relevancy']:.3f}")
            if 'aggregate_faithfulness' in totals:
                print(f"  Faithfulness:       {totals['aggregate_faithfulness']:.3f}")
            if 'aggregate_context_precision' in totals:
                print(f"  Context Precision:  {totals['aggregate_context_precision']:.3f}")
            if 'aggregate_context_recall' in totals:
                print(f"  Context Recall:     {totals['aggregate_context_recall']:.3f}")
        
        # Print LINKS metrics if available
        if 'link_accuracy' in totals:
            print(f"\n🔗 Links Mode:")
            print(f"  Link Accuracy:      {totals['link_accuracy']:.1%}")
        
        # Question-level stats
        questions = config_data.get('single question results', {})
        num_questions = len(questions)
        print(f"\n📝 Questions Processed: {num_questions}")
        
        # Timing stats if available
        times = []
        for q_data in questions.values():
            if 'time_elapsed' in q_data:
                times.append(q_data['time_elapsed'])
        
        if times:
            print(f"⏱️  Average Time per Question: {sum(times)/len(times):.2f}s")
            print(f"   Total Time: {sum(times):.1f}s ({sum(times)/60:.1f} minutes)")


def create_dataframe(config_data):
    """Create pandas DataFrame from question results"""
    questions = config_data.get('single question results', {})
    return pd.DataFrame.from_dict(questions, orient='index')


def plot_results(filepath, data):
    """Plot benchmark results using matplotlib"""
    try:
        from plots.benchmark_handler_functions import display_functions
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: Could not import plotting functions")
        print("Make sure plots/benchmark_handler_functions.py is available")
        return
    
    configs = extract_config_results(data)
    
    for config_name, config_data in configs.items():
        df = create_dataframe(config_data)
        
        print(f"\nPlotting results for: {config_name}")
        
        # Check which metrics are available
        ragas_metrics = ['answer_relevancy', 'faithfulness', 
                        'context_precision', 'context_recall']
        available_metrics = [m for m in ragas_metrics if m in df.columns]
        
        if available_metrics:
            display_functions.plot_all_metrics_histograms(df, available_metrics)
        else:
            print("No RAGAS metrics found to plot")


def compare_results(filepaths):
    """Compare results from multiple benchmark runs"""
    all_data = {}
    
    for filepath in filepaths:
        data = load_results(filepath)
        filename = Path(filepath).stem
        all_data[filename] = data
    
    print(f"\n{'='*80}")
    print(f"Comparing {len(filepaths)} benchmark runs")
    print(f"{'='*80}")
    
    # Create comparison table
    comparison = []
    
    for filename, data in all_data.items():
        configs = extract_config_results(data)
        for config_name, config_data in configs.items():
            totals = config_data.get('total_results', {})
            row = {
                'File': filename,
                'Config': config_name,
                'Questions': len(config_data.get('single question results', {}))
            }
            
            # Add RAGAS metrics
            if 'aggregate_answer_relevancy' in totals:
                row['Ans_Rel'] = f"{totals['aggregate_answer_relevancy']:.3f}"
            if 'aggregate_faithfulness' in totals:
                row['Faith'] = f"{totals['aggregate_faithfulness']:.3f}"
            if 'aggregate_context_precision' in totals:
                row['Ctx_Prec'] = f"{totals['aggregate_context_precision']:.3f}"
            if 'aggregate_context_recall' in totals:
                row['Ctx_Rec'] = f"{totals['aggregate_context_recall']:.3f}"
            if 'link_accuracy' in totals:
                row['Link_Acc'] = f"{totals['link_accuracy']:.1%}"
            
            comparison.append(row)
    
    df = pd.DataFrame(comparison)
    print("\n")
    print(df.to_string(index=False))
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze A2rchi benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View summary
  python analyze_benchmark.py bench_out/results.json
  
  # Plot histograms
  python analyze_benchmark.py bench_out/results.json --plot
  
  # Compare multiple runs
  python analyze_benchmark.py bench_out/*.json --compare
        """
    )
    
    parser.add_argument('files', nargs='+', help='Benchmark result JSON file(s)')
    parser.add_argument('--plot', action='store_true', 
                       help='Generate plots of results')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple result files')
    
    args = parser.parse_args()
    
    # Validate files exist
    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"Error: File not found: {filepath}")
            sys.exit(1)
    
    if args.compare and len(args.files) > 1:
        compare_results(args.files)
    elif args.plot:
        for filepath in args.files:
            data = load_results(filepath)
            plot_results(filepath, data)
    else:
        for filepath in args.files:
            data = load_results(filepath)
            print_summary(filepath, data)


if __name__ == '__main__':
    main()







