#!/usr/bin/env python3
"""
Compare Expected vs Actual Outputs from A2rchi Benchmarking

This script helps evaluate benchmarking results by showing:
- The question asked
- A2rchi's actual answer
- The expected (reference) answer
- Retrieved contexts
- RAGAS scores (if available)

Usage:
    python compare_benchmark_outputs.py <results.json>
    python compare_benchmark_outputs.py <results.json> --html output.html
    python compare_benchmark_outputs.py <results.json> --question 1
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime


def load_benchmark_results(filepath):
    """Load and parse benchmark results JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract the configuration data
    config_data = None
    config_name = None
    for key, value in data.items():
        if 'benchmarking' in key and isinstance(value, dict):
            if 'single question results' in value:
                config_data = value
                config_name = key
                break
    
    if not config_data:
        raise ValueError("No benchmarking results found in JSON file")
    
    # Try to find and load the queries file to get expected links
    queries_path = config_data.get('configuration used', {}).get('services', {}).get('benchmarking', {}).get('queries_path')
    expected_links = {}
    
    if queries_path:
        # Try to load queries file
        from pathlib import Path
        queries_file = Path(queries_path)
        if queries_file.exists():
            try:
                with open(queries_file, 'r') as f:
                    queries_data = json.load(f)
                    for q in queries_data:
                        question_text = q.get('question', '')
                        expected_links[question_text] = q.get('link', '')
            except:
                pass
    
    # Add expected links to question results
    for q_data in config_data['single question results'].values():
        question_text = q_data.get('question', '')
        if question_text in expected_links:
            q_data['expected_link'] = expected_links[question_text]
    
    return config_data, config_name, data.get('time', 'Unknown')


def extract_ticket_id(text):
    """Extract JIRA ticket ID from various formats"""
    import re
    # Match patterns like CMSTRANSF-1078, jira_CMSTRANSF-1078, jira_CMSTRANSF-1078.txt
    patterns = [
        r'(CMSTRANSF-\d+)',
        r'(CMSPROD-\d+)',
        r'jira_(CMSTRANSF-\d+)',
        r'jira_(CMSPROD-\d+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1) if match.lastindex == 1 else match.group(1).replace('jira_', '')
    return None


def calculate_retrieval_accuracy(questions):
    """Calculate how many questions retrieved the correct document"""
    total = 0
    correct = 0
    
    for q_data in questions.values():
        expected_link = q_data.get('expected_link', '')
        contexts = q_data.get('contexts', [])
        
        expected_ticket = extract_ticket_id(expected_link)
        if not expected_ticket:
            continue
            
        total += 1
        
        # Check if any retrieved context contains the expected ticket
        for ctx in contexts:
            retrieved_ticket = extract_ticket_id(str(ctx))
            if retrieved_ticket and retrieved_ticket == expected_ticket:
                correct += 1
                break
    
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, correct, total


def format_text_output(config_data, config_name, timestamp, question_num=None):
    """Format results as readable text"""
    questions = config_data['single question results']
    total_results = config_data.get('total_results', {})
    
    output = []
    output.append("=" * 100)
    output.append(f"BENCHMARK RESULTS COMPARISON")
    output.append("=" * 100)
    output.append(f"Configuration: {config_name}")
    output.append(f"Timestamp: {timestamp}")
    output.append(f"Total Questions: {len(questions)}")
    output.append("")
    
    # Calculate and show retrieval accuracy
    ret_accuracy, ret_correct, ret_total = calculate_retrieval_accuracy(questions)
    output.append("🎯 RETRIEVAL ACCURACY:")
    output.append(f"  • Correct Documents Retrieved: {ret_correct}/{ret_total} ({ret_accuracy:.1f}%)")
    output.append("")
    
    # Show aggregate metrics
    if total_results:
        output.append("📊 AGGREGATE RAGAS METRICS:")
        for metric, value in total_results.items():
            if 'aggregate' in metric:
                clean_name = metric.replace('aggregate_', '').replace('_', ' ').title()
                output.append(f"  • {clean_name}: {value:.3f}")
        output.append("")
    
    output.append("=" * 100)
    output.append("")
    
    # Filter to specific question if requested
    question_items = list(questions.items())
    if question_num is not None:
        if 1 <= question_num <= len(question_items):
            question_items = [question_items[question_num - 1]]
        else:
            output.append(f"⚠️  Question {question_num} not found. Showing all questions.")
            output.append("")
    
    # Show each question
    for i, (qid, q_data) in enumerate(question_items, 1):
        output.append("─" * 100)
        output.append(f"QUESTION {i}/{len(questions)}: {qid}")
        output.append("─" * 100)
        output.append("")
        
        # Question
        output.append("❓ QUESTION:")
        output.append(q_data['question'])
        output.append("")
        
        # Retrieval Check
        expected_link = q_data.get('expected_link', '')
        expected_ticket = extract_ticket_id(expected_link)
        contexts = q_data.get('contexts', [])
        
        retrieved_tickets = []
        for ctx in contexts:
            ticket_id = extract_ticket_id(str(ctx))
            if ticket_id and ticket_id not in retrieved_tickets:
                retrieved_tickets.append(ticket_id)
        
        retrieval_match = expected_ticket in retrieved_tickets if expected_ticket else None
        
        if expected_ticket:
            output.append("🎯 RETRIEVAL CHECK:")
            output.append(f"  Expected Document: {expected_ticket}")
            output.append(f"  Retrieved Documents: {', '.join(retrieved_tickets) if retrieved_tickets else 'None'}")
            if retrieval_match:
                output.append(f"  Status: ✅ CORRECT - Expected document was retrieved")
            else:
                output.append(f"  Status: ❌ INCORRECT - Expected document NOT retrieved")
            output.append("")
        
        # A2rchi's Answer
        output.append("🤖 A2RCHI'S ANSWER:")
        output.append(q_data['chat_answer'])
        output.append("")
        
        # Expected Answer
        output.append("✅ EXPECTED ANSWER:")
        output.append(q_data['ground_truth'])
        output.append("")
        
        # Retrieved Contexts
        contexts = q_data.get('contexts', [])
        if contexts:
            output.append(f"📚 RETRIEVED CONTEXTS ({len(contexts)} documents):")
            for j, ctx in enumerate(contexts, 1):
                # Try to parse if it's a string representation of a Document
                if ctx.startswith('page_content='):
                    # Extract just the content part
                    try:
                        content_start = ctx.find("page_content='") + len("page_content='")
                        content_end = ctx.find("' metadata=", content_start)
                        if content_end != -1:
                            ctx_text = ctx[content_start:content_end]
                        else:
                            ctx_text = ctx
                    except:
                        ctx_text = ctx
                else:
                    ctx_text = ctx
                
                # Truncate if too long
                if len(ctx_text) > 300:
                    ctx_text = ctx_text[:300] + "... [truncated]"
                
                output.append(f"  [{j}] {ctx_text}")
            output.append("")
        
        # Document Scores
        doc_scores = q_data.get('document_scores', [])
        if doc_scores:
            output.append(f"🎯 DOCUMENT SIMILARITY SCORES:")
            for j, score in enumerate(doc_scores, 1):
                output.append(f"  [{j}] {score:.4f}")
            output.append("")
        
        # RAGAS Metrics
        ragas_metrics = {
            'answer_relevancy': 'Answer Relevancy',
            'faithfulness': 'Faithfulness',
            'context_precision': 'Context Precision',
            'context_recall': 'Context Recall'
        }
        
        has_ragas = any(metric in q_data for metric in ragas_metrics.keys())
        if has_ragas:
            output.append("📊 RAGAS EVALUATION SCORES:")
            for metric_key, metric_name in ragas_metrics.items():
                if metric_key in q_data:
                    value = q_data[metric_key]
                    if value is not None:
                        output.append(f"  • {metric_name}: {value:.3f}")
                    else:
                        output.append(f"  • {metric_name}: N/A (evaluation failed)")
            output.append("")
        
        # Timing
        time_elapsed = q_data.get('time_elapsed')
        if time_elapsed:
            output.append(f"⏱️  Time Elapsed: {time_elapsed:.2f}s")
            output.append("")
        
        output.append("")
    
    return "\n".join(output)


def format_html_output(config_data, config_name, timestamp):
    """Format results as HTML for easier reading"""
    questions = config_data['single question results']
    total_results = config_data.get('total_results', {})
    
    html = ["""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Benchmark Results Comparison</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .metrics {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .question-card {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .section {
            margin: 20px 0;
        }
        .section-title {
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #667eea;
        }
        .answer-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
            margin: 10px 0;
            white-space: pre-wrap;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }
        .expected-box {
            border-left-color: #28a745;
        }
        .context-box {
            background: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            font-size: 0.85em;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .metric-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        .score-low { color: #dc3545; }
        .score-medium { color: #ffc107; }
        .score-high { color: #28a745; }
    </style>
</head>
<body>
"""]
    
    # Header
    html.append(f"""
    <div class="header">
        <h1>📊 Benchmark Results Comparison</h1>
        <p><strong>Configuration:</strong> {config_name}</p>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p><strong>Questions Processed:</strong> {len(questions)}</p>
    </div>
""")
    
    # Retrieval Accuracy
    ret_accuracy, ret_correct, ret_total = calculate_retrieval_accuracy(questions)
    html.append('<div class="metrics">')
    html.append('<h2>🎯 Retrieval Accuracy</h2>')
    score_class = 'score-low' if ret_accuracy < 50 else 'score-medium' if ret_accuracy < 80 else 'score-high'
    html.append(f"""
        <div class="metric-item" style="max-width: 400px; margin: 0 auto;">
            <div class="metric-value {score_class}">{ret_accuracy:.1f}%</div>
            <div class="metric-label">Correct Documents Retrieved: {ret_correct}/{ret_total}</div>
        </div>
    """)
    html.append('</div>')
    
    # Aggregate Metrics
    if total_results:
        html.append('<div class="metrics">')
        html.append('<h2>Aggregate RAGAS Metrics</h2>')
        html.append('<div class="metrics-grid">')
        for metric, value in total_results.items():
            if 'aggregate' in metric:
                clean_name = metric.replace('aggregate_', '').replace('_', ' ').title()
                score_class = 'score-low' if value < 0.5 else 'score-medium' if value < 0.7 else 'score-high'
                html.append(f"""
                <div class="metric-item">
                    <div class="metric-value {score_class}">{value:.3f}</div>
                    <div class="metric-label">{clean_name}</div>
                </div>
                """)
        html.append('</div></div>')
    
    # Each Question
    for i, (qid, q_data) in enumerate(questions.items(), 1):
        html.append(f'<div class="question-card">')
        html.append(f'<h2>Question {i}: {qid}</h2>')
        
        # Question
        html.append(f'<div class="section">')
        html.append(f'<div class="section-title">❓ Question</div>')
        html.append(f'<p>{q_data["question"]}</p>')
        html.append(f'</div>')
        
        # Retrieval Check
        expected_link = q_data.get('expected_link', '')
        expected_ticket = extract_ticket_id(expected_link)
        contexts = q_data.get('contexts', [])
        
        retrieved_tickets = []
        for ctx in contexts:
            ticket_id = extract_ticket_id(str(ctx))
            if ticket_id and ticket_id not in retrieved_tickets:
                retrieved_tickets.append(ticket_id)
        
        retrieval_match = expected_ticket in retrieved_tickets if expected_ticket else None
        
        if expected_ticket:
            status_class = 'score-high' if retrieval_match else 'score-low'
            status_icon = '✅' if retrieval_match else '❌'
            status_text = 'CORRECT' if retrieval_match else 'INCORRECT'
            
            html.append(f'<div class="section">')
            html.append(f'<div class="section-title">🎯 Retrieval Check</div>')
            html.append(f'<div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">')
            html.append(f'<p><strong>Expected Document:</strong> {expected_ticket}</p>')
            html.append(f'<p><strong>Retrieved Documents:</strong> {", ".join(retrieved_tickets) if retrieved_tickets else "None"}</p>')
            html.append(f'<p><strong class="{status_class}">{status_icon} Status: {status_text}</strong></p>')
            html.append(f'</div>')
            html.append(f'</div>')
        
        # A2rchi's Answer
        html.append(f'<div class="section">')
        html.append(f'<div class="section-title">🤖 A2rchi\'s Answer</div>')
        html.append(f'<div class="answer-box">{q_data["chat_answer"]}</div>')
        html.append(f'</div>')
        
        # Expected Answer
        html.append(f'<div class="section">')
        html.append(f'<div class="section-title">✅ Expected Answer</div>')
        html.append(f'<div class="answer-box expected-box">{q_data["ground_truth"]}</div>')
        html.append(f'</div>')
        
        # RAGAS Metrics
        ragas_metrics = {
            'answer_relevancy': 'Answer Relevancy',
            'faithfulness': 'Faithfulness',
            'context_precision': 'Context Precision',
            'context_recall': 'Context Recall'
        }
        
        html.append(f'<div class="section">')
        html.append(f'<div class="section-title">📊 RAGAS Scores</div>')
        html.append(f'<div class="metrics-grid">')
        for metric_key, metric_name in ragas_metrics.items():
            if metric_key in q_data and q_data[metric_key] is not None:
                value = q_data[metric_key]
                score_class = 'score-low' if value < 0.5 else 'score-medium' if value < 0.7 else 'score-high'
                html.append(f"""
                <div class="metric-item">
                    <div class="metric-value {score_class}">{value:.3f}</div>
                    <div class="metric-label">{metric_name}</div>
                </div>
                """)
        html.append(f'</div></div>')
        
        html.append(f'</div>')
    
    html.append('</body></html>')
    return '\n'.join(html)


def main():
    parser = argparse.ArgumentParser(
        description='Compare expected vs actual outputs from A2rchi benchmarking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View all results in terminal
  python compare_benchmark_outputs.py results.json
  
  # View specific question
  python compare_benchmark_outputs.py results.json --question 3
  
  # Generate HTML report
  python compare_benchmark_outputs.py results.json --html report.html
  
  # Save text output to file
  python compare_benchmark_outputs.py results.json > report.txt
        """
    )
    
    parser.add_argument('results_file', help='Path to benchmark results JSON file')
    parser.add_argument('--html', help='Generate HTML output file')
    parser.add_argument('--question', '-q', type=int, help='Show only specific question number')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.results_file).exists():
        print(f"Error: File '{args.results_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Load results
    try:
        config_data, config_name, timestamp = load_benchmark_results(args.results_file)
    except Exception as e:
        print(f"Error loading results: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Generate output
    if args.html:
        # HTML output
        html_content = format_html_output(config_data, config_name, timestamp)
        with open(args.html, 'w') as f:
            f.write(html_content)
        print(f"✅ HTML report generated: {args.html}")
    else:
        # Text output
        text_content = format_text_output(config_data, config_name, timestamp, args.question)
        print(text_content)


if __name__ == '__main__':
    main()

