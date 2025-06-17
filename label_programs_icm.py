#!/usr/bin/env python3
"""
ICM Script to Label Programs from ericflo/logic-qa-llm-judge-v01 Dataset

This script uses Internal Coherence Maximization to automatically categorize
programs based on their content and structure.
"""

import json
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from datetime import datetime
from icm_implementation import ICM, ICMConfig, DataPoint

def create_program_dataset(programs, questions, sample_size=None):
    """
    Create ICM dataset from programs and questions.
    
    Each DataPoint will contain:
    - input_text: The program code (truncated for context)
    - metadata: Full program, question, and other info
    """
    dataset = []
    
    # Sample if requested
    if sample_size and sample_size < len(programs):
        import random
        indices = random.sample(range(len(programs)), sample_size)
    else:
        indices = range(len(programs))
    
    for i, idx in enumerate(indices):
        program = programs[idx]
        question = questions[idx]
        
        # Create a condensed version for ICM input
        # Include class name, docstring, and question
        program_lines = program.split('\n')
        
        # Extract key information
        class_name = ""
        docstring = ""
        imports = []
        
        for j, line in enumerate(program_lines):
            if line.strip().startswith('class '):
                class_name = line.strip()
            elif '"""' in line and not docstring:
                # Start of docstring
                start_idx = j
                for k in range(j, min(j+20, len(program_lines))):
                    docstring += program_lines[k] + '\n'
                    if k > j and '"""' in program_lines[k]:
                        break
            elif line.strip().startswith('import ') or line.strip().startswith('from '):
                imports.append(line.strip())
        
        # Create condensed input text for ICM
        input_text = f"""Program Analysis:
Class: {class_name}
Imports: {', '.join(imports[:5])}
Docstring: {docstring[:300]}...
Question: {question[:200]}..."""
        
        # Create DataPoint
        data_point = DataPoint(
            id=i,
            input_text=input_text,
            metadata={
                'full_program': program,
                'question': question,
                'class_name': class_name,
                'program_length': len(program),
                'original_index': idx
            }
        )
        
        dataset.append(data_point)
    
    return dataset

def run_icm_labeling(label_scheme='domain', model_name=None, backend=None, sample_size=None, 
                     dataset_name=None, split=None, program_column=None, question_column=None):
    """
    Run ICM to label programs.
    
    Args:
        label_scheme: Which labeling scheme to use
            - 'domain': 6 problem domain categories
            - 'complexity': 2 complexity levels
            - 'challenge_type': 4 challenge types
        model_name: Model to use (default: Qwen/Qwen3-1.7B-Base)
        backend: Backend to use (default: transformers)
        sample_size: Number of samples to process
        dataset_name: Hugging Face dataset name (default: ericflo/logic-qa-llm-judge-v01)
        split: Dataset split to use (default: train)
        program_column: Name of the program/code column (default: program)
        question_column: Name of the question/prompt column (default: question)
    """
    
    # Set defaults
    if dataset_name is None:
        dataset_name = "ericflo/logic-qa-llm-judge-v01"
    if split is None:
        split = "train"
    if program_column is None:
        program_column = "program"
    if question_column is None:
        question_column = "question"
    
    print(f"Loading dataset: {dataset_name} (split: {split})...")
    dataset = load_dataset(dataset_name)
    
    # Check if split exists
    if split not in dataset:
        available_splits = list(dataset.keys())
        print(f"Error: Split '{split}' not found. Available splits: {available_splits}")
        if available_splits:
            split = available_splits[0]
            print(f"Using split: {split}")
        else:
            raise ValueError(f"No splits found in dataset {dataset_name}")
    
    data = dataset[split]
    
    # Check if columns exist
    available_columns = data.column_names
    if program_column not in available_columns:
        print(f"Error: Column '{program_column}' not found. Available columns: {available_columns}")
        raise ValueError(f"Column '{program_column}' not found in dataset")
    if question_column not in available_columns:
        print(f"Error: Column '{question_column}' not found. Available columns: {available_columns}")
        raise ValueError(f"Column '{question_column}' not found in dataset")
    
    programs = data[program_column]
    questions = data[question_column]
    
    print(f"Total items in dataset: {len(programs)}")
    print(f"Using columns: {program_column} (code), {question_column} (prompts)")
    
    # Configure labels based on scheme
    if label_scheme == 'domain':
        label_names = [
            "Physics/Astronomy",
            "Cryptography/Security", 
            "Mathematics/Computation",
            "Logic/Puzzle",
            "Scheduling/Optimization",
            "Knowledge/Trivia"
        ]
        num_labels = 6
        initial_examples = 12  # 2 per category
        alpha = 65.0
    elif label_scheme == 'complexity':
        label_names = ["Simple", "Complex"]
        num_labels = 2
        initial_examples = 6
        alpha = 70.0
    elif label_scheme == 'challenge_type':
        label_names = [
            "Quantum/Physics",
            "Cryptography",
            "Navigation/Optimization",
            "General Puzzles"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 60.0
    elif label_scheme == 'quality':
        label_names = [
            "Poor Quality",
            "Below Average",
            "Decent",
            "Well Crafted",
            "Exceptional"
        ]
        num_labels = 5
        initial_examples = 10
        alpha = 70.0
    elif label_scheme == 'difficulty':
        label_names = [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Expert"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 70.0
    elif label_scheme == 'primary_skill':
        label_names = [
            "Mathematical",
            "Algorithmic",
            "Data Structures",
            "Domain Knowledge",
            "Pattern Recognition",
            "Logical Reasoning"
        ]
        num_labels = 6
        initial_examples = 12
        alpha = 65.0
    elif label_scheme == 'time_estimate':
        label_names = [
            "5-10 minutes",
            "10-30 minutes",
            "30-60 minutes",
            "1-2 hours",
            "2+ hours"
        ]
        num_labels = 5
        initial_examples = 10
        alpha = 65.0
    elif label_scheme == 'solution_approach':
        label_names = [
            "Direct/Formulaic",
            "Algorithmic",
            "Mathematical Insight",
            "Creative/Lateral"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 60.0
    elif label_scheme == 'prerequisites':
        label_names = [
            "None",
            "Basic Programming",
            "CS Fundamentals",
            "Advanced CS",
            "Specialized Knowledge"
        ]
        num_labels = 5
        initial_examples = 10
        alpha = 65.0
    elif label_scheme == 'concept_count':
        label_names = [
            "Single Concept",
            "2-3 Concepts",
            "Multiple Concepts",
            "Many Concepts"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 65.0
    elif label_scheme == 'educational_value':
        label_names = [
            "Low",
            "Medium",
            "High",
            "Exceptional"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 65.0
    elif label_scheme == 'implementation_length':
        label_names = [
            "One-liner",
            "Short (<20 lines)",
            "Medium (20-50 lines)",
            "Long (50-100 lines)",
            "Very Long (100+ lines)"
        ]
        num_labels = 5
        initial_examples = 10
        alpha = 60.0
    elif label_scheme == 'real_world_relevance':
        label_names = [
            "Academic/Theoretical",
            "Industry Relevant",
            "Direct Application"
        ]
        num_labels = 3
        initial_examples = 6
        alpha = 60.0
    elif label_scheme == 'algorithm_type':
        label_names = [
            "Brute Force",
            "Greedy",
            "Dynamic Programming",
            "Divide and Conquer",
            "Graph/Tree",
            "Mathematical",
            "Other"
        ]
        num_labels = 7
        initial_examples = 14
        alpha = 65.0
    elif label_scheme == 'interview_suitability':
        label_names = [
            "Too Easy",
            "Junior Level",
            "Mid Level", 
            "Senior Level",
            "Too Complex"
        ]
        num_labels = 5
        initial_examples = 10
        alpha = 65.0
    elif label_scheme == 'clarity':
        label_names = [
            "Ambiguous",
            "Clear with Effort",
            "Crystal Clear"
        ]
        num_labels = 3
        initial_examples = 6
        alpha = 60.0
    elif label_scheme == 'testability':
        label_names = [
            "Hard to Test",
            "Testable",
            "Easily Testable"
        ]
        num_labels = 3
        initial_examples = 6
        alpha = 55.0
    elif label_scheme == 'math_level':
        label_names = [
            "No Math",
            "Elementary",
            "High School",
            "Undergraduate",
            "Graduate",
            "Research Level"
        ]
        num_labels = 6
        initial_examples = 12
        alpha = 70.0
    elif label_scheme == 'optimization_potential':
        label_names = [
            "No Optimization Needed",
            "Minor Optimizations",
            "Significant Optimizations",
            "Critical Optimizations"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 60.0
    elif label_scheme == 'insight_required':
        label_names = [
            "Straightforward",
            "Needs Observation",
            "Requires Insight",
            "Deep Insight Required"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 65.0
    elif label_scheme == 'abstraction_level':
        label_names = [
            "Concrete",
            "Moderate Abstraction",
            "High Abstraction"
        ]
        num_labels = 3
        initial_examples = 6
        alpha = 60.0
    elif label_scheme == 'parallelizable':
        label_names = [
            "Inherently Sequential",
            "Partially Parallelizable",
            "Highly Parallelizable"
        ]
        num_labels = 3
        initial_examples = 6
        alpha = 55.0
    elif label_scheme == 'space_complexity':
        label_names = [
            "O(1) Constant",
            "O(log n) Logarithmic",
            "O(n) Linear",
            "O(n²) or Higher"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 70.0
    elif label_scheme == 'edge_case_complexity':
        label_names = [
            "Few Edge Cases",
            "Moderate Edge Cases",
            "Many Edge Cases"
        ]
        num_labels = 3
        initial_examples = 6
        alpha = 60.0
    elif label_scheme == 'debugging_difficulty':
        label_names = [
            "Easy to Debug",
            "Moderate Debugging",
            "Hard to Debug",
            "Nightmare to Debug"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 55.0
    elif label_scheme == 'competition_level':
        label_names = [
            "Practice/Easy",
            "Division 2",
            "Division 1", 
            "Finals Level"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 65.0
    elif label_scheme == 'training_quality':
        label_names = [
            "Reject - Poor Quality",
            "Reject - Ambiguous",
            "Borderline",
            "Good Training Example",
            "Excellent Training Example"
        ]
        num_labels = 5
        initial_examples = 10
        alpha = 70.0
    elif label_scheme == 'problem_similarity':
        label_names = [
            "Unique/Novel",
            "Has Similar Problems",
            "Common Pattern",
            "Very Common/Duplicate"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 65.0
    elif label_scheme == 'curriculum_stage':
        label_names = [
            "Stage 1 - Fundamentals",
            "Stage 2 - Basic Application",
            "Stage 3 - Intermediate Concepts",
            "Stage 4 - Advanced Techniques",
            "Stage 5 - Complex Integration",
            "Stage 6 - Expert Synthesis",
            "Stage 7 - Research Level"
        ]
        num_labels = 7
        initial_examples = 14
        alpha = 70.0
    elif label_scheme == 'data_diversity':
        label_names = [
            "Overrepresented Type",
            "Common Type",
            "Uncommon Type",
            "Rare Type",
            "Unique Contribution"
        ]
        num_labels = 5
        initial_examples = 10
        alpha = 60.0
    elif label_scheme == 'annotation_confidence':
        label_names = [
            "Low Confidence",
            "Medium Confidence",
            "High Confidence",
            "Ground Truth Quality"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 65.0
    elif label_scheme == 'concept_coverage':
        label_names = [
            "Single Core Concept",
            "Multiple Core Concepts",
            "Cross-Domain Concepts",
            "Novel Concept Combination"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 60.0
    elif label_scheme == 'model_difficulty':
        label_names = [
            "Models Solve Easily",
            "Models Sometimes Struggle",
            "Models Often Fail",
            "Models Consistently Fail"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 65.0
    elif label_scheme == 'generalization_value':
        label_names = [
            "Narrow/Specific",
            "Some Transfer",
            "Good Transfer",
            "Excellent Transfer"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 60.0
    elif label_scheme == 'ambiguity_level':
        label_names = [
            "Unambiguous",
            "Minor Ambiguity",
            "Significant Ambiguity",
            "Highly Ambiguous"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 65.0
    elif label_scheme == 'benchmark_suitability':
        label_names = [
            "Training Only",
            "Good for Validation",
            "Good for Testing",
            "Benchmark Quality"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 65.0
    elif label_scheme == 'reasoning_depth':
        label_names = [
            "Surface Level",
            "Single Step",
            "Multi-Step",
            "Deep Chain",
            "Complex Web"
        ]
        num_labels = 5
        initial_examples = 10
        alpha = 65.0
    elif label_scheme == 'error_analysis':
        label_names = [
            "Common Success",
            "Occasional Errors",
            "Error Prone",
            "Systematic Failure Point"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 60.0
    elif label_scheme == 'few_shot_suitability':
        label_names = [
            "Zero-shot Clear",
            "Benefits from 1-2 Examples",
            "Needs Several Examples",
            "Requires Many Examples"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 60.0
    elif label_scheme == 'contamination_risk':
        label_names = [
            "Low Risk",
            "Medium Risk",
            "High Risk",
            "Known Contaminated"
        ]
        num_labels = 4
        initial_examples = 8
        alpha = 70.0
    else:
        raise ValueError(f"Unknown label scheme: {label_scheme}")
    
    # Set defaults if not provided
    if model_name is None:
        model_name = "Qwen/Qwen3-1.7B-Base"
    if backend is None:
        backend = "transformers"
    if sample_size is None:
        sample_size = 100  # Default sample size
    
    print(f"\nUsing label scheme: {label_scheme}")
    print(f"Labels: {label_names}")
    print(f"Model: {model_name}")
    print(f"Backend: {backend}")
    print(f"Sample size: {sample_size if sample_size != -1 else 'Full dataset'}")
    
    # Create ICM dataset
    print(f"\nCreating ICM dataset...")
    # Handle -1 as full dataset
    effective_sample_size = None if sample_size == -1 else sample_size
    icm_dataset = create_program_dataset(programs, questions, effective_sample_size)
    
    # Configure ICM
    config = ICMConfig(
        model_name=model_name,
        backend=backend,
        initial_examples=initial_examples,
        alpha=alpha,
        num_labels=num_labels,
        label_names=label_names,
        max_context_length=8192,  # Reduced for efficiency
        max_new_tokens=32,
        temperature=0.1,
        initial_temperature=8.0,
        final_temperature=0.01,
        cooling_rate=0.98
    )
    
    # Run ICM
    print(f"\nRunning ICM labeling...")
    print(f"This may take a while depending on dataset size and GPU availability...")
    
    icm = ICM(config)
    results = icm.run(icm_dataset, max_iterations=len(icm_dataset) * 2)
    
    # Calculate metrics
    final_score, predictability, inconsistencies = icm.calculate_score(results)
    
    print(f"\n=== ICM RESULTS ===")
    print(f"Final Score: {final_score:.2f}")
    print(f"Mutual Predictability: {predictability:.2f}")
    print(f"Logical Inconsistencies: {inconsistencies}")
    print(f"Acceptance Rate: {sum(icm.acceptance_history) / len(icm.acceptance_history):.2%}")
    
    # Analyze results
    label_counts = {}
    labeled_programs = []
    
    for data_point, label_idx in results:
        label_name = label_names[label_idx]
        label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
        labeled_programs.append({
            'original_index': data_point.metadata['original_index'],
            'class_name': data_point.metadata['class_name'],
            'question_preview': data_point.metadata['question'][:100] + '...',
            'program_length': data_point.metadata['program_length'],
            'assigned_label': label_name,
            'label_index': label_idx
        })
    
    # Print label distribution
    print(f"\n=== LABEL DISTRIBUTION ===")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")
    
    # Show examples from each category
    print(f"\n=== SAMPLE CLASSIFICATIONS ===")
    for label in label_names:
        print(f"\n{label}:")
        examples = [p for p in labeled_programs if p['assigned_label'] == label][:3]
        for ex in examples:
            print(f"  - {ex['class_name']}")
            print(f"    Question: {ex['question_preview']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("icm_results")
    output_dir.mkdir(exist_ok=True)
    
    # Create dataset identifier for filename
    dataset_id = dataset_name.replace('/', '_').replace('-', '_')
    
    # Save detailed results
    results_file = output_dir / f"{dataset_id}_{label_scheme}_{timestamp}.json"
    results_data = {
        'dataset_info': {
            'name': dataset_name,
            'split': split,
            'program_column': program_column,
            'question_column': question_column,
            'sample_size': sample_size if sample_size != -1 else 'full'
        },
        'config': config.__dict__,
        'metrics': {
            'final_score': float(final_score),
            'predictability': float(predictability),
            'inconsistencies': int(inconsistencies),
            'total_items': len(results),
            'acceptance_rate': float(sum(icm.acceptance_history) / len(icm.acceptance_history))
        },
        'label_distribution': label_counts,
        'labeled_programs': labeled_programs,
        'score_history': [float(s) for s in icm.score_history]
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save as CSV for easy viewing
    csv_file = output_dir / f"{dataset_id}_{label_scheme}_{timestamp}.csv"
    df = pd.DataFrame(labeled_programs)
    df.to_csv(csv_file, index=False)
    
    print(f"\n=== RESULTS SAVED ===")
    print(f"JSON: {results_file}")
    print(f"CSV: {csv_file}")
    
    return results, label_counts

def main():
    """Main function to run ICM labeling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Label programs using ICM")
    # All available schemes
    schemes = [
        # Core classification
        'domain', 'complexity', 'challenge_type', 'quality', 'difficulty',
        'primary_skill', 'time_estimate', 'solution_approach', 'prerequisites',
        'concept_count', 'educational_value', 'implementation_length', 
        'real_world_relevance', 'algorithm_type', 'interview_suitability',
        'clarity', 'testability', 'math_level', 'optimization_potential',
        'insight_required', 'abstraction_level', 'parallelizable',
        'space_complexity', 'edge_case_complexity', 'debugging_difficulty',
        'competition_level',
        # ML training specific
        'training_quality', 'problem_similarity', 'curriculum_stage',
        'data_diversity', 'annotation_confidence', 'concept_coverage',
        'model_difficulty', 'generalization_value', 'ambiguity_level',
        'benchmark_suitability', 'reasoning_depth', 'error_analysis',
        'few_shot_suitability', 'contamination_risk'
    ]
    
    parser.add_argument(
        '--scheme', 
        choices=schemes,
        default='domain',
        help='Which labeling scheme to use (default: domain)'
    )
    parser.add_argument(
        '--list-schemes',
        action='store_true',
        help='List all available labeling schemes and exit'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Number of programs to label (default: 100, use -1 for all)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-1.7B-Base',
        help='Model to use for classification (default: Qwen/Qwen3-1.7B-Base)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['transformers', 'vllm', 'auto'],
        default='transformers',
        help='Backend to use (default: transformers)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='ericflo/logic-qa-llm-judge-v01',
        help='Hugging Face dataset name (default: ericflo/logic-qa-llm-judge-v01)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split to use (default: train)'
    )
    parser.add_argument(
        '--program-column',
        type=str,
        default='program',
        help='Name of the column containing programs/code (default: program)'
    )
    parser.add_argument(
        '--question-column',
        type=str,
        default='question',
        help='Name of the column containing questions/prompts (default: question)'
    )
    
    args = parser.parse_args()
    
    # Handle --list-schemes
    if args.list_schemes:
        print("\n=== AVAILABLE LABELING SCHEMES ===\n")
        
        scheme_descriptions = {
            # Core classification schemes
            'domain': 'Problem domain (6 labels: Physics, Crypto, Math, Logic, Scheduling, Knowledge)',
            'complexity': 'Binary complexity (2 labels: Simple vs Complex)',
            'challenge_type': 'Challenge category (4 labels: Quantum, Crypto, Navigation, Puzzles)',
            'quality': 'Quality assessment (5 labels: Poor to Exceptional)',
            'difficulty': 'Difficulty level (4 labels: Beginner to Expert)',
            'primary_skill': 'Main skill tested (6 labels: Math, Algorithmic, Data Structures, etc.)',
            'time_estimate': 'Expected solving time (5 labels: 5-10min to 2+ hours)',
            'solution_approach': 'Solution type (4 labels: Direct, Algorithmic, Mathematical, Creative)',
            'prerequisites': 'Required knowledge (5 labels: None to Specialized)',
            'concept_count': 'Number of concepts (4 labels: Single to Many)',
            'educational_value': 'Learning potential (4 labels: Low to Exceptional)',
            'implementation_length': 'Code length (5 labels: One-liner to 100+ lines)',
            'real_world_relevance': 'Practical application (3 labels: Academic to Direct Use)',
            'algorithm_type': 'Algorithm classification (7 labels: Brute Force, DP, Greedy, etc.)',
            'interview_suitability': 'Interview level (5 labels: Too Easy to Too Complex)',
            'clarity': 'Problem clarity (3 labels: Ambiguous to Crystal Clear)',
            'testability': 'Testing difficulty (3 labels: Hard to Easily Testable)',
            'math_level': 'Mathematics required (6 labels: None to Research Level)',
            'optimization_potential': 'Optimization needs (4 labels: None to Critical)',
            'insight_required': 'Insight level (4 labels: Straightforward to Deep Insight)',
            'abstraction_level': 'Abstract thinking (3 labels: Concrete to High Abstraction)',
            'parallelizable': 'Parallelization potential (3 labels: Sequential to Highly Parallel)',
            'space_complexity': 'Memory usage (4 labels: O(1) to O(n²)+)',
            'edge_case_complexity': 'Edge case handling (3 labels: Few to Many)',
            'debugging_difficulty': 'Debug complexity (4 labels: Easy to Nightmare)',
            'competition_level': 'Competition difficulty (4 labels: Practice to Finals)',
            # ML training specific schemes
            'training_quality': 'ML training suitability (5 labels: Reject to Excellent)',
            'problem_similarity': 'Uniqueness for training (4 labels: Unique to Duplicate)',
            'curriculum_stage': 'Learning progression (7 labels: Stage 1-7)',
            'data_diversity': 'Dataset diversity contribution (5 labels: Overrepresented to Unique)',
            'annotation_confidence': 'Label reliability (4 labels: Low to Ground Truth)',
            'concept_coverage': 'Concept complexity (4 labels: Single to Novel Combination)',
            'model_difficulty': 'Model performance expectation (4 labels: Easy to Consistently Fail)',
            'generalization_value': 'Transfer learning potential (4 labels: Narrow to Excellent)',
            'ambiguity_level': 'Problem ambiguity (4 labels: Unambiguous to Highly Ambiguous)',
            'benchmark_suitability': 'Dataset split recommendation (4 labels: Training to Benchmark)',
            'reasoning_depth': 'Reasoning complexity (5 labels: Surface to Complex Web)',
            'error_analysis': 'Model error likelihood (4 labels: Common Success to Systematic Failure)',
            'few_shot_suitability': 'Few-shot learning needs (4 labels: Zero-shot to Many Examples)',
            'contamination_risk': 'Training data contamination (4 labels: Low to Known Contaminated)'
        }
        
        for scheme in schemes:
            desc = scheme_descriptions.get(scheme, 'No description available')
            print(f"  {scheme:15} - {desc}")
        
        print("\n=== CORE CLASSIFICATION ===")
        print("General categorization schemes (26 total)")
        
        print("\n=== ML TRAINING SPECIFIC ===")
        print("Machine learning dataset curation schemes (14 total):")
        print("  - Rejection Sampling: training_quality, ambiguity_level")
        print("  - Clustering/Similarity: problem_similarity, data_diversity") 
        print("  - Curriculum Learning: curriculum_stage, reasoning_depth")
        print("  - Model Evaluation: model_difficulty, error_analysis")
        print("  - Data Quality: annotation_confidence, contamination_risk")
        
        print("\nExample usage:")
        print("  # Rejection sampling for high-quality training data")
        print("  uv run label_programs_icm.py --scheme training_quality")
        print("  ")
        print("  # Curriculum learning progression")
        print("  uv run label_programs_icm.py --scheme curriculum_stage --sample-size 200")
        print("  ")
        print("  # Find similar/duplicate problems for deduplication")
        print("  uv run label_programs_icm.py --scheme problem_similarity")
        return
    
    results, label_counts = run_icm_labeling(
        label_scheme=args.scheme,
        model_name=args.model,
        backend=args.backend,
        sample_size=args.sample_size,
        dataset_name=args.dataset,
        split=args.split,
        program_column=args.program_column,
        question_column=args.question_column
    )
    
    print("\nLabeling complete! Check the icm_results directory for detailed outputs.")

if __name__ == "__main__":
    main()
