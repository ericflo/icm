#!/usr/bin/env python3
"""
Run ICM with real Qwen3-4B model on ACTUAL datasets from Hugging Face
Uses real TruthfulQA, GSM8K, and other datasets
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from icm_implementation import (
    ICM, ICMConfig, DataPoint,
    create_truthfulness_dataset,
    create_math_correctness_dataset,
    create_comparison_dataset
)

def load_real_truthfulqa(sample_size=50, seed=42):
    """Load real TruthfulQA dataset and format for ICM"""
    print("Loading TruthfulQA dataset from Hugging Face...")
    
    # TruthfulQA has multiple choice questions with correct/incorrect answers
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    
    random.seed(seed)
    
    # Convert to ICM format - we need question-claim pairs
    icm_examples = []
    
    for item in dataset:
        question = item['question']
        
        # TruthfulQA structure: mc1_targets and mc2_targets contain choices and labels
        # mc2_targets has binary labels for all choices (can have multiple correct)
        # mc1_targets has exactly one correct answer
        
        # Use mc2_targets for more nuanced true/false labels
        if 'mc2_targets' in item:
            choices = item['mc2_targets']['choices']
            labels = item['mc2_targets']['labels']
            
            # Get correct answers (label = 1)
            correct_answers = [choices[i] for i, label in enumerate(labels) if label == 1]
            # Get incorrect answers (label = 0)
            incorrect_answers = [choices[i] for i, label in enumerate(labels) if label == 0]
        else:
            # Fallback to mc1_targets
            choices = item['mc1_targets']['choices']
            labels = item['mc1_targets']['labels']
            correct_idx = labels.index(1) if 1 in labels else 0
            correct_answers = [choices[correct_idx]]
            incorrect_answers = [choices[i] for i in range(len(choices)) if i != correct_idx]
        
        # Add some correct claims (limit to 2 per question to balance dataset)
        for answer in correct_answers[:2]:
            icm_examples.append((question, answer, None))
        
        # Add some incorrect claims (limit to 2 per question)
        for answer in incorrect_answers[:2]:
            icm_examples.append((question, answer, None))
    
    # Shuffle and sample
    random.shuffle(icm_examples)
    icm_examples = icm_examples[:sample_size]
    
    print(f"Loaded {len(icm_examples)} examples from TruthfulQA")
    return create_truthfulness_dataset(icm_examples)

def load_real_gsm8k(sample_size=50, seed=42):
    """Load real GSM8K dataset and format for ICM"""
    print("Loading GSM8K dataset from Hugging Face...")
    
    dataset = load_dataset("gsm8k", "main", split="train")
    
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(sample_size // 2, len(dataset)))
    
    icm_examples = []
    
    for idx in indices:
        item = dataset[idx]
        question = item['question']
        
        # Extract the correct answer from the solution
        solution = item['answer']
        # GSM8K answers end with #### followed by the numeric answer
        answer_parts = solution.split('####')
        if len(answer_parts) == 2:
            correct_answer = answer_parts[1].strip()
            correct_solution = answer_parts[0].strip()
            
            # Add correct solution
            icm_examples.append((
                question,
                correct_solution + f" The answer is {correct_answer}.",
                correct_answer,
                None
            ))
            
            # Create an incorrect solution with wrong answer
            wrong_answer = str(int(float(correct_answer) * 1.5 + 10))  # Deliberately wrong
            wrong_solution = f"Let me calculate: The answer must be {wrong_answer}."
            
            icm_examples.append((
                question,
                wrong_solution,
                wrong_answer,
                None
            ))
    
    print(f"Loaded {len(icm_examples)} examples from GSM8K")
    return create_math_correctness_dataset(icm_examples)

def load_real_comparison_data(sample_size=50, seed=42):
    """Load real comparison data - using Anthropic HH-RLHF dataset"""
    print("Loading comparison dataset from Hugging Face...")
    
    # Anthropic's helpful-harmless dataset has chosen/rejected pairs
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    
    random.seed(seed)
    indices = random.sample(range(min(1000, len(dataset))), min(sample_size, 1000))
    
    icm_examples = []
    
    for idx in indices:
        item = dataset[idx]
        
        # Extract the human query and the two responses
        chosen = item['chosen']
        rejected = item['rejected']
        
        # Parse the conversation to get the query and responses
        # Format is usually "Human: {query}\n\nAssistant: {response}\n\nHuman: ..."
        if "Human:" in chosen and "Assistant:" in chosen:
            # Extract first human query
            human_start = chosen.find("Human:") + len("Human:")
            human_end = chosen.find("\n\nAssistant:")
            if human_end > human_start:
                query = chosen[human_start:human_end].strip()
                
                # Extract assistant responses
                chosen_start = chosen.find("Assistant:", human_end) + len("Assistant:")
                chosen_end = chosen.find("\n\nHuman:", chosen_start)
                if chosen_end == -1:
                    chosen_end = len(chosen)
                chosen_response = chosen[chosen_start:chosen_end].strip()
                
                # Do the same for rejected
                rejected_start = rejected.find("Assistant:", human_end) + len("Assistant:")
                rejected_end = rejected.find("\n\nHuman:", rejected_start)
                if rejected_end == -1:
                    rejected_end = len(rejected)
                rejected_response = rejected[rejected_start:rejected_end].strip()
                
                # Add the comparison (chosen is better than rejected)
                icm_examples.append((
                    query,
                    chosen_response,
                    rejected_response,
                    None  # ICM will figure out which is better
                ))
    
    # Limit size and ensure we have good examples
    icm_examples = [ex for ex in icm_examples if ex[0] and ex[1] and ex[2]][:sample_size]
    
    print(f"Loaded {len(icm_examples)} examples from HH-RLHF")
    return create_comparison_dataset(icm_examples)

def load_real_datasets_alternative(task, sample_size=50):
    """Alternative datasets if primary ones fail to load"""
    
    if task == "truthfulness":
        # Use MMLU questions as alternative
        print("Trying MMLU as alternative...")
        dataset = load_dataset("cais/mmlu", "all", split="test")
        
        examples = []
        for i in range(min(sample_size, len(dataset))):
            item = dataset[i]
            question = item['question']
            
            # Add correct answer
            correct_idx = item['answer']
            correct_answer = item['choices'][correct_idx]
            examples.append((question, correct_answer, None))
            
            # Add incorrect answer
            incorrect_idx = (correct_idx + 1) % len(item['choices'])
            incorrect_answer = item['choices'][incorrect_idx]
            examples.append((question, incorrect_answer, None))
        
        return create_truthfulness_dataset(examples[:sample_size])
    
    elif task == "math":
        # Use MathQA as alternative
        print("Trying MathQA as alternative...")
        dataset = load_dataset("math_qa", split="train")
        
        examples = []
        for i in range(min(sample_size // 2, 100)):
            item = dataset[i]
            problem = item['Problem']
            
            # Parse rationale and answer
            rationale = item['Rationale']
            correct_answer = item['correct']
            
            # Add correct solution
            examples.append((
                problem,
                f"{rationale} The answer is {correct_answer}.",
                correct_answer,
                None
            ))
            
            # Add incorrect solution
            options = eval(item['options'])
            wrong_answers = [opt for opt in options.values() if opt != correct_answer]
            if wrong_answers:
                wrong_answer = random.choice(wrong_answers)
                examples.append((
                    problem,
                    f"By calculation, the answer is {wrong_answer}.",
                    wrong_answer,
                    None
                ))
        
        return create_math_correctness_dataset(examples[:sample_size])
    
    else:  # comparison
        # Create synthetic comparisons from a dataset with quality annotations
        print("Creating comparison data from CommitPackFT...")
        dataset = load_dataset("bigcode/commitpackft", "python", split="train")
        
        examples = []
        # Take pairs of examples and treat one as better
        for i in range(0, min(sample_size * 2, 200), 2):
            if i + 1 < len(dataset):
                item1 = dataset[i]
                item2 = dataset[i + 1]
                
                # Use commit message as query
                query = f"Write code to: {item1.get('commit_message', 'implement this feature')[:100]}"
                
                # Use different code samples as responses
                response_a = item1.get('new_contents', '')[:500]
                response_b = item2.get('new_contents', '')[:500]
                
                if response_a and response_b and response_a != response_b:
                    examples.append((query, response_a, response_b, None))
        
        return create_comparison_dataset(examples[:sample_size])

def run_experiment(task, model_name="Qwen/Qwen3-4B", sample_size=50, max_iterations=None):
    """Run ICM experiment on specified task with real data"""
    
    print(f"\n{'='*60}")
    print(f"Running ICM on REAL {task.upper()} data")
    print(f"Model: {model_name}")
    print(f"Sample size: {sample_size}")
    print(f"{'='*60}\n")
    
    # Load appropriate dataset with error handling
    try:
        if task == "truthfulness":
            dataset = load_real_truthfulqa(sample_size)
            consistency_type = "general"
            alpha = 50.0
        elif task == "math":
            dataset = load_real_gsm8k(sample_size)
            consistency_type = "math_correctness" 
            alpha = 30.0
        elif task == "comparison":
            dataset = load_real_comparison_data(sample_size)
            consistency_type = "asymmetry"
            alpha = 40.0
        else:
            raise ValueError(f"Unknown task: {task}")
    except Exception as e:
        print(f"Error loading primary dataset: {e}")
        print("Trying alternative dataset...")
        dataset = load_real_datasets_alternative(task, sample_size)
        consistency_type = "general" if task == "truthfulness" else consistency_type
        alpha = 40.0  # Default alpha for alternatives
    
    print(f"\nDataset loaded successfully!")
    print(f"Number of examples: {len(dataset)}")
    
    # Show a few examples
    print("\nSample data points:")
    for i in range(min(3, len(dataset))):
        print(f"\nExample {i+1}:")
        print(f"Input: {dataset[i].input_text[:150]}...")
        if hasattr(dataset[i], 'metadata'):
            print(f"Metadata: {list(dataset[i].metadata.keys())}")
    
    # Configure ICM
    config = ICMConfig(
        model_name=model_name,
        backend="auto",  # Will use vLLM if available
        initial_examples=min(8, len(dataset) // 3),
        alpha=alpha,
        initial_temperature=10.0,
        final_temperature=0.01,
        cooling_rate=0.99,
        max_context_length=16384,  # Qwen3-4B supports long context
        temperature=0.1,
        max_new_tokens=64
    )
    
    # Special config for comparison task
    if task == "comparison":
        config.label_names = ["B is better", "A is better"]
    
    print(f"\nICM Configuration:")
    print(f"  Initial examples: {config.initial_examples}")
    print(f"  Alpha (mutual predictability weight): {config.alpha}")
    print(f"  Consistency type: {consistency_type}")
    print(f"  Max context length: {config.max_context_length}")
    
    # Initialize ICM
    icm = ICM(config)
    icm.consistency_checker.consistency_type = consistency_type
    
    # Run ICM
    print("\nRunning ICM algorithm on real data...")
    start_time = datetime.now()
    
    if max_iterations is None:
        max_iterations = min(len(dataset) * 2, 100)  # Cap at 100 for large datasets
    
    labeled_data = icm.run(dataset, max_iterations=max_iterations)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Analyze results
    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Labeled {len(labeled_data)} examples")
    
    # Calculate final metrics
    final_score, predictability, inconsistencies = icm.calculate_score(labeled_data)
    
    print(f"\nFinal Metrics:")
    print(f"  Score: {final_score:.2f}")
    print(f"  Mutual Predictability: {predictability:.2f}")
    print(f"  Inconsistencies: {inconsistencies}")
    print(f"  Acceptance Rate: {sum(icm.acceptance_history) / len(icm.acceptance_history) * 100:.1f}%")
    
    # Show detailed results
    print(f"\nDetailed Results (first 10):")
    for i, (dp, label) in enumerate(labeled_data[:10]):
        print(f"\n{i+1}. Label: {config.label_names[label]}")
        if task == "truthfulness":
            question = dp.metadata.get('question', 'N/A')[:80]
            claim = dp.metadata.get('claim', 'N/A')[:80]
            print(f"   Q: {question}...")
            print(f"   A: {claim}...")
        elif task == "math":
            problem = dp.metadata.get('problem', 'N/A')[:80]
            answer = dp.metadata.get('answer', 'N/A')
            print(f"   Problem: {problem}...")
            print(f"   Answer: {answer}")
        elif task == "comparison":
            query = dp.metadata.get('query', 'N/A')[:80]
            print(f"   Query: {query}...")
    
    # Save detailed results
    results_dir = Path("icm_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"REAL_{task}_{model_name.replace('/', '_')}_{timestamp}.json"
    
    results = {
        "task": task,
        "model": model_name,
        "dataset_size": len(dataset),
        "dataset_type": "REAL_DATA",
        "config": config.__dict__,
        "metrics": {
            "final_score": float(final_score),
            "mutual_predictability": float(predictability),
            "inconsistencies": int(inconsistencies),
            "runtime_seconds": elapsed,
            "acceptance_rate": sum(icm.acceptance_history) / len(icm.acceptance_history) if icm.acceptance_history else 0
        },
        "labeled_data": [
            {
                "id": dp.id,
                "label": int(label),
                "label_name": config.label_names[label],
                "metadata": dp.metadata,
                "input_preview": dp.input_text[:200] + "..." if len(dp.input_text) > 200 else dp.input_text
            }
            for dp, label in labeled_data
        ],
        "score_history": [float(s) for s in icm.score_history[-50:]]  # Last 50 scores
    }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run ICM with real Qwen3-4B model on REAL datasets")
    parser.add_argument("--task", choices=["truthfulness", "math", "comparison", "all"],
                        default="truthfulness", help="Task to run")
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                        help="Model to use (default: Qwen/Qwen3-4B)")
    parser.add_argument("--sample-size", type=int, default=50,
                        help="Number of examples to sample from dataset (default: 50)")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Maximum iterations (default: 2x dataset size, capped at 100)")
    
    args = parser.parse_args()
    
    tasks = ["truthfulness", "math", "comparison"] if args.task == "all" else [args.task]
    
    all_results = {}
    for task in tasks:
        try:
            results = run_experiment(task, args.model, args.sample_size, args.max_iterations)
            all_results[task] = results
        except KeyboardInterrupt:
            print("\nExperiment interrupted by user")
            break
        except Exception as e:
            print(f"\nError running {task}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY OF REAL DATA EXPERIMENTS")
        print(f"{'='*60}")
        for task, results in all_results.items():
            print(f"\n{task.upper()}:")
            print(f"  Dataset size: {results['dataset_size']}")
            print(f"  Final Score: {results['metrics']['final_score']:.2f}")
            print(f"  Runtime: {results['metrics']['runtime_seconds']:.1f}s")
            print(f"  Inconsistencies: {results['metrics']['inconsistencies']}")
            print(f"  Acceptance Rate: {results['metrics']['acceptance_rate']:.1f}%")

if __name__ == "__main__":
    main()