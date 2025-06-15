"""
ICM Practical Examples and Usage Guide
Demonstrates real-world usage of ICM with different tasks and models
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import numpy as np
from datetime import datetime

# Import ICM components
from icm_implementation import (
    ICMConfig, DataPoint, ICM,
    create_truthfulness_dataset,
    create_math_correctness_dataset,
    create_comparison_dataset,
    VLLMBackend, TransformersBackend
)


class ICMExperiment:
    """Wrapper class for running ICM experiments"""
    
    def __init__(self, task_name: str, model_name: str = "Qwen/Qwen3-4B", backend: str = "auto"):
        self.task_name = task_name
        self.model_name = model_name
        self.backend = backend
        self.results_dir = Path("icm_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_truthfulqa_sample(self) -> List[DataPoint]:
        """Load sample TruthfulQA-style data"""
        # In practice, you would load from the actual dataset
        # This is a demonstration with sample data
        samples = [
            ("What happens if you eat watermelon seeds?",
             "Watermelon seeds will grow in your stomach", None),
            ("What happens if you eat watermelon seeds?",
             "Nothing harmful happens; they pass through your system", None),
            ("Can you see the Great Wall of China from space?",
             "Yes, the Great Wall is visible from space with the naked eye", None),
            ("Can you see the Great Wall of China from space?",
             "No, it's not visible from space without aid", None),
            ("What happens when you crack your knuckles?",
             "Cracking knuckles causes arthritis", None),
            ("What happens when you crack your knuckles?",
             "Cracking knuckles releases gas bubbles and is generally harmless", None),
            ("How long should you wait after eating before swimming?",
             "You must wait 30 minutes after eating before swimming", None),
            ("How long should you wait after eating before swimming?",
             "There's no medical need to wait after eating before swimming", None),
        ]
        
        return create_truthfulness_dataset(samples)
        
    def load_gsm8k_sample(self) -> List[DataPoint]:
        """Load sample GSM8K-style math problems"""
        samples = [
            ("Jenny has 5 apples. She gives 2 to her friend. How many does she have left?",
             "Jenny starts with 5 apples. She gives away 2 apples. So she has 5 - 2 = 3 apples left.",
             "3", None),
            ("Jenny has 5 apples. She gives 2 to her friend. How many does she have left?",
             "Jenny has 5 apples and gives 2, so 5 + 2 = 7 apples.",
             "7", None),
            ("A store sells pens for $2 each. How much do 6 pens cost?",
             "Each pen costs $2. For 6 pens: 6 × $2 = $12.",
             "12", None),
            ("A store sells pens for $2 each. How much do 6 pens cost?",
             "6 pens at $2 each means 6 - 2 = $4.",
             "4", None),
            ("Tom reads 20 pages per day. How many pages will he read in a week?",
             "Tom reads 20 pages daily. In a week (7 days): 20 × 7 = 140 pages.",
             "140", None),
            ("Tom reads 20 pages per day. How many pages will he read in a week?",
             "20 pages per day for a week is 20 + 7 = 27 pages.",
             "27", None),
        ]
        
        return create_math_correctness_dataset(samples)
        
    def load_alpaca_sample(self) -> List[DataPoint]:
        """Load sample Alpaca-style comparison data"""
        samples = [
            ("Write a haiku about spring",
             "Cherry blossoms bloom\nGentle breeze carries petals\nSpring awakens life",
             "spring is nice flowers grow trees are green",
             None),
            ("Explain photosynthesis to a child",
             "Plants are like tiny factories that make their own food! They use sunlight, water, and air to create energy, just like how you eat food for energy. The green parts of plants capture sunlight and turn it into plant food.",
             "Photosynthesis is the process by which plants convert light energy into chemical energy through complex biochemical reactions involving chlorophyll.",
             None),
            ("How do I make a paper airplane?",
             "1. Take a sheet of paper\n2. Fold it in half lengthwise, then unfold\n3. Fold the top corners to the center\n4. Fold the angled edges to the center again\n5. Fold in half along the original crease\n6. Create wings by folding each side down",
             "make plane with paper",
             None),
            ("What's the best way to learn a new language?",
             "The best way to learn a new language combines multiple approaches: 1) Daily practice with apps or lessons, 2) Immersion through media like movies and music, 3) Speaking practice with native speakers, 4) Regular reading and writing exercises. Consistency is key - even 15 minutes daily is better than sporadic long sessions.",
             "just watch tv shows",
             None),
        ]
        
        return create_comparison_dataset(samples)
        
    def run_experiment(self, dataset: List[DataPoint], config: Optional[ICMConfig] = None) -> dict:
        """Run ICM experiment and return results"""
        if config is None:
            config = ICMConfig(
                model_name=self.model_name,
                backend=self.backend,
                initial_examples=8,
                alpha=50.0,
                temperature=0.1,
                max_context_length=8192
            )
            
        # Adjust config based on task
        if self.task_name == "math":
            config.alpha = 30.0  # Lower weight for math tasks
            config.label_names = ["Incorrect", "Correct"]
        elif self.task_name == "comparison":
            config.alpha = 40.0
            config.label_names = ["False", "True"]
            
        print(f"\nRunning ICM for {self.task_name} task")
        print(f"Model: {config.model_name}")
        print(f"Backend: {config.backend}")
        print(f"Dataset size: {len(dataset)}")
        print(f"Initial examples: {config.initial_examples}")
        print(f"Alpha (predictability weight): {config.alpha}")
        
        # Initialize ICM
        icm = ICM(config)
        
        # Run algorithm
        start_time = datetime.now()
        labeled_data = icm.run(dataset, max_iterations=len(dataset))
        end_time = datetime.now()
        
        # Calculate final metrics
        final_score, predictability, inconsistencies = icm.calculate_score(labeled_data)
        
        # Prepare results
        results = {
            "task": self.task_name,
            "model": config.model_name,
            "backend": config.backend,
            "dataset_size": len(dataset),
            "config": {
                "initial_examples": config.initial_examples,
                "alpha": config.alpha,
                "initial_temperature": config.initial_temperature,
                "final_temperature": config.final_temperature,
                "cooling_rate": config.cooling_rate,
                "label_names": config.label_names
            },
            "results": {
                "final_score": float(final_score),
                "mutual_predictability": float(predictability),
                "inconsistencies": int(inconsistencies),
                "runtime_seconds": (end_time - start_time).total_seconds()
            },
            "labeled_data": [
                {
                    "id": dp.id,
                    "input": dp.input_text[:200] + "..." if len(dp.input_text) > 200 else dp.input_text,
                    "label": int(label),
                    "label_name": config.label_names[label]
                }
                for dp, label in labeled_data
            ],
            "score_history": [float(s) for s in icm.score_history[-20:]],  # Last 20 scores
            "acceptance_rate": sum(icm.acceptance_history) / len(icm.acceptance_history) if icm.acceptance_history else 0
        }
        
        return results
        
    def save_results(self, results: dict):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"{self.task_name}_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"\nResults saved to: {filename}")
        
        # Also save a summary CSV
        summary_file = self.results_dir / "experiment_summary.csv"
        summary_data = {
            "timestamp": timestamp,
            "task": results["task"],
            "model": results["model"],
            "dataset_size": results["dataset_size"],
            "final_score": results["results"]["final_score"],
            "inconsistencies": results["results"]["inconsistencies"],
            "runtime": results["results"]["runtime_seconds"],
            "acceptance_rate": results["acceptance_rate"]
        }
        
        df = pd.DataFrame([summary_data])
        if summary_file.exists():
            existing_df = pd.read_csv(summary_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_csv(summary_file, index=False)
        
    def analyze_results(self, results: dict):
        """Analyze and print experiment results"""
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS ANALYSIS")
        print("="*60)
        
        print(f"\nTask: {results['task'].upper()}")
        print(f"Model: {results['model']}")
        print(f"Runtime: {results['results']['runtime_seconds']:.2f} seconds")
        
        print(f"\nFinal Metrics:")
        print(f"  - Score: {results['results']['final_score']:.2f}")
        print(f"  - Mutual Predictability: {results['results']['mutual_predictability']:.2f}")
        print(f"  - Inconsistencies: {results['results']['inconsistencies']}")
        print(f"  - Acceptance Rate: {results['acceptance_rate']:.2%}")
        
        # Analyze label distribution
        labels = [item['label'] for item in results['labeled_data']]
        label_counts = {i: labels.count(i) for i in range(len(results['config']['label_names']))}
        
        print(f"\nLabel Distribution:")
        for label_idx, count in label_counts.items():
            label_name = results['labeled_data'][0]['label_name'] if label_idx < len(results['labeled_data']) else f"Label {label_idx}"
            percentage = (count / len(labels)) * 100
            print(f"  - {label_name}: {count} ({percentage:.1f}%)")
            
        # Show sample results
        print(f"\nSample Labeled Examples:")
        for item in results['labeled_data'][:5]:
            print(f"\n  ID {item['id']}:")
            print(f"  Input: {item['input'][:100]}...")
            print(f"  Label: {item['label_name']}")
            

def main():
    """Main function to run experiments"""
    parser = argparse.ArgumentParser(description="Run ICM experiments")
    parser.add_argument("--task", choices=["truthfulness", "math", "comparison", "all"],
                        default="all", help="Task to run")
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                        help="Model to use (default: Qwen/Qwen3-4B)")
    parser.add_argument("--backend", choices=["vllm", "transformers", "auto"],
                        default="auto", help="Backend to use")
    parser.add_argument("--small", action="store_true",
                        help="Use small dataset for testing")
    
    args = parser.parse_args()
    
    tasks = ["truthfulness", "math", "comparison"] if args.task == "all" else [args.task]
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"RUNNING {task.upper()} TASK")
        print(f"{'='*60}")
        
        experiment = ICMExperiment(task, args.model, args.backend)
        
        # Load appropriate dataset
        if task == "truthfulness":
            dataset = experiment.load_truthfulqa_sample()
        elif task == "math":
            dataset = experiment.load_gsm8k_sample()
        else:  # comparison
            dataset = experiment.load_alpaca_sample()
            
        # Use smaller subset if requested
        if args.small:
            dataset = dataset[:4]
            
        # Configure ICM
        config = ICMConfig(
            model_name=args.model,
            backend=args.backend,
            initial_examples=min(4, len(dataset) // 2),
            max_context_length=8192 if args.small else 16384
        )
        
        # Run experiment
        try:
            results = experiment.run_experiment(dataset, config)
            experiment.analyze_results(results)
            experiment.save_results(results)
        except Exception as e:
            print(f"Error running {task}: {e}")
            import traceback
            traceback.print_exc()
            

def compare_backends():
    """Compare vLLM vs transformers performance"""
    print("Comparing vLLM vs Transformers backends...")
    
    # Small dataset for comparison
    dataset = create_truthfulness_dataset([
        ("Is the Earth round?", "Yes, the Earth is spherical", None),
        ("Is the Earth round?", "No, the Earth is flat", None),
        ("Can humans breathe underwater?", "No, humans need air to breathe", None),
        ("Can humans breathe underwater?", "Yes, humans have gills", None),
    ])
    
    results = {}
    
    for backend in ["transformers", "vllm"]:
        try:
            print(f"\nTesting {backend} backend...")
            experiment = ICMExperiment("backend_comparison", backend=backend)
            
            config = ICMConfig(
                backend=backend,
                initial_examples=2,
                alpha=20.0,
                max_context_length=4096
            )
            
            result = experiment.run_experiment(dataset, config)
            results[backend] = result
            
        except Exception as e:
            print(f"Failed to test {backend}: {e}")
            results[backend] = None
            
    # Compare results
    print("\n" + "="*60)
    print("BACKEND COMPARISON RESULTS")
    print("="*60)
    
    for backend, result in results.items():
        if result:
            print(f"\n{backend.upper()}:")
            print(f"  Runtime: {result['results']['runtime_seconds']:.2f}s")
            print(f"  Final Score: {result['results']['final_score']:.2f}")
            print(f"  Inconsistencies: {result['results']['inconsistencies']}")
        else:
            print(f"\n{backend.upper()}: Failed to run")


if __name__ == "__main__":
    # Check if running specific comparison
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--compare-backends":
        compare_backends()
    else:
        main()