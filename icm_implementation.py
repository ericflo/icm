"""
Internal Coherence Maximization (ICM) Implementation
Based on the paper "Unsupervised Elicitation of Language Models"

This implementation supports both vLLM and transformers backends for efficient inference.
Default model: Qwen3-4B
"""

import torch
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from collections import defaultdict
import logging
from tqdm import tqdm
import json
from pathlib import Path

# Backend imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM not available, falling back to transformers")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ICMConfig:
    """Configuration for ICM algorithm"""
    # Model configuration
    model_name: str = "Qwen/Qwen3-4B"
    backend: str = "auto"  # "vllm", "transformers", or "auto"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ICM algorithm parameters
    initial_examples: int = 8  # K in the paper
    initial_temperature: float = 10.0  # T_0
    final_temperature: float = 0.01  # T_min
    cooling_rate: float = 0.99  # β
    alpha: float = 50.0  # Weight for mutual predictability
    
    # Sampling parameters
    max_context_length: int = 32768
    max_new_tokens: int = 64
    temperature: float = 0.1
    top_p: float = 0.95
    
    # Consistency fix parameters
    max_consistency_iterations: int = 10
    
    # Task configuration
    num_labels: int = 2  # Binary classification by default
    label_names: List[str] = field(default_factory=lambda: ["False", "True"])


@dataclass
class DataPoint:
    """Single data point for ICM"""
    id: int
    input_text: str
    label: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelBackend:
    """Abstract backend for model inference"""
    def __init__(self, config: ICMConfig):
        self.config = config
        
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        raise NotImplementedError
        
    def get_log_probs(self, prompts: List[str], targets: List[str]) -> List[float]:
        raise NotImplementedError


class VLLMBackend(ModelBackend):
    """vLLM backend for efficient batch inference"""
    def __init__(self, config: ICMConfig):
        super().__init__(config)
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not available")
            
        self.model = LLM(
            model=config.model_name,
            trust_remote_code=True,
            max_model_len=config.max_context_length,
            dtype="auto",
            gpu_memory_utilization=0.9
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens
        )
        
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        outputs = self.model.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
        
    def get_log_probs(self, prompts: List[str], targets: List[str]) -> List[float]:
        """Get log probabilities for target completions"""
        # Create prompts with targets appended
        full_prompts = [p + t for p, t in zip(prompts, targets)]
        
        # Use vLLM to get log probs
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            prompt_logprobs=len(self.tokenizer.encode(targets[0]))  # Approximate
        )
        
        outputs = self.model.generate(prompts, sampling_params)
        
        # Extract log probs for target tokens
        log_probs = []
        for output, target in zip(outputs, targets):
            target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
            prompt_logprobs = output.prompt_logprobs
            
            if prompt_logprobs:
                total_logprob = sum(
                    prompt_logprobs[i].get(token, -float('inf'))
                    for i, token in enumerate(target_tokens)
                    if i < len(prompt_logprobs)
                )
                log_probs.append(total_logprob)
            else:
                log_probs.append(-float('inf'))
                
        return log_probs


class TransformersBackend(ModelBackend):
    """Transformers backend for compatibility"""
    def __init__(self, config: ICMConfig):
        super().__init__(config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype="auto",
            device_map="auto" if config.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if config.device == "cpu":
            self.model = self.model.to(config.device)
            
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_context_length
        )
        
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # Decode only the generated part
        generated = []
        for i, output in enumerate(outputs):
            input_len = inputs['input_ids'][i].shape[0]
            generated_ids = output[input_len:]
            generated.append(self.tokenizer.decode(generated_ids, skip_special_tokens=True))
            
        return generated
        
    def get_log_probs(self, prompts: List[str], targets: List[str]) -> List[float]:
        """Get log probabilities for target completions"""
        log_probs = []
        
        for prompt, target in zip(prompts, targets):
            # Tokenize prompt and target separately
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            
            # Combine them
            input_ids = torch.tensor([prompt_ids + target_ids])
            
            if self.config.device == "cuda":
                input_ids = input_ids.cuda()
                
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
                
            # Calculate log probs for target tokens
            target_logprobs = []
            for i, target_id in enumerate(target_ids):
                # Position in the full sequence
                pos = len(prompt_ids) + i - 1
                if pos < logits.shape[1]:
                    token_logits = logits[0, pos]
                    token_logprobs = F.log_softmax(token_logits, dim=-1)
                    target_logprobs.append(token_logprobs[target_id].item())
                    
            # Sum log probs
            total_logprob = sum(target_logprobs) if target_logprobs else -float('inf')
            log_probs.append(total_logprob)
            
        return log_probs


class LogicalConsistency:
    """Handle logical consistency checking between labels"""
    
    def __init__(self, consistency_type: str = "general"):
        self.consistency_type = consistency_type
        self.inconsistencies = defaultdict(list)
        
    def check_consistency(
        self, 
        x_i: DataPoint, 
        y_i: int, 
        x_j: DataPoint, 
        y_j: int
    ) -> bool:
        """
        Check if two labeled examples are logically consistent.
        Returns True if consistent, False otherwise.
        """
        if self.consistency_type == "asymmetry":
            # For comparative tasks (e.g., A > B vs B > A)
            if "comparison" in x_i.metadata and "comparison" in x_j.metadata:
                comp_i = x_i.metadata["comparison"]
                comp_j = x_j.metadata["comparison"]
                
                # Check if they're comparing the same items in opposite order
                if (comp_i[0] == comp_j[1] and comp_i[1] == comp_j[0]):
                    # They should have opposite labels
                    return y_i != y_j
                    
        elif self.consistency_type == "math_correctness":
            # For mathematical correctness (same problem, different solutions)
            if "problem_id" in x_i.metadata and "problem_id" in x_j.metadata:
                if x_i.metadata["problem_id"] == x_j.metadata["problem_id"]:
                    # If answers are different, can't both be correct
                    if (x_i.metadata.get("answer") != x_j.metadata.get("answer") and
                        y_i == 1 and y_j == 1):  # Both labeled as correct
                        return False
                        
        # Default: no specific constraints
        return True
        
    def count_inconsistencies(self, labeled_data: List[Tuple[DataPoint, int]]) -> int:
        """Count total inconsistencies in the dataset"""
        count = 0
        self.inconsistencies.clear()
        
        for i, (x_i, y_i) in enumerate(labeled_data):
            for j, (x_j, y_j) in enumerate(labeled_data[i+1:], i+1):
                if not self.check_consistency(x_i, y_i, x_j, y_j):
                    count += 1
                    self.inconsistencies[i].append(j)
                    self.inconsistencies[j].append(i)
                    
        return count
        
    def get_inconsistent_pairs(self, labeled_data: List[Tuple[DataPoint, int]]) -> List[Tuple[int, int]]:
        """Get all inconsistent pairs of indices"""
        pairs = []
        for i in self.inconsistencies:
            for j in self.inconsistencies[i]:
                if i < j:  # Avoid duplicates
                    pairs.append((i, j))
        return pairs


class ICM:
    """Main ICM algorithm implementation"""
    
    def __init__(self, config: ICMConfig, backend: Optional[ModelBackend] = None):
        self.config = config
        
        # Initialize backend
        if backend:
            self.backend = backend
        else:
            self.backend = self._create_backend()
            
        # Initialize components
        self.consistency_checker = LogicalConsistency()
        self.iteration = 0
        self.temperature = config.initial_temperature
        
        # Tracking
        self.score_history = []
        self.acceptance_history = []
        
    def _create_backend(self) -> ModelBackend:
        """Create appropriate backend based on config"""
        if self.config.backend == "vllm" or (self.config.backend == "auto" and VLLM_AVAILABLE):
            logger.info("Using vLLM backend")
            return VLLMBackend(self.config)
        else:
            logger.info("Using transformers backend")
            return TransformersBackend(self.config)
            
    def format_in_context_prompt(
        self, 
        x_i: DataPoint, 
        labeled_data: List[Tuple[DataPoint, int]],
        exclude_idx: Optional[int] = None
    ) -> str:
        """Format prompt with in-context examples for mutual predictability"""
        prompt_parts = []
        
        # Add labeled examples as context
        for idx, (x, y) in enumerate(labeled_data):
            if exclude_idx is not None and idx == exclude_idx:
                continue
                
            label_name = self.config.label_names[y]
            example = f"Input: {x.input_text}\nLabel: {label_name}\n"
            prompt_parts.append(example)
            
        # Add the query
        prompt_parts.append(f"Input: {x_i.input_text}\nLabel:")
        
        return "\n".join(prompt_parts)
        
    def calculate_mutual_predictability(self, labeled_data: List[Tuple[DataPoint, int]]) -> float:
        """Calculate mutual predictability score P_θ(D)"""
        if len(labeled_data) <= 1:
            return 0.0
            
        total_log_prob = 0.0
        
        # Process in batches for efficiency
        prompts = []
        targets = []
        
        for i, (x_i, y_i) in enumerate(labeled_data):
            # Create prompt excluding current example
            prompt = self.format_in_context_prompt(x_i, labeled_data, exclude_idx=i)
            target = f" {self.config.label_names[y_i]}"
            
            prompts.append(prompt)
            targets.append(target)
            
        # Get log probabilities
        log_probs = self.backend.get_log_probs(prompts, targets)
        total_log_prob = sum(log_probs)
        
        return total_log_prob
        
    def calculate_score(self, labeled_data: List[Tuple[DataPoint, int]]) -> float:
        """Calculate overall scoring function U(D)"""
        # Mutual predictability
        predictability = self.calculate_mutual_predictability(labeled_data)
        
        # Logical consistency
        inconsistencies = self.consistency_checker.count_inconsistencies(labeled_data)
        
        # Combined score
        score = self.config.alpha * predictability - inconsistencies
        
        return score, predictability, inconsistencies
        
    def consistency_fix(
        self, 
        labeled_data: List[Tuple[DataPoint, int]]
    ) -> List[Tuple[DataPoint, int]]:
        """Fix logical inconsistencies (Algorithm 2)"""
        data = labeled_data.copy()
        
        for iteration in range(self.config.max_consistency_iterations):
            inconsistent_pairs = self.consistency_checker.get_inconsistent_pairs(data)
            
            if not inconsistent_pairs:
                break
                
            # Sample a random inconsistent pair
            i, j = random.choice(inconsistent_pairs)
            x_i, _ = data[i]
            x_j, _ = data[j]
            
            # Try all consistent label combinations
            best_score = -float('inf')
            best_labels = None
            
            for y_i in range(self.config.num_labels):
                for y_j in range(self.config.num_labels):
                    if self.consistency_checker.check_consistency(x_i, y_i, x_j, y_j):
                        # Create temporary dataset with new labels
                        temp_data = data.copy()
                        temp_data[i] = (x_i, y_i)
                        temp_data[j] = (x_j, y_j)
                        
                        # Calculate score
                        score, _, _ = self.calculate_score(temp_data)
                        
                        if score > best_score:
                            best_score = score
                            best_labels = (y_i, y_j)
                            
            # Apply best labels if they improve score
            if best_labels:
                current_score, _, _ = self.calculate_score(data)
                if best_score > current_score:
                    data[i] = (x_i, best_labels[0])
                    data[j] = (x_j, best_labels[1])
                    
        return data
        
    def predict_label(self, x: DataPoint, context: List[Tuple[DataPoint, int]]) -> int:
        """Predict label for a single example given context"""
        prompt = self.format_in_context_prompt(x, context)
        
        # Get model prediction
        outputs = self.backend.generate([prompt])
        prediction = outputs[0].strip()
        
        # Parse prediction
        for i, label_name in enumerate(self.config.label_names):
            if label_name.lower() in prediction.lower():
                return i
                
        # Default to random if parsing fails
        return random.randint(0, self.config.num_labels - 1)
        
    def run(
        self, 
        unlabeled_data: List[DataPoint],
        max_iterations: Optional[int] = None
    ) -> List[Tuple[DataPoint, int]]:
        """Run ICM algorithm (Algorithm 1)"""
        if max_iterations is None:
            max_iterations = len(unlabeled_data)
            
        # Initialize with K random examples
        labeled_data = []
        remaining_data = unlabeled_data.copy()
        random.shuffle(remaining_data)
        
        for _ in range(min(self.config.initial_examples, len(remaining_data))):
            x = remaining_data.pop()
            y = random.randint(0, self.config.num_labels - 1)
            labeled_data.append((x, y))
            
        # Initial consistency fix
        labeled_data = self.consistency_fix(labeled_data)
        
        # Main loop
        progress_bar = tqdm(range(max_iterations), desc="ICM iterations")
        
        for n in progress_bar:
            # Update temperature
            self.temperature = max(
                self.config.final_temperature,
                self.config.initial_temperature / (1 + self.config.cooling_rate * np.log(n + 1))
            )
            
            # Sample an example (could be from labeled or unlabeled)
            if remaining_data and random.random() < 0.7:  # 70% chance to pick unlabeled
                x_i = random.choice(remaining_data)
                is_new = True
            else:
                idx = random.randint(0, len(labeled_data) - 1)
                x_i, _ = labeled_data[idx]
                is_new = False
                
            # Predict label
            y_i_hat = self.predict_label(x_i, labeled_data)
            
            # Create temporary dataset
            if is_new:
                temp_data = labeled_data + [(x_i, y_i_hat)]
            else:
                temp_data = labeled_data.copy()
                for i, (x, y) in enumerate(temp_data):
                    if x.id == x_i.id:
                        temp_data[i] = (x, y_i_hat)
                        break
                        
            # Fix inconsistencies
            temp_data = self.consistency_fix(temp_data)
            
            # Calculate score difference
            old_score, _, _ = self.calculate_score(labeled_data)
            new_score, _, inconsistencies = self.calculate_score(temp_data)
            delta = new_score - old_score
            
            # Accept or reject
            accept = False
            if delta > 0:
                accept = True
            else:
                # Metropolis criterion
                if random.random() < np.exp(delta / self.temperature):
                    accept = True
                    
            if accept:
                labeled_data = temp_data
                if is_new and x_i in remaining_data:
                    remaining_data.remove(x_i)
                    
            # Update tracking
            self.score_history.append(new_score)
            self.acceptance_history.append(accept)
            
            # Update progress bar
            progress_bar.set_postfix({
                'score': f'{new_score:.2f}',
                'inconsistencies': inconsistencies,
                'temperature': f'{self.temperature:.3f}',
                'labeled': len(labeled_data)
            })
            
        return labeled_data


# Utility functions for common tasks
def create_truthfulness_dataset(questions_claims: List[Tuple[str, str, bool]]) -> List[DataPoint]:
    """Create dataset for truthfulness task (like TruthfulQA)"""
    dataset = []
    for i, (question, claim, is_true) in enumerate(questions_claims):
        text = f"Question: {question}\nClaim: {claim}\nI think this Claim is"
        dp = DataPoint(
            id=i,
            input_text=text,
            label=int(is_true) if is_true is not None else None,
            metadata={"question": question, "claim": claim}
        )
        dataset.append(dp)
    return dataset


def create_math_correctness_dataset(problems_solutions: List[Tuple[str, str, str, bool]]) -> List[DataPoint]:
    """Create dataset for mathematical correctness (like GSM8K)"""
    dataset = []
    for i, (problem, solution, answer, is_correct) in enumerate(problems_solutions):
        text = f"Question: {problem}\nClaim: {solution}\nI think this Claim is"
        dp = DataPoint(
            id=i,
            input_text=text,
            label=int(is_correct) if is_correct is not None else None,
            metadata={
                "problem": problem,
                "solution": solution,
                "answer": answer,
                "problem_id": hash(problem)
            }
        )
        dataset.append(dp)
    return dataset


def create_comparison_dataset(comparisons: List[Tuple[str, str, str, bool]]) -> List[DataPoint]:
    """Create dataset for comparison tasks (like Alpaca)"""
    dataset = []
    for i, (query, response_a, response_b, a_is_better) in enumerate(comparisons):
        text = (f"Query: {query}\n"
                f"Response A: {response_a}\n"
                f"Response B: {response_b}\n"
                f"Claim: Response A is more helpful and harmless than Response B\n"
                f"I think this Claim is")
        dp = DataPoint(
            id=i,
            input_text=text,
            label=int(a_is_better) if a_is_better is not None else None,
            metadata={
                "query": query,
                "response_a": response_a,
                "response_b": response_b,
                "comparison": ("A", "B")
            }
        )
        dataset.append(dp)
    return dataset


# Example usage and testing
if __name__ == "__main__":
    # Example: TruthfulQA-style task
    print("ICM Implementation Test")
    print("=" * 50)
    
    # Create sample data
    sample_data = [
        ("What are some cities that get less rainfall than Boston?",
         "San Francisco and Los Angeles get less rainfall than Boston.",
         None),  # To be labeled by ICM
        ("What is the capital of France?",
         "The capital of France is London.",
         None),
        ("How many planets are in our solar system?",
         "There are 8 planets in our solar system.",
         None),
        ("What color is the sky on a clear day?",
         "The sky is green on a clear day.",
         None),
    ]
    
    # Create dataset
    dataset = create_truthfulness_dataset(sample_data)
    
    # Configure ICM
    config = ICMConfig(
        model_name="Qwen/Qwen3-4B",
        backend="auto",
        initial_examples=2,
        alpha=30.0,  # Adjusted for small dataset
        max_context_length=8192  # Reduced for efficiency
    )
    
    # Run ICM
    print(f"Running ICM with {config.model_name}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Initial random examples: {config.initial_examples}")
    print()
    
    icm = ICM(config)
    labeled_data = icm.run(dataset, max_iterations=10)
    
    # Display results
    print("\nResults:")
    print("-" * 50)
    for data_point, label in labeled_data:
        label_name = config.label_names[label]
        print(f"ID: {data_point.id}")
        print(f"Input: {data_point.input_text[:100]}...")
        print(f"Label: {label_name}")
        print()
    
    # Show final statistics
    final_score, predictability, inconsistencies = icm.calculate_score(labeled_data)
    print(f"Final Score: {final_score:.2f}")
    print(f"Mutual Predictability: {predictability:.2f}")
    print(f"Inconsistencies: {inconsistencies}")
    
    # Save results
    results = {
        "config": config.__dict__,
        "labeled_data": [
            {
                "id": dp.id,
                "input": dp.input_text,
                "label": label,
                "metadata": dp.metadata
            }
            for dp, label in labeled_data
        ],
        "final_metrics": {
            "score": final_score,
            "predictability": predictability,
            "inconsistencies": inconsistencies
        }
    }
    
    with open("icm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to icm_results.json")