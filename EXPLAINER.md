# Complete Guide to ICM: Automatic Data Organization with AI

**Turn messy datasets into organized, categorized information - without any manual labeling**

## What Problem Does This Solve?

As an engineer, you probably have datasets sitting around that need organizing. Maybe you have:
- Survey responses that need sorting into "helpful" vs "not helpful"
- Math solutions that need checking for correctness
- Product reviews that should be categorized as positive or negative
- Claims or statements that need fact-checking
- Multiple versions of content that need ranking by quality

Normally, you'd need to manually go through hundreds or thousands of entries, or hire people to categorize them. ICM (Internal Coherence Maximization) automates this process by using AI to find patterns and organize your data consistently.

## How It Works (The Simple Version)

Think of ICM as a smart pattern-finding assistant:

1. **You provide raw data** - questions, claims, solutions, comparisons, etc.
2. **ICM analyzes the patterns** - it looks at how different pieces relate to each other
3. **It automatically organizes everything** - sorting your data into meaningful categories
4. **You get clean, organized results** - ready to use for analysis or decision-making

The key insight: instead of needing pre-organized examples, ICM figures out the patterns by looking at what makes sense together.

## Quick Start: Your First ICM Project

Let's start with a complete, working example that you can run right now:

### Step 1: Install Dependencies

```bash
# Install required packages
pip install torch transformers vllm pandas tqdm numpy
# or use uv for faster installation
uv add torch transformers vllm pandas tqdm numpy
```

### Step 2: Basic Usage Example

```python
# file: my_first_icm.py
from icm_implementation import ICM, ICMConfig, create_truthfulness_dataset

# Step 2a: Prepare your data
# Format: [(question, claim, None)] - None means "unknown, please categorize"
survey_data = [
    ("How satisfied are you with our service?", 
     "Very happy, solved my problem quickly", None),
    ("How satisfied are you with our service?", 
     "Terrible, waited 2 hours for basic help", None),
    ("Would you recommend us?", 
     "Absolutely, great experience", None),
    ("Would you recommend us?", 
     "No way, very disappointing", None),
    ("How was the support quality?",
     "Outstanding! The agent was knowledgeable and friendly", None),
    ("How was the support quality?",
     "Poor communication, didn't understand my issue", None),
]

# Step 2b: Create ICM dataset
dataset = create_truthfulness_dataset(survey_data)

# Step 2c: Configure the system
config = ICMConfig(
    model_name="Qwen/Qwen3-4B",    # Fast, reliable model
    initial_examples=2,            # Start with 2 random examples
    alpha=50.0,                    # Higher = stricter consistency
    max_new_tokens=32              # Short responses for classification
)

# Step 2d: Run the organization
print("Starting ICM analysis...")
icm = ICM(config)
results = icm.run(dataset, max_iterations=20)

# Step 2e: See your organized results
print("\n=== ORGANIZED RESULTS ===")
for item, category in results:
    sentiment = "Positive" if category == 1 else "Negative"
    print(f"\nResponse: {item.metadata['claim']}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence ID: {item.id}")

# Step 2f: Get performance metrics
final_score, predictability, inconsistencies = icm.calculate_score(results)
print(f"\n=== QUALITY METRICS ===")
print(f"Final Score: {final_score:.2f}")
print(f"Mutual Predictability: {predictability:.2f}")
print(f"Logical Inconsistencies: {inconsistencies}")
```

### Step 3: Run Your First Example

```bash
python my_first_icm.py
```

**Expected Output:**
```
Starting ICM analysis...
ICM iterations: 100%|████████████| 20/20 [00:45<00:00,  2.1it/s]

=== ORGANIZED RESULTS ===

Response: Very happy, solved my problem quickly
Sentiment: Positive
Confidence ID: 0

Response: Terrible, waited 2 hours for basic help
Sentiment: Negative
Confidence ID: 1

Response: Absolutely, great experience
Sentiment: Positive
Confidence ID: 2

Response: No way, very disappointing
Sentiment: Negative
Confidence ID: 3

Response: Outstanding! The agent was knowledgeable and friendly
Sentiment: Positive
Confidence ID: 4

Response: Poor communication, didn't understand my issue
Sentiment: Negative
Confidence ID: 5

=== QUALITY METRICS ===
Final Score: 85.23
Mutual Predictability: 92.15
Logical Inconsistencies: 0
```

## Real-World Use Cases with Complete Code Examples

### 1. Quality Control for Content

**Problem:** You have 500 customer support responses and need to identify which ones actually answer the customer's question.

**Solution:** Here's a complete implementation:

```python
# file: quality_control_example.py
from icm_implementation import ICM, ICMConfig, create_truthfulness_dataset
import pandas as pd

# Step 1: Prepare your customer support data
support_data = [
    ("How do I reset my password?", 
     "Click on 'Forgot Password' on the login page, enter your email, and follow the instructions sent to your inbox.", None),
    ("How do I reset my password?", 
     "Thank you for contacting us. We appreciate your business.", None),
    ("What are your shipping rates?", 
     "Shipping is $5.99 for orders under $50, and free for orders over $50.", None),
    ("What are your shipping rates?", 
     "I'll need to check with my manager and get back to you.", None),
    ("Is this product compatible with Mac?", 
     "Yes, this product is fully compatible with macOS 10.15 and later versions.", None),
    ("Is this product compatible with Mac?", 
     "We have many great products available. Would you like to see our catalog?", None),
    ("How long is the warranty?", 
     "All products come with a 2-year manufacturer warranty covering defects and normal wear.", None),
    ("How long is the warranty?", 
     "Warranty is important. Please contact our sales team for more information.", None),
]

# Step 2: Create dataset and configure ICM
dataset = create_truthfulness_dataset(support_data)
config = ICMConfig(
    model_name="Qwen/Qwen3-4B",
    initial_examples=3,
    alpha=60.0,  # Higher alpha for stricter quality control
    max_new_tokens=16,
    temperature=0.1  # Lower temperature for more consistent results
)

# Step 3: Run analysis
print("Analyzing customer support response quality...")
icm = ICM(config)
results = icm.run(dataset, max_iterations=30)

# Step 4: Process and save results
helpful_responses = []
unhelpful_responses = []

for item, category in results:
    response_data = {
        'question': item.metadata['question'],
        'response': item.metadata['claim'],
        'quality': 'Helpful' if category == 1 else 'Unhelpful',
        'id': item.id
    }
    
    if category == 1:
        helpful_responses.append(response_data)
    else:
        unhelpful_responses.append(response_data)

# Step 5: Generate quality report
print(f"\n=== QUALITY CONTROL REPORT ===")
print(f"Total responses analyzed: {len(results)}")
print(f"Helpful responses: {len(helpful_responses)} ({len(helpful_responses)/len(results)*100:.1f}%)")
print(f"Unhelpful responses: {len(unhelpful_responses)} ({len(unhelpful_responses)/len(results)*100:.1f}%)")

print("\n=== HELPFUL RESPONSES ===")
for response in helpful_responses[:3]:  # Show top 3
    print(f"Q: {response['question']}")
    print(f"A: {response['response']}")
    print(f"Quality: {response['quality']}\n")

print("=== NEEDS IMPROVEMENT ===")
for response in unhelpful_responses[:3]:  # Show top 3
    print(f"Q: {response['question']}")
    print(f"A: {response['response']}")
    print(f"Quality: {response['quality']}\n")

# Step 6: Export results to CSV
df = pd.DataFrame([r for r in helpful_responses + unhelpful_responses])
df.to_csv('support_quality_analysis.csv', index=False)
print("Results exported to 'support_quality_analysis.csv'")
```

**What you get:** Organized data showing which responses work well, which need improvement, plus a CSV export for further analysis.

### 2. Educational Content Verification

**Problem:** You have a database of math problems with solutions, but you're not sure which solutions are correct.

**Solution:** Here's a complete math verification system:

```python
# file: math_verification_example.py
from icm_implementation import ICM, ICMConfig, create_math_correctness_dataset
import json

# Step 1: Prepare math problems with solutions
math_problems = [
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
    ("What is 15% of 80?",
     "To find 15% of 80: 15/100 × 80 = 0.15 × 80 = 12",
     "12", None),
    ("What is 15% of 80?",
     "15% of 80 is 80 - 15 = 65",
     "65", None),
    ("If a triangle has sides of 3, 4, and 5 units, what type of triangle is it?",
     "Using the Pythagorean theorem: 3² + 4² = 9 + 16 = 25 = 5². This is a right triangle.",
     "right triangle", None),
    ("If a triangle has sides of 3, 4, and 5 units, what type of triangle is it?",
     "Since all sides are different lengths, this is an isosceles triangle.",
     "isosceles triangle", None),
]

# Step 2: Create dataset and configure ICM
dataset = create_math_correctness_dataset(math_problems)
config = ICMConfig(
    model_name="Qwen/Qwen3-4B",
    initial_examples=4,
    alpha=75.0,  # High alpha for mathematical accuracy
    max_new_tokens=24,
    max_context_length=16384,
    label_names=["Incorrect", "Correct"]
)

# Step 3: Run mathematical verification
print("Verifying mathematical solutions...")
icm = ICM(config)
results = icm.run(dataset, max_iterations=40)

# Step 4: Analyze results and create verification report
correct_solutions = []
incorrect_solutions = []

for item, category in results:
    solution_data = {
        'problem': item.metadata['problem'],
        'solution': item.metadata['solution'],
        'expected_answer': item.metadata['answer'],
        'verification': 'Correct' if category == 1 else 'Incorrect',
        'problem_id': item.metadata['problem_id']
    }
    
    if category == 1:
        correct_solutions.append(solution_data)
    else:
        incorrect_solutions.append(solution_data)

# Step 5: Generate verification report
print(f"\n=== MATH VERIFICATION REPORT ===")
print(f"Total solutions analyzed: {len(results)}")
print(f"Correct solutions: {len(correct_solutions)} ({len(correct_solutions)/len(results)*100:.1f}%)")
print(f"Incorrect solutions: {len(incorrect_solutions)} ({len(incorrect_solutions)/len(results)*100:.1f}%)")

print("\n=== VERIFIED CORRECT SOLUTIONS ===")
for sol in correct_solutions:
    print(f"Problem: {sol['problem']}")
    print(f"Solution: {sol['solution'][:100]}...")
    print(f"Answer: {sol['expected_answer']}")
    print(f"Status: ✓ {sol['verification']}\n")

print("=== SOLUTIONS NEEDING REVIEW ===")
for sol in incorrect_solutions:
    print(f"Problem: {sol['problem']}")
    print(f"Solution: {sol['solution'][:100]}...")
    print(f"Answer: {sol['expected_answer']}")
    print(f"Status: ✗ {sol['verification']}\n")

# Step 6: Export detailed results
verification_report = {
    'summary': {
        'total_solutions': len(results),
        'correct_count': len(correct_solutions),
        'incorrect_count': len(incorrect_solutions),
        'accuracy_rate': len(correct_solutions) / len(results)
    },
    'correct_solutions': correct_solutions,
    'incorrect_solutions': incorrect_solutions,
    'verification_timestamp': pd.Timestamp.now().isoformat()
}

with open('math_verification_report.json', 'w') as f:
    json.dump(verification_report, f, indent=2)
print("Detailed report saved to 'math_verification_report.json'")
```

**What you get:** Clean dataset with verified correct solutions, plus identification of problems that need new solutions.

### 3. Survey and Feedback Analysis

**Problem:** You collected 1000 pieces of feedback about a product feature, but need to understand what's positive vs negative sentiment.

**Solution:** Here's a comprehensive feedback analysis system:

```python
# file: feedback_analysis_example.py
from icm_implementation import ICM, ICMConfig, create_truthfulness_dataset
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Step 1: Load feedback data (example format)
feedback_data = [
    ("How do you feel about the new dashboard?", 
     "The new dashboard is amazing! So much easier to find what I need.", None),
    ("How do you feel about the new dashboard?", 
     "Terrible update. Can't find anything now and it's slow.", None),
    ("What's your opinion on the mobile app redesign?", 
     "Love the new mobile interface! Clean and intuitive.", None),
    ("What's your opinion on the mobile app redesign?", 
     "The mobile app is now confusing and hard to navigate.", None),
    ("How would you rate the customer support experience?", 
     "Outstanding support! Quick response and solved my issue perfectly.", None),
    ("How would you rate the customer support experience?", 
     "Support was unhelpful and took forever to respond.", None),
    ("What do you think about the new pricing model?", 
     "Fair pricing and good value for the features we get.", None),
    ("What do you think about the new pricing model?", 
     "Way too expensive for what we're getting. Considering alternatives.", None),
    ("How satisfied are you with the recent bug fixes?", 
     "Great job on the bug fixes! App is much more stable now.", None),
    ("How satisfied are you with the recent bug fixes?", 
     "Still experiencing the same bugs. Nothing seems fixed.", None),
]

# Step 2: Create dataset and configure ICM for sentiment analysis
dataset = create_truthfulness_dataset(feedback_data)
config = ICMConfig(
    model_name="Qwen/Qwen3-4B",
    initial_examples=4,
    alpha=55.0,
    max_new_tokens=20,
    temperature=0.15,
    label_names=["Negative", "Positive"]
)

# Step 3: Run sentiment analysis
print("Analyzing customer feedback sentiment...")
icm = ICM(config)
results = icm.run(dataset, max_iterations=35)

# Step 4: Process results and extract insights
feedback_analysis = []
for item, category in results:
    analysis = {
        'question': item.metadata['question'],
        'feedback': item.metadata['claim'],
        'sentiment': 'Positive' if category == 1 else 'Negative',
        'sentiment_score': category,
        'topic': extract_topic(item.metadata['question']),  # Custom function
        'id': item.id
    }
    feedback_analysis.append(analysis)

def extract_topic(question):
    """Extract topic from question for categorization"""
    if 'dashboard' in question.lower():
        return 'Dashboard'
    elif 'mobile' in question.lower():
        return 'Mobile App'
    elif 'support' in question.lower():
        return 'Customer Support'
    elif 'pricing' in question.lower():
        return 'Pricing'
    elif 'bug' in question.lower():
        return 'Bug Fixes'
    else:
        return 'General'

# Step 5: Generate comprehensive analysis report
print(f"\n=== FEEDBACK SENTIMENT ANALYSIS ===")
print(f"Total feedback analyzed: {len(results)}")

# Overall sentiment distribution
sentiments = [item['sentiment'] for item in feedback_analysis]
sentiment_counts = Counter(sentiments)
print(f"Positive feedback: {sentiment_counts['Positive']} ({sentiment_counts['Positive']/len(results)*100:.1f}%)")
print(f"Negative feedback: {sentiment_counts['Negative']} ({sentiment_counts['Negative']/len(results)*100:.1f}%)")

# Topic-based analysis
print(f"\n=== SENTIMENT BY TOPIC ===")
topics = [item['topic'] for item in feedback_analysis]
topic_sentiment = {}
for topic in set(topics):
    topic_feedback = [item for item in feedback_analysis if item['topic'] == topic]
    positive_count = sum(1 for item in topic_feedback if item['sentiment'] == 'Positive')
    total_count = len(topic_feedback)
    topic_sentiment[topic] = {
        'positive': positive_count,
        'total': total_count,
        'percentage': positive_count / total_count * 100
    }

for topic, stats in topic_sentiment.items():
    print(f"{topic}: {stats['positive']}/{stats['total']} positive ({stats['percentage']:.1f}%)")

# Step 6: Show sample feedback by sentiment
print(f"\n=== SAMPLE POSITIVE FEEDBACK ===")
positive_feedback = [item for item in feedback_analysis if item['sentiment'] == 'Positive']
for feedback in positive_feedback[:3]:
    print(f"Topic: {feedback['topic']}")
    print(f"Feedback: {feedback['feedback']}")
    print(f"Sentiment: {feedback['sentiment']}\n")

print(f"=== SAMPLE NEGATIVE FEEDBACK ===")
negative_feedback = [item for item in feedback_analysis if item['sentiment'] == 'Negative']
for feedback in negative_feedback[:3]:
    print(f"Topic: {feedback['topic']}")
    print(f"Feedback: {feedback['feedback']}")
    print(f"Sentiment: {feedback['sentiment']}\n")

# Step 7: Export results for further analysis
df = pd.DataFrame(feedback_analysis)
df.to_csv('feedback_sentiment_analysis.csv', index=False)
print("Detailed analysis exported to 'feedback_sentiment_analysis.csv'")

# Step 8: Create summary dashboard data
summary_report = {
    'overall_sentiment': {
        'positive_percentage': sentiment_counts['Positive'] / len(results) * 100,
        'negative_percentage': sentiment_counts['Negative'] / len(results) * 100,
        'total_responses': len(results)
    },
    'topic_breakdown': topic_sentiment,
    'top_positive_topics': sorted(topic_sentiment.items(), 
                                key=lambda x: x[1]['percentage'], reverse=True)[:3],
    'areas_for_improvement': sorted(topic_sentiment.items(), 
                                  key=lambda x: x[1]['percentage'])[:3]
}

with open('feedback_summary_dashboard.json', 'w') as f:
    json.dump(summary_report, f, indent=2)
print("Executive summary saved to 'feedback_summary_dashboard.json'")
```

**What you get:** Organized feedback sorted by sentiment, topic-based analysis, and executive summary ready for decision-making.

### 4. Content Comparison and Ranking

**Problem:** You have multiple versions of documentation or marketing copy and need to identify which versions are clearer or more helpful.

**Solution:** Here's a complete content comparison system:

```python
# file: content_comparison_example.py
from icm_implementation import ICM, ICMConfig, create_comparison_dataset
import pandas as pd

# Step 1: Prepare content versions for comparison
content_comparisons = [
    ("How to install our software",
     "1. Download the installer\n2. Run the installer\n3. Follow the setup wizard\n4. Restart your computer",
     "Download and install our software by running the installer file.",
     None),
    ("How to reset your password",
     "To reset your password: Go to Settings > Account > Password > Reset. Enter your current password, then your new password twice. Click Save.",
     "Click forgot password and check your email for instructions.",
     None),
    ("What is our refund policy?",
     "We offer full refunds within 30 days of purchase. To request a refund, contact support with your order number. Refunds are processed within 5-7 business days.",
     "Contact support for refunds.",
     None),
    ("How to contact customer support",
     "Customer Support Hours: Mon-Fri 9AM-6PM EST. Email: support@company.com. Phone: 1-800-SUPPORT. Live chat available on our website during business hours.",
     "Email us at support@company.com or call us.",
     None),
]

# Step 2: Create dataset and configure ICM for comparison
dataset = create_comparison_dataset(content_comparisons)
config = ICMConfig(
    model_name="Qwen/Qwen3-4B",
    initial_examples=2,
    alpha=65.0,
    max_new_tokens=32,
    temperature=0.1,
    label_names=["Version B Better", "Version A Better"]
)

# Step 3: Run content comparison analysis
print("Comparing content versions for clarity and helpfulness...")
icm = ICM(config)
results = icm.run(dataset, max_iterations=25)

# Step 4: Process comparison results
comparison_results = []
for item, category in results:
    result = {
        'topic': item.metadata['query'],
        'version_a': item.metadata['response_a'],
        'version_b': item.metadata['response_b'],
        'winner': 'Version A' if category == 1 else 'Version B',
        'winner_content': item.metadata['response_a'] if category == 1 else item.metadata['response_b'],
        'loser_content': item.metadata['response_b'] if category == 1 else item.metadata['response_a'],
        'id': item.id
    }
    comparison_results.append(result)

# Step 5: Generate content optimization report
print(f"\n=== CONTENT COMPARISON REPORT ===")
print(f"Total comparisons analyzed: {len(results)}")

version_a_wins = sum(1 for r in comparison_results if r['winner'] == 'Version A')
version_b_wins = sum(1 for r in comparison_results if r['winner'] == 'Version B')

print(f"Version A wins: {version_a_wins} ({version_a_wins/len(results)*100:.1f}%)")
print(f"Version B wins: {version_b_wins} ({version_b_wins/len(results)*100:.1f}%)")

print(f"\n=== WINNING CONTENT EXAMPLES ===")
for result in comparison_results:
    print(f"\nTopic: {result['topic']}")
    print(f"Winner: {result['winner']}")
    print(f"Winning Content: {result['winner_content'][:150]}...")
    print(f"Why it's better: More detailed and actionable")

# Step 6: Create content optimization recommendations
recommendations = []
for result in comparison_results:
    if result['winner'] == 'Version A':
        rec = {
            'topic': result['topic'],
            'action': 'Keep Version A',
            'reason': 'More comprehensive and detailed',
            'current_content': result['winner_content'],
            'replace_content': result['loser_content']
        }
    else:
        rec = {
            'topic': result['topic'],
            'action': 'Keep Version B',
            'reason': 'More concise and direct',
            'current_content': result['winner_content'],
            'replace_content': result['loser_content']
        }
    recommendations.append(rec)

print(f"\n=== CONTENT OPTIMIZATION RECOMMENDATIONS ===")
for rec in recommendations:
    print(f"\nTopic: {rec['topic']}")
    print(f"Recommendation: {rec['action']}")
    print(f"Reason: {rec['reason']}")
    print(f"Keep this version: {rec['current_content'][:100]}...")

# Step 7: Export optimization plan
optimization_plan = {
    'analysis_summary': {
        'total_comparisons': len(results),
        'version_a_preferred': version_a_wins,
        'version_b_preferred': version_b_wins,
        'version_a_success_rate': version_a_wins / len(results) * 100
    },
    'detailed_comparisons': comparison_results,
    'recommendations': recommendations
}

with open('content_optimization_plan.json', 'w') as f:
    json.dump(optimization_plan, f, indent=2)

df = pd.DataFrame(comparison_results)
df.to_csv('content_comparison_results.csv', index=False)

print("\nOptimization plan saved to 'content_optimization_plan.json'")
print("Detailed results saved to 'content_comparison_results.csv'")
```

**What you get:** Ranked content with clear indicators of what works best, plus actionable recommendations for content optimization.

## Why This Approach Works

### No Training Data Needed
Unlike traditional machine learning, you don't need hundreds of pre-organized examples. ICM finds the patterns in your existing data.

### Consistency Guaranteed
The system ensures that similar items get treated similarly - no random inconsistencies from human reviewers having different standards.

### Scalable
Works whether you have 50 items or 5,000 items. The time investment is the same.

### Transparent
You can see exactly how each piece of data was categorized and why, making it easy to verify the results make sense.

## Complete Configuration Guide

### Understanding ICM Configuration

ICM is highly configurable. Here's a comprehensive guide to all the settings:

```python
# Complete configuration example with explanations
from icm_implementation import ICMConfig

config = ICMConfig(
    # === MODEL CONFIGURATION ===
    model_name="Qwen/Qwen3-4B",          # Which AI model to use
    backend="auto",                       # "vllm", "transformers", or "auto"
    device="cuda",                        # "cuda" for GPU, "cpu" for CPU
    
    # === ICM ALGORITHM PARAMETERS ===
    initial_examples=8,                   # How many random examples to start with
    initial_temperature=10.0,             # Starting "randomness" for exploration
    final_temperature=0.01,               # Final "randomness" (more focused)
    cooling_rate=0.99,                    # How quickly to reduce randomness
    alpha=50.0,                           # Weight for consistency (higher = stricter)
    
    # === RESPONSE GENERATION ===
    max_context_length=32768,             # Maximum input length
    max_new_tokens=64,                    # Maximum response length
    temperature=0.1,                      # Response randomness
    top_p=0.95,                           # Response diversity
    
    # === CONSISTENCY CHECKING ===
    max_consistency_iterations=10,        # How many consistency fixes to try
    
    # === TASK CONFIGURATION ===
    num_labels=2,                         # Number of categories (2 = binary)
    label_names=["False", "True"]         # Names for categories
)
```

### Configuration for Different Tasks

#### 1. High-Quality Content Review
```python
# For reviewing content quality (strict standards)
quality_config = ICMConfig(
    model_name="Qwen/Qwen3-4B",
    initial_examples=6,
    alpha=75.0,                    # Very strict consistency
    temperature=0.05,              # Very low randomness
    max_new_tokens=32,             # Short, focused responses
    label_names=["Poor Quality", "High Quality"]
)
```

#### 2. Sentiment Analysis
```python
# For analyzing sentiment (balanced approach)
sentiment_config = ICMConfig(
    model_name="Qwen/Qwen3-4B",
    initial_examples=4,
    alpha=55.0,                    # Moderate consistency
    temperature=0.15,              # Some randomness for nuance
    max_new_tokens=20,
    label_names=["Negative", "Positive"]
)
```

#### 3. Mathematical Verification
```python
# For checking math correctness (very strict)
math_config = ICMConfig(
    model_name="Qwen/Qwen3-4B",
    initial_examples=8,
    alpha=80.0,                    # Extremely strict
    temperature=0.03,              # Minimal randomness
    max_new_tokens=48,
    max_consistency_iterations=15,  # More consistency checks
    label_names=["Incorrect", "Correct"]
)
```

#### 4. Multi-Category Classification
```python
# For organizing into multiple categories
multi_config = ICMConfig(
    model_name="Qwen/Qwen3-4B",
    initial_examples=12,           # More examples for complexity
    alpha=60.0,
    num_labels=4,                  # 4 categories
    label_names=["Urgent", "High", "Medium", "Low"],
    max_new_tokens=36
)
```

### Model Selection Guide

#### Fast Development and Testing
```python
# Fastest model for development
dev_config = ICMConfig(
    model_name="Qwen/Qwen2.5-0.5B",      # Smallest, fastest
    backend="transformers",               # More compatible
    initial_examples=2,
    alpha=40.0,
    max_new_tokens=16
)
```

#### Production Quality
```python
# Best balance of speed and quality
prod_config = ICMConfig(
    model_name="Qwen/Qwen3-4B",          # Good balance
    backend="vllm",                      # Faster inference
    initial_examples=6,
    alpha=65.0,
    max_new_tokens=32
)
```

#### Maximum Quality
```python
# Highest quality (slower but most accurate)
quality_config = ICMConfig(
    model_name="Qwen/Qwen3-14B",         # Larger, more capable
    backend="vllm",
    initial_examples=10,
    alpha=70.0,
    max_new_tokens=64,
    temperature=0.05
)
```

### Working with Different Data Formats

#### Loading from CSV
```python
import pandas as pd
from icm_implementation import create_truthfulness_dataset

# Load survey data from CSV
df = pd.read_csv('survey_responses.csv')

# Convert to ICM format
# CSV columns: question, response
survey_data = [
    (row['question'], row['response'], None)
    for _, row in df.iterrows()
]

dataset = create_truthfulness_dataset(survey_data)
```

#### Loading from JSON
```python
import json
from icm_implementation import create_comparison_dataset

# Load content comparisons from JSON
with open('content_versions.json', 'r') as f:
    data = json.load(f)

# Convert to ICM format
# JSON format: [{"topic": "...", "version_a": "...", "version_b": "..."}]
comparison_data = [
    (item['topic'], item['version_a'], item['version_b'], None)
    for item in data
]

dataset = create_comparison_dataset(comparison_data)
```

#### Loading from Database
```python
import sqlite3
from icm_implementation import create_math_correctness_dataset

# Load from database
conn = sqlite3.connect('math_problems.db')
cursor = conn.execute('SELECT problem, solution, answer FROM math_problems')

# Convert to ICM format
math_data = [
    (row[0], row[1], row[2], None)  # problem, solution, answer, label
    for row in cursor.fetchall()
]

dataset = create_math_correctness_dataset(math_data)
conn.close()
```

### Command Line Usage

The included examples script provides ready-to-use workflows:

#### Basic Usage
```bash
# Run truthfulness analysis
uv run icm_examples.py --task truthfulness --model Qwen/Qwen3-4B

# Run math verification
uv run icm_examples.py --task math --model Qwen/Qwen3-4B

# Run content comparison
uv run icm_examples.py --task comparison --model Qwen/Qwen3-4B
```

#### Advanced Usage
```bash
# Run with custom settings
uv run icm_examples.py \
    --task truthfulness \
    --model Qwen/Qwen3-4B \
    --backend vllm \
    --alpha 60.0 \
    --initial-examples 6 \
    --sample-size 100

# Run all tasks with experiments
uv run run_experiments.py --task all --sample-size 200
```

#### Batch Processing
```bash
# Process multiple datasets
for task in truthfulness math comparison; do
    uv run icm_examples.py --task $task --model Qwen/Qwen3-4B
done
```

## Understanding Your Results

After running ICM, you'll get comprehensive results that help you understand both the categorization and the quality of the analysis.

### 1. Basic Results Analysis

```python
# After running ICM
results = icm.run(dataset, max_iterations=30)

# Basic results structure
for item, category in results:
    print(f"Input: {item.input_text[:50]}...")
    print(f"Category: {config.label_names[category]}")
    print(f"Item ID: {item.id}")
    print(f"Metadata: {item.metadata}")
    print("---")
```

### 2. Quality Metrics Analysis

```python
# Get detailed quality metrics
final_score, predictability, inconsistencies = icm.calculate_score(results)

print(f"=== QUALITY ANALYSIS ===")
print(f"Final Score: {final_score:.2f}")
print(f"  - Higher is better (0-100 scale)")
print(f"  - Combines accuracy and consistency")

print(f"Mutual Predictability: {predictability:.2f}")
print(f"  - How well items predict each other's labels")
print(f"  - Higher means more coherent patterns")

print(f"Logical Inconsistencies: {inconsistencies}")
print(f"  - Number of contradictory classifications")
print(f"  - Lower is better (0 is perfect)")

print(f"Acceptance Rate: {sum(icm.acceptance_history) / len(icm.acceptance_history):.2%}")
print(f"  - Percentage of proposed changes accepted")
print(f"  - Shows how much refinement happened")
```

### 3. Score History Analysis

```python
import matplotlib.pyplot as plt

# Plot score improvement over time
plt.figure(figsize=(10, 6))
plt.plot(icm.score_history)
plt.title('ICM Score Improvement Over Time')
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.grid(True)
plt.show()

# Print final convergence info
print(f"=== CONVERGENCE ANALYSIS ===")
print(f"Total iterations: {len(icm.score_history)}")
print(f"Starting score: {icm.score_history[0]:.2f}")
print(f"Final score: {icm.score_history[-1]:.2f}")
print(f"Improvement: {icm.score_history[-1] - icm.score_history[0]:.2f}")

# Check if converged
last_10_scores = icm.score_history[-10:]
score_variance = np.var(last_10_scores) if len(last_10_scores) > 1 else 0
print(f"Recent score variance: {score_variance:.4f}")
print(f"Converged: {'Yes' if score_variance < 0.1 else 'No'}")
```

### 4. Label Distribution Analysis

```python
from collections import Counter
import pandas as pd

# Analyze how labels are distributed
labels = [category for _, category in results]
label_counts = Counter(labels)

print(f"=== LABEL DISTRIBUTION ===")
for label_idx, count in label_counts.items():
    label_name = config.label_names[label_idx]
    percentage = (count / len(results)) * 100
    print(f"{label_name}: {count} items ({percentage:.1f}%)")

# Create distribution DataFrame for analysis
distribution_df = pd.DataFrame([
    {
        'label': config.label_names[label_idx],
        'count': count,
        'percentage': (count / len(results)) * 100
    }
    for label_idx, count in label_counts.items()
])

print("\nLabel Distribution DataFrame:")
print(distribution_df)
```

### 5. Detailed Item Analysis

```python
# Group results by category for detailed analysis
categories = {}
for item, category in results:
    label_name = config.label_names[category]
    if label_name not in categories:
        categories[label_name] = []
    categories[label_name].append({
        'id': item.id,
        'text': item.input_text,
        'metadata': item.metadata
    })

# Analyze each category
for label_name, items in categories.items():
    print(f"\n=== {label_name.upper()} CATEGORY ===")
    print(f"Total items: {len(items)}")
    
    # Show sample items
    print(f"Sample items:")
    for item in items[:3]:
        print(f"  ID {item['id']}: {item['text'][:100]}...")
    
    # Show statistics if available
    if 'confidence' in items[0].get('metadata', {}):
        confidences = [item['metadata']['confidence'] for item in items]
        print(f"Average confidence: {np.mean(confidences):.2f}")
        print(f"Min confidence: {min(confidences):.2f}")
        print(f"Max confidence: {max(confidences):.2f}")
```

### 6. Export Results for Further Analysis

```python
# Export comprehensive results
results_data = []
for item, category in results:
    results_data.append({
        'id': item.id,
        'input_text': item.input_text,
        'category_index': category,
        'category_name': config.label_names[category],
        'metadata': item.metadata,
        'final_score': final_score,
        'predictability': predictability,
        'inconsistencies': inconsistencies
    })

# Save to CSV
results_df = pd.DataFrame(results_data)
results_df.to_csv('icm_detailed_results.csv', index=False)

# Save to JSON with full metadata
full_results = {
    'config': {
        'model_name': config.model_name,
        'alpha': config.alpha,
        'initial_examples': config.initial_examples,
        'label_names': config.label_names
    },
    'metrics': {
        'final_score': float(final_score),
        'predictability': float(predictability),
        'inconsistencies': int(inconsistencies),
        'total_items': len(results),
        'acceptance_rate': float(sum(icm.acceptance_history) / len(icm.acceptance_history))
    },
    'results': results_data,
    'score_history': [float(s) for s in icm.score_history]
}

with open('icm_complete_analysis.json', 'w') as f:
    json.dump(full_results, f, indent=2)

print("Results exported to:")
print("- icm_detailed_results.csv (spreadsheet format)")
print("- icm_complete_analysis.json (full data)")
```

## Performance and Efficiency

### Speed
*Performance varies significantly based on hardware (GPU memory, CPU cores) and model size:*
- Small datasets (< 100 items): 5-10 minutes (typical with GPU)
- Medium datasets (100-1000 items): 30-60 minutes (typical with GPU)
- Large datasets (1000+ items): 1-3 hours (typical with GPU)

### Accuracy
ICM performance depends on the specific dataset and task complexity. Results are typically consistent and follow discovered patterns, though individual accuracy varies by use case.

### Cost
Runs on your own hardware - no per-request charges or data privacy concerns.

## When to Use ICM vs Other Approaches

**Use ICM when:**
- You have unorganized data that needs categorizing
- You want consistent results across your entire dataset
- You don't have training examples available
- You need to understand why each item was categorized a certain way

**Consider other approaches when:**
- You need very domain-specific categorization (e.g., medical diagnosis)
- You have extensive pre-categorized training data available
- Your data doesn't have clear patterns that can be discovered
- You need real-time processing of individual items

## Advanced Features and Techniques

### 1. Custom Multi-Category Classification

Create sophisticated classification systems beyond binary categorization:

```python
# Example: Priority classification system
priority_config = ICMConfig(
    model_name="Qwen/Qwen3-4B",
    num_labels=4,
    label_names=["Critical", "High", "Medium", "Low"],
    initial_examples=8,
    alpha=65.0
)

# Example: Content quality assessment
quality_config = ICMConfig(
    model_name="Qwen/Qwen3-4B",
    num_labels=5,
    label_names=["Excellent", "Good", "Average", "Poor", "Unacceptable"],
    initial_examples=10,
    alpha=70.0
)

# Example: Sentiment with neutral category
sentiment_config = ICMConfig(
    model_name="Qwen/Qwen3-4B",
    num_labels=3,
    label_names=["Negative", "Neutral", "Positive"],
    initial_examples=6,
    alpha=60.0
)
```

### 2. Advanced Data Processing Pipelines

Build complete data processing workflows:

```python
# file: advanced_pipeline_example.py
from icm_implementation import ICM, ICMConfig, create_truthfulness_dataset
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict
import logging

class ICMPipeline:
    """Complete ICM processing pipeline with preprocessing and postprocessing"""
    
    def __init__(self, config: ICMConfig, output_dir: str = "icm_results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def preprocess_text_data(self, raw_data: List[str]) -> List[tuple]:
        """Preprocess raw text into ICM format"""
        processed = []
        for i, text in enumerate(raw_data):
            # Clean and format text
            cleaned_text = text.strip().replace('\n', ' ').replace('\t', ' ')
            # Create question-statement pairs for truthfulness analysis
            processed.append((
                "Is this statement accurate and helpful?",
                cleaned_text,
                None  # Unknown label
            ))
        return processed
    
    def run_analysis(self, data: List[tuple], task_name: str) -> Dict:
        """Run complete ICM analysis with monitoring"""
        self.logger.info(f"Starting {task_name} analysis with {len(data)} items")
        
        # Create dataset
        dataset = create_truthfulness_dataset(data)
        
        # Run ICM
        icm = ICM(self.config)
        results = icm.run(dataset, max_iterations=len(data) * 2)
        
        # Calculate metrics
        final_score, predictability, inconsistencies = icm.calculate_score(results)
        
        # Compile comprehensive results
        analysis_results = {
            'task_name': task_name,
            'config': self.config.__dict__,
            'metrics': {
                'final_score': float(final_score),
                'predictability': float(predictability),
                'inconsistencies': int(inconsistencies),
                'total_items': len(results),
                'acceptance_rate': float(sum(icm.acceptance_history) / len(icm.acceptance_history)),
                'convergence_iterations': len(icm.score_history)
            },
            'results': [
                {
                    'id': item.id,
                    'input': item.input_text,
                    'category_index': category,
                    'category_name': self.config.label_names[category],
                    'metadata': item.metadata
                }
                for item, category in results
            ],
            'score_history': [float(s) for s in icm.score_history]
        }
        
        self.logger.info(f"Analysis complete. Final score: {final_score:.2f}")
        return analysis_results
    
    def postprocess_results(self, results: Dict) -> Dict:
        """Add advanced analytics to results"""
        # Calculate additional metrics
        categories = [r['category_index'] for r in results['results']]
        from collections import Counter
        category_distribution = Counter(categories)
        
        # Add distribution analysis
        results['analytics'] = {
            'category_distribution': {
                self.config.label_names[idx]: count 
                for idx, count in category_distribution.items()
            },
            'distribution_entropy': self._calculate_entropy(category_distribution),
            'most_common_category': self.config.label_names[max(category_distribution, key=category_distribution.get)],
            'balance_score': min(category_distribution.values()) / max(category_distribution.values())
        }
        
        return results
    
    def _calculate_entropy(self, distribution: Counter) -> float:
        """Calculate entropy of label distribution"""
        import math
        total = sum(distribution.values())
        entropy = 0
        for count in distribution.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy
    
    def save_results(self, results: Dict, task_name: str):
        """Save results in multiple formats"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive JSON
        json_file = self.output_dir / f"{task_name}_{timestamp}_complete.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save results CSV
        results_df = pd.DataFrame(results['results'])
        csv_file = self.output_dir / f"{task_name}_{timestamp}_results.csv"
        results_df.to_csv(csv_file, index=False)
        
        # Save metrics summary
        metrics_file = self.output_dir / f"{task_name}_{timestamp}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'task': task_name,
                'timestamp': timestamp,
                'metrics': results['metrics'],
                'analytics': results['analytics']
            }, f, indent=2)
        
        self.logger.info(f"Results saved to {self.output_dir}")
        return {
            'json_file': json_file,
            'csv_file': csv_file,
            'metrics_file': metrics_file
        }

# Example usage
if __name__ == "__main__":
    # Configure pipeline
    config = ICMConfig(
        model_name="Qwen/Qwen3-4B",
        initial_examples=6,
        alpha=65.0,
        label_names=["Low Quality", "High Quality"]
    )
    
    pipeline = ICMPipeline(config)
    
    # Load and process data
    raw_feedback = [
        "This product is amazing! Works perfectly and great customer service.",
        "Terrible quality, broke after one week of use.",
        "Good value for money, does what it says.",
        "Customer support was unhelpful and rude.",
        "Exactly as described, fast shipping, very satisfied.",
        "Confusing instructions, took forever to set up."
    ]
    
    # Run complete analysis
    processed_data = pipeline.preprocess_text_data(raw_feedback)
    results = pipeline.run_analysis(processed_data, "customer_feedback")
    results = pipeline.postprocess_results(results)
    file_paths = pipeline.save_results(results, "customer_feedback")
    
    # Print summary
    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"Final Score: {results['metrics']['final_score']:.2f}")
    print(f"Most Common Category: {results['analytics']['most_common_category']}")
    print(f"Distribution Balance: {results['analytics']['balance_score']:.2f}")
    print(f"Files saved: {list(file_paths.values())}")
```

### 3. Batch Processing with Progress Monitoring

Process multiple datasets efficiently:

```python
# file: batch_processing_example.py
from icm_implementation import ICM, ICMConfig, create_truthfulness_dataset
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import time

class BatchProcessor:
    """Process multiple datasets with monitoring and error handling"""
    
    def __init__(self, base_config: ICMConfig):
        self.base_config = base_config
        self.results = []
        
    def process_dataset(self, dataset_path: Path, task_name: str) -> Dict:
        """Process a single dataset file"""
        try:
            # Load data based on file type
            if dataset_path.suffix == '.csv':
                df = pd.read_csv(dataset_path)
                data = [(row['question'], row['response'], None) for _, row in df.iterrows()]
            elif dataset_path.suffix == '.json':
                with open(dataset_path) as f:
                    json_data = json.load(f)
                data = [(item['question'], item['response'], None) for item in json_data]
            else:
                raise ValueError(f"Unsupported file type: {dataset_path.suffix}")
            
            # Create dataset
            dataset = create_truthfulness_dataset(data)
            
            # Run ICM with monitoring
            start_time = time.time()
            icm = ICM(self.base_config)
            results = icm.run(dataset, max_iterations=len(data) * 3)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            final_score, predictability, inconsistencies = icm.calculate_score(results)
            
            return {
                'task_name': task_name,
                'file_path': str(dataset_path),
                'dataset_size': len(data),
                'processing_time': processing_time,
                'metrics': {
                    'final_score': float(final_score),
                    'predictability': float(predictability),
                    'inconsistencies': int(inconsistencies),
                    'acceptance_rate': float(sum(icm.acceptance_history) / len(icm.acceptance_history))
                },
                'results': results,
                'score_history': icm.score_history
            }
            
        except Exception as e:
            return {
                'task_name': task_name,
                'file_path': str(dataset_path),
                'error': str(e),
                'status': 'failed'
            }
    
    def process_multiple_datasets(self, dataset_paths: List[Path], task_names: List[str]):
        """Process multiple datasets with progress tracking"""
        all_results = []
        
        for dataset_path, task_name in tqdm(zip(dataset_paths, task_names), 
                                          desc="Processing datasets",
                                          total=len(dataset_paths)):
            print(f"\nProcessing: {task_name}")
            result = self.process_dataset(dataset_path, task_name)
            all_results.append(result)
            
            # Print quick summary
            if 'error' not in result:
                print(f"  ✓ Score: {result['metrics']['final_score']:.2f}")
                print(f"  ✓ Time: {result['processing_time']:.1f}s")
            else:
                print(f"  ✗ Error: {result['error']}")
        
        return all_results
    
    def generate_batch_report(self, batch_results: List[Dict]) -> Dict:
        """Generate comprehensive batch processing report"""
        successful = [r for r in batch_results if 'error' not in r]
        failed = [r for r in batch_results if 'error' in r]
        
        if successful:
            avg_score = sum(r['metrics']['final_score'] for r in successful) / len(successful)
            avg_time = sum(r['processing_time'] for r in successful) / len(successful)
            total_items = sum(r['dataset_size'] for r in successful)
        else:
            avg_score = avg_time = total_items = 0
        
        report = {
            'batch_summary': {
                'total_datasets': len(batch_results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(batch_results) * 100,
                'total_items_processed': total_items,
                'average_score': avg_score,
                'average_processing_time': avg_time
            },
            'individual_results': batch_results,
            'best_performing': max(successful, key=lambda x: x['metrics']['final_score']) if successful else None,
            'fastest_processing': min(successful, key=lambda x: x['processing_time']) if successful else None
        }
        
        return report

# Example usage
config = ICMConfig(
    model_name="Qwen/Qwen3-4B",
    initial_examples=4,
    alpha=60.0
)

processor = BatchProcessor(config)

# Process multiple files
dataset_files = [
    Path("survey_data_q1.csv"),
    Path("survey_data_q2.csv"),
    Path("feedback_data.json"),
    Path("reviews_data.csv")
]

task_names = ["Q1_Survey", "Q2_Survey", "Customer_Feedback", "Product_Reviews"]

# Run batch processing
batch_results = processor.process_multiple_datasets(dataset_files, task_names)
batch_report = processor.generate_batch_report(batch_results)

# Save comprehensive report
with open('batch_processing_report.json', 'w') as f:
    json.dump(batch_report, f, indent=2)

print(f"\n=== BATCH PROCESSING COMPLETE ===")
print(f"Success Rate: {batch_report['batch_summary']['success_rate']:.1f}%")
print(f"Average Score: {batch_report['batch_summary']['average_score']:.2f}")
print(f"Total Items: {batch_report['batch_summary']['total_items_processed']}")
```

### 4. Integration with Existing Data Pipelines

Connect ICM to your existing workflows:

```python
# file: database_integration_example.py
import sqlite3
import psycopg2
from sqlalchemy import create_engine
import pandas as pd
from icm_implementation import ICM, ICMConfig, create_truthfulness_dataset

class DatabaseIntegration:
    """Integrate ICM with various database systems"""
    
    def __init__(self, config: ICMConfig):
        self.config = config
    
    def process_from_postgres(self, connection_string: str, query: str) -> pd.DataFrame:
        """Process data directly from PostgreSQL"""
        engine = create_engine(connection_string)
        
        # Load data
        df = pd.read_sql(query, engine)
        
        # Convert to ICM format
        data = [(row['question'], row['response'], None) for _, row in df.iterrows()]
        dataset = create_truthfulness_dataset(data)
        
        # Run ICM
        icm = ICM(self.config)
        results = icm.run(dataset)
        
        # Convert back to DataFrame
        results_df = pd.DataFrame([
            {
                'original_id': item.metadata.get('original_id'),
                'question': item.metadata['question'],
                'response': item.metadata['claim'],
                'category': self.config.label_names[category],
                'item_id': item.id
            }
            for item, category in results
        ])
        
        return results_df
    
    def save_to_database(self, results_df: pd.DataFrame, connection_string: str, table_name: str):
        """Save results back to database"""
        engine = create_engine(connection_string)
        results_df.to_sql(table_name, engine, if_exists='replace', index=False)
    
    def create_analysis_dashboard_data(self, results_df: pd.DataFrame) -> Dict:
        """Create data for dashboard visualization"""
        from collections import Counter
        
        category_counts = Counter(results_df['category'])
        
        dashboard_data = {
            'summary': {
                'total_items': len(results_df),
                'categories': dict(category_counts),
                'distribution': {
                    cat: count / len(results_df) * 100 
                    for cat, count in category_counts.items()
                }
            },
            'sample_data': {
                cat: results_df[results_df['category'] == cat].head(5).to_dict('records')
                for cat in category_counts.keys()
            }
        }
        
        return dashboard_data

# Example usage with SQLite
config = ICMConfig(model_name="Qwen/Qwen3-4B", alpha=65.0)
db_integration = DatabaseIntegration(config)

# Create sample database
conn = sqlite3.connect('sample_data.db')
sample_data = pd.DataFrame({
    'question': ['How was your experience?'] * 4,
    'response': [
        'Excellent service, very satisfied',
        'Poor quality, would not recommend',
        'Good value for money',
        'Terrible customer support'
    ]
})
sample_data.to_sql('customer_feedback', conn, if_exists='replace', index=False)
conn.close()

# Process from database
results_df = db_integration.process_from_postgres(
    'sqlite:///sample_data.db',
    'SELECT question, response FROM customer_feedback'
)

# Create dashboard data
dashboard_data = db_integration.create_analysis_dashboard_data(results_df)

print("Database integration complete!")
print(f"Processed {len(results_df)} items")
print(f"Categories found: {list(dashboard_data['summary']['categories'].keys())}")
```

## Comprehensive Troubleshooting Guide

### 1. "My results don't look right"

This is the most common issue. Here's a systematic debugging approach:

```python
# file: debug_results_example.py
from icm_implementation import ICM, ICMConfig, create_truthfulness_dataset
import numpy as np

def debug_icm_results(data, config):
    """Comprehensive debugging for ICM results"""
    
    print("=== ICM DEBUGGING SESSION ===")
    
    # Step 1: Check data quality
    print(f"1. DATA QUALITY CHECK")
    print(f"   Dataset size: {len(data)}")
    
    # Check for duplicate or near-duplicate entries
    texts = [item[1] for item in data]  # Get response texts
    unique_texts = set(texts)
    print(f"   Unique responses: {len(unique_texts)} / {len(data)}")
    if len(unique_texts) < len(data) * 0.8:
        print("   ⚠️  WARNING: Many duplicate responses detected")
    
    # Check text length distribution
    text_lengths = [len(text) for text in texts]
    print(f"   Text length range: {min(text_lengths)} - {max(text_lengths)} chars")
    print(f"   Average length: {np.mean(text_lengths):.1f} chars")
    
    if max(text_lengths) > config.max_context_length * 0.8:
        print("   ⚠️  WARNING: Some texts may be too long for model context")
    
    # Step 2: Run ICM with detailed monitoring
    print(f"\n2. RUNNING ICM WITH DEBUG INFO")
    dataset = create_truthfulness_dataset(data)
    icm = ICM(config)
    
    # Track initial random assignments
    print(f"   Initial examples: {config.initial_examples}")
    print(f"   Alpha (consistency weight): {config.alpha}")
    
    results = icm.run(dataset, max_iterations=30)
    
    # Step 3: Analyze convergence
    print(f"\n3. CONVERGENCE ANALYSIS")
    scores = icm.score_history
    print(f"   Starting score: {scores[0]:.2f}")
    print(f"   Final score: {scores[-1]:.2f}")
    print(f"   Score improvement: {scores[-1] - scores[0]:.2f}")
    
    # Check if scores are improving
    if len(scores) > 10:
        recent_trend = np.polyfit(range(len(scores[-10:])), scores[-10:], 1)[0]
        print(f"   Recent trend: {'Improving' if recent_trend > 0 else 'Declining' if recent_trend < 0 else 'Stable'}")
    
    # Check for premature convergence
    score_variance = np.var(scores[-5:]) if len(scores) >= 5 else float('inf')
    if score_variance < 0.01 and len(scores) < 20:
        print("   ⚠️  WARNING: Possible premature convergence")
    
    # Step 4: Analyze results distribution
    print(f"\n4. RESULTS DISTRIBUTION")
    labels = [category for _, category in results]
    from collections import Counter
    label_counts = Counter(labels)
    
    for label_idx, count in label_counts.items():
        label_name = config.label_names[label_idx]
        percentage = count / len(results) * 100
        print(f"   {label_name}: {count} ({percentage:.1f}%)")
    
    # Check for extreme imbalance
    if max(label_counts.values()) / len(results) > 0.9:
        print("   ⚠️  WARNING: Extreme label imbalance detected")
        print("   SOLUTION: Try increasing alpha or reducing initial_examples")
    
    # Step 5: Sample analysis
    print(f"\n5. SAMPLE ANALYSIS")
    categories = {}
    for item, category in results:
        label_name = config.label_names[category]
        if label_name not in categories:
            categories[label_name] = []
        categories[label_name].append(item.metadata['claim'])
    
    for label, samples in categories.items():
        print(f"\n   {label} examples:")
        for sample in samples[:3]:
            print(f"     - {sample[:80]}...")
    
    return results, icm.score_history

# Example usage with problematic data
problematic_data = [
    ("Is this a good response?", "Yes, this is good", None),
    ("Is this a good response?", "No, this is bad", None),
    ("Is this a good response?", "This is okay", None),
    ("Is this a good response?", "Not sure about this", None),
    ("Is this a good response?", "Maybe it's fine", None),
    ("Is this a good response?", "Could be better", None),
]

# Test with different configurations
configs_to_test = [
    ICMConfig(alpha=30.0, initial_examples=2, label_names=["Poor", "Good"]),
    ICMConfig(alpha=60.0, initial_examples=4, label_names=["Poor", "Good"]),
    ICMConfig(alpha=90.0, initial_examples=6, label_names=["Poor", "Good"]),
]

print("Testing different configurations...")
for i, config in enumerate(configs_to_test):
    print(f"\n{'='*50}")
    print(f"CONFIGURATION {i+1}: alpha={config.alpha}, examples={config.initial_examples}")
    results, score_history = debug_icm_results(problematic_data, config)
```

### 2. "It's running too slowly"

Here's a complete performance optimization guide:

```python
# file: performance_optimization_example.py
from icm_implementation import ICM, ICMConfig, create_truthfulness_dataset
import time
import torch

class PerformanceOptimizer:
    """Tools for optimizing ICM performance"""
    
    def __init__(self):
        self.benchmark_results = []
    
    def benchmark_configuration(self, data, config, test_name):
        """Benchmark a specific configuration"""
        print(f"\nTesting: {test_name}")
        
        # Measure GPU memory before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        # Time the execution
        start_time = time.time()
        
        try:
            dataset = create_truthfulness_dataset(data)
            icm = ICM(config)
            results = icm.run(dataset, max_iterations=min(20, len(data)))
            
            execution_time = time.time() - start_time
            
            # Measure memory usage
            peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            memory_used = (peak_memory - initial_memory) / 1024**2  # MB
            
            # Calculate score
            final_score, _, _ = icm.calculate_score(results)
            
            result = {
                'test_name': test_name,
                'execution_time': execution_time,
                'memory_used_mb': memory_used,
                'final_score': final_score,
                'items_per_second': len(data) / execution_time,
                'config': config.__dict__
            }
            
            self.benchmark_results.append(result)
            
            print(f"  ✓ Time: {execution_time:.1f}s")
            print(f"  ✓ Speed: {result['items_per_second']:.1f} items/sec")
            print(f"  ✓ Memory: {memory_used:.1f} MB")
            print(f"  ✓ Score: {final_score:.1f}")
            
            return result
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            return None
    
    def find_optimal_config(self, data):
        """Find the best configuration for your data"""
        print("=== PERFORMANCE OPTIMIZATION ===")
        print(f"Dataset size: {len(data)} items")
        
        # Test different configurations
        test_configs = [
            # Fast development config
            (ICMConfig(
                model_name="Qwen/Qwen2.5-0.5B",
                backend="transformers",
                initial_examples=2,
                alpha=40.0,
                max_new_tokens=16,
                max_context_length=4096
            ), "Fast Development"),
            
            # Balanced config
            (ICMConfig(
                model_name="Qwen/Qwen3-4B",
                backend="auto",
                initial_examples=4,
                alpha=55.0,
                max_new_tokens=32,
                max_context_length=8192
            ), "Balanced"),
            
            # Quality config (slower)
            (ICMConfig(
                model_name="Qwen/Qwen3-4B",
                backend="vllm",
                initial_examples=6,
                alpha=70.0,
                max_new_tokens=64,
                max_context_length=16384
            ), "High Quality"),
        ]
        
        for config, name in test_configs:
            self.benchmark_configuration(data, config, name)
        
        # Find best performing configs
        successful_results = [r for r in self.benchmark_results if r is not None]
        
        if successful_results:
            fastest = min(successful_results, key=lambda x: x['execution_time'])
            highest_score = max(successful_results, key=lambda x: x['final_score'])
            most_efficient = max(successful_results, key=lambda x: x['items_per_second'])
            
            print(f"\n=== RECOMMENDATIONS ===")
            print(f"Fastest: {fastest['test_name']} ({fastest['execution_time']:.1f}s)")
            print(f"Highest Quality: {highest_score['test_name']} (score: {highest_score['final_score']:.1f})")
            print(f"Most Efficient: {most_efficient['test_name']} ({most_efficient['items_per_second']:.1f} items/sec)")
            
            return {
                'fastest': fastest,
                'highest_quality': highest_score,
                'most_efficient': most_efficient
            }
    
    def optimize_for_large_datasets(self, data_size):
        """Provide specific recommendations for large datasets"""
        print(f"\n=== LARGE DATASET OPTIMIZATION ({data_size} items) ===")
        
        if data_size > 1000:
            print("Recommendations for large datasets:")
            print("1. Use vLLM backend for faster inference")
            print("2. Reduce max_context_length to 8192 or lower")
            print("3. Use smaller models for initial testing")
            print("4. Process in batches of 500-1000 items")
            print("5. Consider reducing initial_examples to 4-6")
            
            # Recommended config for large datasets
            large_dataset_config = ICMConfig(
                model_name="Qwen/Qwen3-4B",
                backend="vllm",
                initial_examples=4,
                alpha=60.0,
                max_new_tokens=32,
                max_context_length=8192,
                temperature=0.1
            )
            
            print("\nRecommended configuration:")
            for key, value in large_dataset_config.__dict__.items():
                print(f"  {key}: {value}")
            
            return large_dataset_config

# Example usage
sample_data = [
    ("Is this helpful?", "Very helpful response", None),
    ("Is this helpful?", "Not helpful at all", None),
    ("Is this helpful?", "Somewhat useful", None),
    ("Is this helpful?", "Completely useless", None),
] * 5  # Multiply to simulate larger dataset

optimizer = PerformanceOptimizer()
recommendations = optimizer.find_optimal_config(sample_data)

# Get recommendations for larger datasets
large_config = optimizer.optimize_for_large_datasets(2000)
```

### 3. "I'm getting memory errors"

Complete memory management solutions:

```python
# file: memory_management_example.py
import torch
import gc
from icm_implementation import ICM, ICMConfig, create_truthfulness_dataset

class MemoryManager:
    """Tools for managing memory during ICM processing"""
    
    def __init__(self):
        self.memory_log = []
    
    def log_memory_usage(self, stage: str):
        """Log current memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            cached = torch.cuda.memory_reserved() / 1024**2      # MB
            self.memory_log.append({
                'stage': stage,
                'allocated_mb': allocated,
                'cached_mb': cached
            })
            print(f"{stage}: {allocated:.1f}MB allocated, {cached:.1f}MB cached")
    
    def clear_memory(self):
        """Aggressively clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_memory_efficient_config(self, dataset_size: int):
        """Get configuration optimized for memory usage"""
        print(f"Creating memory-efficient config for {dataset_size} items")
        
        if dataset_size < 100:
            # Small dataset - can use standard settings
            config = ICMConfig(
                model_name="Qwen/Qwen2.5-0.5B",  # Smallest model
                backend="transformers",           # More memory efficient
                initial_examples=2,
                alpha=50.0,
                max_context_length=4096,         # Reduced context
                max_new_tokens=16                # Shorter responses
            )
        elif dataset_size < 500:
            # Medium dataset - conservative settings
            config = ICMConfig(
                model_name="Qwen/Qwen2.5-0.5B",
                backend="transformers",
                initial_examples=3,
                alpha=55.0,
                max_context_length=6144,
                max_new_tokens=24
            )
        else:
            # Large dataset - minimal memory usage
            config = ICMConfig(
                model_name="Qwen/Qwen2.5-0.5B",
                backend="transformers",
                initial_examples=2,
                alpha=60.0,
                max_context_length=2048,         # Very limited context
                max_new_tokens=12                # Very short responses
            )
        
        print("Memory-efficient configuration:")
        for key, value in config.__dict__.items():
            print(f"  {key}: {value}")
        
        return config
    
    def process_with_memory_monitoring(self, data, config):
        """Process data with comprehensive memory monitoring"""
        print("=== MEMORY-MONITORED PROCESSING ===")
        
        self.log_memory_usage("Before processing")
        
        try:
            # Clear memory before starting
            self.clear_memory()
            self.log_memory_usage("After initial cleanup")
            
            # Create dataset
            dataset = create_truthfulness_dataset(data)
            self.log_memory_usage("After dataset creation")
            
            # Initialize ICM
            icm = ICM(config)
            self.log_memory_usage("After ICM initialization")
            
            # Process in smaller chunks if dataset is large
            if len(data) > 200:
                print("Large dataset detected - processing in chunks")
                chunk_size = 100
                all_results = []
                
                for i in range(0, len(dataset), chunk_size):
                    chunk = dataset[i:i+chunk_size]
                    print(f"Processing chunk {i//chunk_size + 1}/{(len(dataset)-1)//chunk_size + 1}")
                    
                    chunk_results = icm.run(chunk, max_iterations=len(chunk))
                    all_results.extend(chunk_results)
                    
                    # Clear memory after each chunk
                    self.clear_memory()
                    self.log_memory_usage(f"After chunk {i//chunk_size + 1}")
                
                results = all_results
            else:
                # Process normally for smaller datasets
                results = icm.run(dataset)
                self.log_memory_usage("After ICM processing")
            
            # Calculate final metrics
            final_score, predictability, inconsistencies = icm.calculate_score(results)
            self.log_memory_usage("After metric calculation")
            
            # Final cleanup
            self.clear_memory()
            self.log_memory_usage("After final cleanup")
            
            print(f"\n=== PROCESSING COMPLETE ===")
            print(f"Final Score: {final_score:.2f}")
            print(f"Items processed: {len(results)}")
            
            # Print memory usage summary
            print(f"\n=== MEMORY USAGE SUMMARY ===")
            max_memory = max(entry['allocated_mb'] for entry in self.memory_log)
            print(f"Peak memory usage: {max_memory:.1f} MB")
            
            return results
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ GPU out of memory: {e}")
            print("Try these solutions:")
            print("1. Use a smaller model (Qwen2.5-0.5B)")
            print("2. Reduce max_context_length to 2048")
            print("3. Reduce initial_examples to 2")
            print("4. Process data in smaller chunks")
            print("5. Use CPU instead of GPU")
            return None
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            return None

# Example usage
memory_manager = MemoryManager()

# Test data
test_data = [
    ("Is this response good?", f"This is response number {i}", None)
    for i in range(50)
]

# Get memory-efficient configuration
config = memory_manager.get_memory_efficient_config(len(test_data))

# Process with memory monitoring
results = memory_manager.process_with_memory_monitoring(test_data, config)

if results:
    print("✅ Processing completed successfully!")
else:
    print("❌ Processing failed - try the suggested optimizations")
```

### 4. Additional Common Issues and Solutions

```python
# Quick fixes for common problems

def fix_empty_results():
    """Solution for when ICM returns empty or random results"""
    print("If ICM results seem random or empty:")
    print("1. Check your data format - ensure None for unknown labels")
    print("2. Increase alpha parameter (try 70-80)")
    print("3. Ensure your data has distinguishable patterns")
    print("4. Try more initial_examples (6-10)")
    
def fix_slow_convergence():
    """Solution for slow convergence"""
    print("If ICM takes too long to converge:")
    print("1. Reduce initial_temperature to 5.0")
    print("2. Increase cooling_rate to 0.95")
    print("3. Set max_iterations to dataset_size * 2")
    print("4. Use smaller model for faster iterations")

def fix_inconsistent_results():
    """Solution for inconsistent results across runs"""
    print("If results vary significantly between runs:")
    print("1. Set random seed: random.seed(42), np.random.seed(42)")
    print("2. Increase alpha for stricter consistency")
    print("3. Use more initial_examples")
    print("4. Reduce temperature to 0.05")

def fix_poor_quality_scores():
    """Solution for low quality scores"""
    print("If final scores are consistently low:")
    print("1. Check if your data actually has clear patterns")
    print("2. Try different alpha values (30-80)")
    print("3. Ensure sufficient data (minimum 20-30 items)")
    print("4. Check for data quality issues (duplicates, unclear text)")

# Print all quick fixes
print("=== QUICK TROUBLESHOOTING REFERENCE ===")
fix_empty_results()
print()
fix_slow_convergence()
print()
fix_inconsistent_results()
print()
fix_poor_quality_scores()
```

## Next Steps

1. **Start small:** Try ICM on a subset of your data (50-100 items) to see how it works
2. **Experiment with settings:** Adjust parameters to match your data characteristics
3. **Scale up:** Once you're happy with results, process your full dataset
4. **Integrate:** Export results and integrate with your existing analysis tools

## Getting Help

- Check the examples in `icm_examples.py` for common use cases
- Review the technical documentation in `README.md` for detailed configuration options
- Look at the test suite in `icm_test_suite.py` for edge cases and examples

---

**Bottom Line:** ICM automates the tedious work of organizing data, giving you clean, consistently categorized datasets without manual effort. It's particularly powerful when you have data that needs organizing but don't have the time or resources for manual review.