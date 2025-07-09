# English Prompting Guide with LangChain OpenAI

## 1. Install Dependencies

### Install langchain_openai and related packages:
```bash
pip install langchain-openai langchain python-dotenv
```

Or install all dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

## 2. Setup Environment Variables

### Create .env file:
```bash
# Copy template file
cp sample.env .env
```

### Update .env with actual information:
```env
SHUBI_API_KEY=your_actual_api_key_here
SHUBI_URL=your_actual_shubi_url_here
```

## 3. Sample Code Usage

### Basic setup:
```python
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    model="claude-3-7-sonnet-20250219", 
    temperature=0,
    api_key=os.environ.get("SHUBI_API_KEY"), 
    base_url=os.environ.get("SHUBI_URL")
)
```

## 4. Run Prompting Examples

### Run prompting demonstrations:
```bash
cd scripts
python prompting_example.py
```

### Run evaluation comparison:
```bash
cd scripts
python run_evaluation.py
```

### Run evaluation with options:
```bash
# Test with 10 samples
python run_evaluation.py --limit 10

# Test only zero-shot and few-shot
python run_evaluation.py --methods zero_shot few_shot

# Increase delay between API calls
python run_evaluation.py --delay 2.0
```

## 5. Standard Format (Consistent with Fine-tuning)

### Labels:
- **0**: No-clickbait (factual, objective, clear information)  
- **1**: Clickbait (sensational, attention-grabbing, exaggerated, vague)

### Mapping with Fine-tuning:
- Fine-tuning BERT/LLM: `num_labels=2` with 0/1
- Dataset: `"label": 0/1` and `"truth_class": "no-clickbait"/"clickbait"`
- Prompting: Output consistent format 0/1

## 6. Prompting Techniques Demonstrated

### 1. Zero-shot Prompting
- No example provided
- Relies only on instructions and definitions
- Tests model's understanding of context

**Example:**
```
Headline: "Oscar's biggest loser finally wins... on 21st try"
Result: 1 - Clickbait
```

### 2. Few-shot Prompting  
- Provides training data examples
- Pattern recognition approach
- Consistent format learning
- Usually outperforms zero-shot

**Training Examples Used:**
- "Trump vows 35% tax for US firms that move jobs overseas" → No-clickbait
- "Bet you didn't know government jobs paid so well :p" → Clickbait
- "John Glenn, American Hero of the Space Age, Dies at 95" → No-clickbait
- "Trump says Happy New Year in the most Trump way" → Clickbait

### 3. Chain of Thought (CoT) Prompting
- Step-by-step reasoning process
- Detailed analysis methodology
- Transparent decision making
- Best for explainability

**Analysis Steps:**
1. Identify emotional/attention-grabbing keywords
2. Evaluate information specificity (facts vs. withheld info)
3. Check for curiosity gaps
4. Assess overall tone and structure
5. Final classification with reasoning

## 7. Sample Output

```
=== ENGLISH PROMPTING METHODS DEMONSTRATION ===

=== ZERO-SHOT PROMPTING EXAMPLE ===
Test headline: Oscar's biggest loser finally wins... on 21st try
Zero-shot result: 1

=== FEW-SHOT PROMPTING EXAMPLE ===
Test headline: The curious case of the billion-dollar lithium mine that sold on the cheap
Few-shot result:
Label: 1
Reason: This headline creates curiosity by describing something as "curious" and mentioning it was sold "on the cheap" without explaining why or providing specific details.

=== CHAIN OF THOUGHT PROMPTING ===
Test headline: Wow, Instagram has a lot of followers.
Chain of Thought result:
Step 1 - Keywords: "Wow" is an emotional exclamation designed to grab attention
Step 2 - Information specificity: Very vague - doesn't specify how many followers or provide concrete numbers
Step 3 - Curiosity gaps: Creates curiosity about the actual number without revealing it
Step 4 - Tone/structure: Casual, clickbait-style structure
Step 5 - Final classification: 1 (Clickbait) - uses emotional language and withholds specific information
```

## 8. Customization

### Change model:
```python
llm = ChatOpenAI(
    model="claude-opus-4-20250514",  # or other available model
    temperature=0.7,  # adjust creativity
    # ... other params
)
```

### Customize prompts:
- Edit examples in few-shot prompting
- Modify CoT analysis steps
- Adjust prompt templates for specific domains

## 9. Troubleshooting

### Common errors:
1. **Missing API Key**: Check .env file
2. **Network Error**: Verify SHUBI_URL
3. **Model Unavailable**: Use `python scripts/test_available_models.py` to find working models
4. **Parse Errors**: LLM returns unexpected format

### Debug tips:
```python
# Print raw response if parsing fails
print(f"Raw response: {response.content}")
```

## 10. Evaluation and Metrics

### Metrics measured:
- **Accuracy**: Overall correct prediction rate
- **Precision**: Precision for each class
- **Recall**: Ability to find all positive cases
- **F1-Score**: Balanced score between precision and recall

### Sample evaluation results:
```
📈 FINAL EVALUATION COMPARISON
====================================================================
   method  valid_samples  total_samples  accuracy  precision  recall  f1_score
      cot             18             20     0.889      0.889   0.889     0.889
 few_shot             19             20     0.842      0.842   0.842     0.842
zero_shot             17             20     0.765      0.765   0.765     0.765

🏆 BEST METHOD: cot
   📊 F1-Score: 0.889
   🎯 Accuracy: 0.889
```

### Insights from evaluation:
1. **Chain of Thought** often performs best due to structured reasoning
2. **Few-shot** outperforms **Zero-shot** with training examples
3. **CoT** provides best explainability for decisions
4. Success rate varies by headline complexity

## 11. Training Data Examples Used

The prompting methods use real examples from the training dataset:

### No-clickbait Examples:
- "Live Nation CEO Michael Rapino: 'I don't want to be in the secondary biz at all'"
- "Cherokee Nation files first-of-its-kind opioid lawsuit against Wal-Mart, CVS and Walgreens"
- "Trump: 'We must fight' hard-line conservative Freedom Caucus in 2018 midterm elections"

### Clickbait Examples:
- "What happens when you don't shave for a whole year? Natural beauty, explains this blogger 🙌"
- "The best fast food, picked by the world's top chefs"
- "This woman sneaking pictures of The Rock's butt is you, me, and your grandma"

## 12. Extension Ideas

### Experiment with different approaches:
- A/B test various prompt formulations
- Measure performance on larger datasets
- Fine-tune prompts based on error analysis

### Production optimization:
- Implement response caching
- Batch processing for multiple headlines
- Add error handling and retry logic
- API rate limiting management

### Domain adaptation:
- Adapt prompts for different content types
- Multi-language support
- Industry-specific clickbait detection 