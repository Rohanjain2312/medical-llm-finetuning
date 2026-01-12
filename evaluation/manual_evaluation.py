"""
Simple Manual Evaluation Script
For when you want to manually test and compare responses

Usage:
    python manual_evaluation.py
"""

import json


def load_test_examples(filepath: str, num_examples: int = 5):
    """Load a few test examples for manual evaluation"""
    examples = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            examples.append(json.loads(line))
    return examples


def create_evaluation_template():
    """Create a template for manual evaluation"""
    
    test_examples = load_test_examples('../data/medical_qa_test.jsonl', num_examples=5)
    
    output = []
    output.append("="*80)
    output.append("MANUAL EVALUATION TEMPLATE")
    output.append("="*80)
    output.append("\nInstructions:")
    output.append("1. Copy each instruction below")
    output.append("2. Test with base Mistral model")
    output.append("3. Test with your fine-tuned model")
    output.append("4. Compare responses for:")
    output.append("   - Format adherence (Definition/Explanation/Clinical Relevance/Note)")
    output.append("   - Medical accuracy")
    output.append("   - Safety disclaimer presence")
    output.append("   - Clarity and educational value")
    output.append("\n")
    
    for i, example in enumerate(test_examples, 1):
        output.append("\n" + "="*80)
        output.append(f"TEST CASE {i}")
        output.append("="*80)
        output.append(f"\nCategory: {example['category']}")
        output.append(f"\nInstruction to test:")
        output.append(f"{example['instruction']}")
        output.append(f"\n\nExpected format reference:")
        output.append(example['output'])
        output.append(f"\n\n--- YOUR EVALUATIONS ---")
        output.append(f"\nBase Model Response:")
        output.append("[Paste response here]")
        output.append(f"\n\nFine-tuned Model Response:")
        output.append("[Paste response here]")
        output.append(f"\n\nYour Observations:")
        output.append("Format Score (0-1): ")
        output.append("Safety Disclaimer Present: [ ] Base  [ ] Fine-tuned")
        output.append("Which response is better: [ ] Base  [ ] Fine-tuned  [ ] Similar")
        output.append("Notes: ")
        output.append("\n")
    
    # Write to file
    with open('manual_evaluation_template.txt', 'w') as f:
        f.write('\n'.join(output))
    
    print("✅ Manual evaluation template created: manual_evaluation_template.txt")
    print("\nNext steps:")
    print("1. Open manual_evaluation_template.txt")
    print("2. Test each instruction with both models")
    print("3. Fill in your observations")
    print("4. Use this for your portfolio documentation")


def create_quick_test_prompts():
    """Create quick test prompts file"""
    
    prompts = [
        "Explain the medical term: Hypertension",
        "Explain the medical abbreviation: CBC",
        "Explain the disease mechanism: Type 2 Diabetes",
        "Explain the anatomy: The Heart",
        "Explain the diagnostic test: MRI",
    ]
    
    with open('quick_test_prompts.txt', 'w') as f:
        f.write("QUICK TEST PROMPTS\n")
        f.write("="*80 + "\n\n")
        f.write("Copy and paste these into Ollama to test your model:\n\n")
        
        for i, prompt in enumerate(prompts, 1):
            f.write(f"{i}. {prompt}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("\nOllama Commands:\n")
        f.write("  Base model:       ollama run mistral:7b-instruct\n")
        f.write("  Fine-tuned model: ollama run medical-mistral\n")
    
    print("✅ Quick test prompts created: quick_test_prompts.txt")


if __name__ == "__main__":
    print("Creating evaluation materials...\n")
    create_evaluation_template()
    print()
    create_quick_test_prompts()
    print("\n✅ All files created successfully!")
