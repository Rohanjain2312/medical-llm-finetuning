"""
Medical LLM Evaluation Script
Compares base Mistral 7B vs fine-tuned medical model

Usage:
    python evaluate_model.py --model_path <path_to_gguf> --test_data medical_qa_test.jsonl
"""

import json
import argparse
import time
from typing import List, Dict
import requests


class OllamaEvaluator:
    """Evaluates models via Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    def query_model(self, model_name: str, prompt: str, max_tokens: int = 512) -> Dict:
        """Query a model via Ollama API"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "response": ""}
    
    def evaluate_format_adherence(self, response: str) -> Dict:
        """Check if response follows the structured format"""
        checks = {
            "has_definition": "**Definition**:" in response,
            "has_explanation": "**Explanation**:" in response,
            "has_clinical_relevance": "**Clinical Relevance**:" in response,
            "has_safety_note": "**Note**:" in response and "educational information" in response.lower(),
        }
        
        score = sum(checks.values()) / len(checks)
        return {"score": score, "checks": checks}
    
    def evaluate_response_quality(self, response: str, expected_category: str) -> Dict:
        """Evaluate response quality with simple heuristics"""
        metrics = {}
        
        # Length check (medical responses should be substantive)
        word_count = len(response.split())
        metrics["word_count"] = word_count
        metrics["appropriate_length"] = 100 <= word_count <= 600
        
        # Structure check
        sections = response.count("**")
        metrics["has_structure"] = sections >= 6  # At least 3 bold headers
        
        # Safety check
        safety_terms = ["educational", "consult healthcare", "medical decisions"]
        metrics["has_safety_disclaimer"] = any(term in response.lower() for term in safety_terms)
        
        # Medical terminology (basic check)
        if expected_category == "medical_terminology":
            metrics["uses_medical_terms"] = any(term in response.lower() for term in 
                ["condition", "tissue", "organ", "symptom", "patient"])
        
        return metrics


def load_test_data(filepath: str) -> List[Dict]:
    """Load test dataset"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def run_evaluation(base_model: str, finetuned_model: str, test_data_path: str, num_samples: int = 10):
    """Run comparative evaluation"""
    
    print("="*80)
    print("MEDICAL LLM EVALUATION")
    print("="*80)
    print(f"\nBase Model: {base_model}")
    print(f"Fine-tuned Model: {finetuned_model}")
    print(f"Test Data: {test_data_path}")
    print(f"Number of samples: {num_samples}\n")
    
    # Load test data
    test_data = load_test_data(test_data_path)
    print(f"✅ Loaded {len(test_data)} test examples")
    
    # Sample diverse examples
    samples = test_data[:num_samples]
    
    # Initialize evaluator
    evaluator = OllamaEvaluator()
    
    # Results storage
    results = {
        "base_model": {"responses": [], "metrics": []},
        "finetuned_model": {"responses": [], "metrics": []}
    }
    
    print("\n" + "="*80)
    print("RUNNING EVALUATIONS")
    print("="*80 + "\n")
    
    for i, example in enumerate(samples, 1):
        instruction = example['instruction']
        expected_output = example['output']
        category = example['category']
        
        print(f"\n[{i}/{num_samples}] Evaluating: {instruction}")
        print("-" * 80)
        
        # Query base model
        print("  Querying base model...")
        base_response = evaluator.query_model(base_model, instruction)
        base_text = base_response.get('response', '')
        
        # Query fine-tuned model
        print("  Querying fine-tuned model...")
        ft_response = evaluator.query_model(finetuned_model, instruction)
        ft_text = ft_response.get('response', '')
        
        # Evaluate both
        base_format = evaluator.evaluate_format_adherence(base_text)
        ft_format = evaluator.evaluate_format_adherence(ft_text)
        
        base_quality = evaluator.evaluate_response_quality(base_text, category)
        ft_quality = evaluator.evaluate_response_quality(ft_text, category)
        
        # Store results
        results["base_model"]["responses"].append({
            "instruction": instruction,
            "response": base_text,
            "expected": expected_output,
            "category": category
        })
        results["base_model"]["metrics"].append({
            "format": base_format,
            "quality": base_quality
        })
        
        results["finetuned_model"]["responses"].append({
            "instruction": instruction,
            "response": ft_text,
            "expected": expected_output,
            "category": category
        })
        results["finetuned_model"]["metrics"].append({
            "format": ft_format,
            "quality": ft_quality
        })
        
        # Print quick comparison
        print(f"  Base format score: {base_format['score']:.2f}")
        print(f"  Fine-tuned format score: {ft_format['score']:.2f}")
        print(f"  ✅ Improvement: {ft_format['score'] - base_format['score']:+.2f}")
    
    # Calculate aggregate metrics
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80 + "\n")
    
    base_avg_format = sum(m['format']['score'] for m in results['base_model']['metrics']) / num_samples
    ft_avg_format = sum(m['format']['score'] for m in results['finetuned_model']['metrics']) / num_samples
    
    print(f"Average Format Adherence:")
    print(f"  Base Model:       {base_avg_format:.2%}")
    print(f"  Fine-tuned Model: {ft_avg_format:.2%}")
    print(f"  Improvement:      {(ft_avg_format - base_avg_format):.2%}")
    
    # Count safety disclaimers
    base_safety = sum(1 for m in results['base_model']['metrics'] 
                     if m['quality'].get('has_safety_disclaimer', False))
    ft_safety = sum(1 for m in results['finetuned_model']['metrics'] 
                   if m['quality'].get('has_safety_disclaimer', False))
    
    print(f"\nSafety Disclaimer Presence:")
    print(f"  Base Model:       {base_safety}/{num_samples} ({base_safety/num_samples:.0%})")
    print(f"  Fine-tuned Model: {ft_safety}/{num_samples} ({ft_safety/num_samples:.0%})")
    
    # Save detailed results
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Detailed results saved to: {output_file}")
    
    return results


def generate_sample_outputs(results: Dict, output_file: str = "sample_outputs.txt"):
    """Generate human-readable comparison file"""
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MEDICAL LLM EVALUATION - SAMPLE OUTPUTS\n")
        f.write("="*80 + "\n\n")
        
        for i, (base, ft) in enumerate(zip(results['base_model']['responses'], 
                                           results['finetuned_model']['responses']), 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"EXAMPLE {i}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"INSTRUCTION:\n{base['instruction']}\n\n")
            f.write(f"CATEGORY: {base['category']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("BASE MODEL RESPONSE:\n")
            f.write("-" * 80 + "\n")
            f.write(base['response'] + "\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("FINE-TUNED MODEL RESPONSE:\n")
            f.write("-" * 80 + "\n")
            f.write(ft['response'] + "\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("EXPECTED OUTPUT (REFERENCE):\n")
            f.write("-" * 80 + "\n")
            f.write(base['expected'] + "\n\n")
    
    print(f"✅ Sample outputs saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate medical LLM models")
    parser.add_argument("--base_model", default="mistral:7b-instruct", 
                       help="Base model name in Ollama")
    parser.add_argument("--finetuned_model", default="medical-mistral", 
                       help="Fine-tuned model name in Ollama")
    parser.add_argument("--test_data", default="../data/medical_qa_test.jsonl",
                       help="Path to test data")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of test samples to evaluate")
    
    args = parser.parse_args()
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        print("✅ Ollama is running\n")
    except Exception as e:
        print(f"❌ Error: Ollama is not running or not accessible")
        print(f"   Please start Ollama with: ollama serve")
        print(f"   Error details: {e}")
        exit(1)
    
    # Run evaluation
    results = run_evaluation(
        base_model=args.base_model,
        finetuned_model=args.finetuned_model,
        test_data_path=args.test_data,
        num_samples=args.num_samples
    )
    
    # Generate readable output
    generate_sample_outputs(results)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - evaluation_results.json (detailed metrics)")
    print("  - sample_outputs.txt (human-readable comparisons)")
    print("\nUse these files to document your model's improvements!")
