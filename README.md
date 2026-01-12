# Medical LLM Fine-Tuning

End-to-end pipeline for fine-tuning Mistral 7B on medical terminology education using LoRA/PEFT, demonstrating practical domain-specific LLM customization.

**Live Model**: [medical-mistral-7b on HuggingFace](https://huggingface.co/rohanjain2312/medical-mistral-7b)

## Overview

This project fine-tunes Mistral 7B to create a Medical Terminology Education Assistant that provides consistent, structured explanations of medical concepts. The goal was to transform a general-purpose LLM into a specialized educational tool with predictable output formatting.

### The Challenge

General-purpose LLMs produce inconsistent medical explanations:
- Variable formatting and structure
- Missing safety disclaimers
- Unpredictable response quality
- No standardization across queries

### The Solution

Fine-tuned model that delivers 100% format adherence with:
- Structured 4-part responses (Definition → Explanation → Clinical Relevance → Safety Note)
- Consistent medical terminology
- Built-in educational disclaimers
- Professional medical education tone

## What I Built

### Complete ML Pipeline

1. **Data Generation**: Created 200 synthetic medical education examples across 5 categories
2. **Model Fine-Tuning**: Implemented LoRA fine-tuning on Mistral 7B using Unsloth
3. **Model Evaluation**: Built automated comparison framework between base and fine-tuned models
4. **Production Deployment**: Quantized to GGUF and deployed via Ollama
5. **Public Release**: Published to HuggingFace for community access

### Technical Implementation

**Architecture**:
- Base: Mistral 7B Instruct v0.3
- Method: LoRA (rank=32, alpha=32)
- Training: 150 examples, 5 epochs, ~15 minutes on T4 GPU
- Output: GGUF q4_k_m quantized model (4.4GB)

**Training Results**:
- Final Loss: 0.6179
- Trainable Parameters: 1.14% (83.9M / 7.3B)
- Format Adherence: 100% (vs 0% base model)

## What I Learned

### 1. Parameter-Efficient Fine-Tuning

**Key Insight**: LoRA enables effective fine-tuning with minimal resources.

- Trained only 1.14% of model parameters using Low-Rank Adaptation
- Achieved significant behavior changes without full model retraining
- Learned to balance LoRA rank (capacity) vs efficiency tradeoffs
- Discovered 4-bit quantization maintains quality while reducing memory

**Practical Skills**:
- Configured LoRA adapters for target modules (attention + MLP layers)
- Optimized hyperparameters: learning rate (3e-4), scheduler (cosine), warmup steps
- Implemented gradient accumulation for larger effective batch sizes on limited hardware

### 2. Instruction Fine-Tuning & Prompt Engineering

**Key Insight**: Small, high-quality datasets can dramatically alter model behavior.

- Designed instruction-response format for consistent outputs
- Created synthetic training data following medical education standards
- Learned importance of system prompts in guiding model behavior
- Understood how to structure data for instruction-following models

**Practical Skills**:
- Formatted data in Mistral's chat template format
- Engineered system prompts for consistent output structure
- Balanced dataset across multiple medical categories
- Validated data quality through manual review

### 3. Model Quantization & Deployment

**Key Insight**: Production deployment requires balancing quality, size, and speed.

- Converted fine-tuned model to GGUF format for efficient inference
- Applied q4_k_m quantization reducing model size from ~14GB to 4.4GB
- Deployed locally using Ollama for fast, private inference
- Published to HuggingFace for broader accessibility

**Practical Skills**:
- Used Unsloth for efficient GGUF conversion
- Configured Ollama Modelfiles for custom model serving
- Set up local inference endpoints
- Managed model versioning and distribution

### 4. LLM Evaluation Methodology

**Key Insight**: Evaluating fine-tuned models requires both automated metrics and qualitative assessment.

- Built automated evaluation comparing base vs fine-tuned outputs
- Developed format adherence scoring system
- Conducted side-by-side output comparisons
- Learned that small dataset fine-tuning can achieve 100% adherence

**Practical Skills**:
- Designed evaluation metrics for structured outputs
- Implemented automated testing via Ollama API
- Created reproducible evaluation pipelines
- Documented measurable improvements

### 5. End-to-End ML Engineering

**Key Insight**: Real ML projects require skills beyond model training.

- Managed complete pipeline from data to deployment
- Version controlled code, configs, and documentation
- Created reproducible workflows for others to follow
- Balanced technical implementation with clear communication

**Practical Skills**:
- Organized project structure following best practices
- Documented setup, usage, and limitations clearly
- Published work for portfolio demonstration
- Integrated multiple tools (Colab, Ollama, HuggingFace, Git)

## Repository Structure

```
medical-llm-finetuning/
├── data/                            # Training and test datasets
├── notebooks/                       # Fine-tuning notebook (Colab)
├── models/                          # GGUF model (download from HF)
├── evaluation/                      # Evaluation scripts
├── Modelfile                        # Ollama configuration
└── requirements.txt                 # Dependencies
```

## Quick Start

### Training
1. Open `notebooks/medical_llm_finetuning.ipynb` in Google Colab
2. Enable T4 GPU runtime
3. Run all cells (~15 minutes)
4. Download generated GGUF file

### Deployment
```bash
# Download model from HuggingFace
# Place in models/ directory

# Create Ollama model
ollama create medical-mistral -f Modelfile

# Test
ollama run medical-mistral "Explain the medical term: Hypertension"
```

## Example: Base vs Fine-Tuned

**Prompt**: "Explain the medical term: Hypertension"

**Base Model**: Unstructured paragraph, inconsistent format, no safety disclaimer

**Fine-Tuned Model**:
```
**Definition**: Hypertension is persistently elevated blood pressure in the arteries.

**Explanation**: Blood pressure is measured as systolic over diastolic pressure 
(e.g., 120/80 mmHg). Hypertension is defined as blood pressure consistently at or 
above 130/80 mmHg. The heart must work harder to pump blood through vessels with 
increased resistance, which can damage arteries and organs over time.

**Clinical Relevance**: Major risk factor for heart disease, stroke, and kidney disease. 
Often called the "silent killer" because it typically has no symptoms until serious 
complications occur.

**Note**: This is educational information only. Always consult healthcare professionals 
for medical decisions.
```

## Limitations & Future Work

**Current Limitations**:
- 150 training examples covers common but not comprehensive medical knowledge
- Educational purposes only, not validated for clinical use
- English language only
- Synthetic data may miss real-world edge cases

**Potential Improvements**:
- Expand to 500+ examples for broader coverage
- Add retrieval-augmented generation for current medical information
- Implement multilingual support
- Fine-tune specialized subdomains (cardiology, neurology)

## Technical Details

**Training Configuration**:
- Epochs: 5
- Batch Size: 2 (gradient accumulation: 4)
- Learning Rate: 3e-4 (cosine scheduler)
- Optimizer: AdamW 8-bit
- Framework: Unsloth + TRL

**Dataset Categories**:
- Medical Terminology (50 train, 13 test)
- Abbreviations (40 train, 16 test)
- Disease Mechanisms (25 train, 8 test)
- Anatomy & Physiology (20 train, 6 test)
- Diagnostic Tests (15 train, 7 test)

## Acknowledgments

- **Base Model**: Mistral AI for Mistral 7B Instruct
- **Training Framework**: Unsloth for efficient fine-tuning
- **Deployment**: Ollama for local model serving
- **Development Assistance**: Built with guidance from Claude (Anthropic)

## Contact

**Rohan Jain**
- LinkedIn: [linkedin.com/in/jaroh23](https://www.linkedin.com/in/jaroh23/)
- GitHub: [@Rohanjain2312](https://github.com/Rohanjain2312)
- HuggingFace: [@rohanjain2312](https://huggingface.co/rohanjain2312)

## License

Apache 2.0