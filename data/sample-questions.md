### 1. "Why did you choose LoRA over full fine-tuning?"

**Answer**: "LoRA allows training only 1.14% of parameters while achieving similar results to full fine-tuning. With limited GPU resources (Colab T4), full fine-tuning 7 billion parameters would be impractical. LoRA also prevents catastrophic forgetting of the base model's general capabilities while adding domain-specific behavior."

---

### 2. "How did you handle the lack of real medical data?"

**Answer**: "I generated 200 synthetic examples following medical education standards. Each example followed a consistent structure with Definition, Explanation, Clinical Relevance, and Safety Notes. While synthetic data has limitations, it was sufficient to teach the model output formatting and consistent terminology usage, which was my primary goal."

---

### 3. "What was your biggest challenge in this project?"

**Answer**: "Balancing dataset size with training effectiveness. With only 150 training examples, I had to optimize hyperparameters carefully—increasing LoRA rank to 32 for more capacity, using 5 epochs, and implementing cosine scheduling. The key was ensuring the model learned the structure without overfitting."

---

### 4. "How do you evaluate whether your fine-tuning was successful?"

**Answer**: "I used both automated and manual evaluation. Automated metrics measured format adherence—checking for required sections in responses. Manual evaluation compared base vs fine-tuned outputs side-by-side. The fine-tuned model achieved 100% format adherence vs 0% for base model, clearly demonstrating improvement."

---

### 5. "Why GGUF quantization? What are the tradeoffs?"

**Answer**: "GGUF with q4_k_m quantization reduces model size from 14GB to 4.4GB while maintaining quality. This enables local deployment on consumer hardware via Ollama. The tradeoff is slight quality degradation, but for structured text generation, 4-bit quantization preserves format adherence while making inference 3-4x faster."

---

### 6. "How would you improve this project with more resources?"

**Answer**: "Three key improvements: First, expand to 500+ examples covering more medical specialties. Second, implement RAG to provide current medical information beyond training data. Third, add multilingual support and create a web interface for easier access. I'd also benchmark against other medical LLMs."

---

### 7. "What did you learn about instruction fine-tuning?"

**Answer**: "Small, high-quality datasets can dramatically change model behavior. The key is consistency—every training example followed the exact same format. I also learned that system prompts are crucial for guiding behavior, and that structured outputs require explicit formatting in training data."

---

### 8. "How would you deploy this in production?"

**Answer**: "Current Ollama deployment works for local use. For production, I'd containerize with Docker, add API rate limiting, implement logging and monitoring, set up model versioning, and add input validation. I'd also add a feedback loop to collect user ratings and continuously improve the model."

---

### 9. "What are the safety considerations for medical AI?"

**Answer**: "This model is strictly educational, not diagnostic. Every response includes disclaimers. In production, I'd add content filtering, implement audit trails, ensure HIPAA compliance if handling patient data, and clearly communicate limitations. Medical AI should augment, not replace, professional judgment."

---

### 10. "How did you validate your training wasn't overfitting?"

**Answer**: "I monitored validation loss during training—it stayed close to training loss, indicating good generalization. I also tested on held-out examples and manually verified the model could handle variations of prompts it hadn't seen. With only 150 examples, some overfitting is expected, but the model generalizes well to the test set."

---

**Pro Tip**: Have the HuggingFace link and GitHub ready to show live demos during interviews.