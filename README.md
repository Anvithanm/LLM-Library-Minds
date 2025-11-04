# LLM for Book Q&A System: RAG vs Fine-Tuning

A comprehensive comparison of **Retrieval-Augmented Generation (RAG)** and **Fine-Tuning** approaches for question-answering on a 20-book dataset.

---

## Overview

This project implements and compares two popular approaches for building a question-answering system trained on 20 books:

1. **RAG (Retrieval-Augmented Generation)**: Retrieves relevant context from a vector database and generates answers
2. **Fine-Tuning**: Trains an LLM directly on the book content using LoRA

### Key Features

- Works with 20 books from Hugging Face dataset
- FAISS vector store for efficient similarity search
- LoRA-based efficient fine-tuning
- Comprehensive evaluation metrics
- Relevance filtering to prevent hallucinations
- Weights & Biases integration for experiment tracking
---

## Dataset

**Source**: [IsmaelMousa/books](https://huggingface.co/datasets/IsmaelMousa/books)

**Dataset Details**:
- Number of books: 20 (selected from the dataset)
- Format: Text with metadata (title, author)
- Average book length: ~50,000-100,000 words

**Sample Data**:
```python
{
  'title': 'Book Title',
  'author': 'Author Name',
  'EN': 'Full book content...'
}
```

---

## Approach 1: RAG (Retrieval-Augmented Generation)

### Architecture

```
User Query
    ↓
Vector Search (FAISS)
    ↓
Retrieve Top-K Documents
    ↓
Relevance Filtering
    ↓
LLM + Context → Answer
```

### Components

1. **Text Splitting**: Books split into 1000-char chunks (200-char overlap)
2. **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
3. **Vector Store**: FAISS (Facebook AI Similarity Search)
4. **LLM**: Mistral-7B-Instruct or TinyLlama
5. **Retrieval**: Similarity search with relevance thresholds

### Key Features

- **Relevance Filtering**: Prevents answering irrelevant questions
- **Semantic Similarity Check**: Cosine similarity threshold (0.3)
- **Distance Threshold**: FAISS L2 distance check (1.0)
- **Gibberish Detection**: Filters nonsensical queries

### Implementation

```python
# Create vector store
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    documents=split_documents,
    embedding=embeddings
)

# Query
results = vectorstore.similarity_search(query, k=4)
```

### Pros & Cons

**Pros**:
- Quick setup (minutes)
- Easy to update (just re-index)
- Explainable (see source documents)
- No GPU training needed

 **Cons**:
- Slower inference (retrieval + generation)
- Dependent on retrieval quality
- Requires vector database storage

---

## Approach 2: Fine-Tuning

### Architecture

```
Books Dataset
    ↓
Prepare Training Data (Q&A pairs)
    ↓
LoRA Fine-Tuning
    ↓
Fine-tuned Model → Answer
```

### Components

1. **Base Model**: TinyLlama-1.1B or Mistral-7B
2. **Fine-Tuning Method**: LoRA (Low-Rank Adaptation)
3. **Training**: Instruction-tuning format
4. **Optimization**: 4-bit quantization with BitsAndBytes

### Training Configuration

```python
# LoRA Config
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Args
training_args = SFTConfig(
    output_dir=output_dir,

    # Training hyperparameters
    num_train_epochs=3,  # Number of epochs
    per_device_train_batch_size=4,  # Batch size
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16

    # Optimizer settings
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    weight_decay=0.01,

    # Memory optimization
    fp16=True,  # Mixed precision training
    gradient_checkpointing=True,

    # W&B Integration
    report_to="wandb",  #  Enable W&B logging

    # Logging and saving
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,

    # Other settings
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataset_text_field="text",
    max_length=512,
)
```

### Training Data Format

```
### Instruction:
Summarize the following passage from 'Book Title' by Author:

### Input:
[Book passage text...]

### Response:
[Expected output...]
```

### Pros & Cons

 **Pros**:
- Faster inference (no retrieval)
- Learns writing style and patterns
- Consistent outputs
- Smaller deployment footprint

 **Cons**:
- Requires GPU training (2-3 hours)
- Hard to update (need to retrain)
- Less explainable
- Can hallucinate

---

### Results

| Metric | RAG | Fine-Tuning | Winner |
|--------|-----|-------------|--------|
| Accuracy | 62.27% | 87% | 🏆 Fine-Tuning |
| Speed (avg) | 3.2s | 2.1s | 🏆 Fine-Tuning |
| Faithfulness | 71.93% | na| 🏆 RAG |
| Explainability | ✅ Yes | ❌ No | 🏆 RAG |
| Update Ease | ✅ Easy | ❌ Hard | 🏆 RAG |


---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- GPU with 16GB+ VRAM (for fine-tuning)
- CUDA 11.8+ (for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/llm-book-qa.git
cd llm-book-qa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Requirements

```txt
# Core dependencies
transformers>=4.35.0
datasets>=2.14.0
torch>=2.0.0
accelerate>=0.24.0

# RAG dependencies
langchain>=0.1.0
langchain-community>=0.0.10
faiss-cpu>=1.7.4  # or faiss-gpu for GPU
sentence-transformers>=2.2.2

# Fine-tuning dependencies
peft>=0.7.0
bitsandbytes>=0.41.0
trl>=0.7.11

# Evaluation
scikit-learn>=1.3.0
numpy>=1.24.0

# Experiment tracking
wandb>=0.16.0

# Utilities
huggingface_hub>=0.19.0
```

---

## Results

### Performance Summary

**Dataset**: 20 books, ~1.5M words total

#### RAG Results

- Retrieval accuracy: 100.00%
- Average response time: 3.2s
- Faithfulness: 71.93%
- Semantic Similarity : 11.66%

#### Fine-Tuning Results
- Accuracy: 87%

### Best Use Cases

**Use RAG when**:
- ✅ Need source citations
- ✅ Knowledge changes frequently
- ✅ Limited GPU resources
- ✅ Quick prototyping needed

**Use Fine-Tuning when**:
- ✅ Specific style/tone required
- ✅ Static knowledge base
- ✅ Fast inference critical
- ✅ Creative generation needed

---

## Weights & Biases Integration

### Setup

```python
import wandb

# Login (one-time)
wandb.login()  # Enter API key from wandb.ai/authorize

# Initialize project
wandb.init(
    project="llm-book-qa",
    name="experiment-1",
    config={
        "approach": "rag",  # or "finetuning"
        "model": "mistral-7b",
        "num_books": 20
    }
)
```

### What W&B Tracks

- 📉 Training/validation loss curves
- 💻 GPU utilization and memory
- ⚙️ All hyperparameters
- 📊 Custom metrics (accuracy, F1, etc.)
- 🔄 Model versions and checkpoints
- 📈 Comparison across experiments

### View Dashboard

After training, view results at:
```
https://wandb.ai/anvithanm22-northeastern-university/llm-finetuning-books/workspace?nw=nwuseranvithanm22
```

### Features Used

- Real-time training monitoring
- Hyperparameter sweeps
- Model artifact versioning
- Experiment comparison
- Collaborative reports

---

## Resources

### Documentation
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [FAISS Documentation](https://faiss.ai/)
- [PEFT (LoRA) Guide](https://huggingface.co/docs/peft)
- [W&B Documentation](https://docs.wandb.ai/)

### Tutorials
- [RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Fine-Tuning LLMs](https://huggingface.co/blog/fine-tune-llms)
- [LoRA Explained](https://huggingface.co/blog/lora)

### Papers
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [FAISS: Efficient Similarity Search](https://arxiv.org/abs/1702.08734)

---


## Acknowledgments

- **Dataset**: [IsmaelMousa/books](https://huggingface.co/datasets/IsmaelMousa/books)
- **Models**: Mistral AI, TinyLlama teams
- **Frameworks**: LangChain, Hugging Face, FAISS
- **Tools**: Weights & Biases for experiment tracking

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{llm-book-qa-2024,
  author = {Your Name},
  title = {LLM Book QA: RAG vs Fine-Tuning Comparison},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/llm-book-qa}
}
```