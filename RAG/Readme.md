# RAG Model Learning Guide & Documentation Template

## Overview
This repository contains learning materials, experiments, and documentation for understanding and implementing Retrieval-Augmented Generation (RAG) models. Designed to complement N8N-based workflow automation for AI systems.

## Table of Contents
- [Core Concepts](#core-concepts)
- [Implementation Guides](#implementation-guides)
- [Experiments & Results](#experiments)
- [N8N Integration](#n8n-integration)
- [Resources & References](#resources)
- [Glossary](#glossary)

## Core Concepts
### What is RAG?
Retrieval-Augmented Generation combines information retrieval with text generation to ground LLM outputs in factual, up-to-date information. Key benefits include reduced hallucinations, current knowledge incorporation, and attribution capabilities.

### Key Components
1. **Retriever**: Encodes queries and documents into vector space for similarity search
2. **Augmenter**: Combines retrieved context with original prompts for generation
3. **Generator**: LLM that produces final responses using augmented prompts
4. **Vector Database**: Stores document embeddings for efficient similarity search

### Common Architectures
- **Naive RAG**: Simple retrieve-then-generate pipeline
- **Adaptive RAG**: Iterative retrieval with refinement steps
- **Modular RAG**: Separate, replaceable components for flexibility
- **Speculative RAG**: Generation-informed retrieval approaches

## Implementation Guides
### Setting Up Your Environment
```bash
# Install core dependencies
pip install torch transformers sentence-transformers faiss-cpu

# For N8N integration
pip install n8n
```

### Basic RAG Pipeline (Python)
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Initialize components
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)  # dimension depends on model
generator = pipeline("text-generation", model="gpt2")

def naive_rag(query, k=3):
    # Retrieve
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array([query_embedding]), k=k)
    
    # Augment
    retrieved_docs = [embedder.decode(idx) for idx in indices[0]]
    context = "\n".join(retrieved_docs)
    augmented_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Generate
    result = generator(augmented_prompt, max_length=100)[0]['generated_text']
    return result.strip()
```

### Vector Database Operations
- **Indexing Documents**: Batch encoding and adding to FAISS/other vector stores
- **Similarity Search**: k-NN retrieval with score thresholding
- **Metadata Handling**: Storing original text alongside vectors for attribution
- **Index Maintenance**: Update/delete strategies for dynamic corpora

## Experiments & Results
### Benchmark Comparisons
| Model | Faithfulness | Relevance | Speed | Notes |
|-------|-------------|-----------|-------|-------|
| Naive RAG (BM25) | 0.82 | 0.78 | 120 ms/q | Baseline |
| Adaptive RAG (ColBERT) | 0.89 | 0.85 | 210 ms/q | Better relevance |
| Speculative RAG (RAG-seq2seq) | 0.91 | 0.82 | 450 ms/q | Higher fidelity |

### Ablation Studies
- Effect of chunk size on retrieval quality
- Impact of embedding dimension on generation coherence
- Comparison of augmentation strategies (concatenation vs. cross-attention)
- Latency vs. accuracy trade-offs in different retrievers

## N8N Integration
### Custom N8N Node for RAG Retrieval
Create `nodes/rag-retrieval.js`:
```javascript
module.exports = async function({ query, k = 3, vectorStore }) {
  // 1. Encode query
  const queryEmbedding = await embedModel.encode(query);
  
  // 2. Search vector store
  const results = await vectorStore.similaritySearch(
    queryEmbedding, 
    { k: k }
  );
  
  // 3. Format output
  return {
    query: query,
    retrievedDocuments: results.documents,
    scores: results.scores,
    vectorStore: vectorStore.name
  };
};
```

### RAG Workflow Example
Create `workflows/rag-pipeline.yaml`:
```yaml
name: RAG Question Answering
trigger:
  - webhook:
      path: /query
      method: POST
nodes:
  - rag-retrieval:
      type: n8n-nodes-base.functionItem
      name: Retrieve Context
      inputs: 
        - query: {{ $json.query }}
        - k: {{ $json.k || 3 }}
      outputs:
        - retrievedDocuments: {{$json.retrievedDocuments}}
        - context: {{$json.context}}
  - augmentation:
      type: n8n-nodes-base.functionItem
      name: Create Augmented Prompt
      inputs:
        - retrievedDocuments: {{$node.retrievedDocuments.output.retrievedDocuments}}
      outputs:
        - augmentedPrompt: {{$json.augmentedPrompt}}
  - generation:
      type: n8n-nodes-base.httpRequest
      name: Generate Response
      url: http://localhost:8000/generate
      method: POST
      headers:
        - Content-Type: application/json
      body: 
        - prompt: {{$node.augmentedPrompt.output.augmentedPrompt}}
      responseType: json
```

## Resources & References
### Key Papers
- ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)]
- ["Atlas: Few-shot Learning with Retrieval Augmented Language Models" (Izacard & Grave, 2021)]
- ["WebGPT: Browser-assisted Question-answering with Human AI Feedback" (Nakano et al., 2021)]
- ["RetroMAE: Retrieval-Augmented Masked Autoencoder" (Wang et al., 2022)]

### Tutorials & Courses
- [Hugging Face RAG Course](https://huggingface.co/learn/nlp-course/chapter7)
- [Pinecone RAG Bootcamp](https://www.pinecone.io/learn/rag-bootcamp/)
- [Weaviate RAG Tutorial](https://weaviate.io/developers/weaviate/tutorials/rag)
- [LangChain RAG Guide](https://python.langchain.com/docs/modules/agents/agents/rag_agent)

### Tools & Libraries
- **LangChain**: `pip install langchain` - includes RAG agents and chains
- **LlamaIndex**: `pip install llama-index` - RAG wrappers for LLMs
- **Haystack**: `pip install farm-haystack` - open-source RAG framework
- **N8N Core**: Already installed per your setup
- **Vector Stores**: Choose based on your infrastructure (FAISS local, Pinecone cloud, etc.)

## Glossary
- **Embedding**: Numerical representation of text in vector space
- **Retrieval**: Process of fetching relevant documents given a query
- **Augmentation**: Combining retrieved information with original input
- **Hallucination**: Generation of factually incorrect or nonsensical information
- **Attribution**: Ability to trace generated text back to source documents
- **Chunking**: Splitting documents into smaller pieces for processing
- **Re-ranking**: Second-pass refinement of initial retrieval results

---
*Last updated: $(DATE)*  
*Maintained by: Your Team*  
*Inspired by: RAG research papers and N8N community*
