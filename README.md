# Retrieval-Augmented Generation for AI-Generated Content: A Survey
This repo is constructed for collecting and categorizing papers about RAG according to our survey paper: *Retrieval-Augmented Generation for AI-Generated Content: A Survey*. Considering the rapid growth of this field, we will continue to update both arxiv paper and this repo.

# Overview
<div aligncenter><img width="900" alt="image" src="https://github.com/hymie122/RAG-Survey/blob/main/RAG_Overview.jpg">

# Catalogue
## Methods Taxonomy
### RAG Foundations
<div aligncenter><img width="900" alt="image" src="https://github.com/hymie122/RAG-Survey/blob/main/RAG_Foundations.png">

  - Query-based RAG
    
    [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)
    
    [KILT: a Benchmark for Knowledge Intensive Language Tasks](https://arxiv.org/abs/2009.02252)
  - Latent Representation-based RAG

  - Logit-based RAG

  - Speculative RAG

### RAG Enhancements
<div aligncenter><img width="900" alt="image" src="https://github.com/hymie122/RAG-Survey/blob/main/RAG_Enhancements.png">

  - Input Enhancement
    
    - Query Transformations
      
      [Query2doc: Query Expansion with Large Language Models](https://aclanthology.org/2023.emnlp-main.585)

      [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://doi.org/10.18653/v1/2023.acl-long.99)
    - Data Augmentation

      [Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models](https://proceedings.mlr.press/v202/huang23i.html)

  - Retriever Enhancement
    
    - Recursive Retrieve

      [Query Expansion by Prompting Large Language Models](https://doi.org/10.48550/arXiv.2305.03653)

      [Large Language Models Know Your Contextual Search Intent: A Prompting Framework for Conversational Search](https://aclanthology.org/2023.findings-emnlp.86)
      
    - Chunk Optimization

      [LlamaIndex](https://github.com/jerryjliu/llama_index)
      
    - Finetune Retriever

      [C-Pack: Packaged Resources To Advance General Chinese Embedding](https://arxiv.org/abs/2309.07597)

      [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216)

      [LM-Cocktail: Resilient Tuning of Language Models via Model Merging](https://arxiv.org/abs/2311.13534)

      [Retrieve Anything To Augment Large Language Models](https://arxiv.org/abs/2310.07554)

      [Replug: Retrieval-augmented black-box language models](https://arxiv.org/abs/2301.12652)

      [When Language Model Meets Private Library](https://doi.org/10.18653/v1/2022.findings-emnlp.21)

      [EditSum: {A} Retrieve-and-Edit Framework for Source Code Summarization](https://doi.org/10.1109/ASE51524.2021.9678724)

      [Synchromesh: Reliable Code Generation from Pre-trained Language Models](https://openreview.net/forum?id=KmtVD97J43e)

      [Retrieval Augmented Convolutional Encoder-decoder Networks for Video Captioning](https://doi.org/10.1145/3539225)
   
    - Hybrid Retrieve

      [RAP-Gen: Retrieval-Augmented Patch Generation with CodeT5 for Automatic Program Repair](https://doi.org/10.1145/3611643.3616256)

      [ReACC: A Retrieval-Augmented Code Completion Framework](https://doi.org/10.18653/v1/2022.acl-long.431)

      [Retrieval-based neural source code summarization](https://doi.org/10.1145/3377811.3380383)

      [BashExplainer: Retrieval-Augmented Bash Code Comment Generation based on Fine-tuned CodeBERT](https://doi.org/10.1109/ICSME55016.2022.00016)

      [Retrieval-Augmented Score Distillation for Text-to-3D Generation](https://doi.org/10.48550/arXiv.2402.02972)
   
    - Re-ranking

      [Re2G: Retrieve, Rerank, Generate](https://doi.org/10.18653/v1/2022.naacl-main.194)

      [Passage Re-ranking with BERT](http://arxiv.org/abs/1901.04085)

      [AceCoder: Utilizing Existing Code to Enhance Code Generation](https://arxiv.org/abs/2303.17780)

      [XRICL: Cross-lingual Retrieval-Augmented In-Context Learning for Cross-lingual Text-to-SQL Semantic Parsing](https://doi.org/10.18653/v1/2022.findings-emnlp.384)
   
    - Meta-data Filtering

      [PineCone](https://www.pinecone.io)
  - Generator Enhancement

    - Prompt Engineering
   
    - Decoding Tuning
   
    - Finetune Generator
   
  - Result Enhancement

    - Rewrite Output

  - RAG Pipeline Enhancement
    
    - Adaptive Retrieval
      
      - Rule-Baesd
    
      - Model-Based
   
    - Iterative RAG

    
    


## Applications Taxonomy
<div aligncenter><img width="900" alt="image" src="https://github.com/hymie122/RAG-Survey/blob/main/Applications.png">
  
### RAG for Text
  - Qusetion Answering

  - Fact verification

  - Commonsense Reasoning

  - Human-Machine Conversation

  - Neural Machine Translation

  - Event Extraction

  - Summarization

### RAG for Code
  - Code Generation

  - Code Summary

  - Code Completion

  - Automatic Program Repair

  - Text-to-SQL and Code-based Semantic Parsing

  - Others

### RAG for Audio
  - Audio Generation

  - Audio Captioning

### RAG for Image
  - Image Generation

    [Retrievegan: Image synthesis via differentiable patch retrieval](https://arxiv.org/abs/2007.08513)

    [Instance-conditioned gan](https://arxiv.org/abs/2109.05070)

    [Memory-driven text-to-image generation](https://arxiv.org/abs/2208.07022)

    [RE-IMAGEN: RETRIEVAL-AUGMENTED TEXT-TO-IMAGE GENERATOR](https://arxiv.org/abs/2209.14491)

    [KNN-Diffusion: Image Generation via Large-Scale Retrieval](https://arxiv.org/abs/2204.02849)

    [Retrieval-Augmented Diffusion Models](https://arxiv.org/abs/2204.11824)

    [Text-Guided Synthesis of Artistic Images with Retrieval-Augmented Diffusion Models](https://arxiv.org/abs/2207.13038)

    [X&Fuse: Fusing Visual Information in Text-to-Image Generation](https://arxiv.org/abs/2303.01000)

  - Image Captioning

    [Memory-augmented image captioning](https://ojs.aaai.org/index.php/AAAI/article/view/16220)

    [Retrieval-enhanced adversarial training with dynamic memory-augmented attention for image paragraph captioning](https://www.sciencedirect.com/science/article/pii/S0950705120308595)

    [Retrieval-Augmented Transformer for Image Captioning](https://arxiv.org/abs/2207.13162)

    [Retrieval-augmented image captioning](https://arxiv.org/abs/2302.08268)

    [Reveal: Retrieval-augmented visual-language pre-training with multi-source multimodal knowledge memory](https://arxiv.org/abs/2212.05221)

    [SmallCap: Lightweight Image Captioning Prompted With Retrieval Augmentation](https://arxiv.org/abs/2209.15323)

    [Cross-Modal Retrieval and Semantic Refinement for Remote Sensing Image Captioning](https://www.mdpi.com/2072-4292/16/1/196)

### RAG for Video
  - Video Captioning

  - Video Generation

### RAG for 3D
  - Text-to-3D

### RAG for Knowledge
  - Knowledge Base Question Answering

  - Knowledge Graph Completion

### RAG for Science
  - Drug Discovery
    
    [Retrieval-based controllable molecule generation](https://arxiv.org/abs/2208.11126)
    
    [Prompt-based 3d molecular diffusion models for structure-based drug design](https://openreview.net/forum?id=FWsGuAFn3n)
    
    [A protein-ligand interaction- focused 3d molecular generative framework for generalizable structure- based drug design](https://chemrxiv.org/engage/chemrxiv/article-details/6482d9dbbe16ad5c57af1937)
    
  - Medical Applications
    
    [Genegpt: Augmenting large language models with domain tools for improved access to biomedical information](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10153281/)
    
    [Retrieval-augmented large language models for adolescent idiopathic scoliosis patients in shared decision-making](https://dl.acm.org/doi/abs/10.1145/3584371.3612956)

## Benchmark
  [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://doi.org/10.48550/arXiv.2309.01431)
  
  [CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models](https://doi.org/10.48550/arXiv.2401.17043)
  
  [ARES: An Automated Evaluation Framework for Retrieval-AugmentedGeneration Systems](https://doi.org/10.48550/arXiv.2311.09476)
  
  [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://doi.org/10.48550/arXiv.2309.15217)


## Citing
if you find this work useful, please cite our paper:
```

```


