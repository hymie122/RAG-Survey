# Retrieval-Augmented Generation for AI-Generated Content: A Survey
This repo is constructed for collecting and categorizing papers about RAG according to our survey paper: [*Retrieval-Augmented Generation for AI-Generated Content: A Survey*](https://arxiv.org/abs/2402.19473). Considering the rapid growth of this field, we will continue to update both [paper](https://arxiv.org/abs/2402.19473) and this repo.

# Overview
<div aligncenter><img width="900" alt="image" src="https://github.com/hymie122/RAG-Survey/blob/main/RAG_Overview.jpg">

# Catalogue
## Methods Taxonomy
### RAG Foundations
<div aligncenter><img width="900" alt="image" src="https://github.com/hymie122/RAG-Survey/blob/main/RAG_Foundations.png">

  - Query-based RAG
    
    [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)
    
    [KILT: a Benchmark for Knowledge Intensive Language Tasks](https://arxiv.org/abs/2009.02252)
    
    [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)

    [REPLUG: Retrieval-Augmented Black-Box Language Models](https://arxiv.org/abs/2301.12652)

      [In-Context Retrieval-Augmented Language Models](https://arxiv.org/abs/2302.00083)
    
    [When Language Model Meets Private Library](https://arxiv.org/abs/2210.17236)
    



    [Retrieval-Augmented Score Distillation for Text-to-3D Generation](https://doi.org/10.48550/arXiv.2402.02972)
    
  - Latent Representation-based RAG

    [EditSum: A Retrieve-and-Edit Framework for Source Code Summarization](https://doi.org/10.1109/ASE51524.2021.9678724)

    [A Retrieve-and-Edit Framework for Predicting Structured Outputs](https://proceedings.neurips.cc/paper/2018/hash/cd17d3ce3b64f227987cd92cd701cc58-Abstract.html)

    [DecAF: Joint Decoding of Answers and Logical Forms for Question Answering over Knowledge Bases](https://openreview.net/pdf?id=XHc5zRPxqV9)

    [A Protein-Ligand Interaction-focused 3D Molecular Generative Framework for Generalizable Structure-based Drug Design](https://chemrxiv.org/engage/chemrxiv/article-details/6482d9dbbe16ad5c57af1937)

  - Logit-based RAG

    [Syntax-Aware Retrieval Augmented Code Generation](https://aclanthology.org/2023.findings-emnlp.90)

    [Memory-augmented image captioning](https://aaai.org/papers/01317-memory-augmented-image-captioning/)

    [Retrieval-based neural source code summarization](https://doi.org/10.1145/3377811.3380383)

  - Speculative RAG

    [REST: Retrieval-Based Speculative Decoding](https://doi.org/10.48550/arXiv.2311.08252)
    
    [GPTCache](https://github.com/zilliztech/GPTCache)

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
   
      [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)

      [Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models](https://doi.org/10.48550/arXiv.2310.06117)

      [Active Prompting with Chain-of-Thought for Large Language Models](https://doi.org/10.48550/arXiv.2302.12246)

      [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](http://papers.nips.cc/paper\_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)

      [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://aclanthology.org/2023.emnlp-main.825)

      [Lost in the Middle: How Language Models Use Long Contexts](https://doi.org/10.48550/arXiv.2307.03172)

      [ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model](https://doi.org/10.1109/ICCV51070.2023.00040)

      [Automatic Semantic Augmentation of Language Model Prompts (for Code Summarization)](https://arxiv.org/abs/2304.06815)

      [Retrieval-Based Prompt Selection for Code-Related Few-Shot Learning](https://doi.org/10.1109/ICSE48619.2023.00205)

      [XRICL: Cross-lingual Retrieval-Augmented In-Context Learning for Cross-lingual Text-to-SQL Semantic Parsing](https://doi.org/10.18653/v1/2022.findings-emnlp.384)

      [Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models](https://proceedings.mlr.press/v202/huang23i.html)
   
    - Decoding Tuning

      [InferFix: End-to-End Program Repair with LLMs](https://doi.org/10.1145/3611643.3613892)

      [Synchromesh: Reliable Code Generation from Pre-trained Language Models](https://openreview.net/forum?id=KmtVD97J43e)

      [A Protein-Ligand Interaction-focused 3D Molecular Generative Framework for Generalizable Structure-based Drug Design](https://chemrxiv.org/engage/chemrxiv/article-details/6482d9dbbe16ad5c57af1937)
   
    - Finetune Generator
   
      [Improving Language Models by Retrieving from Trillions of Tokens](https://proceedings.mlr.press/v162/borgeaud22a.html)

      [When Language Model Meets Private Library](https://doi.org/10.18653/v1/2022.findings-emnlp.21)

      [Concept-Aware Video Captioning: Describing Videos With Effective Prior Information](https://doi.org/10.1109/TIP.2023.3307969)

      [Animate-A-Story: Storytelling with Retrieval-Augmented Video Generation](https://doi.org/10.48550/arXiv.2307.06940)

      [Retrieval-Augmented Score Distillation for Text-to-3D Generation](https://doi.org/10.48550/arXiv.2402.02972)
   
  - Result Enhancement

    - Rewrite Output
   
      [Automated Code Editing with Search-Generate-Modify](https://doi.org/10.48550/arXiv.2306.06490)

      [Repair Is Nearly Generation: Multilingual Program Repair with LLMs](https://doi.org/10.1609/aaai.v37i4.25642)

      [Case-based Reasoning for Natural Language Queries over Knowledge Bases](https://doi.org/10.18653/v1/2021.emnlp-main.755)

  - RAG Pipeline Enhancement
    
    - Adaptive Retrieval
      
      - Rule-Baesd
     
        [Active retrieval augmented generation](https://arxiv.org/abs/2305.06983)

        [Efficient Nearest Neighbor Language Models](https://doi.org/10.18653/v1/2021.emnlp-main.461)

        [When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories](https://doi.org/10.18653/v1/2023.acl-long.546)

        [How Can We Know When Language Models Know? On the Calibration of Language Models for Question Answering](https://doi.org/10.1162/tacl\_a\_00407)

        [Large Language Models Struggle to Learn Long-Tail Knowledge](https://proceedings.mlr.press/v202/kandpal23a.html)
    
      - Model-Based
        
        [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://doi.org/10.48550/arXiv.2310.11511)

        [Investigating the Factual Knowledge Boundary of Large Language Models with Retrieval Augmentation](https://doi.org/10.48550/arXiv.2307.11019)

        [Self-Knowledge Guided Retrieval Augmentation for Large Language Models](https://aclanthology.org/2023.findings-emnlp.691)
   
    - Iterative RAG
   
      [RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation](https://aclanthology.org/2023.emnlp-main.151)

      [Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy](https://aclanthology.org/2023.findings-emnlp.620)
    
    


## Applications Taxonomy
<div aligncenter><img width="900" alt="image" src="https://github.com/hymie122/RAG-Survey/blob/main/Applications.png">
  
### RAG for Text
  - Question Answering

    [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://doi.org/10.18653/v1/2021.eacl-main.74)

    [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)

    [Knowledge Graph Based Synthetic Corpus Generation for Knowledge-Enhanced Language Model Pre-training](https://doi.org/10.18653/v1/2021.naacl-main.278)

    [Atlas: Few-shot Learning with Retrieval Augmented Language Models](http://jmlr.org/papers/v24/23-0037.html)

    [Improving Language Models by Retrieving from Trillions of Tokens](https://proceedings.mlr.press/v162/borgeaud22a.html)

    [Self-Knowledge Guided Retrieval Augmentation for Large Language Models](https://aclanthology.org/2023.findings-emnlp.691)

    [Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering](https://doi.org/10.48550/arXiv.2306.04136)

    [Think-on-Graph: Deep and Responsible Reasoning of Large Language Model with Knowledge Graph](https://doi.org/10.48550/arXiv.2307.07697)

    [Nonparametric Masked Language Modeling](https://doi.org/10.18653/v1/2023.findings-acl.132)

    [CL-ReLKT: Cross-lingual Language Knowledge Transfer for Multilingual Retrieval Question Answering](https://doi.org/10.18653/v1/2022.findings-naacl.165)

    [One Question Answering Model for Many Languages with Cross-lingual Dense Passage Retrieval](https://proceedings.neurips.cc/paper/2021/hash/3df07fdae1ab273a967aaa1d355b8bb6-Abstract.html)

    [Entities as Experts: Sparse Memory Access with Entity Supervision](https://arxiv.org/abs/2004.07202)

    [When to Read Documents or QA History: On Unified and Selective Open-domain QA](https://doi.org/10.18653/v1/2023.findings-acl.401)

  - Fact verification

    [CONCRETE: Improving Cross-lingual Fact-checking with Cross-lingual Retrieval](https://aclanthology.org/2022.coling-1.86)

  - Commonsense Reasoning

    [KG-BART: Knowledge Graph-Augmented {BART} for Generative Commonsense Reasoning](https://doi.org/10.1609/aaai.v35i7.16796)

  - Human-Machine Conversation

    [Grounded Conversation Generation as Guided Traverses in Commonsense Knowledge Graphs](https://doi.org/10.18653/v1/2020.acl-main.184)

    [Skeleton-to-Response: Dialogue Generation Guided by Retrieval Memory](https://doi.org/10.18653/v1/n19-1124)

    [Internet-Augmented Dialogue Generation](https://doi.org/10.18653/v1/2022.acl-long.579)

    [BlenderBot 3: a deployed conversational agent that continually learns to responsibly engage](https://doi.org/10.48550/arXiv.2208.03188)

    [A Model of Cross-Lingual Knowledge-Grounded Response Generation for Open-Domain Dialogue Systems](https://doi.org/10.18653/v1/2021.findings-emnlp.33)

  - Neural Machine Translation

    [Neural Machine Translation with Monolingual Translation Memory](https://doi.org/10.18653/v1/2021.acl-long.567)

    [Nearest Neighbor Machine Translation](https://openreview.net/forum?id=7wCBOfJ8hJM)

    [Training Language Models with Memory Augmentation](https://doi.org/10.18653/v1/2022.emnlp-main.382)

  - Event Extraction

    [Retrieval-Augmented Generative Question Answering for Event Argument Extraction](https://doi.org/10.18653/v1/2022.emnlp-main.307)

  - Summarization

    [Retrieval-Augmented Multilingual Keyphrase Generation with Retriever-Generator Iterative Training](https://doi.org/10.18653/v1/2022.findings-naacl.92)

    [Unlimiformer: Long-Range Transformers with Unlimited Length Input](https://doi.org/10.48550/arXiv.2305.01625)

### RAG for Code
  - Code Generation

    [Retrieval Augmented Code Generation and Summarization](https://doi.org/10.18653/v1/2021.findings-emnlp.232)

    [When Language Model Meets Private Library](https://doi.org/10.18653/v1/2022.findings-emnlp.21)

    [DocPrompting: Generating Code by Retrieving the Docs](https://openreview.net/pdf?id=ZTCxT2t2Ru)

    [CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://aclanthology.org/2023.emnlp-main.68)

    [AceCoder: Utilizing Existing Code to Enhance Code Generation](https://arxiv.org/abs/2303.17780)

    [The impact of lexical and grammatical processing on generating code from natural language](https://doi.org/10.18653/v1/2022.findings-acl.173)

  - Code Summary

    [Retrieval-based neural source code summarization](https://doi.org/10.1145/3377811.3380383)

    [Retrieve and Refine: Exemplar-based Neural Comment Generation](https://doi.org/10.1145/3324884.3416578)

    [RACE: Retrieval-augmented Commit Message Generation](https://doi.org/10.18653/v1/2022.emnlp-main.372)

    [BashExplainer: Retrieval-Augmented Bash Code Comment Generation based on Fine-tuned CodeBERT](https://doi.org/10.1109/ICSME55016.2022.00016)

  - Code Completion

    [ReACC: A Retrieval-Augmented Code Completion Framework](https://doi.org/10.18653/v1/2022.acl-long.431)

    [RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation](https://aclanthology.org/2023.emnlp-main.151)

    [CoCoMIC: Code Completion By Jointly Modeling In-file and Cross-file Context](https://doi.org/10.48550/arXiv.2212.10007)

  - Automatic Program Repair

    [Repair Is Nearly Generation: Multilingual Program Repair with LLMs](https://doi.org/10.1609/aaai.v37i4.25642)

    [Retrieval-Based Prompt Selection for Code-Related Few-Shot Learning](https://doi.org/10.1109/ICSE48619.2023.00205)

    [InferFix: End-to-End Program Repair with LLMs](https://doi.org/10.1145/3611643.3613892)

  - Text-to-SQL and Code-based Semantic Parsing

    [XRICL: Cross-lingual Retrieval-Augmented In-Context Learning for Cross-lingual Text-to-SQL Semantic Parsing](https://doi.org/10.18653/v1/2022.findings-emnlp.384)

    [Synchromesh: Reliable Code Generation from Pre-trained Language Models](https://openreview.net/forum?id=KmtVD97J43e)

    [Leveraging Code to Improve In-context Learning for Semantic Parsing](https://arxiv.org/abs/2311.09519)

  - Others

    [Leveraging training data in few-shot prompting for numerical reasoning](https://arxiv.org/abs/2305.18170)

### RAG for Audio
  - Audio Generation

    [Retrieval-Augmented Text-to-Audio Generation](https://doi.org/10.48550/arXiv.2309.08051)

    [Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://doi.org/10.1109/ICASSP49357.2023.10095969)

    [Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://doi.org/10.1109/ICASSP49357.2023.10095969)

  - Audio Captioning

    [RECAP: Retrieval-Augmented Audio Captioning](https://doi.org/10.48550/arXiv.2309.09836)

    [Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://doi.org/10.1109/ICASSP49357.2023.10095969)

    [CNN architectures for large-scale audio classification](https://doi.org/10.1109/ICASSP.2017.7952132)

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

    [Retrieval Augmented Convolutional Encoder-decoder Networks for Video Captioning](https://doi.org/10.1145/3539225)

    [Concept-Aware Video Captioning: Describing Videos With Effective Prior Information](https://doi.org/10.1109/TIP.2023.3307969)

  - Video Generation

    [Animate-A-Story: Storytelling with Retrieval-Augmented Video Generation](https://doi.org/10.48550/arXiv.2307.06940)

    [Frozen in Time: {A} Joint Video and Image Encoder for End-to-End Retrieval](https://doi.org/10.1109/ICCV48922.2021.00175)

### RAG for 3D
  - Text-to-3D

    [ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model](https://doi.org/10.1109/ICCV51070.2023.00040)

    [AMD: Anatomical Motion Diffusion with Interpretable Motion Decomposition and Fusion](https://arxiv.org/abs/2312.12763)

    [Retrieval-Augmented Score Distillation for Text-to-3D Generation](https://doi.org/10.48550/arXiv.2402.02972)

### RAG for Knowledge
  - Knowledge Base Question Answering

    [ReTraCk: A Flexible and Efficient Framework for Knowledge Base Question Answering](https://doi.org/10.18653/v1/2021.acl-demo.39)

    [Case-based Reasoning for Natural Language Queries over Knowledge Bases](https://doi.org/10.18653/v1/2021.emnlp-main.755)

    [Logical Form Generation via Multi-task Learning for Complex Question Answering over Knowledge Bases](https://aclanthology.org/2022.coling-1.145)

  - Knowledge Graph Completion

    [Retrieval-Enhanced Generative Model for Large-Scale Knowledge Graph Completion](https://doi.org/10.1145/3539618.3592052)

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
@misc{zhao2024retrievalaugmented,
      title={Retrieval-Augmented Generation for AI-Generated Content: A Survey}, 
      author={Penghao Zhao and Hailin Zhang and Qinhan Yu and Zhengren Wang and Yunteng Geng and Fangcheng Fu and Ling Yang and Wentao Zhang and Bin Cui},
      year={2024},
      eprint={2402.19473},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


