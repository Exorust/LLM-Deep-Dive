<div align="center">
  <img src="img/master.png" alt="Robot Image">
  <h1>The LLM Deep Dive</h1>
  <p align="center">
    🐦 <a href="https://twitter.com/charoori_ai">Follow me on Twitter</a> •
    📧 <a href="mailto:chandrahas.aroori@gmail.com?subject=LLM%20Cookbook">Contact on Email</a>
  </p>
</div>
<br/>

In an age of GPT, I'm going to handwrite the best links I've used to learn LLMs.

**Welcome.**

PS: This is for people trying to go deeper. If you want something kind of basic, look elsewhere.

#### ◻️How to use this guide?
Start by going through the Table of contents. See what you've already read and what you haven't.
Then, start with Easy links in each section. 
Each area has multiple types of subtopics each of which will go more in depth. In the event there are no articles, feel free to email for additions or raise a PR.

#### ◻️Table of contents
- [🟩 Model Architecture](#-model-architecture)
  - [◻️Transformer Architecture](#️transformer-architecture)
    - [Tokenization](#tokenization)
    - [Positional Encoding](#positional-encoding)
      - [Rotational Positional Encoding](#rotational-positional-encoding)
      - [Rotary Positional Encoding](#rotary-positional-encoding)
  - [◻️GPT Architecture](#️gpt-architecture)
  - [◻️Attention](#️attention)
  - [◻️Loss](#️loss)
    - [Cross-Entropy Loss](#cross-entropy-loss)
- [🟩 Agentic LLMs](#-agentic-llms)
- [🟩 Methodology](#-methodology)
  - [◻️Distillation](#️distillation)
- [🟩 Datasets](#-datasets)
- [🟩 Pipeline](#-pipeline)
  - [◻️Training](#️training)
  - [◻️Inference](#️inference)
    - [RAG](#rag)
  - [◻️Prompting](#️prompting)
- [🟩 FineTuning](#-finetuning)
  - [◻️Quantized FineTuning](#️quantized-finetuning)
  - [◻️LoRA](#️lora)
  - [◻️DPO](#️dpo)
  - [◻️ORPO](#️orpo)
  - [◻️RLHF](#️rlhf)
- [🟩 Quantization](#-quantization)
  - [◻️Post Training Quantization](#️post-training-quantization)
    - [Static/Dynamic Quantization](#staticdynamic-quantization)
    - [GPTQ](#gptq)
    - [GGUF](#gguf)
    - [LLM.int8()](#llmint8)
  - [◻️Quantization Aware Training → 1BIT LLM](#️quantization-aware-training--1bit-llm)
- [🟩 RL in LLM](#-rl-in-llm)
- [🟩 Coding](#-coding)
  - [◻️Torch Fundamentals](#️torch-fundamentals)
- [🟩 Deployment](#-deployment)
- [🟩 Engineering](#-engineering)
  - [◻️Flash Attention 2](#Flash-Attention-2)
  - [◻️KV Cache](#️kv-cache)
  - [◻️Batched Inference](#️batched-inference)
  - [◻️Python Advanced](#️python-advanced)
    - [Decorators](#decorators)
    - [Context Managers](#context-managers)
  - [◻️Triton Kernels](#️triton-kernels)
  - [◻️CuDA](#️cuda)
  - [◻️JAX / XLA JIT compilers](#️jax--xla-jit-compilers)
  - [◻️Model Exporting (vLLM, Llama.cpp, QLoRA)](#️model-exporting-vllm-llamacpp-qlora)
  - [◻️ML Debugging](#️ml-debugging)
- [🟩 Benchmarks](#-benchmarks)
- [🟩 Modifications](#-modifications)
  - [◻️Model Merging](#️model-merging)
    - [Linear Mapping](#linear-mapping)
    - [SLERP](#slerp)
    - [TIES](#ties)
    - [DARE](#dare)
  - [◻️MoE](#️moe)
- [🟩 Misc Algorithms](#-misc-algorithms)
  - [◻️Chained Matrix Unit](#️chained-matrix-unit)
  - [◻️Gradient Checkpointing](#️gradient-checkpointing)
  - [◻️Chunked Cross Entropy](#️chunked-cross-entropy)
  - [◻️BPE](#️bpe)
- [🟩 Explainability](#-explainability)
  - [◻️Sparse Autoencoders](#️sparse-autoencoders)
  - [◻️Task Vectors](#️task-vectors)
  - [◻️Counterfactuals](#️counterfactuals)
- [🟩 MultiModal Transformers](#-multimodal-transformers)
  - [◻️Audio](#️audio)
    - [Whisper Models](#whisper-models)
    - [Diarization](#diarization)
- [🟩 Adversarial methods](#-adversarial-methods)
- [🟩 Misc](#-misc)
- [🟩 Add to the guide:](#-add-to-the-guide)




### 🟩 Model Architecture
This section talks about the key aspects of LLM architecture.
> 📝 Try to cover basics of Transformers, then understand the GPT architecture before diving deeper into other concepts
- [Numbers every LLM Developer should know](https://github.com/ray-project/llm-numbers#1-mb-gpu-memory-required-for-1-token-of-output-with-a-13b-parameter-model)![Easy](https://img.shields.io/badge/difficulty-Easy-green)
#### ◻️Transformer Architecture
- [Jay Alamar - Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) ![Easy](https://img.shields.io/badge/difficulty-Easy-green)
- [Umar Jamil: Attention](https://www.youtube.com/watch?v=bCz4OMemCcA&) ![Easy](https://img.shields.io/badge/difficulty-Easy-green)
- [Large Scale Transformer model training with Tensor Parallel (TP)](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)![Easy](https://img.shields.io/badge/difficulty-Easy-green)
- [RoPE (Rotary positional embeddings) explained: The positional workhorse of modern LLMs](https://www.youtube.com/watch?v=GQPOtyITy54&t=66s)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Rotary Embeddings: A Relative Revolution | EleutherAI Blog](https://blog.eleuther.ai/rotary-embeddings/)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)

##### Tokenization
- [Tokenization in large language models, explained](https://seantrott.substack.com/p/tokenization-in-large-language-models)![Easy](https://img.shields.io/badge/difficulty-Easy-green)
- [LLM Tokenizers Explained: BPE Encoding, WordPiece and SentencePiece](https://www.youtube.com/watch?v=hL4ZnAWSyuU)![Easy](https://img.shields.io/badge/difficulty-Easy-green)
- [SentencePiece Tokenizer Demystified](https://towardsdatascience.com/sentencepiece-tokenizer-demystified-d0a3aac19b15/)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
##### Positional Encoding
###### Rotational Positional Encoding
###### Rotary Positional Encoding
-[Rotary Positional Encoding Explained](https://medium.com/@ngiengkianyew/understanding-rotary-positional-encoding-40635a4d078e)

#### ◻️GPT Architecture
- [Jay Alamar - Illustrated GPT2](https://jalammar.github.io/illustrated-gpt2/) ![Easy](https://img.shields.io/badge/difficulty-Easy-green)
- [Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20](https://github.com/karpathy/llm.c/discussions/481) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Umar Jamil: Llama Explained](https://www.youtube.com/watch?v=Mn_9W1nCFLo) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Umar Jamil: Llama 2 from Scratch](https://www.youtube.com/watch?v=oM4VmoabDAI) ![Hard](https://img.shields.io/badge/difficulty-Hard-red)

#### ◻️Attention
- [Multi-Head Attention (MHA), Multi-Query Attention (MQA), Grouped Query Attention (GQA) Explained](https://www.youtube.com/watch?v=o68RRGxAtDo)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- 
#### ◻️Loss
##### Cross-Entropy Loss
- [Cross Entropy in Large Language Models (LLMs)](https://medium.com/ai-assimilating-intelligence/cross-entropy-in-large-language-models-llms-4f1c842b5fca)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
---
### 🟩 Agentic LLMs
-[Agentic LLMs Deep Dive](https://www.aimon.ai/posts/deep-dive-into-agentic-llm-frameworks)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
This section talks about various aspects of the Agentic LLMs

---
### 🟩 Methodology
This section tries to cover various methodologies used in LLMs. 
#### ◻️Distillation
- [LLM distillation demystified: a complete guide](https://snorkel.ai/blog/llm-distillation-demystified-a-complete-guide/)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Distilling step-by-step: Outperforming larger language models with less training data and smaller model sizes](https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
---
### 🟩 Datasets

---
### 🟩 Pipeline
#### ◻️Training
#### ◻️Inference
##### RAG
- [Introduction to Facebook AI Similarity Search (Faiss)](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
#### ◻️Prompting

---
### 🟩 FineTuning
-[Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
#### ◻️Quantized FineTuning
- [Umar Jamil: Quantization](https://www.youtube.com/watch?v=0VdNflU08yA) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
#### ◻️LoRA
- [Umar Jamil: LoRA Explained](https://www.youtube.com/watch?v=PXWYUTMt-AU) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
#### ◻️DPO
- [Umar Jamil: DPO Explained](https://www.youtube.com/watch?v=hvGa5Mba4c8) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
#### ◻️ORPO
#### ◻️RLHF
- [Umar Jamil: RLHF Explained](https://www.youtube.com/watch?v=qGyFrqc34yc) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Policy Gradients: The Foundation of RLHF](https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
---
### 🟩 Quantization
- [HuggingFace Quantization Overview](https://huggingface.co/docs/transformers/main/en/quantization/overview)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
#### ◻️Post Training Quantization
##### Static/Dynamic Quantization
##### GPTQ
##### GGUF
##### LLM.int8()
#### ◻️Quantization Aware Training → 1BIT LLM

---
### 🟩 RL in LLM

---
### 🟩 Coding
#### ◻️Torch Fundamentals

---
### 🟩 Deployment
- [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)![Hard](https://img.shields.io/badge/difficulty-Hard-red)
- [LLM Inference Optimizations — Continuous Batching and Selective Batching, Orca](https://medium.com/byte-sized-ai/inference-optimizations-1-continuous-batching-03408c673098)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [[vLLM] LLM Inference Optimizations: Chunked Prefill and Decode-Maximal Batching](https://medium.com/byte-sized-ai/llm-inference-optimizations-2-chunked-prefill-764407b3a67a)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [LLM Inference Series: 2. The two-phase process behind LLMs’ responses](https://medium.com/@plienhar/llm-inference-series-2-the-two-phase-process-behind-llms-responses-1ff1ff021cd5)![Hard](https://img.shields.io/badge/difficulty-Hard-red)
- [LLM Inference Series: 4. KV caching, a deeper look](https://medium.com/@plienhar/llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8)![Hard](https://img.shields.io/badge/difficulty-Hard-red)
- [How KV caches impact time to first token for LLMs](https://www.glean.com/blog/glean-kv-caches-llm-latency)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Generation with LLMs](https://charoori.notion.site/Generation-with-LLMs-17d311b8ed1e819b99a3e79112e00ca6)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)


---
### 🟩 Engineering
- [ML Engineering; Used for training BLOOM](https://github.com/stas00/ml-engineering/tree/master) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)

- [Low Level Technicals of LLMs](https://www.youtube.com/watch?v=pRM_P6UfdIc) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Fixing bugs in Llama, Mistral, Gemma](https://www.youtube.com/watch?v=TKmfBnW0mQA) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [PyTorch Conference Mini Talk](https://www.youtube.com/watch?v=PdtKkc5jB4g) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [PyTorch Engineers Meeting Talk](https://www.youtube.com/watch?v=MQwryfkydc0) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Hugging Face Collab Blog](https://huggingface.co/blog/unsloth-trl) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Summary of Designing Machine Learning Systems](https://github.com/serodriguez68/designing-ml-systems-summary)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [System Design for Recommendations and Search](https://eugeneyan.com/writing/system-design-for-discovery/)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Recommender Systems, Not Just Recommender Models](https://medium.com/nvidia-merlin/recommender-systems-not-just-recommender-models-485c161c755e)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Blueprints for recommender system architectures: 10th anniversary edition](https://amatria.in/blog/RecsysArchitectures)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
#### ◻️Flash Attention 2
- [Flash Attention Machine Learning](https://www.youtube.com/watch?v=N1EZpa7lZc8)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [FLASHATTENTION: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
#### ◻️KV Cache
#### ◻️Batched Inference
#### ◻️Python Advanced
##### Decorators
##### Context Managers
#### ◻️Triton Kernels
#### ◻️CuDA
- [CUDA / GPU Mode lecture Talk](https://www.youtube.com/watch?v=hfb_AIhDYnA) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
#### ◻️JAX / XLA JIT compilers
#### ◻️Model Exporting (vLLM, Llama.cpp, QLoRA)
-[QLoRA: Fine-Tuning Large Language Models (LLM’s)](https://medium.com/@dillipprasad60/qlora-explained-a-deep-dive-into-parametric-efficient-fine-tuning-in-large-language-models-llms-c1a4794b1766)![Hard](https://img.shields.io/badge/difficulty-Hard-red)
-[]()
#### ◻️ML Debugging

---
### 🟩 Benchmarks

---
### 🟩 Modifications
#### ◻️Model Merging
-[An Introduction to Model Merging for LLMs](https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
##### Linear Mapping
##### SLERP
-[Merging tokens to accelerate LLM inference with SLERP](https://medium.com/towards-data-science/merging-tokens-to-accelerate-llm-inference-with-slerp-38a32bf7f194)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
##### TIES
##### DARE
#### ◻️MoE

---
### 🟩 Misc Algorithms
#### ◻️Chained Matrix Unit
#### ◻️Gradient Checkpointing
#### ◻️Chunked Cross Entropy
#### ◻️BPE

---
### 🟩 Explainability
#### ◻️Sparse Autoencoders
- [Sparse AutoEncoders Explained](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html) ![Easy](https://img.shields.io/badge/difficulty-Easy-green)
#### ◻️Task Vectors
#### ◻️Counterfactuals

---
### 🟩 MultiModal Transformers
#### ◻️Audio
##### Whisper Models
- [Whisper Model Explained](https://www.notta.ai/en/blog/how-to-use-whisper)
##### Diarization

---
### 🟩 Adversarial methods

---

### 🟩 Misc
- [Tweet on what to learn in ML (RT by Karpathy)](https://x.com/youraimarketer/status/1778992208697258152) ![Hard](https://img.shields.io/badge/difficulty-Hard-red)
- [Schedule - CS 685, Spring 2024, UMass Amherst](https://people.cs.umass.edu/~miyyer/cs685/schedule.html)![Hard](https://img.shields.io/badge/difficulty-Hard-red)
---


### 🟩 Add to the guide:
Add links you find useful through pull requests. 
<!-- Use the following code for sample links:
- [Link 1](http://example.com) ![Hard](https://img.shields.io/badge/difficulty-Hard-red)
- [Link 2](http://example.com) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Link 3](http://example.com) ![Easy](https://img.shields.io/badge/difficulty-Easy-green) -->
