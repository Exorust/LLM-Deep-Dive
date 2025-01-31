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
  - [◻️Flash Attention 2](#️flash-attention-2)
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
#### ◻️Transformer Architecture
- [Jay Alamar - Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) ![Easy](https://img.shields.io/badge/difficulty-Easy-green)
- [Umar Jamil: Attention](https://www.youtube.com/watch?v=bCz4OMemCcA&) ![Easy](https://img.shields.io/badge/difficulty-Easy-green)

##### Tokenization
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

#### ◻️Loss
##### Cross-Entropy Loss

---
### 🟩 Agentic LLMs
-[Agentic LLMs Deep Dive](https://www.aimon.ai/posts/deep-dive-into-agentic-llm-frameworks)
This section talks about various aspects of the Agentic LLMs

---
### 🟩 Methodology
This section tries to cover various methodologies used in LLMs. 
#### ◻️Distillation

---
### 🟩 Datasets

---
### 🟩 Pipeline
#### ◻️Training
#### ◻️Inference
##### RAG
#### ◻️Prompting

---
### 🟩 FineTuning
#### ◻️Quantized FineTuning
- [Umar Jamil: Quantization](https://www.youtube.com/watch?v=0VdNflU08yA) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
#### ◻️LoRA
- [Umar Jamil: LoRA Explained](https://www.youtube.com/watch?v=PXWYUTMt-AU) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
#### ◻️DPO
- [Umar Jamil: DPO Explained](https://www.youtube.com/watch?v=hvGa5Mba4c8) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
#### ◻️ORPO
#### ◻️RLHF
- [Umar Jamil: RLHF Explained](https://www.youtube.com/watch?v=qGyFrqc34yc) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)

---
### 🟩 Quantization
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
- [Achieve 23x LLM Inference Throughput & Reduce p50 Latency](https://charoori.notion.site/Achieve-23x-LLM-Inference-Throughput-Reduce-p50-Latency-17d311b8ed1e818daf2bf1287647e99f)![Hard](https://img.shields.io/badge/difficulty-Hard-red)
- [LLM Inference Optimizations — Continuous Batching (Dynamic Batching) and Selective Batching, Orca | by Don Moon | Byte-Sized AI | Medium](https://charoori.notion.site/LLM-Inference-Optimizations-Continuous-Batching-Dynamic-Batching-and-Selective-Batching-Orca--17d311b8ed1e81718f56f7a26ec8f5b8)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [LLM Inference Optimizations - Chunked Prefills and Decode Maximal Batching | by Don Moon | Byte-Sized AI](https://charoori.notion.site/LLM-Inference-Optimizations-Chunked-Prefills-and-Decode-Maximal-Batching-by-Don-Moon-Byte-Size-17d311b8ed1e81d49ec0fedbd351821b)![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [LLM Inference Series: 2. The two-phase process behind LLMs' responses | by Pierre Lienhart | Medium](https://charoori.notion.site/LLM-Inference-Series-2-The-two-phase-process-behind-LLMs-responses-by-Pierre-Lienhart-Medium-17d311b8ed1e812d9a80e14442d55eb1)![Hard](https://img.shields.io/badge/difficulty-Hard-red)
- [LLM Inference Series: 4. KV caching, a deeper look | by Pierre Lienhart | Medium](https://charoori.notion.site/LLM-Inference-Series-4-KV-caching-a-deeper-look-by-Pierre-Lienhart-Medium-17d311b8ed1e81a499d7f1cd07cd97c8)![Hard](https://img.shields.io/badge/difficulty-Hard-red)



---
### 🟩 Engineering
- [ML Engineering; Used for training BLOOM](https://github.com/stas00/ml-engineering/tree/master) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)

- [Low Level Technicals of LLMs](https://www.youtube.com/watch?v=pRM_P6UfdIc) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Fixing bugs in Llama, Mistral, Gemma](https://www.youtube.com/watch?v=TKmfBnW0mQA) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [PyTorch Conference Mini Talk](https://www.youtube.com/watch?v=PdtKkc5jB4g) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [PyTorch Engineers Meeting Talk](https://www.youtube.com/watch?v=MQwryfkydc0) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Hugging Face Collab Blog](https://huggingface.co/blog/unsloth-trl) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
#### ◻️Flash Attention 2
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
#### ◻️ML Debugging

---
### 🟩 Benchmarks

---
### 🟩 Modifications
#### ◻️Model Merging
##### Linear Mapping
##### SLERP
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
---


### 🟩 Add to the guide:
Add links you find useful through pull requests. 
<!-- Use the following code for sample links:
- [Link 1](http://example.com) ![Hard](https://img.shields.io/badge/difficulty-Hard-red)
- [Link 2](http://example.com) ![Medium](https://img.shields.io/badge/difficulty-Medium-yellow)
- [Link 3](http://example.com) ![Easy](https://img.shields.io/badge/difficulty-Easy-green) -->
