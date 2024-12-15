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

#### How to use this guide?
Start with Easy links in each section. 
Each area has multiple types of subtopics each of which will go more in depth. In the event there are no articles, feel free to email for additions or raise a PR.

#### Table of contents
- [This section talks about various aspects of the Agentic LLMs](#this-section-talks-about-various-aspects-of-the-agentic-llms)
  - [🟩 Methodology](#-methodology)
    - [Distillation](#distillation)
  - [🟩 Datasets](#-datasets)
  - [🟩 Pipeline](#-pipeline)
    - [Training](#training)
    - [Inference](#inference)
      - [RAG](#rag)
    - [Prompting](#prompting)
  - [🟩 FineTuning](#-finetuning)
    - [Quantized FineTuning](#quantized-finetuning)
    - [DPT](#dpt)
    - [ORPO](#orpo)
  - [🟩 Quantization](#-quantization)
    - [Post Training Quantization](#post-training-quantization)
      - [Static/Dynamic Quantization](#staticdynamic-quantization)
      - [GPTQ](#gptq)
      - [GGUF](#gguf)
      - [LLM.int8()](#llmint8)
    - [Quantization Aware Training → 1BIT LLM](#quantization-aware-training--1bit-llm)
  - [🟩 RL in LLM](#-rl-in-llm)
  - [🟩 Coding](#-coding)
    - [Torch Fundamentals](#torch-fundamentals)
  - [🟩 Engineering](#-engineering)
    - [Flash Attention 2](#flash-attention-2)
    - [KV Cache](#kv-cache)
    - [Inference → Batched?](#inference--batched)
    - [Python Advanced](#python-advanced)
      - [Decorators](#decorators)
      - [Context Managers](#context-managers)
    - [Triton Kernels](#triton-kernels)
    - [CuDA](#cuda)
    - [JAX / XLA JIT compilers](#jax--xla-jit-compilers)
    - [Model Exporting (vLLM, Llama.cpp, QLoRA)](#model-exporting-vllm-llamacpp-qlora)
    - [ML Debugging](#ml-debugging)
  - [🟩 Benchmarks](#-benchmarks)
  - [🟩 Modifications](#-modifications)
    - [Model Merging](#model-merging)
      - [Linear Mapping](#linear-mapping)
      - [SLERP](#slerp)
      - [TIES](#ties)
      - [DARE](#dare)
    - [MoE](#moe)
  - [🟩 Misc Algorithms](#-misc-algorithms)
    - [Chained Matrix Unit](#chained-matrix-unit)
    - [Gradient Checkpointing](#gradient-checkpointing)
    - [Chunked Cross Entropy](#chunked-cross-entropy)
    - [BPE](#bpe)
  - [🟩 Explainability](#-explainability)
    - [Sparse Autoencoders](#sparse-autoencoders)
    - [Task Vectors](#task-vectors)
    - [Counterfactuals](#counterfactuals)
  - [🟩 MultiModal Transformers](#-multimodal-transformers)
    - [Audio](#audio)
      - [Whisper Models](#whisper-models)
      - [Diarization](#diarization)
  - [🟩 Adversarial methods](#-adversarial-methods)
  - [🟩 Misc](#-misc)
  - [🟩 Add to the guide:](#-add-to-the-guide)




### 🟩 Model Architecture
This section talks about the key aspects of LLM architecture.
> 📝 Try to cover basics of Transformers, then understand the GPT architecture before diving deeper into other concepts
#### Transformer Architecture
- [Jay Alamar - Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) ![Easy](https://img.shields.io/badge/difficulty-Easy-green)
##### Tokenization
##### Positional Encoding
###### Rotational Positional Encoding
###### Rotary Positional Encoding

#### GPT Architecture

#### Architecture Llama


#### Attention

#### Loss
##### Cross-Entropy Loss

---
### 🟩 Agentic LLMs
This section talks about various aspects of the Agentic LLMs
---
### 🟩 Methodology
This section tries to cover various methodologies used in LLMs. 
#### Distillation

---
### 🟩 Datasets

---
### 🟩 Pipeline
#### Training
#### Inference
##### RAG
#### Prompting

---
### 🟩 FineTuning
#### Quantized FineTuning
#### DPT
#### ORPO

---
### 🟩 Quantization
#### Post Training Quantization
##### Static/Dynamic Quantization
##### GPTQ
##### GGUF
##### LLM.int8()
#### Quantization Aware Training → 1BIT LLM

---
### 🟩 RL in LLM

---
### 🟩 Coding
#### Torch Fundamentals

---
### 🟩 Engineering
#### Flash Attention 2
#### KV Cache
#### Inference → Batched?
#### Python Advanced
##### Decorators
##### Context Managers
#### Triton Kernels
#### CuDA
#### JAX / XLA JIT compilers
#### Model Exporting (vLLM, Llama.cpp, QLoRA)
#### ML Debugging

---
### 🟩 Benchmarks

---
### 🟩 Modifications
#### Model Merging
##### Linear Mapping
##### SLERP
##### TIES
##### DARE
#### MoE

---
### 🟩 Misc Algorithms
#### Chained Matrix Unit
#### Gradient Checkpointing
#### Chunked Cross Entropy
#### BPE

---
### 🟩 Explainability
#### Sparse Autoencoders
- [Sparse AutoEncoders Explained](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html) ![Easy](https://img.shields.io/badge/difficulty-Easy-green)
#### Task Vectors
#### Counterfactuals

---
### 🟩 MultiModal Transformers
#### Audio
##### Whisper Models
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