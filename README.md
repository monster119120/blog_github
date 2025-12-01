# 工业界大模型技术全栈记录

## 📖 简介

回看读博期间研究的大模型技术，不得不说视野非常小。自从进入大厂基座模型组，深感工业界已经领先学术界一大截。本项目旨在记录工业界真实大模型训练和推理的方方面面，希望能够对在读的同学提供一些帮助，也是对自己学习知识的总结。

本项目内容将持续更新，涵盖数据、训练、推理部署、评估以及应用等全栈技术。

- **Github**: [Industrial_LLM_tutorial](https://github.com/monster119120/Industrial_LLM_tutorial) - 欢迎 Star 和 PR！
- **知乎专栏**: [大模型全栈记录](https://www.zhihu.com/column/c_1934673782448062552)
- **微信公众号**: **大模型全栈开发**
- **个人经历**: [博士之路](https://www.zhihu.com/column/c_1934959737918697853)
- **论文笔记**: [大模型论文阅读笔记](https://www.zhihu.com/column/c_1939016923317769755)

---

## 📚 目录

1. [大模型数据](#1-大模型数据)
2. [大模型训练](#2-大模型训练)
3. [大模型推理部署](#3-大模型推理部署)
4. [大模型评估](#4-大模型评估)
5. [大模型应用](#5-大模型应用)

---

## 1. 大模型数据

数据是大模型的基石。本章节涵盖数据的获取、清洗、合成等流程。

- **[数据下载](https://github.com/monster119120/blog_github/blob/main/1_data/download.md)**
- **[数据预处理](https://github.com/monster119120/blog_github/blob/main/1_data/preprocess.md)**
- **[数据去重](https://github.com/monster119120/blog_github/blob/main/1_data/deduplicate.md)**
- **[DeepSeek CodeV2 Math 数据](https://github.com/monster119120/blog_github/blob/main/1_data/deepseek_codev2_math.md)**

### 核心流程
- **原始数据爬取**: 网页、书籍、代码、多语言数据等。
- **数据清洗**: 格式标准化、去重（MinHash, LSH）、分类、打分。
- **数据采样**: 不同领域数据的配比。
- **数据合成**: 预训练、SFT、RL 数据的合成策略。

---

## 2. 大模型训练

本章节深入探讨大模型训练的算法与基础设施。

### 算法 (Algorithm)
- **[MoE (Mixture of Experts)](https://github.com/monster119120/blog_github/blob/main/2_training/algo/moe_algo.md)**
- **[MoE 剪枝](https://github.com/monster119120/blog_github/blob/main/2_training/algo/moe_pruning.md)**
- **[100B MoE 超参](https://github.com/monster119120/blog_github/blob/main/2_training/algo/100b_moe_hyper_param.md)**
- **[Post-training](https://github.com/monster119120/blog_github/blob/main/2_training/algo/post_train.md)**
- **[PPO](https://github.com/monster119120/blog_github/blob/main/2_training/algo/ppo.md)**
- **[Reward Rule](https://github.com/monster119120/blog_github/blob/main/2_training/algo/reward_rule.md)**
- **长文训练**:
    - [位置编码基础理论](https://github.com/monster119120/blog_github/blob/main/2_training/algo/long_context/大模型长文训练（一）位置编码基础理论.md)
    - [长度外推](https://github.com/monster119120/blog_github/blob/main/2_training/algo/long_context/大模型长文训练（二）长度外推.md)
    - [YaRN代码详解](https://github.com/monster119120/blog_github/blob/main/2_training/algo/long_context/大模型长文训练（三）YaRN代码详解.md)
- **Attention 变体**: 
    - [NSA (Native Sparse Attention)](https://github.com/monster119120/blog_github/blob/main/2_training/algo/nsa/Native_Sparse_Attention（一）图解.md)

### 基础设施 (Infra)
- **Megatron-LM 系列**:
    - [Megatron-LM 详解](https://github.com/monster119120/blog_github/blob/main/2_training/infra/megatron_detail.md)
    - [代码结构分析](https://github.com/monster119120/blog_github/blob/main/2_training/infra/megatron/Megatron-LM（一）代码结构分析.md)
    - [代码运行流程](https://github.com/monster119120/blog_github/blob/main/2_training/infra/megatron/Megatron-LM（二）代码运行流程.md)
    - [代码调试指南](https://github.com/monster119120/blog_github/blob/main/2_training/infra/megatron/Megatron-LM（三）代码调试指南.md)
- **并行策略**: CP, TP, EP, SP, Pipeline Parallelism。
- **加速技术**:
    - [Flash Attention v1](https://github.com/monster119120/blog_github/blob/main/2_training/infra/flash_attn/五张图片看懂Flash Attention v1（一）.md)
    - [Flash Attention v2](https://github.com/monster119120/blog_github/blob/main/2_training/infra/flash_attn/Flash%20Attention%20v2（一）.md)
    - [Flash Attention v3](https://github.com/monster119120/blog_github/blob/main/2_training/infra/flash_attn/Flash%20Attention%20v3（一）%20.md)
    - [Ring Attention](https://github.com/monster119120/blog_github/blob/main/2_training/infra/ring_attn/ring_attn（一）.md)
    - Deepspeed, Torchtiton

---

## 3. 大模型推理部署

关注大模型的高效推理与服务化部署。

### 算法
- KV Cache 裁剪
- 投机采样 (Speculative Decoding)
- 量化 (Quantization)
- RAG (Retrieval Augmented Generation)

### 基础设施
- **[推理算法](https://github.com/monster119120/blog_github/blob/main/3_inference/algo/)**
- **[推理架构](https://github.com/monster119120/blog_github/blob/main/3_inference/infra/)**
- vLLM, SGLang
- Continuous Batching, Paged Attention

---

## 4. 大模型评估

- **[评估概览](https://github.com/monster119120/blog_github/blob/main/4_evaluation/README.md)**
- Pretrain 评估
- Posttrain 评估

---

## 5. 大模型应用

- **[应用概览](https://github.com/monster119120/blog_github/blob/main/5_application/README.md)**
- Agent & MCP
- Deep Research
- 搜索增强

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=monster119120/Industrial_LLM_tutorial&type=Date)](https://www.star-history.com/#monster119120/Industrial_LLM_tutorial&Date)