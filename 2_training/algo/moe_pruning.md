Structured LLM Pruning

**重点看**：

* 同构型剪枝 (模型前后同构) 优于 结构化剪枝 (整块剪枝) 优于 非结构化剪枝 (造成稀疏矩阵，部署困难)
* 工业界论文，简单有效的方案最好
* 关注实验结论

#### [层剪枝] [需要 Retrain] [Submission to ICLR'25] Xiaodong Chen, Yuxuan Hu, Jing Zhang, Yanling Wang, Cuiping Li, and Hong Chen. 2025. Streamlining Redundant Layers to Compress Large Language Models. arXiv:2403.19135 [cs].
**Takeway**：LLM-Streamline comprises two components: layer pruning and layer replacement. First, certain contiguous redundant layers are pruned from the LLMs based on cosine similarity importance metric; Then, a lightweight network is trained on a small subset of SlimPajama to replace the pruned layers to restore the model’s performance.

#### [层剪枝] [需要 Retrain] [Submission to ICLR'25] Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, and Daniel A. Roberts. 2024. The Unreasonable Ineffectiveness of the Deeper Layers. arXiv:2403.17887 [cs].
**Takeway**：A simple [layer/depth](https://github.com/liyunqianggyn/Awesome-LLMs-Pruning/blob/main/concepts/other_concepts.md/#layer-or-depth-pruning) pruning to remove n consecutive or contiguous layers from popular families of open-weight pretrained LLMs by minimizing the angular distance between layers' representations. Parameter-efficient finetuning method is applied to further reduce computational resources of finetuning.

#### [需要 LoRA Retrain] [Submission to ICLR'24] Song Guo, Jiahang Xu, Li Lyna Zhang, and Mao Yang. 2023. Compresso: Structured Pruning with Collaborative Prompting Learns Compact Large Language Models. arXiv:2310.05015 [cs].
**Takeway**：Retrain LLMs' weights with lightweight [LoRA](https://github.com/liyunqianggyn/Awesome-LLMs-Pruning/blob/main/concepts/details/LoRA.md), and optimize *structured-pruning masks* with efficient trainable parameters in differentiable way on instruction-tuning [Alpaca](https://github.com/gururise/AlpacaDataCleaned) dataset. Collaborative prompt is used to help pruning task.

#### [裁剪 Attention 模块] [不需要 Retrain] Shwai He, Guoheng Sun, Zheyu Shen, and Ang Li. 2024. What Matters in Transformers? Not All Attention is Needed.
**主要结论**：

* Attention Drop: Minimal Performance Impact with High Efficiency
* Block and MLP Drop: Significant Performance Degradation with Moderate Speedup

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=14d8fc7cbcf74fedb4be83c4963e51bc&docGuid=5M2jpZlqa2mCqL "")
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=8a1ba5af75fd4bceac04c7c5571b12d7&docGuid=5M2jpZlqa2mCqL "")
#### [裁剪 Block 整层] [ICLR-workshop 24] Bo-Kyeong Kim, Geonmin Kim, Tae-Ho Kim, Thibault Castells, Shinkook Choi, Junho Shin, and Hyoung-Kyu Song. 2024. Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods. arXiv:2402.02834 [cs].
**Takeaway**: First identify unimportant Transformer blocks (bigger and coarse units), then perform one-shot pruning with Perplexity (PPL) as pruning criteria and light [LoRA](https://github.com/liyunqianggyn/Awesome-LLMs-Pruning/blob/main/concepts/details/LoRA.md) retraining. Show fast inference and good zero-shot capabilities.

**思考启发**：看下 PPL 如何作为 层重要性 的 indicator 的

#### [层裁剪] [部分层 Retrain] [Submission to ICLR'25] Yao Lu, Yujie Fang, Zeyu Wang, Hao Cheng, Jiaheng Wei, Dongwei Xu, Qi Xuan, Xiaoniu Yang, and Zhaowei Zhu. 2024. Reassessing Layer Pruning in LLMs: New Insights and Methods.
**Takeaway**：Validate seven different layer selection metrics including Random, Reverse-order, Magnitude, Taylor, Perplexity and Cosine Similarity (BI). Reverse-order pruning is simple yet effective. LoRA performs worse than a simple partial-layer fine-tuning. Iterative pruning offers no benefit compared to one-shot pruning.

**思考启发**：评估了不同的层裁剪选择策略

#### [NIPS'23] [需要 LoRA Retrain] Xinyin Ma, Gongfan Fang, and Xinchao Wang. 2023. LLM-Pruner: On the Structural Pruning of Large Language Models. arXiv:2305.11627 [cs].
**Takeaway**：First discover all coupled structures following [Depgraph](https://arxiv.org/abs/2301.12900), then estimate grouped importance of coupled structure on calibration, then prune less important groups, and last finetune with efficient [LoRA](https://github.com/liyunqianggyn/Awesome-LLMs-Pruning/blob/main/concepts/details/LoRA.md) on [Alpaca](https://github.com/gururise/AlpacaDataCleaned) dataset consists of 50K instruction-response pairs.

#### [Submission to ICLR'25] [整层剪枝] Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. 2024. ShortGPT: Layers in Large Language Models are More Redundant Than You Expect. arXiv:2403.03853 [cs].
**Takeaway**：Delete certain [layers](https://github.com/liyunqianggyn/Awesome-LLMs-Pruning/blob/main/concepts/other_concepts.md/#layer-or-depth-pruning), i.e., transformer blocks (given one block consists of both an Attention and an MLP) in LLMs based on Block Influence (BI) score, a novel metric designed to assess the hidden states transformation of each layer. Layers in LLMs could be more redundant than expected.

#### [工业界论文] [重点看] Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, and Pavlo Molchanov. 2024. Compact Language Models via Pruning and Knowledge Distillation. arXiv:2407.14679 [cs].
#### 
#### [LoRA Retrain] [层剪枝/Attention 剪枝] Shoaib Ahmed Siddiqui, Xin Dong, Greg Heinrich, Thomas Breuel, Jan Kautz, David Krueger, and Pavlo Molchanov. 2024. A deeper look at depth pruning of LLMs. arXiv:2407.16286 [cs].
**Takeaway**：This work explores different block importance metrics including cosine similarity, relativeL1/L2 and Shapleyvalue-based, to take a deeper look at depth pruning of LLMs. Further exam the impact of droping individual Attention and MLP layers. Two simple performance recovery techniques are applied on calibration dataset.



#### Sharath Turuvekere Sreenivas, Saurav Muralidharan, Raviraj Joshi, Marcin Chochowski, Ameya Sunil Mahabaleshwarkar, Gerald Shen, Jiaqi Zeng, Zijia Chen, Yoshi Suhara, Shizhe Diao, Chenhan Yu, Wei-Chun Chen, Hayley Ross, Oluwatobi Olabiyi, Ashwath Aithal, Oleksii Kuchaiev, Daniel Korzekwa, Pavlo Molchanov, Mostofa Patwary, et al. 2024. LLM Pruning and Distillation in Practice: The Minitron Approach. arXiv:2408.11796 [cs].
#### 
#### [不需要 Retrain] [分析各层重要性] [Submission to ICLR'25] Ruihan Xu, Qingpei Guo, Ming Yang, and Shiliang Zhang. 2024. Rethinking the Impact of Heterogeneous Sublayers in Transformers.
**Takeaway**: Instead of pruning entire coarse-grained transformer blocks, this paper proposed a finer granularity depth pruning method that prunes sublayers with treating single transformer block as 2 sublayers, i.e., Multi-Head Attention (MHA) and MLP.



#### Honghe Zhang, XiaolongShi XiaolongShi, Jingwei Sun, and Guangzhong Sun. 2024. Structured Pruning for Large Language Models Using Coupled Components Elimination and Minor Fine-tuning. In Kevin Duh, Helena Gomez, and Steven Bethard, editors, *Findings of the Association for Computational Linguistics: NAACL 2024*, pages 1–12, Mexico City, Mexico. Association for Computational Linguistics.
#### 