如何进行数据去重

# The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale
## 方法流程
**第一步**：collect each document’s 5-grams, obtained using an English word tokenizer

**第二步**：computed MinHashes using 112 hash functions in total, split into 14 buckets of 8 hashes each

**第三步**：Documents with the same 8 MinHashes in any bucket are considered duplicates of each other

**第四步**：We then perform a transitive clustering step where documents A, B and C will be in the same duplicate cluster if A and C are duplicates and B and C are duplicates, even if A and B do not have 8 matching MinHashes in any bucket with each other



论文提出，使用上述去重流程后，相对于随机从未去重的数据里随机采样进行预训练，发现提升不大，作者分析了原因：不能把所有数据合起来一起去重，需要分 snapshot 进行去重： **individually deduplicating each snapshot**

以下分析基于论文中给出的实验过程与结果，尽可能梳理作者在这几段中想要表达的主要思想、实验设计及结论。

---

## 1. 初始设想：全局去重会提高数据质量和模型性能
作者最初认为，对整个数据集（包含多个时间快照）做一次完整的、全局的去重（MinHash 去重），应该能显著提升训练数据的“纯净度”，从而提高下游任务的指标成绩。为此，他们采用迭代的方式，从最新的快照（2023-50）开始一路往前，直至最早的快照。

* 这样做的直观理由是：如果新快照里某些内容在旧快照中已出现过，就把旧快照里这些重复内容删除掉，以确保没有大规模重复文本。
* 结果：大量的原始数据被删除。最极端的情况下，有些老快照中的数据被删除了 90% 之多，最后总共剩下约 4 万亿个 token。

然而，作者发现，在对这一全局去重后的数据（他们抽样了 3500 亿个 token）进行训练时，模型性能“并没有太大提高”，而且在一系列综合任务上，表现还远不及其基线（RefinedWeb）。这与“去重能显著提升效果”的最初假设相悖。

---

## 2. 针对“数据质量”的进一步探究：原本保留 vs 原本删除
既然全局去重削减了大量数据，却得不到理想的效果，作者想深入看看到底被“保留下来的数据”质量如何，以及被“删除掉的数据”质量又是怎样。于是他们选择了一个较老的快照（2013-48），并把其去重过程中的“保留部分”和“被删除部分”分别拿出来:

1. 保留部分（originally kept data）：全局去重后，这部分只剩大约最初 10% 的数据（约 310 亿个 token）。
2. 删除部分（originally removed data）：被全局去重算法从该快照中剔除掉的约 4600 亿 tokens，作者又重新在其内部做了一次“单独去重”（与其他快照无关），抽取了其中 1710 亿个 token。

作者随后分别用这两个数据子集训练模型，并在图 4（Fig. 4）中做了比较。结果相当出乎意料：

* 那个只剩 10% 的、被保留下来的数据，质量居然比那 90% 的“被删除数据”还要差。
* 作者通过人工查看也发现：被保留的文本中包含了更多广告、“关键词堆积”、糟糕格式化的内容，反而被删除的数据质量相对更高些。

这十分有趣，因为原以为“全局去重”会把低质量、重复的内容剔除干净，谁料想保留的内容里却出现了更多无效或低质文本。由此说明，全局去重可能会“错杀好数据”，并没有带来理想的质量提升。

---

## 3. 新思路：单独对每个快照去重，减少“过度清洗”
既然“把所有快照一锅端”地全局去重会导致数据分布不平衡、甚至保留了很多不良文本，作者就改用一种更“温和”的策略：

* 对每一个时间快照独立地做去重，不再跨快照互相比对。
* 这样每个快照都能“保留”更多本快照内部“独一无二”的数据，避免被其他快照中相似内容而误删。

结果：采用独立去重的汇总数据总规模达约 2 万亿个 token（也比全局去重剩下的 4 万亿 tokens 要更小一些，但显著多于之前被大量删除的情形）。

* 在此基础上，作者从所有快照中抽样训练，最终结果与 RefinedWeb 的表现相当（见 Fig. 5），明显优于此前的全局去重版本。

换言之，这表明“每个快照自己内部去重”对模型性能更有益处。

---

## 4. 原因探讨：大批量重复 vs 少量重复
作者进一步提出一个推断：真正影响模型性能（并且确实应该去除）的，是那些“在许多快照中重复出现的，规模非常大的文档集群”（动辄成百上千份的几乎相同内容）。去掉这些“大规模重复”确实会让模型的训练数据更为合理。但如果细化到“只在少数快照里出现次数不多的小重复”，则去除它们的收益很小，甚至会丢失语料的多样性，破坏有益的“自然噪声”，从而反而损害了模型性能。

所以，与其一刀切地做大规模全局去重，不如针对性地先过滤那些庞大的重复集群，对其余小规模重复不再严苛处理，或者换成更有针对性的“数据质量过滤器”，减少无关或糟糕内容即可。

---

## 5. 后续尝试：更轻量的全局去重也不如独立去重效果
作者表示，他们也尝试了一些“轻度的全局去重”策略，希望在不大幅删减数据的前提下，只去除最明显、最冗余的部分。然而，这些方法依旧没能在下游性能上超过“对每个快照独立去重”的方案，最终结论仍然是：

> 独立去重 + 面向大规模重复的针对性过滤 > 轻度/重度的全局去重
作者在附录 E.3 里有更细节的描述与结果，但核心结论就是，全局去重并不总是好事，真的要结合数据特点、使用场景、重复模式来做更精细的处理。

---

## 总结
简而言之，作者想要表达的是：

1. 全局去重在理论上很诱人，但实际操作中很可能会“剔除过度”，让真正高质量、少量重复的数据也被误删除；并且由于各种偏差，最终保留的文本反而包含了很多低质或广告性内容。
2. 独立对每个快照进行去重的做法，相对更稳妥，避免全局去重导致的“误删好数据”，最终在模型训练指标上表现更优。
3. 去重真正要解决的核心问题，大多集中在“重复次数极多的大规模重复集群”。而对于那些只在少量快照中出现的小规模重复，可能没必要严格去除，因为那通常会带来对数据多样性的损害。
4. 即使尝试更轻量的全局去重，也无法超越简单的“每个快照独立去重”。作者推测，更合理的做法是仅对海量重复内容进行剔除，然后使用更有针对性的过滤策略，而非一味地把所有快照一起算作一个整体去 deduplicate。

从这些分析可以看出，论文的中心思想是：在庞大规模文本数据的去重与过滤中，“适度的去重”与“多样性保持”需要平衡，全局化粗暴去重会让原本看似“干净”的结果反而变得不如预期。作者采用了实验与对照的方法，详尽展示了“保留 vs. 删除”对数据质量和模型训练的影响，为后续构建大规模语言模型的数据预处理提供了重要的实证参考。

### 相关工具
* 关于 Educational 的分类器代码：[https://github.com/huggingface/cosmopedia/tree/main/classification](https://github.com/huggingface/cosmopedia/tree/main/classification)

# [https://developer.nvidia.com/blog/mastering-llm-techniques-data-preprocessing/](https://developer.nvidia.com/blog/mastering-llm-techniques-data-preprocessing/)
### 方法流程
* 精确去重：为每个文档生成hash签名，按hash将文档分组到桶，每个桶保留一个文档。
* 模糊去重：使用minhash签名和位置敏感哈希（LSH）识别相似文档。计算minhash签名，使用LSH将相似的文档分组到桶中，每个文档可能属于一或多个桶，计算同一个桶内文档的Jaccard相似性，将相似度矩阵转化为图，识别图中的连通分量。连通分量中的文档是模糊相似的，从数据集中删除识别的重复项。
* 语义去重：使用embedding模型捕获语义，使用聚类对语义相似的内容进行分组。使用预训练模型对数据点嵌入，使用k-means聚类将嵌入聚到k个簇中，每个簇中两两计算余弦相似度，超过阈值则认为语义重复。每个簇的每组语义重复，只保留一个代表性数据。

### 相关工具
NVIDIA NeMo Curator

[https://developer.nvidia.com/nemo-curator](https://developer.nvidia.com/nemo-curator)

[https://github.com/NVIDIA/NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator)

### 主要结论
精确去重适用于完全相同的文档，缺点是无法分辨语义上有微小不同的等效文档。

模糊去重对于识别少量修改的内容、检测部分文档重叠、查找格式不同但内容相似的文档很有价值，在计算效率和检测能力之间取得平衡。

语义去重在识别改写的内容、同一材料的翻译版本和概念相同的信息很有价值。

# D4: Improving LLM Pretraining via Document De-Duplication and Diversification, in NIPS 2023
### 方法流程
将每个文档输入到125M的OPT模型中得到文档嵌入，并使用token的最后一层嵌入。

SemDeDup：首先使用k-means对嵌入空间进行聚类，并在每个聚类中去除彼此的epsilonballs内的点。我们使用的算法与Abbas相同。

Prototypicality：SSL原型方法，首先使用k-means聚类对嵌入空间进行聚类，按距离最近质心长度从小到大进行排序，丢弃数据点，这样就会丢弃最“原型”的数据点，丰富高方差的离群值。

D4：我们发现了很多重复驱动集群的实例：模板文本集群上极度冗余的信息集群，它们不会被minhash删除。这些嵌入空间的区域往往很密集，导致k-means在重复文本上浪费宝贵的聚类分配。这种有偏差的集群也可能对prototypicality有效性产生不良影响。因此我们提出策略：

1. 使用SemDeDup算法，从源数据集D上选择比率为Rdedup的数据，得到小数据集D‘
2. 使用k-means对D’数据进行聚类
3. 在D‘上应用SSL原型方法，选择比率为Rproto

### 相关工具
OPT论文[https://arxiv.org/pdf/2205.01068](https://arxiv.org/pdf/2205.01068)

MetaSeq代码[https://github.com/facebookresearch/metaseq](https://github.com/facebookresearch/metaseq)

SemDeDup代码[https://github.com/facebookresearch/SemDeDup/](https://github.com/facebookresearch/SemDeDup/)

### 主要结论
在数据有限的情况下，使用精心选择的子数据进行多epoch训练，比使用大量数据进行单epoch训练好。

总体效果：精心选择数据+多epoch > 大量数据+单epoch > 随机选择数据+多epoch

可以使用D4对单个数据源多样化和去重，然后对处理后数据进行混合提供额外的多样性。

研究的最大规模是在6.7B模型上使用100Btoken进行训练，但发现模型规模越大，使用D4算法选择的数据提供的效率提升越大。

# [How to Train Data-Efficient LLMs](https://arxiv.org/pdf/2402.09668)
### 方法流程
### 相关工具
### 主要结论
# [SlimPajama-DC: Understanding Data Combinations for LLM Training](https://arxiv.org/abs/2309.10818)
### 方法流程
### 相关工具
### 主要结论
# Data, Data Everywhere: A Guide for Pretraining Dataset Construction, in EMNLP 2024
### 方法流程
### 相关工具
### 主要结论
