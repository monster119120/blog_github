# 公开预训练数据资源汇总

本文档旨在汇总目前主流且公开可获取的大模型预训练数据集，涵盖英文、中文、代码、数学及书籍学术资源等多个类别。

## 一、英文基础数据
主要来源于 Common Crawl 项目，经过深度清洗和优化。

*   **Hugging Face FineWeb**
    *   **链接**：[Blog Post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
    *   **简介**：
        *   **FineWeb 完整版**：包含 15T tokens，占用 44TB 磁盘空间。
        *   **FineWeb-Edu**：专注于教育内容的高质量子集，包含 1.3T（极高教育质量）和 5.4T（高教育质量）tokens。

*   **SlimPajama**
    *   **链接**：[Cerebras Blog](https://cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)
    *   **简介**：RedPajama 的清洗与去重版本，包含 627B tokens，通过去除重复项大幅提升了训练效率。

*   **The Pile**
    *   **链接**：[Hugging Face Dataset](https://huggingface.co/datasets/EleutherAI/pile)
    *   **简介**：EleutherAI 发布的经典大规模数据集，包含由 22 个不同子集构成的 825GB 文本数据。

*   **DCLM (DataComp for Language Models)**
    *   **链接**：[Hugging Face](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | [ArXiv Paper](https://arxiv.org/abs/2406.11794)
    *   **简介**：由苹果公司参与构建的高质量数据集，经过精细清洗和优化，旨在建立语言模型数据的新基准。

## 二、专项数据（代码与数学）

### 代码数据
*   **StarCoderData**
    *   **链接**：[Hugging Face Dataset](https://huggingface.co/datasets/bigcode/starcoderdata)
    *   **简介**：约 250B tokens。包含 783GB 的代码（涵盖 86 种编程语言）、54GB GitHub Issues、13GB Jupyter Notebooks 以及 32GB GitHub Commits。

*   **The Stack v2**
    *   **链接**：[Hugging Face Dataset](https://huggingface.co/datasets/bigcode/the-stack-v2-train-smol-ids/tree/main/data)
    *   **简介**：BigCode 项目发布的超大规模代码数据集，包含约 900B tokens。

### 数学数据
*   **OpenWebMath**
    *   **链接**：[GitHub](https://github.com/keirp/OpenWebMath)
    *   **简介**：14.7B tokens。从 Common Crawl 的 2000 多亿个 HTML 文件中筛选并提取出 630 万份与数学相关的文档，质量极高。

## 三、中文数据
**数据主要来源**：网页数据、书籍、学术资料、社交媒体、百科数据。

*   **WuDaoCorpora (悟道文本数据集)**
    *   **链接**：[SciDB](https://www.scidb.cn/en/detail?dataSetId=c6a3fe684227415a9db8e21bac4a15ab)
    *   **简介**：开源部分约 200GB，覆盖教育、科技等 50 多个行业领域，是早期高质量中文语料的代表。

*   **CCI2-Data**
    *   **链接**：[Hugging Face](https://huggingface.co/datasets/BAAI/CCI2-Data?row=23)
    *   **简介**：北京智源人工智能研究院发布，规模约 501GB。数据类型丰富，包括网页、公众号、博客、百科、问答及试题。

*   **Wanjuan 1.0 (书生·万卷)**
    *   **链接**：[GitHub](https://github.com/opendatalab/WanJuan1.0/blob/main/WanJuan1.0-CN.md)
    *   **简介**：总量超 1TB。由上海 AI 实验室构建，包含文本、图文、视频多模态数据。文本部分覆盖科技、文学、法律等领域，经过细粒度清洗和价值对齐，具有多元融合和高安全性的特征。

*   **MNBVC**
    *   **链接**：[Hugging Face](https://huggingface.co/datasets/liwu/MNBVC)
    *   **简介**：超大规模中文语料库，数据源极其广泛。涵盖新闻、小说、学术论文、聊天记录、甚至火星文等亚文化内容，旨在保留中文互联网的多样性。

*   **SkyPile-150B**
    *   **链接**：[ModelScope](https://www.modelscope.cn/datasets/modelscope/SkyPile-150B)
    *   **简介**：昆仑万维发布，包含约 1500 亿 tokens（纯文本 620GB）。基于 2.33 亿个中国互联网网页构建，经过 fastText 和 BERT 等工具的严格过滤和去重。

## 四、书籍与学术资源

### 电子书与文献库
*   **Project Gutenberg**：[官网](https://www.gutenberg.org/) - 拥有 75,000+ 本免费公版电子书。
*   **Books1**：[下载链接](https://hyper.ai/datasets/13642) - 标准预训练常用书籍数据集。
*   **Books3**：[Hugging Face](https://huggingface.co/datasets/defunct-datasets/the_pile_books3) - The Pile 中的书籍子集，规模较大（注意：因版权原因，部分链接可能已失效）。
*   **中文书籍整理**：[GitHub](https://github.com/shjwudp/shu/blob/master/index.csv) - 约 380 本中文书籍索引。
*   **TigerBot 开源数据**：[GitHub](https://github.com/TigerResearch/TigerBot#%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%E9%9B%86) - 包含 100G 数据，其中中文书籍占比约 16% (12G)。
*   **中华古诗词**：[GitHub](https://github.com/chinese-poetry/chinese-poetry/tree/master) - 收录唐宋两朝近 1.4 万诗人/词人的作品，包含 5.5 万首唐诗和 26 万首宋诗等。

### 学术与大型归档 (需注意版权合规性)
*   **Anna’s Archive**：[官网](https://annas-archive.org/) - 全球最大的开放数据图书馆之一，整合了 Libgen、Sci-Hub 等资源，规模达 PB 级别。
*   **Sci-Hub**：[数据库](https://sci-hub.se/database) - 收集了超过 8800 万份研究论文。
*   **Z-Library**：[官网](https://z-lib.id/) - 世界上最大的电子书和文章集合之一。

### 参考：中图法书籍分类简表
在构建书籍数据集时，通常参考以下分类体系进行数据配比：

| 代码 | 分类名称 | 代码 | 分类名称 |
| :--- | :--- | :--- | :--- |
| **A** | 马列主义、毛泽东思想等 | **N** | 自然科学总论 |
| **B** | 哲学、宗教 | **O** | 数理科学和化学 |
| **C** | 社会科学总论 | **P** | 天文学、地球科学 |
| **D** | 政治、法律 | **Q** | 生物科学 |
| **E** | 军事 | **R** | 医药、卫生 |
| **F** | 经济 | **S** | 农业科学 |
| **G** | 文化科学、教育、体育 | **T** | 工业技术 |
| **H** | 语言、文字 | **U** | 交通运输 |
| **I** | 文学 | **V** | 航空、航天 |
| **J** | 艺术 | **X** | 环境科学、安全科学 |
| **K** | 历史、地理 | **Z** | 综合性图书 |

## 五、参考：主流模型预训练数据配比

1.  **LLaMA-1**：
    *   主要依赖 CommonCrawl (67%)，辅以 GitHub (4.5%)、Wikipedia (4.5%)、Books (4.5%)、ArXiv (2.5%)、StackExchange (2%) 等。
2.  **GPT-3**：
    *   采用了加权采样策略，CommonCrawl (60% 权重)、WebText2 (22%)、Books1 (8%)、Books2 (8%)、Wikipedia (3%)。

## 六、参考文献
1.  [arXiv:2411.07715](https://arxiv.org/html/2411.07715v1)
2.  [Volcengine Developer Article](https://developer.volcengine.com/articles/7399549896811872294)