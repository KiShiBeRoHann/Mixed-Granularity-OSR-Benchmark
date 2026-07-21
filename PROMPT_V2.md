# 🚀 开放世界视觉识别：混合粒度与相对拓扑 OSR 基准体系 (Benchmark)

## 一、 基准定位与"靶场"拓扑设计 (Benchmark & Topology Design)

本研究的核心目标是**构建 OSR/OOD 领域首个"混合粒度与相对拓扑（Relative-Topology）"大一统评测基准（Benchmark）**。我们打破了传统 OSR "平级分类"的扁平化设定，在多个主流数据集上构建了从微缩图像到高清图像的立体防守靶场。

**1. 核心数据集与分类头规模：**

* **轻量级验证 (CIFAR-100)：** 严格固定为 **20 个分类头**。硬编码锁定（`seed_46`）。
* **大规模高清验证 (FGVC-Aircraft, ImageNet, iNaturalist)：** 采用 `build.py` 中的动态 LISP 树状解析与互斥拓扑抽签锁，自动抽取独立子树，强制固定为 **60 个分类头**，确保模型在相同容量下面临等价的分类与防守压力。

**2. 灵活的层级划分逻辑 (Hierarchy Splits)：**

* **L2 模式 (Family -> Variant)：** 支持所有数据集。划分为粗粒度组与细粒度组，包含粗粒度超类级别的未见变体泛化，以及细粒度级别的同族内斗（Near-Variant）。
* **L3 模式 (Maker -> Family -> Variant)：** 支持 Aircraft, ImageNet, iNaturalist。在 L2 基础上增加中等粒度（Medium），构建更极其复杂的递进式拓扑关系（跨族内斗 Near-Family）。

## 二、 核心评估指标体系 (Evaluation Metrics)

针对不同层级的基准靶场，我们设计了极其严密的指标计算分支，重点考察模型在"未见变体泛化"与"不同距离未知类拒识"上的表现。

### 阈值无关指标（ACC + AUROC）

**三层架构靶场 (L3: Aircraft, ImageNet, iNaturalist)：**

* **分类泛化表现 (Accuracy)：**
  * `Global Acc`: 全局准确率（联合基础 ID、Coarse Gen 和 Medium Gen）。
  * `Coarse Gen Acc (Maker)`: 粗粒度（制造商）级别的未见变体泛化准确率。
  * `Medium Gen Acc (Family)`: 中粒度（家族）级别的未见变体泛化准确率。

* **开集拒识表现 (AUROC)：**
  * `Strict Global AUROC`: 严苛全局拒识率（统合跨族、同族、隔离远端）。
  * `Near-Family AUROC (跨族)`: 细粒度相近但属于不同家族的 OOD 拒识能力。
  * `Near-Variant AUROC (同族)`: 极端困难的同族内斗 OOD 拒识能力。
  * `Strict Far AUROC (隔离)`: 远端无关联 OOD 的拒识能力。

**两层架构靶场 (L2: CIFAR-100 & FGVC-Aircraft/ImageNet/iNaturalist)：**

* **分类泛化表现 (Accuracy)：**
  * `Global Acc`: 全局准确率。
  * `Gen Acc`: 粗粒度未见变体的泛化准确率。

* **开集拒识表现 (AUROC)：**
  * `Strict Global AUROC`: 联合 Near-OOD (同族) 与 Far-OOD 的全局严苛拒识率。

### 阈值依赖指标（CV-RR / UN-RR）

为量化"泛化与拒识"的实际业务折衷，我们引入基于 TPR 锚定（默认 @TPR95）的阈值依赖指标：

* **CV-RR (Class-Variant Rejection Rate / 已知类内未见变体拒绝率)：** 在保证 95% ID 样本通过的前提下，衡量粗粒度未见变体（Coarse Gen）中被错误拒绝的比例。**越低越好**——反映模型是否「防守过度」导致已知超类下的未见子类被误杀。
* **UN-RR (Unknown Near-Rejection Rate / 未知近邻拒绝率)：** 同等阈值下，衡量近邻 OOD（Near-Variant / Near-Family）中被正确拒绝的比例。**越高越好**——反映模型对近距离未知类的实际防守效能。

> 这两个指标共同揭示核心矛盾：一个理想模型应同时做到**低 CV-RR（不误伤同类变体）**与**高 UN-RR（有效拦截近邻未知类）**。

## 三、 参战模型与统一数据流适配 (Baselines & Framework)

为了保证基准对比的绝对公平，所有 Baseline 均接入统一的 `get_mixed_granularity_loaders` 数据流，并针对高清图像引入了 `AdaptiveAvgPool2d((1, 1))` 与 `Resize(224)` 的智能预处理分支。

**1. 基础视觉防守派 (Traditional OSR)：**

* **VGG_MLS：** 基于 VGG16-BN 骨干 + 最大 Logit 得分的 MLS 基准。
* **ARPL + CS：** 经典的对抗性互惠点学习基准。
* **DeF (F-DEF & C-DEF with SupCon)：** 基于监督对比学习与双重防御机制的高阶基准。

**2. 多模态大模型降维打击派 (Vision-Language Models)：**

* **ZS-CLIP (Vanilla)：** 纯零样本下的视觉-语言特征基准（提供最纯粹的文本先验能力底座）。
* **CMA (Cross-Modal Agent)：** Zero-Shot 推理方法 —— 引入中性文本 Agent 与向量三角惩罚进行评测，无需训练。
* **Linear Probe CLIP：** 冻结视觉骨干，重训分类头。
* **FSMOE (Few-Shot Mixture of Experts)：** 引入基于多模态大模型的专家混合路由机制进行评测。
* **CoOp / CoCoOp：** 基于提示学习的自适应特征微调评测。
* **A2Pt：** 结合视觉 Prompt + 文本 Prompt 的多模态提示学习方法。

## 四、 核心研究动机与突破口 (The Motivation for the Benchmark)

传统的 OSR 方法在追求高 AUROC 时，往往忽略了现实世界数据的层级从属关系。当我们把主流 SOTA 方法置于本基准中时，数据无情揭示：现有策略极易陷入"防守过度（导致类内变体泛化 Acc 暴跌）"**或**"防守失效（导致 Near-Variant/Family AUROC 极低）"的困境。

本 Benchmark 的提出，正是为了量化并揭示这一"泛化与拒识"的矛盾，强制要求模型在粗细粒度分类以及远近端 OOD 拒识上达到动态平衡，从而为下一代安全、鲁棒的多模态防御机制提供严谨的试金石。

## 五、 基础设施 (Infrastructure)

* **`build.py`**：LISP 树解析 + 互斥拓扑抽签，是 ImageNet / iNaturalist / Aircraft 划分 JSON 的前置生成脚本。
* **`run.sh`**：统一实验启动器，自动管理多 seed 循环、L2/L3 路径分叉与日志归档。
* **`datasets/__init__.py`**：`get_mixed_granularity_loaders()` 统一数据流工厂，所有模型共用的唯一数据入口。
* **参考实现**：`CoOp/` 子目录为 CoOp/CoCoOp 的原始官方代码库；根目录 `train_coop.py` / `train_cocoop.py` 是接入本基准统一数据流后的重实现版本。
