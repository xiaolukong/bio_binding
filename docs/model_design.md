# DNA-Protein Binding Affinity Transformer — 模型设计文档 v0.2

## 1. 任务定义

给定一段 DNA 序列和一段蛋白质序列，预测二者结合的紧密程度（连续 scalar，对应 log Kd 等亲和力指标）。

训练采用 **Ranking-based InfoNCE**：每个 data point 包含 1 个 positive 和 7 个 negative，模型通过学习将 positive 的 affinity score 排在所有 negative 之上来获得对 binding 强弱的判别能力。

---

## 2. 上游 Encoder（训练时 frozen）

| 模态 | 模型 | Output Dim | 说明 |
|------|------|-----------|------|
| DNA | DNABERT-2 | **512** | HuggingFace 开源，BPE tokenizer |
| Protein | ESM-2 (`esm2_t6_8M_UR50D`) | **960** | Meta 开源，每个氨基酸一个 token，6层/8M参数的轻量版本 |

Encoder 在训练过程中保持 frozen，仅作为特征提取器。后期如需 fine-tune，可解冻并使用更低的学习率（见第 6 节）。

---

## 3. 序列长度与显存预算

**硬件**：GTX 5060，8GB VRAM

**Batch 结构**：4 data points × 8 sequences（1 pos + 7 neg）= **32 sequences / batch**

| DNA tokens | Protein tokens | L_total | Activation 显存 | 可行 |
|:----------:|:--------------:|:-------:|:---------------:|:----:|
| 128 | 128 | 259 | 1.14 GB | ✓ |
| **256** | **256** | **515** | **2.84 GB** | **✓ 推荐** |
| 300 | 300 | 603 | 3.55 GB | ✓ |
| 400 | 400 | 803 | 5.41 GB | ✓ |
| 512 | 512 | 1027 | 7.91 GB | ✗ |

（模型参数本身约 44M，fp16 仅占 0.08 GB，瓶颈在 O(L²) 的 attention activation。）

**确认值**：`max_dna_len = 60`，`max_prot_len = 514`，`max_seq_len = 577`
- DNA 序列固定长度 60 tokens（无需 padding/截断）
- Protein 序列固定长度 512 aa（ESM-2 按氨基酸逐 token 编码）
- 实测显存：activation 3.31 GB + 模型权重 0.08 GB + overhead 0.3 GB = **~3.7 GB**，8GB VRAM 余量充足，不需要 gradient checkpointing，也不需要将 d_model 降到 512

---

## 4. 模型结构

### 4.1 整体流程

```
DNA sequence  ──[frozen DNABERT-2]──► DNA embedding    (L_dna  × 768)
Prot sequence ──[frozen ESM-2 t6]──► Prot embedding   (L_prot × 320)
                                              │
                        ┌─────────────────────┴─────────────────────┐
               Linear(512 → 768)                          Linear(960 → 768)
               + Segment Emb A                            + Segment Emb B
                        └─────────────────────┬─────────────────────┘
                                              │
               Prepend [CLS], insert [SEP], append [END]:
               [CLS] | DNA_proj | [SEP] | Prot_proj | [END]
                                              │
                              + Learnable Positional Embedding
                                              │
                          Transformer Encoder × 6 layers
                          (Pre-LN: LayerNorm → Attention → Add
                                   LayerNorm → FFN → Add)
                                              │
                              h_cls  (CLS 位置的输出)
                                              │
                                    LayerNorm(768)
                                              │
                             Linear(768 → 768) + GELU + Dropout(0.1)
                                              │
                                      Linear(768 → 1)
                                              │
                                    affinity scalar s
```

### 4.2 Token 序列结构

```
Index:   0      1 … L_dna    L_dna+1    L_dna+2 … L_dna+1+L_prot    L_dna+2+L_prot
Token:  [CLS]  DNA tokens    [SEP]      Prot tokens                  [END]
SegEmb:   A       A            A             B                           B
PosEmb:   0       1…L_dna   L_dna+1    L_dna+2…                    L_total-1
```

**Segment Embedding 的作用**：给每个 token 叠加一个可学习的 modality 标记（A=DNA侧，B=Protein侧），使模型明确区分两种模态，弥补位置编码无法表达跨模态语义的不足。代价极低（2 个 768 维 embedding），但对 cross-modal attention 质量有明显帮助。

`[CLS]` 和 `[SEP]` 归入 Segment A，`[END]` 归入 Segment B（或单独设第三类，可调）。

---

## 5. Hyperparameter 汇总

### 5.1 上游 Encoder（外部固定）

| 参数 | 值 |
|------|----|
| `d_dna` | 512（DNABERT-2） |
| `d_protein` | 960（ESM-2 t6_8M） |

### 5.2 模型核心参数

| 参数 | 值 | 说明 |
|------|----|------|
| `d_model` | **768** | projection 后统一维度，与 DNABERT-2 输出同维，DNA 侧 projection 退化为 identity 可选 |
| `n_heads` | **12** | 768 / 12 = 64 dim per head |
| `n_layers` | **6** | v1 baseline，后续可扩展到 8-12 |
| `d_ffn` | **3072** | 4 × d_model |
| `dropout` | **0.1** | attention dropout 和 FFN dropout 统一 |
| `max_dna_len` | **60** | DNA token 上限（固定序列长度） |
| `max_prot_len` | **514** | Protein token 上限（固定序列长度） |
| `max_seq_len` | **577** | CLS(1) + DNA(60) + SEP(1) + Prot(514) + END(1) |
| `activation` | **GELU** | FFN 激活 |
| `norm` | **Pre-LN** | LayerNorm 置于 attention/FFN 之前，训练更稳定 |

### 5.3 Special Tokens & Embeddings

| Token | Segment | 初始化 |
|-------|---------|--------|
| `[CLS]` | A | 随机，可学习 |
| `[SEP]` | A | 随机，可学习 |
| `[END]` | B | 随机，可学习 |
| Segment Emb A | — | 随机，可学习，shape (768,) |
| Segment Emb B | — | 随机，可学习，shape (768,) |
| Positional Emb | — | 随机，可学习，shape (577, 768) |

### 5.4 Projection Head

```
h_cls → LayerNorm(768) → Linear(768, 768) → GELU → Dropout(0.1) → Linear(768, 1) → scalar s
```

CLS head 前的 LayerNorm 用于稳定 Pre-LN Transformer 最后一层输出的 scale，是 BERT/RoBERTa 的标准做法。

---

## 6. 训练设计

### 6.1 数据组织

每个训练样本（data point）包含：
- 1 个 **positive pair**：(DNA, Protein) 已知强结合，对应较低（更负）的 log Kd
- 7 个 **negative pairs**：同一 Protein 搭配不结合或弱结合的 DNA（或反之）

每个 batch：4 data points × 8 sequences = **32 forward passes**（模型对每条 (DNA, Prot) 对独立推理，输出 scalar）。

### 6.2 Loss 函数：Ranking InfoNCE

对第 $i$ 个 data point，模型对 8 条序列分别输出 affinity score $s_1, s_2, \dots, s_8$，其中 $s_1$ 对应 positive。

$$\mathcal{L}_i = -\log \frac{\exp(s_1 / \tau)}{\sum_{j=1}^{8} \exp(s_j / \tau)}$$

总 loss 对 batch 内 4 个 data points 取平均：

$$\mathcal{L} = \frac{1}{4} \sum_{i=1}^{4} \mathcal{L}_i$$

**温度参数 $\tau$**：初始设为 **0.07**（标准 InfoNCE 默认值）。可设为可学习参数（log 空间参数化，防止崩塌），或保持固定。建议先固定，稳定后再放开。

**直觉**：模型学到的不是绝对 Kd 值，而是"给定同一组候选，哪个最可能结合"的相对排序能力，这与 Kd 数据的噪声特性更匹配。

### 6.3 优化器与调度

| 参数 | 值 |
|------|----|
| Optimizer | AdamW |
| Learning rate（projection + transformer） | **1e-4** |
| Learning rate（encoder fine-tune，后期可选） | **1e-5** |
| Weight decay | **0.01** |
| LR schedule | Linear warmup（前 5% steps）+ Cosine decay |
| Warmup steps | `max(500, 0.05 × total_steps)` |
| Gradient clipping | **1.0**（max norm） |
| Mixed precision | **bf16**（GTX 5060 Blackwell 支持，比 fp16 更稳定） |
| Batch size | **10 data points × 8 sequences = 80 sequences**（GPU 12GB，实测占用约 8.7 GB） |

### 6.4 评估指标

| 指标 | 说明 |
|------|------|
| **AUROC** | 主要指标，衡量 positive 得分高于 random negative 的概率 |
| **AUPRC** | 正负样本不平衡时比 AUROC 更敏感 |
| **Top-1 Accuracy** | 每个 data point 中，positive 是否得分最高 |
| **Spearman ρ** | 如果有绝对 Kd 值，评估预测 score 与 Kd 的排序相关性 |

### 6.5 训练流程

```
Phase 1（主训练）：
  - Encoder frozen
  - 训练 projection + transformer + head
  - 监控 val AUROC，patience=10 epochs early stopping

Phase 2（可选 fine-tune）：
  - 解冻 Encoder，使用分层学习率（encoder lr = 1e-5，其余 1e-4）
  - 仅在 Phase 1 收敛后进行
  - 需要更多数据才不会过拟合
```

---

## 7. 潜在问题 & 设计权衡

| 问题 | 影响 | 缓解方案 |
|------|------|---------|
| Self-attention O(L²) 复杂度 | L>512 时显存压力大 | 已将 max_seq_len 控制在 515，余量充足 |
| 模态不均衡（DNA vs Protein 语义空间差异大） | attention 偏向某一模态 | Segment Embedding 区分模态（已纳入设计） |
| CLS 聚合能力依赖数据量 | 数据少时过拟合 | 可改为 DNA mean pool + Prot mean pool concat → MLP（备选） |
| τ 固定可能次优 | 影响 loss landscape 尖锐程度 | 先固定 0.07，稳定后设为可学习 log_τ |
| Negative 采样质量 | Hard negative 太少则模型不能学到细粒度区分 | 后期可加入 hard negative mining |
| Pre-LN 会使最后一层输出 scale 偏大 | 影响 CLS head 初始 loss | head 前加 LayerNorm（已纳入设计，见 5.4） |

---

## 8. 参数规模总结

| 模块 | 参数量 |
|------|--------|
| DNA Projection（512→768） | 0.39M |
| Protein Projection（960→768） | 0.74M |
| Segment Embeddings（2×768） | 0.001M |
| Positional Embedding（515×768） | 0.40M |
| Transformer（6层，d=768，ffn=3072） | 42.5M |
| CLS Head（768→768→1） | 0.59M |
| Special Tokens（3×768） | 0.002M |
| **合计（可训练）** | **~44M** |
| DNABERT-2（frozen） | 117M |
| ESM-2 t6_8M（frozen） | 8M |

---

## 9. 已确认事项

- [x] DNA Encoder：DNABERT-2，`d_dna = 512`
- [x] Protein Encoder：ESM-2 t6_8M，`d_protein = 960`
- [x] 输出：连续 affinity scalar（回归方向，配合 ranking loss）
- [x] Loss：Ranking InfoNCE，τ = 0.07（固定，后期可学习）
- [x] Batch：4 data points × 8（1 pos + 7 neg）= 32 sequences
- [x] 序列长度：DNA=60 tokens（固定），Protein=514 tokens（固定），总长 577
- [x] Segment Embedding：使用，区分 DNA / Protein 模态
- [x] 训练策略：Phase 1 frozen encoder，Phase 2 可选 fine-tune
- [x] 硬件：GTX 5060（8GB），bf16 混合精度
