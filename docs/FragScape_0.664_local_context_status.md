# FragScape 0.664 「平衡点」模型定位与状态核查

> 核查日期：2026-10-02（清理与改名后同日复核）
> 目标肽：`GFPGTPGLPGVK`（human COL2A1 / P02458，残基 276–287）
> 核查目录：`/home/jilinan/FragScape`（原名 `/home/jilinan/PepCleaver_release`）

---

## 1. 结论（TL;DR）

**是。** 本目录（原名 `/home/jilinan/PepCleaver_release`，现为 `/home/jilinan/FragScape`）就是最终定下能跑出 0.664 的那个平衡点模型所在目录，
模型本体与所需的局部上下文嵌入都还在磁盘上，且可 100% 复现：

| 项目 | 值 |
|------|-----|
| **平衡点模型** | `weights/fragscape.pt`（FragScape, max_len=12, 4,806,497 参数） |
| **编码方式** | ±20 残基局部上下文（COL2A1[255:307]，取 [20:32]） |
| **所需嵌入缓存** | `data/local_context_embeddings.pkl`（20,289 条，dict: peptide→(L,480)） |
| **实测复现值** | **0.6644**（与原报告 §11.4 的 0.6644 完全一致） |
| **test AUC** | 0.7513 |

之所以「重新跑变成 0.5509」，不是模型丢了，而是**跑错了推理路径**——用的是「独立片段」退化
路径（见 §4），而不是这个局部上下文模型。

---

## 2. 关键文件与指纹（可用于备份/校验）

| 文件 | 大小 | 修改时间 | md5 |
|------|------|----------|-----|
| `weights/fragscape.pt` | 19,250,647 B | 2026-08-11 10:45 | `3e35ccddfadd57ea7e64da2b7032cd56` |
| `data/local_context_embeddings.pkl` | 468,534,773 B | 2026-08-11 10:42 | `620df24dc2e11b46cb350971bc4e8140` |

> 注：`pepcleaver_v2.pt`（md5 `8293f493cea9a04e00ae583451eca29d`）与
> `pepcleaver_v4_10_12.pt`（md5 `a75dfc2d188929a6b00334a061028cf1`）是本次清理**之前**
> 同目录中的旧权重，现已移出（见 §5）。上表两个文件才是本仓库唯一保留的模型产物。

---

## 3. 如何复现 0.6644

仓库原先**没有任何脚本**会加载平衡点权重 `weights/fragscape.pt`
（`scripts/predict_long.py` 硬编码加载的是另一个权重，且使用独立片段编码），
因此本次新增了一个专用复现脚本 `scripts/predict.py`：

```bash
/home/jilinan/miniconda3/bin/python scripts/predict.py --peptides GFPGTPGLPGVK
```

实测输出：

```
device : cuda
weights: .../weights/fragscape.pt
model  : FragScape(max_len=12)  params=4,806,497
cache  : .../data/local_context_embeddings.pkl  (20,289 peptides)

peptide          len          source      prob   flags
--------------------------------------------------------------
GFPGTPGLPGVK      12 cache(localctx)    0.6644   >=0.6
```

脚本同时支持：多条肽（`--peptides A,B,C`）、以及当目标肽不在缓存中时用
`--protein <FASTA>` 现场用 ESM-2 重算 ±20 窗口嵌入（需 `fair-esm`）。

> 环境提示：系统默认 `python3` 的 torch 缺 CUDA 运行库（`libcublasLt.so`），会 ImportError；
> 请使用 `/home/jilinan/miniconda3/bin/python`（torch 2.6.0+cu124，CUDA 可用）。

---

## 4. 0.5509 到底从哪来

0.5509 是**旧的「独立片段」退化路径**的结果，与平衡点模型无关：

- 直接来源：旧文件 `data/col2a1_10_12_predictions.csv` 第 480 行
  （**该文件已在本次清理中移出仓库**）
  ```
  GFPGTPGLPGVK,12,275,0.550917
  ```
  该文件生成于 **2026-08-03**，用的是「仅短肽(2–9)训练的模型 + 独立片段编码」
  （见 `docs/FragScape_v2_v4_evolution_report.md` §3、§7；报告把该值记为 v1.0 = 0.5509）。
- 另一条会得到相近低值的路径：`scripts/predict_long.py` → `weights/pepcleaver_v2.pt`
  + 独立片段嵌入（无上下文）。**该脚本与权重同样已在本次清理中移出仓库。**

三条编码路径对照（报告 §11.4）：

| 模型/编码 | 概率 | ≥0.6 | 说明 |
|-----------|------|------|------|
| v1.0（仅短肽）独立片段 | 0.5509 | ✗ | 12 肽被截断/补零后打分，无上下文 |
| v2.1（短+长联合）独立片段 | 0.4495 | ✗ | 对应旧权重 `pepcleaver_v4_10_12.pt`（已移出） |
| **v2.1 局部上下文（±20）** | **0.6644** | **✓** | **平衡点模型 = `fragscape.pt`** |

---

## 5. 清理与改名后的状态（2026-10-02 复核）

- 目录已由 `PepCleaver_release` 改名为 **`FragScape`**；README / docs / model / scripts 中
  的品牌文本已统一改为 FragScape。所有被移除的文件**均可恢复**，位于：
  `/home/jilinan/.fragscape_trash`
- 本次只保留能复现 0.6644 的最小链路：`model/fragscape.py`、`model/components.py`、
  `scripts/predict.py`、`weights/fragscape.pt`、`data/local_context_embeddings.pkl`，
  外加 README / LICENSE / requirements.txt / CITATION.cff / docs。
- 风险仍在：这两个大文件（19 MB + 447 MB）**依旧未纳入 git**（HEAD 仍是公开的 v1.0.0
  发布版 `f9923d0`，且 `.git` 的 remote 仍指向
  `github.com/SUAT-ZhiweiFengLab/PepCleaver.git`）。
- 建议：立即把以下两个文件做离线备份，或（用 Git LFS）纳入版本控制：
  - `weights/fragscape.pt`
  - `data/local_context_embeddings.pkl`

---

## 6. 勘误（供文档修订参考）

`docs/FragScape_v1.0_v2.1_comprehensive_report.md` §12.1 把权重写成
`weights/pepcleaver_v4_10_12.pt`（那是 0.4495 的「独立片段」模型），
而 §11.4 的 0.6644 平衡点实际对应 `weights/fragscape.pt`
（§11 末尾脚注「推荐使用的可靠模型」指的就是它）。两者不要混用。

---

## 7. 追加修订（2026-10-02 晚 · 发布到 GitHub 之后）

发布版**不再包含** `data/local_context_embeddings.pkl`（468,534,773 B）。
原因：它是纯派生数据，占掉了克隆体积的绝大部分；现已加入 `.gitignore`
（`data/*.pkl`）。仓库改为只让 `weights/fragscape.pt`（19 MB）走 Git LFS，
全新克隆从 ~466 MB 降到 ~19 MB。

**关键结论：没有缓存一样能复现 0.6644。** 实测脚本
`/home/jilinan/fragscape_verify_nocache.py`（用 `/home/jilinan/miniforge3/bin/python`
跑，ESM-2 权重取 `esm2_t12_35M_UR50D`）：

| 对比项 | 值 |
|--------|-----|
| 缓存向量 vs ESM-2 重算向量，max \|Δ\| | `2.264977e-06` |
| 缓存向量 vs ESM-2 重算向量，mean \|Δ\| | `3.781555e-07` |
| 余弦相似度 | `0.999999999999` |
| 用**缓存**向量打分 | `0.664465` |
| 用 **ESM-2 重算**向量打分 | **`0.664465`** |

因此 §2 的 md5 表依然有效（可用于离线备份校验），但该 pkl 已不再是复现 0.6644
的必需品。现行复现命令（无需任何缓存文件）：

```bash
/home/jilinan/miniforge3/bin/python scripts/predict.py \
    --peptides GFPGTPGLPGVK --protein data/human_col2a1.fsa
# GFPGTPGLPGVK      12      esm2(+-20)    0.6644   >=0.6
```

> 环境提示：`--protein` 这条路径需要 `fair-esm`。系统 `miniconda3` 里没装，
> 但有 esm 的解释器包括 `/home/jilinan/miniforge3/bin/python`、
> `/home/lc/PepCleaver_release/.venv/bin/python`、`/home/lc/.venv/bin/python`。
> 首次运行会自动下载 ESM-2 t12 35M 权重（~134 MB）到
> `~/.cache/torch/hub/checkpoints/`，之后可完全离线。

§5 中「两个大文件依旧未纳入 git」的风险项，至此已经解决：权重进了 LFS，
缓存改为按需重算。

