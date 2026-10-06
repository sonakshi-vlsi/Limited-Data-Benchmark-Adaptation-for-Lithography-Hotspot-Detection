# Limited-Data-Benchmark-Adaptation-for-Lithography-Hotspot-Detection
This work investigates whether a convolutional neural network (CNN) pretrained on one ICCAD-12 benchmark (source) can be adapted to a different benchmark (target) using only a small fraction. 

**Can a hotspot detector trained on one process be reused on a new process using only a handful of labelled examples?**

This project pretrains a lightweight CNN on one ICCAD-12 benchmark and adapts it to four unseen benchmarks using only **5–20 % of their training data**. It compares four approaches: training from scratch, zero-shot transfer, head-only fine-tuning and full fine-tuning.

![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![Dataset](https://img.shields.io/badge/Dataset-ICCAD--12-blue) ![Experiments](https://img.shields.io/badge/Experiments-113-green)

---

## Key results

| | |
|---|---|
| **Up to +0.35 balanced accuracy** | Fine-tuning vs. training from scratch at a 5 % data budget (ICCAD-1) |
| **35× more hotspots found** | ICCAD-4 hotspot recall rose from **2.3 % → 81.9 %**: same architecture, same data, only the starting weights differ |
| **0.939 average balanced accuracy** | Across all 5 benchmarks, with **0.905 average recall** |
| **0.986 with zero target data** | Zero-shot transfer from ICCAD-3 to ICCAD-5 matches fully fine-tuned performance |

---

## Why this matters

As features shrink below the wavelength of light used to print them, some layout shapes print wrong. These failures, called **lithography hotspots**, show up as shorts or opens on the wafer. Full lithography simulation finds them accurately, but it is too slow for large designs, so CNN classifiers are used to screen layout clips instead.

The catch is that these classifiers need a lot of labelled data for each process node. When a foundry moves to a new node, it has almost no labelled hotspots, because labelling them requires the same expensive simulation the classifier is meant to replace. This is a **cold-start problem**.

This project measures how far a model trained on one data-rich benchmark can be reused on another with very little labelled data.

---

## Dataset: ICCAD-12

The ICCAD-12 dataset contains grayscale layout clip images, each labelled **Hotspot (HS)** or **Non-Hotspot (NHS)**, with a fixed train/test split. All images were resized to 64×64 and normalised to [0, 1].

| Benchmark | Train HS | Train NHS | Test HS | Test NHS |
|---|---:|---:|---:|---:|
| ICCAD-1 | 99 | 340 | 226 | 4,679 |
| ICCAD-2 | 174 | 5,285 | 498 | 41,298 |
| **ICCAD-3 (source)** | **909** | **4,643** | **1,808** | **46,333** |
| ICCAD-4 | 95 | 4,452 | 177 | 31,890 |
| ICCAD-5 | 26 | 2,716 | 41 | 19,327 |

Hotspots make up only **0.2–1.9 %** of the samples. A model that always predicts "no hotspot" would score about 99 % plain accuracy and be useless. For that reason, **balanced accuracy** (the mean of sensitivity and specificity) is the main metric.

The data is scarcer than the percentages suggest. On ICCAD-5, a 5 % budget means **one** labelled hotspot.

---

## Method

### Model

The lightweight CNN is reproduced from the reference baseline (Verma et al.):

```
Input 64×64×1
→ [Conv2D 3×3, 12 filters, ELU → BatchNorm → MaxPool 2×2] × 3
→ Flatten → Dropout 0.3 → Dense(1, sigmoid)
```

- About **12.9 k trainable parameters**
- Nadam optimiser with binary cross-entropy loss

### Pipeline

1. **Pretrain on the source benchmark.** The model is trained on ICCAD-3, the largest benchmark, for 10 epochs. It reaches 96.8 % balanced accuracy on the ICCAD-3 test set.
2. **Build limited-data subsets.** For each target benchmark (ICCAD-1, 2, 4, 5), stratified subsets are drawn at **5 %, 10 % and 20 %** of the training set. Each subset keeps the original HS:NHS ratio and is sampled with 3 random seeds.
3. **Compare four approaches:**

| Approach | What it does |
|---|---|
| Scratch | Same architecture with random weights, trained only on the target subset |
| Zero-shot | ICCAD-3 model evaluated directly on the target, with no target training |
| Head-only fine-tuning | Convolutional layers frozen; only the final dense layer is retrained |
| Full fine-tuning | All layers retrained at a reduced learning rate |

4. **Evaluate.** Every model is tested on the **complete, unmodified** target test split, so the natural class imbalance is preserved. Results are averaged over the three seeds.

The full study has **113 experimental conditions**:

- 4 targets × 3 budgets × 3 seeds × 3 trained approaches = 108 runs
- 4 zero-shot evaluations
- 1 source-model evaluation

---

## Results

### Best result per benchmark (full fine-tuning, 20 % budget)

| Benchmark | Balanced Acc. | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| ICCAD-1 | 0.945 | 0.397 | 0.960 | 0.561 |
| ICCAD-2 | 0.883 | 0.676 | 0.773 | 0.666 |
| ICCAD-3 (source) | 0.968 | 0.485 | 0.977 | 0.648 |
| ICCAD-4 | 0.912 | 0.336 | 0.840 | 0.454 |
| ICCAD-5 | 0.987 | 0.443 | 0.976 | 0.609 |
| **Average** | **0.939** | **0.467** | **0.905** | **0.588** |

### Scratch vs. full fine-tuning (balanced accuracy)

| Target | 5 % Scratch | 5 % Full FT | 20 % Scratch | 20 % Full FT |
|---|---:|---:|---:|---:|
| ICCAD-1 | 0.587 | **0.899** | 0.745 | **0.945** |
| ICCAD-2 | 0.548 | **0.865** | 0.865 | **0.883** |
| ICCAD-4 | 0.513 | **0.856** | 0.554 | **0.912** |
| ICCAD-5 | 0.500 | **0.986** | 0.593 | **0.987** |

With little data, the scratch models collapse to always predicting "no hotspot". On ICCAD-4 at the 20 % budget, the scratch model found **4 of 177** hotspots, while the fine-tuned model found **145 of 177**.

### Zero-shot transfer (no target data)

| Target | Balanced Acc. | Recall |
|---|---:|---:|
| ICCAD-1 | 0.508 | 0.018 |
| ICCAD-2 | 0.886 | 0.789 |
| ICCAD-4 | 0.893 | 0.842 |
| ICCAD-5 | **0.986** | 0.976 |

How well zero-shot transfer works depends heavily on the target. It is near random on ICCAD-1 and near perfect on ICCAD-5. The benchmarks where zero-shot does worst are the ones that gain the most from fine-tuning.

### Ablation: head-only vs. full fine-tuning (20 % budget)

| Target | Head-only | Full FT | Δ |
|---|---:|---:|---:|
| ICCAD-1 | 0.905 | 0.945 | +0.040 |
| ICCAD-2 | 0.880 | 0.883 | +0.003 |
| ICCAD-4 | 0.803 | 0.912 | **+0.109** |
| ICCAD-5 | 0.986 | 0.987 | +0.001 |

Which approach is better depends on the benchmark. Head-only fine-tuning is also riskier when the frozen features don't match the target: on ICCAD-2 at the 5 % budget, its balanced accuracy ranged from 0.71 to 0.97 across seeds.

<!-- Add figures here once exported from the report/notebook, e.g.:
![Balanced accuracy vs data budget](figures/adaptation_curves.png)
![Confusion matrices, scratch vs adapted](figures/confusion_matrices.png)
-->

---

## Takeaways

1. **Fine-tuning a pretrained model clearly beats training from scratch when target data is scarce.** The advantage is largest at the smallest budgets.
2. **Check zero-shot transfer first.** It costs almost nothing to run. For some benchmark pairs (ICCAD-3 → ICCAD-5), no fine-tuning is needed at all.
3. **The best fine-tuning depth depends on the target.** Use head-only fine-tuning only when zero-shot results already show the source and target are well matched.
4. **Low precision, high recall is the right trade-off here.** A missed hotspot becomes a real defect on the wafer, while a false alarm only triggers a cheap extra check.

---

## Limitations and future work

**Limitations:**

- Only one source benchmark (ICCAD-3) was tested.
- Training was capped at 15–50 epochs per condition because of compute limits.
- Only one CNN architecture was used.

**Future work:**

- A full 5×5 cross-benchmark transfer matrix
- Partial freezing, e.g. unfreezing only the last convolutional block
- Few-shot meta-learning

---

## Tools

Python · TensorFlow / Keras · NumPy · Google Colab

## How to run

1. Open the notebook in Google Colab.
2. Point the dataset path to your copy of ICCAD-12.
3. Run all cells.
4. Results are written per benchmark, budget, seed and approach.

---

## Team

Course project for **AI-ML for IC Design (Digital Assignment 2)**, VIT Chennai.

- **Sonakshi Agrawal**
- Edupuganti Vyshnavi
- Manda Krishna Hriday
- Abhinav M A

## Reference

A. Verma, K. A. Rao, D. S. Hegde, *Lithography Hotspot Detection using Deep Learning*, IIT Bombay, 2024. This is the baseline CNN architecture. The full reference list is in the report.
