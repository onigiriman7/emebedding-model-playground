# Semantic Geometry Distillation (SGD)

**Efficient, controllable document clustering by distilling LLM judgments into a small adapter on top of a frozen embedder.**

> Status: research draft. Experiments are proposed, not yet run. All "target" numbers below are hypotheses, not results.

## The Problem in One Example

Take three documents:

1. "Apple released a new AI-powered laptop"
2. "Apple's quarterly revenue increased"
3. "Microsoft released a new AI-powered laptop"

Group them by **company** and (1, 2) go together. Group them by **event type** or **product** and (1, 3) go together. Neither is "correct" — it depends on what you want.

- **Standard embedding + K-means** is cheap, but uses one fixed, general notion of similarity.
- **LLMs** can reason about the notion you want, but calling them repeatedly over a large corpus is slow and expensive.

## The Idea

Keep the pretrained embedder `E` **frozen**. Learn only a small transformation on top of it, conditioned on the user's objective `q`:

```
x  ->  z = E(x)  ->  z' = f_theta(z, q)  ->  cluster (e.g. spherical K-means)
```

- A small, actively chosen set of LLM pairwise judgments trains `f_theta`.
- After training, **no LLM calls** are needed at inference, and **no re-embedding** when `q` changes.
- New objective (topic, entity, event, risk, sentiment...) = train a new small adapter, not a new embedding model.

## How It Differs from Prior Work

| Method | Embedder updated? | Supervision | Task conditioning | Cross-K consistency |
|---|---|---|---|---|
| ClusterLLM | Yes (fine-tuned) | Discrete hard triplets | Via data / instructions | Not addressed |
| Prefix-instruction embeddings | No (prompted) | Instruction prefix | Prefix into decoder-only embedder | Not addressed |
| PPLL | Yes (contrastive) | LLM pseudo-labels | Not per-query | Not addressed |
| Co-evolving LLM + embedder | Yes (co-trained) | Preference-optimized interpretations | Implicit | Not addressed |
| **SGD** | **No (frozen)** | **Graded scores + active sampling** | **Explicit adapter `f_theta(z, q)`** | **Explicit objective** |

The paper is upfront that LLM-guided clustering is already an active field. No single axis is unique to SGD; the claim is about the **combination** and the **cost profile**.

## Contributions

1. **Parameter-efficient reformulation:** frozen embedder plus a small task-conditioned adapter, so the marginal cost of a new objective is an adapter run, not an embedder fine-tune.
2. **Explicit clusterability objective**, kept separate from relationship matching so it can be ablated.
3. **Graded supervision** (`r_ij` in [0,1]) with uncertainty-driven active sampling, instead of discrete triplets, plus a **multi-resolution consistency** objective.
4. **Direct empirical positioning** against the closest methods on a quality / cost / parameter-efficiency frontier.

## Method at a Glance

**Adapter.** Either a small projection network `g_theta([z; e(q)])` or a cheaper task-conditioned linear map `W_q z`. Outputs are L2-normalized for cosine / spherical K-means.

**Loss.** `L = λ_s L_semantic + λ_c L_cluster + λ_p L_preserve`

| Term | Purpose |
|---|---|
| `L_semantic` | Make cosine similarity `s_ij` match the LLM's relationship score `r_ij` (regression or contrastive form) |
| `L_cluster` | Push intra-cluster similarity above inter-cluster similarity, so the space is easy to partition |
| `L_preserve` | Keep `z'` close to `z` to avoid solutions that only memorize the supervision |

**Active supervision.** With O(N²) pairs, query the LLM only on pairs where the base embedding is most uncertain (similarity near the decision boundary). Loop: embed, pick uncertain pairs, get LLM scores, update the adapter, repeat.

**Why pairwise?** Asking an LLM to assign categories requires a taxonomy that may not exist and may depend on `q`. Asking "are these two related under `q`?" is more fundamental. Graded scores are hypothesized to carry more information per query than triplets.

**Controlling K.** The user picks K freely; representation (`f_theta`) is decoupled from granularity.

**Multi-resolution consistency.** Clusters at K=50 should nest inside their K=10 parents, as a trained and measurable property rather than an accident.

## Cost Model

- **Training:** LLM calls on M selected pairs, with M << N².
- **Inference:** O(N·d) for the transform plus the clustering algorithm. No LLM, no embedder updates.
- **Key metric:** marginal cost of adding a new objective `q` (adapter retrain vs. embedder fine-tune), reported alongside cost per million documents.

## Proposed Research Questions

| RQ | Question |
|---|---|
| RQ1 | Does SGD improve clustering quality over embedding + K-means / spherical K-means / hierarchical? |
| RQ2 | Can SGD match ClusterLLM, prefix-instruction control, PPLL, and co-evolving methods at lower parameter and retraining cost? |
| RQ3 | How many LLM judgments are needed? (M from 0 to 100K; hope for fast saturation) |
| RQ4 | Does active sampling beat random, and does graded scoring beat triplets at matched query budget? |
| RQ5 | Does conditioning enable genuinely different clusterings of the same corpus, and does cross-K nesting hold? |

## Evaluation

- **External:** ARI, NMI, V-measure, clustering F1 (with the caveat that labels encode only one interpretation).
- **Intrinsic:** intra/inter-cluster cosine, silhouette, size balance, stability.
- **LLM-as-evaluator:** must be a different model from the supervision teacher.
- **Human evaluation:** "Given objective `q`, which clustering is more coherent?"
- **Cost:** $ per corpus per stage, marginal $ per new `q`, runtime, GPU memory, CPU cores, LLM calls.

## Planned Ablations

- No transform, and generic (unconditioned) transform
- Random vs. uncertainty-based supervision
- Graded pairwise vs. hard triplets (matched budget)
- Frozen embedder + adapter vs. full embedder fine-tuning (matched parameters/compute)
- Semantic loss only, semantic + cluster, and full objective
- With vs. without multi-resolution consistency (cross-K containment)

## Known Failure Modes

- **Teacher bias:** the adapter inherits the LLM's errors.
- **Overfitting** with very few judgments.
- **Ambiguous objectives:** e.g. "cluster by relevance" is underspecified.
- **Granularity mismatch** across K.
- **Representation bottleneck:** if the frozen embedder discarded the needed information, the adapter cannot recover it. This is the main cost of the frozen design.

## Broader Impact

Cheaper semantic clustering helps knowledge management, legal discovery, literature analysis, and customer feedback work. Clusters can encode embedder and teacher bias, so systems should expose uncertainty and let users inspect representative documents. Clustering is treated as task-dependent organization, not discovery of one true taxonomy.

## Key References

- ClusterLLM (EMNLP 2023)
- Controllable Clustering with LLM-driven Embeddings (EMNLP Industry 2025)
- PPLL: Positive Pairs from LLM-Driven Labels (EMNLP 2025)
- Co-Evolving LLMs and Embedding Models via Density-Guided Preference Optimization (EMNLP 2025)
- BERTopic, SCCL, Spherical K-Means (baselines)