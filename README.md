# embedding-model-playground

**A sandbox for bending, blending, translating, and stress-testing embedding spaces, without retraining the models underneath.**

---

## The Intuition

Think of a pretrained embedding model as a huge, detailed map of meaning. Every document is a pin on that map, and "similar" means "pinned nearby."

The catch: the map was drawn for *general* purposes. Ask it to group these three sentences:

1. "Apple released a new AI-powered laptop"
2. "Apple's quarterly revenue increased"
3. "Microsoft released a new AI-powered laptop"

By **company**, (1, 2) belong together. By **product** or **event**, (1, 3) do. The map has one answer baked in.

Now imagine you don't redraw the map. You put a **lens** over it. A lens is a small, cheap function that stretches the parts of the map you care about and squashes the rest. Change the lens, and the same pins rearrange to answer a different question.

That is the whole playground: **the big model stays frozen, and the interesting work happens in tiny, swappable transformations on top of it.**

## Why This Might Be the Next Big Thing

Fair warning: this is a thesis, not a proven result. Each point below is a bet this repo exists to test.

**1. The economics flip.** Fine-tuning an embedder for every new objective is expensive. A lens is kilobytes to megabytes and trains in minutes. When the marginal cost of a new perspective approaches zero, people can afford thousands of perspectives instead of one.

**2. The heavy model is already commoditized.** Strong embedders are widely available. The differentiation is moving from *the model* to *how you look through it*.

**3. LLMs can be distilled into geometry.** Calling an LLM on every document pair doesn't scale. But a few thousand well-chosen LLM judgments can teach a small lens to reproduce that judgment, and then inference is just a matrix multiply. (This is the idea behind Semantic Geometry Distillation, below.)

**4. Lenses are objects you can compute with.** Once a perspective is a matrix or a small network, it can be added, interpolated, inverted, shared, and versioned like code. That is a new kind of composability for semantic search.

## Core Reference: Semantic Geometry Distillation (SGD)

SGD is the research draft that seeded this repo. In short:

```
x  ->  z = E(x)  ->  z' = f_theta(z, q)  ->  cluster / search
        (frozen)      (small, trained per objective q)
```

- `E` is a frozen pretrained embedder.
- `f_theta(z, q)` is a small adapter conditioned on an objective `q` (topic, entity, event, risk, sentiment...).
- Training uses a *sparse, actively selected* set of LLM pairwise relationship scores `r_ij` in [0, 1], focusing on pairs where the base embedding is most uncertain.
- The loss combines three terms: `L = λ_s L_semantic + λ_c L_cluster + λ_p L_preserve`, matching LLM scores, making the space easy to partition, and staying close to the original so it doesn't memorize the supervision.
- A proposed multi-resolution consistency objective asks that K=50 clusters nest inside their K=10 parents.
- After training, **no LLM calls and no re-embedding** are needed when the objective changes.

Honest status: the experiments in the draft are *proposed, not yet run*. Related methods (ClusterLLM, prefix-instruction embedding control, PPLL, co-evolving LLM/embedder training) already occupy nearby ground. SGD's specific bet is the combination of frozen embedder, graded supervision, explicit clusterability, and cross-resolution consistency, and that bet still needs evidence. Its main known cost: if the frozen embedder discarded the information you need, no lens can recover it.

## Experiment Ideas

Everything below is an experiment to run, ordered roughly from "most likely to work" to "most fun if it does."

### 1. Question-shaped lenses (hypernetwork)
Train a network that *generates* the lens from the text of the objective. Type "group by customer pain point" and space reshapes on the fly, with no per-objective training. Could plausibly replace some rerankers.

### 2. Cross-embedder translation
Different embedding models appear to learn surprisingly similar geometry. Train a small map `T: E_A -> E_B` and test whether you can migrate a vector index without re-embedding, or make two models interoperable.

### 3. Perspective discovery
Flip the problem: instead of asking for a lens, *search* for lenses that make your corpus highly clusterable in ways your existing lenses don't explain. The system proposes slicings you didn't think to ask for.

### 4. Lens algebra
If lenses are matrices, try `W_company - W_product`, interpolate 30% legal with 70% sentiment, or project onto a lens's orthogonal complement to make search *blind* to a concept (topic-agnostic tone search, for example). Question to answer: how linear is this in practice?

### 5. Ambiguity as disagreement
Run a committee of lenses over each document. Where they disagree about a document's neighbors, that document is genuinely ambiguous. Use variance across lenses as an ambiguity score, and route those cases to an LLM or a human.

### 6. Lens spectroscopy
Train many lenses for many concepts and measure which are learnable from very few labels. That profile is a fingerprint of what an embedder knows and what it threw away, a cheap audit before you commit to a model.

### 7. Time lenses
Learn `W_t -> W_{t+1}` to map semantic drift: what a 2012 document "means" in 2026 vocabulary, or where a topic is heading.

### 8. Curved space for free hierarchy
Map flat embeddings into hyperbolic space with learned curvature and see whether taxonomy-like structure emerges without labeled hierarchies. Connects directly to SGD's cross-K consistency objective.

### 9. Retrieval-based editing
Shift a vector along a lens direction (more formal, more urgent), then retrieve the nearest *real* sentence. Style transfer with no generator, and every output is an existing text.

### 10. Geometry as a reward
Use similarity in a lens space as a cheap, differentiable reward when tuning a generator, effectively a tiny reward model per perspective.

### 11. Keyed semantic access
Apply a secret rotation to all vectors. Cosine similarity is preserved so search still works, but the meaningful geometry needs the key. *Not real cryptography*, but possibly a cheap extra layer against embedding-inversion attacks. Needs careful adversarial testing before anyone trusts it.

### 12. On-device context switching
Swap a tiny lens based on device state (for example riding vs. parked, navigating vs. idle) so the same stored memory retrieves differently by context. Because lenses are so small, switching is nearly free, which makes this a natural fit for edge deployment.

## What Success and Failure Look Like

We want this repo to be honest about its own claims.

| Question | If yes | If no |
|---|---|---|
| Can a lens match full embedder fine-tuning at a fraction of the cost? | The economics thesis holds | Frozen embedders are too limiting for some tasks |
| Do lenses compose (add, subtract, blend) predictably? | Lens algebra is real | Perspectives are nonlinear; algebra is a toy |
| Does cross-embedder translation preserve neighborhoods? | Index migration and interoperability become cheap | Embedding geometries are less universal than they look |
| Does graded LLM supervision beat discrete triplets at equal query budget? | SGD's core design choice is validated | Triplets are just as good and simpler |
| Does a hypernetwork generalize to unseen objectives? | Zero-shot perspectives work | Lenses need per-objective data |

Known risks: teacher bias (the lens inherits the LLM's mistakes), overfitting with too few judgments, underspecified objectives ("cluster by relevance"), and the representation bottleneck noted above.

## Suggested Repo Layout

```
embedding-model-playground/
├── lenses/          # adapter implementations (linear, MLP, hypernetwork)
├── supervision/     # LLM pairwise scoring + active sampling
├── objectives/      # semantic, clusterability, preservation, consistency losses
├── experiments/     # one folder per idea above, each with its own README
├── eval/            # ARI/NMI, cross-K containment, cost tracking
└── notebooks/       # intuition-first walkthroughs
```

## Evaluation Principles

- Report quality **and** cost (dollars per corpus, marginal cost per new objective, LLM calls, GPU memory).
- Use an evaluator LLM that is *different* from the teacher LLM.
- Include human judgment: "given this objective, which clustering is more coherent?"
- Never treat benchmark labels as the only valid interpretation.

## Contributing

Pick an experiment, write down the hypothesis and what would falsify it *before* running it, and open a PR with the result, including negative ones. A clean "this didn't work" is as valuable as a win.

## Key References

- ClusterLLM: LLMs as a Guide for Text Clustering (EMNLP 2023)
- Controllable Clustering with LLM-driven Embeddings (EMNLP Industry 2025)
- PPLL: Positive Pairs from LLM-Driven Labels (EMNLP 2025)
- Co-Evolving LLMs and Embedding Models via Density-Guided Preference Optimization (EMNLP 2025)
- BERTopic, SCCL, Spherical K-Means (baselines)