---
title: Overview
description: How semantic routing works, from meaning as geometry to a calibrated decision.
---

Semantic routing sends an input — text, an image, audio — to the right handler based on what it *means*, not on keywords or hand-written rules. People say the same thing a hundred different ways. Semantic routing handles that variety without you enumerating every phrasing.

## Semantic space

Here's the core idea. Imagine a huge coordinate system where every point is a concept. "I need help with my password" sits right next to "Can't log in to my account", and far from "What's the weather forecast?". That's semantic space. Meaning becomes geometry, and similarity becomes a distance you can measure.

- Each point (a vector) represents the meaning of some content.
- Distance between points is semantic difference.
- Similar meanings cluster together, whatever the exact wording.

So synonyms, paraphrases, and related ideas all land close to each other — and you never program the variations by hand.

### Encoders

To put content into semantic space, you *encode* it into a vector. That's the job of an encoder (you'll also hear "embedding model" or "bi-encoder").

```mermaid
flowchart LR
    A["Input Content<br/>(Text, Image, Audio)"] --> B["Bi-Encoder<br/>(Neural Net)"]
    B --> C["Vector Embedding<br/>[x₁, x₂, ..., xₙ]"]
    
    style A fill:#f5f5f5,stroke:#333,stroke-width:1px
    style B fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style C fill:#f1f8e9,stroke:#558b2f,stroke-width:1px
```

An encoder takes the input, analyzes its features, and outputs a fixed-size vector of numbers — usually hundreds or thousands of dimensions. "How's the weather today?" might become `[0.12, -0.34, 0.56, ...]`. "What's the temperature outside?" becomes a different but nearby vector, because the meanings are close.

Semantic Router supports two families:

- **Dense encoders** (`OpenAIEncoder`, `HuggingFaceEncoder`, …) fill every dimension. They capture rich semantic relationships.
- **Sparse encoders** (`AurelioSparseEncoder`, `BM25Encoder`, …) leave most dimensions at zero. They excel at exact keywords and term frequency.

### Multimodal routing

Text is the common case, but anything you can encode, you can route:

- **Images.** `CLIPEncoder` and `VitEncoder` place images in the same space as text, so you can compare across modalities.
- **Audio.** Speech or sound, routed on content or tone.
- **Mixed content.** Text and images together, encoded jointly or separately.

That's what lets you route on what's *in* an image, or on the combined meaning of text plus a picture.

### Making the decision

Once everything is a vector, routing is a similarity calculation. The usual measures:

- **Cosine similarity**: cos(θ) = (A·B)/(||A||·||B||)
- **Euclidean distance**: d(A,B) = √(Σ(Aᵢ-Bᵢ)²)
- **Dot product**: A·B = Σ(Aᵢ·Bᵢ)

An `IndexedRouter` compares the incoming query against every route's example utterances and gives each route a score. Raw similarities are hard to act on, though: 0.8 means something different for every encoder and every set of routes. So the router can be *calibrated*. Fit it on some labelled queries and it learns to turn a score into the probability that the route is right. Then a threshold like 0.9 means "act only when you're 90% sure", and the best route above it wins.

### When similarity isn't enough

Some decisions depend on instructions, not resemblance: "refunds, but not pricing questions before a purchase". A `DecisionRouter` hands those to a small, fast decision model that reads your route descriptions and returns a probability for each route. You can chain the two, so the embedding router answers the easy queries in milliseconds and only the ambiguous ones go to the model.

## The workflow

1. **Define routes**: example utterances for each decision, and a description if a decision model will read it.
2. **Pick an encoder**: based on content type and performance needs.
3. **Build the router**: connect the encoder, routes and an index.
4. **Calibrate it**: fit on labelled queries the router hasn't seen, then set a threshold.
5. **Write handlers**: the logic for each route, and a fallback for "no route".
6. **Route inputs**: encode, score, decide, dispatch.

```mermaid
flowchart TD
    subgraph Setup["Setup Phase"]
        A[Define Routes] --> |Example utterances| B[Select Encoder]
        B --> |Based on content type| C[Configure Router]
        C --> |Connect components| E[Precompute Route Embeddings]
        E --> D[Calibrate on Labelled Queries]
    end
    
    subgraph Runtime["Runtime Phase"]
        F[Receive Input] --> G[Encode Input]
        G --> H[Calculate Similarity]
        H --> I{Confidence > Threshold?}
        I -->|Yes| J[Execute Matched Handler]
        I -->|No| K[Execute Default Handler]
    end
    
    D --> F
    
    classDef setup fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef runtime fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    class A,B,C,D,E setup
    class F,G,H,I,J,K runtime
```

Semantic Router handles the vector math, the scoring, the calibration and the decision. You focus on defining good routes and writing the handlers.
