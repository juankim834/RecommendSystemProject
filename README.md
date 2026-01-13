# RecommendSystemProject

This is for personal practice project

## What is in this system?

### DSSM

This project implements a customized **Dual-Tower DSSM**(Deep Structured Semantic Model) designed for retrieval tasks. The model explicitly separates user and item feature processing to allow for efficient large-scale retrieval using ANN (Approximate Nearest Neighbor).

The architecture consists of two main towers:

1. User Tower (Complex Feature Handling) The user tower is designed to capture dynamic user interests and static profiles:

    - Sequential Features: Processed via a Transformer-based Sequence Encoder.

        - Includes Embedding, Multi-head Attention, and Layer Normalization.

        - Mean Pooling is applied to the output sequence to generate a fixed-size context vector representing the user's historical behavior.

    - Sparse Features: Processed via specific MLP towers (Embedding lookup + Dense layers).

    - Continuous Features: Transformed using Linear layers with ReLU activation.

    - All features are concatenated and normalized (LayerNorm) to form the final User Embedding.

2. Item Tower (Lightweight)

    - Sparse & Continuous Features: Processed via MLP and Linear+ReLU layers respectively.

    - Outputs a normalized Item Embedding.

3. Interaction & Loss

    - Similarity: Calculated using the Dot Product of the normalized user and item vectors (equivalent to Cosine Similarity).

    - Training: Optimized using Cross-Entropy Loss.

#### Code snippet

```mermaid
flowchart LR
    %% ================= User Tower =================
    subgraph U[User Tower]
        U1[Sparse Features\n(User ID / Category)]
        U2[Dense Features\n(Numerical Values)]
        U3[Sequential Features\n(User Behavior Sequence)]

        U1 --> U1E[MLP_Tower\nEmbedding]
        U2 --> U2E[Linear + ReLU]
        U3 --> U3E[Sequence Encoder\nEmbedding Layer\nMulti-Head Attention\nLayerNorm]

        U1E --> UC[Concatenate]
        U2E --> UC
        U3E --> UC

        UC --> UVec[User Representation]
        UVec --> UNorm[L2 Normalization]
    end

    %% ================= Item Tower =================
    subgraph I[Item Tower]
        I1[Sparse Features\n(Item ID / Category)]
        I2[Dense Features\n(Numerical Values)]

        I1 --> I1E[MLP\nEmbedding]
        I2 --> I2E[Linear + ReLU]

        I1E --> IC[Concatenate]
        I2E --> IC

        IC --> IVec[Item Representation]
        IVec --> INorm[L2 Normalization]
    end

    %% ================= Matching =================
    UNorm --> DP[Dot Product Similarity]
    INorm --> DP

    DP --> Loss[Cross-Entropy Loss]
```
