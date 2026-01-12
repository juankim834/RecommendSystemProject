# RecommendSystemProject
This is for personal practice project

## What is in this system?

### DSSM

This project implements a customized **Dual-Tower DSSM **(Deep Structured Semantic Model) designed for retrieval tasks. The model explicitly separates user and item feature processing to allow for efficient large-scale retrieval using ANN (Approximate Nearest Neighbor).

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

flowchart TB
    %% Style definition
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:5,ry:5;
    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef trans fill:#e0f2f1,stroke:#00695c,stroke-width:2px,stroke-dasharray: 5 5;
    classDef pool fill:#b2dfdb,stroke:#004d40,stroke-width:2px,rx:5,ry:5;
    classDef tower fill:#f3e5f5,stroke:#7b1fa2,stroke-width:4px;
    classDef final fill:#ffebee,stroke:#c62828,stroke-width:2px,rx:5,ry:5;

    subgraph Interaction_Layer [Interaction & Loss]
        direction TB
        DotProd((Dot Product))
        Loss[Cross-Entropy Loss]:::final
    end

    subgraph User_Tower [User Tower]
        direction TB
        
        %% Input layer
        U_Seq_In[Input: Sequence Features]:::input
        U_Sparse_In[Input: Sparse Features]:::input
        U_Dense_In[Input: Dense Features]:::input

        %% Processing Layer - Sequence (Transformer)
        subgraph Transformer_Block [Sequence Encoder]
            direction TB
            T_Emb[Embedding Layer]:::trans
            T_MHA[Multi-head Attention]:::trans
            T_Norm[Layer Normalization]:::trans
            
            T_Emb --> T_MHA --> T_Norm
        end
        
        %% Pooling Layer
        T_Pool[Mean Pooling]:::pool

        %% Processing Layer - Others
        U_MLP[MLP Tower / Embedding]:::process
        U_Linear[Linear Layer + ReLU]:::process
        
        %% Concating and Output
        U_Concat[Concat Features]:::process
        U_FinalNorm[Layer Normalization]:::tower

        U_Seq_In --> T_Emb
        T_Norm --> T_Pool
        T_Pool --> U_Concat

        U_Sparse_In --> U_MLP --> U_Concat
        U_Dense_In --> U_Linear --> U_Concat
        
        U_Concat --> U_FinalNorm
    end

    subgraph Item_Tower [Item Tower]
        direction TB
        
        %% Input Layer
        I_Sparse_In[Input: Sparse Features]:::input
        I_Dense_In[Input: Dense Features]:::input
        
        %% Processing Layer
        I_MLP[MLP Processing]:::process
        I_Linear[Linear Layer + ReLU]:::process
        
        %% Concatting and Output
        I_Concat[Concat Features]:::process
        I_FinalNorm[Layer Normalization]:::tower
        
        %% 连线
        I_Sparse_In --> I_MLP --> I_Concat
        I_Dense_In --> I_Linear --> I_Concat
        I_Concat --> I_FinalNorm
    end

    %% Connection of Towers and Processing Layers
    U_FinalNorm -- User Vector --> DotProd
    I_FinalNorm -- Item Vector --> DotProd
    
    DotProd --> Loss
    Label[True Labels]:::input -.-> Loss
    linkStyle default stroke:#333,stroke-width:2px;

