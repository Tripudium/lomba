```mermaid
graph LR
    subgraph Data["Data Processing"]
        LOB[LOB Messages<br/>(Time, Price, Size, Type)] --> Feat[Feature Extraction<br/>(Delta Price, Log Size, Log dt)]
        Feat --> Bin[Binning & Hierarchy<br/>(Quantile Bins, Coarse/Resid)]
    end

    subgraph Model["Model Architecture"]
        Bin --> Embed[Embeddings & Projection]
        Embed --> Mamba[Mamba Backbone<br/>(Selective SSM)]
    end

    subgraph Heads["Output Heads"]
        Mamba --> TPP[TPP Head<br/>(Mixture of Exponentials)]
        Mamba --> Marks[Hierarchical Mark Heads<br/>(Price, Size, Type, Side)]
    end

    TPP --> NextTime[Next Arrival Time]
    Marks --> NextMarks[Next Event Marks]

    style Data fill:#e1f5fe,stroke:#01579b
    style Model fill:#fff3e0,stroke:#ff6f00
    style Heads fill:#e8f5e9,stroke:#2e7d32
```
