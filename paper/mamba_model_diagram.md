```mermaid
graph TD
    subgraph Inputs["Input Features"]
        Fields[Categorical & Continuous Fields<br/>(Price, Size, Type, Side, Level, Time)]
        Dt[Inter-arrival Time<br/>(dt_prev)]
        TimeAbs[Absolute Time<br/>(Optional)]
    end

    subgraph Embedding["Embedding Layer"]
        Embeds[Field Embeddings / Soft Binning]
        Concat[Concatenate]
        Proj[Linear Projection]
        DtProj[Log1p + Linear Projection]
        TimeEnc[Cyclical Encoding + Linear]
    end

    subgraph Backbone["Mamba Backbone"]
        Add[Add Features]
        DropEmb[Dropout]
        MambaStack[Mamba Layers (Stack of N)]
        Norm[LayerNorm]
        DropMlp[Dropout]
    end

    subgraph Heads["Output Heads"]
        TPP[Mixture TPP Head]
        MarkHeads[Hierarchical Mark Heads<br/>(Price, Size, Type, Side, Level, Time)]
    end

    Fields --> Embeds --> Concat --> Proj --> Add
    Dt --> DtProj --> Add
    TimeAbs --> TimeEnc --> Add

    Add --> DropEmb --> MambaStack --> Norm --> DropMlp

    DropMlp --> TPP
    DropMlp --> MarkHeads

    style Inputs fill:#e3f2fd,stroke:#1565c0
    style Embedding fill:#fff8e1,stroke:#ff8f00
    style Backbone fill:#f3e5f5,stroke:#7b1fa2
    style Heads fill:#e8f5e9,stroke:#2e7d32
```
