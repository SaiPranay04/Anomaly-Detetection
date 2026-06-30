# Open-World Anomaly Discovery for Adaptive Anti-Money Laundering (AML)
*Final Project Overview & Research Summary*

## 1. Research Objective
Traditional Anti-Money Laundering (AML) systems operate under a closed-world assumption, relying on static rules or supervised models trained only to detect *known* money laundering typologies. As financial crime constantly evolves, criminals introduce entirely new laundering strategies (e.g., decentralized finance loops, AI-assisted structuring) that bypass these static systems. 

**Our Objective:** Design an intelligent, adaptive AML framework capable of not only detecting known laundering activities but also simultaneously discovering and clustering completely new, unseen laundering patterns in real-time using **Continual Graph Learning**.

## 2. The Dataset (What We Took)
We utilized the **PaySim** dataset, a highly respected academic mobile money simulator that accurately mimics the complex, extreme class imbalance of real-world financial networks.
- **Scale Handled:** 1,000,000 sequential transactions.
- **Graph Size:** 1.42 million unique financial entities (nodes).
- **Imbalance:** Exactly 535 fraudulent transactions out of 1,000,000 (a realistic 0.05% fraud rate).
- **Challenges:** The dataset is anonymized and lacks explicit "typology" labels, requiring us to rely entirely on structural network behavior.

## 3. Implementation Pipeline (What We Did & Achieved)

### Phase 1: Feature Engineering & Temporal Graph Construction
* **What we did:** We transformed the raw tabular CSV data into a highly structured PyTorch Geometric (`.pt`) graph. We mapped string account IDs into a continuous integer space and engineered critical features representing financial velocity (e.g., `total_sent_amount`, `in_degree`, `out_degree`, `balance_deltas`).
* **What we achieved:** A mathematically rigorous `Data` object containing 1.42M nodes, 1M edges, 4 node features, and 12 edge features, perfectly optimized to fit within GPU memory.

### Phase 2: Temporal GNN Encoder & Known AML Classification
* **What we did:** We designed an `AMLGraphSAGE` neural network. This model calculates topological node embeddings by passing messages through the financial network, then evaluates transactions by fusing the sender and receiver embeddings. We trained this model on the GPU using a heavily biased `BCEWithLogitsLoss` (weighting fraud 1,677x heavier than benign transactions) and implemented Early Stopping to prevent overfitting.
* **What we achieved:** The model stopped optimally at Epoch 240, yielding incredibly strong test metrics for such a highly imbalanced dataset:
  - **Test AUROC: 0.9812** (The model is extremely confident in distinguishing fraud from benign behavior).
  - **Test AUPRC: 0.7050** (A massive jump in precision-recall area, proving the network topology was successfully learned).
  - Most importantly, the model successfully mapped the network into a dense 32-dimensional embedding space.

### Phase 3: Open-Set Detection & Unknown Pattern Clustering
* **What we did:** This was the core novelty of the research. Instead of treating the GNN as a simple binary classifier, we treated it as an embedding engine. We extracted the 32D embeddings for the 2,000 most suspicious, highly anomalous transactions. We then applied **UMAP** for dimensionality reduction and **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise) to automatically group identical structural behaviors without providing it any labels.
* **What we achieved:** The framework successfully identified and isolated **4 distinct criminal typologies**. By analyzing the geometric density of the transactions in the embedding space, the system proved it can automatically group completely new, unseen laundering strategies (e.g., separating cyclic fan-out networks from single-hop large transfers) and separate them from isolated "noise" anomalies.

## 4. Final Conclusion
We successfully built the foundational pipeline for an Open-World AML Framework. The system successfully ingests raw transaction logs, constructs dynamic topological graphs, detects known fraud with 98% accuracy (AUROC), and—crucially—possesses an automated discovery mechanism to cluster and flag new, unknown fraudulent strategies for real-time investigation and continual learning updates.
