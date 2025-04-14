
# 🔗 Actor Co-occurrence Link Prediction

This project addresses the problem of **predicting missing links** in an actor co-occurrence network using **Graph Neural Networks (GNNs)**, **Node2Vec embeddings**, and **graph-theoretic features**. The network is constructed from Wikipedia page co-occurrences of actors, and the task is to reconstruct randomly deleted edges from the graph.

## 📌 Problem Statement

Given:
- A graph of actors where an edge represents co-occurrence on the same Wikipedia page.
- A set of node features extracted from the actors’ Wikipedia text.
- A training set of known edges (with binary labels: 1 for real, 0 for negative samples).
- A test set of node pairs for which edge existence is to be predicted.

The goal is to accurately **predict whether a link exists** between given pairs of nodes.

---


## 🧱 Project Structure

```
.
├── data/
│   ├── node_information.csv     # Wikipedia-derived node features
│   ├── train.txt                # Training edge list with labels
│   └── test.txt                 # Test edge list (to predict)
├── utils.py                     # I/O and preprocessing
├── feature_engineering.py       # Node2Vec and stat features
├── feature_embedding.py         # GNN encoder definitions
├── classifier.py                # Link predictor MLP
├── pipeline.py                  # Training, evaluation, and inference logic
├── main.py                      # Entry point
├── output_files/                # Folder for predictions
└── README.md                    # Project documentation
```

---

## 🧠 Model Architecture

The model consists of three major components:

### 1. **Feature Extraction**
- **Textual Features**: 932-dimensional node feature vectors extracted from Wikipedia.
- **Node2Vec Embeddings**: Learned structural embeddings from the graph.
- **Graph-Theoretic Statistics**: Degree, clustering coefficient, PageRank, closeness, and betweenness centrality.

### 2. **Graph Encoder**
Graph Neural Networks are used to encode nodes into latent embeddings. Supported encoders:
- `GCN`
- `GraphSAGE`
- `GAT`
- `GIN`
- `APPNP`
- `TransformerGNN`

### 3. **Link Predictor**
A feedforward neural network takes the embeddings of two nodes and their statistical features, and outputs a **link existence score**.

---


## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python main.py
```

The model will:
- Load and preprocess the data.
- Train a GNN + Link Predictor.
- Evaluate on validation data.
- Generate predictions on the test set.
- Save output to `output_files/`.


---

## 📊 Output

- Validation performance is tracked each epoch.
- Best model (based on validation accuracy) is saved to `best_model.pt`.
- Final predictions are saved as:

```text
output_files/TransformerGNN_predictions.csv
```

---

## 📈 Model Fusion

The final node embeddings used for prediction are a **sum of GNN and Node2Vec embeddings**, combining structural and learned representations:

```python
total_emb = encoder(X, edge_index) + node2vec_embeddings
```

The link predictor also ingests 10 graph-statistical features for each node pair.

---

## 📬 Contact

For questions or contributions, feel free to reach out or open an issue.
