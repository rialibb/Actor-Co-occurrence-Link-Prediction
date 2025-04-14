from pipeline import pipeline







if __name__ == "__main__":
    
    pipeline(
        SEED=0,                          # int: Random seed for reproducibility (e.g., 0, 42)
        NODE_INFO="data/node_information.csv",  # str: Path to node feature file (CSV format)
        TRAIN_FILE="data/train.txt",     # str: Path to training edge list file
        TEST_FILE="data/test.txt",       # str: Path to test edge list file
        VAL_RATIO=0.1,                   # float: Validation split ratio (e.g., 0.1 for 10%)
        NAME='TransformerGNN',           # str: GNN model name — one of {'GCN', 'SAGE', 'GAT', 'GIN', 'APPNP', 'TransformerGNN'}
        HIDDEN_DIM=256,                  # int: Hidden layer dimension in GNN (e.g., 128, 256)
        EMBED_DIM=64,                    # int: Output embedding dimension (e.g., 64, 128)
        NUM_EPOCHS=50,                   # int: Maximum number of training epochs
        LR=0.001,                        # float: Learning rate for optimizer
        PATIENCE=10                      # int: Early stopping patience (stop if no val improvement after this many epochs)
    )