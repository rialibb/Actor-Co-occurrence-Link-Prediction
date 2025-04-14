# Graph Neural Network for Link Prediction on Actor Co-occurrence Graph

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.utils import dropout_adj


from utils import extract_features, load_data, split_train_val, compute_graph_edges
from feature_engineering import extract_stat, compute_nodewise_stat, compute_node_2_vec_embed
from feature_embedding import select_embed_model
from classifier import LinkPredictor



# Device configuration: use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')







def pipeline(SEED=0, 
             NODE_INFO="data/node_information.csv", 
             TRAIN_FILE="data/train.txt",
             TEST_FILE="data/test.txt",
             VAL_RATIO=0.1, 
             NAME='GCN',
             HIDDEN_DIM=256, 
             EMBED_DIM=64, 
             NUM_EPOCHS=50,
             LR=0.001, 
             PATIENCE=10):
    """
    Main pipeline for training, validation, and prediction in the link prediction task.

    This function encapsulates all steps of the project, including preprocessing, GNN model training,
    performance tracking, final test set prediction, and saving of results.

    Inputs:
    - SEED (int): random seed for reproducibility.
    - NODE_INFO (str): path to the CSV file containing node features.
    - TRAIN_FILE (str): file containing labeled node pairs (link or no link).
    - TEST_FILE (str): file containing node pairs to be predicted.
    - VAL_RATIO (float): proportion of data used for validation (default is 10%).
    - NAME (str): name of the GNN model to use (e.g., 'GCN', 'GAT', 'APPNP', etc.).
    - HIDDEN_DIM (int): hidden layer dimension of the GNN.
    - EMBED_DIM (int): final embedding dimension produced by the encoder.
    - NUM_EPOCHS (int): maximum number of training epochs.
    - LR (float): learning rate for the optimizer.
    - PATIENCE (int): number of epochs without improvement before triggering early stopping.

    Outputs:
    - No explicit return (None), but:
        * The best model (based on validation accuracy) is saved to 'best_model.pt'.
        * Predictions on the test set are saved as a CSV file in the 'output_files' directory.
    """

    # Reproducibility: set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # Extract features
    X, id_to_index, feature_dim = extract_features(NODE_INFO)
    X = X.to(device)
    
    # Load train edges with labels and test edges
    train_data, test_edges = load_data(TRAIN_FILE, TEST_FILE, id_to_index)
    train_edges, train_labels = train_data

    # Split train edges into training and validation sets (10% for validation)
    train_edges_final, val_edges_final, train_pos = split_train_val(train_edges, train_labels, val_ratio=VAL_RATIO)

    # Prepare graph adjacency (edge list) for GNN from **training** positive edges only
    edge_index = compute_graph_edges(train_pos)
    edge_index = edge_index.to(device)
    
    # Compute nodewise stats 
    deg_dict, clustering_dict, pagerank_dict, closeness_dict, betweenness_dict = extract_stat(train_edges_final)

    # perform node2vec 
    node2vec_embeddings = compute_node_2_vec_embed(train_edges, test_edges, device)

    # Initialize models
    encoder = select_embed_model(NAME)(in_features=feature_dim, hidden_dim=HIDDEN_DIM, out_dim=EMBED_DIM).to(device)
    predictor = LinkPredictor(embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM).to(device)
    

    # Prepare DataLoader for training and validation edge lists
    train_loader = DataLoader(train_edges_final, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_edges_final, batch_size=128, shuffle=False)

    # Combine parameters of both encoder and predictor for the optimizer
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=1e-4)
    # Binary cross entropy loss with logits (combines Sigmoid + BCELoss)
    criterion = nn.BCEWithLogitsLoss()
    # Learning rate scheduler: 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,  # peak LR 
        steps_per_epoch=len(train_loader),  # number of batches per epoch
        epochs=NUM_EPOCHS,  # total epochs
        anneal_strategy='cos',  # 'cos' 
    )

    # Early stopping parameters
    PATIENCE = 10
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0



    # Training loop
    # ----------------
    for epoch in range(1, NUM_EPOCHS+1):
        encoder.train()
        predictor.train()
        total_loss = 0.0
        total_examples = 0
        train_corr =0

        # Iterate over batches of training edges
        for batch in train_loader:
            # batch is a list of tuples (u_idx, v_idx, label) of length batch_size
            # Default collate will convert it to three tensors: src_list, dst_list, label_list
            src, dst, labels = batch  # each will be a tensor
            src = src.to(device)
            dst = dst.to(device)
            labels = labels.to(device).float()

            # Forward pass: compute node embeddings and then link logits for this batch
            # Compute all node embeddings via GCNEncoder
            edge_index_dropped, _ = dropout_adj(edge_index, p=0.2)
            emb = encoder(X, edge_index_dropped)               # shape [num_nodes, embed_dim]
            total_emb = emb+ node2vec_embeddings
            # Gather embeddings for the source and destination nodes in the batch
            emb_u = total_emb[src]                           # shape [batch_size, embed_dim]
            emb_v = total_emb[dst]                           # shape [batch_size, embed_dim]
            embed_ns = compute_nodewise_stat(deg_dict, clustering_dict, pagerank_dict, closeness_dict, betweenness_dict, src, dst, device)
            # Compute link prediction logits for each pair
            logit = predictor(emb_u, emb_v, embed_ns).squeeze(1) # shape [batch_size], raw scores

            # Compute binary cross-entropy loss
            loss = criterion(logit, labels)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss for reporting
            batch_size = len(labels)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            # Compute accuracy: compare predicted class to true label
            preds = (torch.sigmoid(logit) >= 0.5).long()
            true = labels.long()
            train_corr += (preds == true).sum().item()

        # Step the learning rate scheduler
        scheduler.step()
        avg_train_loss = total_loss / total_examples if total_examples > 0 else 0.0
        train_accuracy = train_corr / total_examples if total_examples > 0 else 0.0

        # Validation loop (evaluate on validation set)
        encoder.eval()
        predictor.eval()
        val_loss_sum = 0.0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                src, dst, labels = batch
                src = src.to(device)
                dst = dst.to(device)
                labels = labels.to(device).float()
                # Forward pass on validation batch
                emb = encoder(X, edge_index)
                total_emb = emb+ node2vec_embeddings
                emb_u = total_emb[src]
                emb_v = total_emb[dst]
                embed_ns = compute_nodewise_stat(deg_dict, clustering_dict, pagerank_dict, closeness_dict, betweenness_dict, src, dst, device)
                logit = predictor(emb_u, emb_v ,embed_ns).squeeze(1)
                loss_val = criterion(logit, labels)
                # Accumulate validation loss
                batch_size = len(labels)
                val_loss_sum += loss_val.item() * batch_size
                val_total += batch_size
                # Compute accuracy: compare predicted class to true label
                preds = (torch.sigmoid(logit) >= 0.5).long()
                true = labels.long()
                val_correct += (preds == true).sum().item()
        avg_val_loss = val_loss_sum / val_total if val_total > 0 else 0.0
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0

        # Print epoch results
        print(f"Epoch {epoch:02d}: Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_accuracy:.4f}")

        # Check for improvement for early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch

            # Save to disk
            torch.save({
                'epoch': best_epoch,
                'encoder_state_dict': encoder.state_dict(),
                'predictor_state_dict': predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc
            }, 'best_model.pt')
        
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping condition
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered (no improvement in {PATIENCE} epochs).")
            break
        
    # Load best model weights from disk
    checkpoint = torch.load('best_model.pt', map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    predictor.load_state_dict(checkpoint['predictor_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with Val Acc = {checkpoint['val_accuracy']:.4f}")

    # Testing: generate predictions for the test set
    # -------------------------------------------------
    encoder.eval()
    predictor.eval()
    test_preds = []
    with torch.no_grad():
        # We can process all test edges in one go (they are not too many), or batch if needed
        test_src_indices = [src for (src, dst) in test_edges]
        test_dst_indices = [dst for (src, dst) in test_edges]
        test_src_tensor = torch.tensor(test_src_indices, dtype=torch.long).to(device)
        test_dst_tensor = torch.tensor(test_dst_indices, dtype=torch.long).to(device)
        # Compute embeddings for all nodes
        emb = encoder(X, edge_index)
        total_emb = emb+ node2vec_embeddings
        # Get embeddings for test pairs
        emb_u = total_emb[test_src_tensor]
        emb_v = total_emb[test_dst_tensor]
        embed_ns = compute_nodewise_stat(deg_dict, clustering_dict, pagerank_dict, closeness_dict, betweenness_dict, test_src_tensor, test_dst_tensor, device)
        # Compute link probabilities
        logit = predictor(emb_u, emb_v, embed_ns).squeeze(1)
        probs = torch.sigmoid(logit)  # probabilities in [0,1]
        # Threshold at 0.5 to get binary predictions (0 or 1)
        pred_labels = (probs >= 0.5).long().cpu().numpy()
        test_preds = pred_labels.tolist()

    # 7. Save predictions to CSV
    # --------------------------
    output_df = pd.DataFrame({
        "ID": list(range(len(test_preds))),
        "Predicted": test_preds
    })
    # create submission folder:
    if not os.path.exists('output_files'):
        os.makedirs('output_files')
    output_file = f"output_files/{NAME}_predictions.csv"
    output_df.to_csv(output_file, index=False)
    print(f"Test set predictions saved to {output_file}")
