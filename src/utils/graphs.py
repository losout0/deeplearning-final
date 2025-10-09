import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def create_database(CONFIG):
    logs_path = os.path.dirname(CONFIG["test_log_file"])

    gqa_path = os.path.join(logs_path, "test_metrics_20250926_145034.csv")
    mha_path = os.path.join(logs_path, "test_metrics_20250926_145752.csv")

    mha_df = pd.read_csv(mha_path)
    gqa_df = pd.read_csv(gqa_path)

    mha_df["model"] = "MHA"
    gqa_df["model"] = "GQA"

    df = pd.concat([mha_df, gqa_df], ignore_index=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df.sort_values(by=["model", "batch_idx"], inplace=True)

    print("Prévia dos dados unidos:")
    print(df.head())
    return df
    

def aggregate_batches(df, block_size=100):
    df = df.copy()
    df["block"] = (df["batch_idx"] - 1) // block_size + 1
    agg = df.groupby("block").agg({
        "loss": "mean",
        "perplexity": "mean",
        "F1": "mean",
        "Precision": "mean",
        "Recall": "mean"
    }).reset_index()
    return agg

def plot_graph(agg_mha, agg_gqa, metrica):
    plt.figure(figsize=(12, 5))
    
    plt.plot(agg_mha["block"], agg_mha[metrica], label="MHA", color="blue")
    plt.plot(agg_gqa["block"], agg_gqa[metrica], label="GQA", color="red")
    plt.xlabel("Block (100 batches)")
    plt.ylabel("Loss")
    plt.title("Loss média por bloco")
    plt.legend()
    plt.show()


def comparative_table(df):
    mean_metrics = df.groupby("model")[["loss", "perplexity", "F1", "Precision",
                                        "Recall", "duration_s", "num_tokens", "memory_MB"]].mean().reset_index()

    mean_metrics.columns = [
        "Model",
        "Loss média",
        "Perplexidade média",
        "F1 média",
        "Precisão média",
        "Recall média",
        "Tempo médio por batch (s)",
        "Tokens médios por batch",
        "Memória média por batch (MB)"
    ]

    mean_metrics["Tokens por segundo"] = mean_metrics["Tokens médios por batch"] / \
        mean_metrics["Tempo médio por batch (s)"]

    return mean_metrics


def confusion_matrix(logs_path, model):
    conf_path = os.path.join(logs_path, "predictions_20250926_145034.csv")
    conf_df = pd.read_csv(conf_path)

    class_counts = conf_df['y_true'].value_counts()
    top_classes = class_counts.head(10).index.tolist()

    filtered_df = conf_df[conf_df['y_true'].isin(
        top_classes) & conf_df['y_pred'].isin(top_classes)]

    cm = confusion_matrix(
        filtered_df['y_true'],
        filtered_df['y_pred'],
        labels=top_classes
    )

    cm_df = pd.DataFrame(cm, index=top_classes, columns=top_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Matriz de Confusão - {model} (Classes Mais Frequentes)")
    plt.show()
