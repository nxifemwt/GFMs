import os

import numpy as np
import pandas as pd
import torch
import wandb
import xgboost as xgb
from Bio.Seq import Seq
from pyfaidx import Fasta
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import autocast
from tqdm import tqdm
from utils import load_model_and_tokenizer

biotypes_of_interest = [
    "protein_coding",
    "lncRNA",
    "processed_pseudogene",
    "unprocessed_pseudogene",
    "snRNA",
    "miRNA",
    "TEC",
    "snoRNA",
    "misc_RNA",
]

hg38 = Fasta("Homo_sapiens.GRCh38.dna.primary_assembly.fa")

def extract_gene_sequences(gencode_file, hg38):
    genes = []
    with open(gencode_file, "r") as f:
        from tqdm import tqdm

        for line in tqdm(f):
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if fields[2] == "gene":
                attributes = dict(item.split("=") for item in fields[8].split(";"))
                gene_type = attributes.get("gene_type", "")
                if gene_type not in biotypes_of_interest:
                    continue
                chrom, start, end, strand = (
                    fields[0],
                    int(fields[3]),
                    int(fields[4]),
                    fields[6],
                )
                chrom = chrom.replace("chr", "")
                if chrom == "M":
                    chrom = "MT"
                try:
                    sequence = str(hg38[chrom][start - 1 : end].seq)
                    if strand == "-":
                        sequence = str(Seq(sequence).reverse_complement())
                    genes.append({"sequence": sequence, "gene_type": gene_type})
                except KeyError:
                    print(
                        f"Warning: Chromosome {chrom} not found in reference genome. Skipping."
                    )
    return pd.DataFrame(genes)


@torch.no_grad()
def generate_embedding(model, tokenizer, genes_df, feature_layer=-1):
    embeddings_list = []
    for seq in tqdm(genes_df["sequence"], desc="Generating embeddings"):
        tokens = tokenizer.tokenize(seq)
        chunks = [tokens[i : i + max_length] for i in range(0, len(tokens), max_length)]

        chunk_embeddings = []
        for chunk in chunks:
            input_ids = tokenizer.convert_tokens_to_ids(chunk)
            input_ids = torch.tensor([input_ids]).cuda()

            if model_name == "nt_500m" or model_name == "nt_50m":
                attention_mask = input_ids != tokenizer.pad_token_id
                with autocast():
                    output = model(
                        input_ids,
                        attention_mask=attention_mask,
                        encoder_attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                output = output["hidden_states"][feature_layer].cpu()
            elif model_name == "hyenadna" or model_name == "caduceus":
                output = (
                    model(input_ids, output_hidden_states=True)
                    .hidden_states[feature_layer]
                    .detach()
                    .cpu()
                )
            elif model_name == "dnabertv2":
                with autocast():
                    output = model(input_ids)[1].detach().cpu()
            elif model_name == "mistral":
                with autocast():
                    output = (
                        model(input_ids, output_hidden_states=True)
                        .hidden_states[feature_layer]
                        .detach()
                        .cpu()
                    )
            elif "genalm" == model_name:
                with autocast():
                    output = model(input_ids).hidden_states[feature_layer]
            chunk_embedding = torch.max(output, dim=1)[0].squeeze().cpu().numpy()
            chunk_embeddings.append(chunk_embedding)

        sequence_embedding = np.mean(chunk_embeddings, axis=0)
        embeddings_list.append(sequence_embedding)

    return embeddings_list


if __name__ == "__main__":
    print(f"Starting main_biotype_gencode.py")
    use_wandb = "WANDB_SWEEP_ID" in os.environ

    if use_wandb:
        run = wandb.init()
        model_name = wandb.config.model_name
        pretrained = wandb.config.pretrained
        tokenizer_type = wandb.config.tokenizer
        embedding_dim = wandb.config.embedding_dim
    else:
        run = None
        model_name = "caduceus"  
        pretrained = False
        tokenizer_type = "char"
        embedding_dim = 4096
        print(
            f"Model Name: {model_name}, Pretrained: {pretrained}, Tokenizer: {tokenizer_type}"
        )

    model, tokenizer, max_length = load_model_and_tokenizer(
        model_name, pretrained, tokenizer_type, embedding_dim
    )

    genes_df = pd.read_csv("./capped_genes_df.csv")
    print(f"Number of samples: {len(genes_df)}")

    embedding_list = generate_embedding(model, tokenizer, genes_df)

    genes_df["embeddings"] = embedding_list

    X = np.stack(genes_df["embeddings"].values)
    np.save(
        f"./embeddings/{model_name}_{'pretrained' if pretrained else 'random'}_{tokenizer_type}.npy",
        X,
    )
    le = LabelEncoder()
    y = le.fit_transform(genes_df["gene_type"])

    print("Unique classes:", le.classes_)
    print("Number of unique classes:", len(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "objective": "multi:softmax",
        "num_class": len(le.classes_),
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 1000,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "verbosity": 2,
        "device": "gpu",
    }
    xgb_model = xgb.XGBClassifier(**params, use_label_encoder=False)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Weighted F1 Score: {f1}")
    if use_wandb:
        run.log({"f1_score": f1})
        run.finish()
