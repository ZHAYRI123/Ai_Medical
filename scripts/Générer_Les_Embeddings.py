import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from tqdm import tqdm

# Paramètres
data_file = "medquad_qa.pkl"
checkpoint_file = "last_index.txt"
output_file = "train_gpt_data.pkl"
batch_size = 8  # ajuster selon la RAM dispo
save_every_batches = 50  # Sauvegarde tous les 50 batches

# Charger les données
df = pd.read_pickle(data_file)

# Initialiser les colonnes si elles n'existent pas
if "Q_FFNN_embeds" not in df.columns:
    df["Q_FFNN_embeds"] = None
if "A_FFNN_embeds" not in df.columns:
    df["A_FFNN_embeds"] = None

# Reprise à partir du checkpoint
start_index = 0
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        start_index = int(f.read())

print(f"Reprise depuis l'index : {start_index}")

# Chargement du modèle
tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/BioRedditBERT-uncased")
model = TFAutoModel.from_pretrained("cambridgeltl/BioRedditBERT-uncased")

# Fonction pour batcher les embeddings
def get_embeddings(text_list):
    inputs = tokenizer(text_list, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings.numpy()

# Traitement par batch
num_rows = len(df)
batches = list(range(start_index, num_rows, batch_size))

for batch_num, start in tqdm(enumerate(batches), total=len(batches), desc="Processing batches"):
    end = min(start + batch_size, num_rows)
    rows = df.iloc[start:end]

    try:
        questions = rows["question"].astype(str).tolist()
        answers = rows["answer"].astype(str).tolist()

        q_embeds = get_embeddings(questions)
        a_embeds = get_embeddings(answers)

        for i, idx in enumerate(range(start, end)):
            df.at[idx, "Q_FFNN_embeds"] = q_embeds[i]
            df.at[idx, "A_FFNN_embeds"] = a_embeds[i]

        # Sauvegarde de l’index actuel
        with open(checkpoint_file, "w") as f:
            f.write(str(end))

        # Sauvegarde partielle
        if batch_num % save_every_batches == 0:
            df.to_pickle(output_file)
            print(f"Sauvegarde intermédiaire à l'index {end}")

    except Exception as e:
        print(f"Erreur entre les index {start} et {end} : {e}")

# Sauvegarde finale
df.to_pickle(output_file)
print("Embeddings terminés et enregistrés dans", output_file)
