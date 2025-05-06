import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from tqdm import tqdm

# Charger les données
df = pd.read_pickle('medquad_qa.pkl')

# Charger BioBERT
tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/BioRedditBERT-uncased")
model = TFAutoModel.from_pretrained("cambridgeltl/BioRedditBERT-uncased")

# Fonction pour générer les embeddings BERT
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="tf", max_length=512, truncation=True, padding="max_length")
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    return cls_embedding.numpy().flatten()

# Appliquer BERT
question_embeddings = []
answer_embeddings = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    q_emb = get_embedding(str(row['question']))
    a_emb = get_embedding(str(row['answer']))
    question_embeddings.append(q_emb)
    answer_embeddings.append(a_emb)

df["Q_FFNN_embeds"] = question_embeddings
df["A_FFNN_embeds"] = answer_embeddings

# Sauvegarder le fichier final
df.to_pickle("train_gpt_data.pkl")
print("Embeddings BERT enregistrés dans train_gpt_data.pkl")
