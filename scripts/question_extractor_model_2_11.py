# Entraîner un petit réseau de neurones (MLP) qui apprend à projeter les questions (Q_FFNN_embeds) dans le même espace que les réponses (A_FFNN_embeds) :
# Entrée : Q_FFNN_embeds (dim 768)
# Sortie : A_FFNN_embeds (dim 768)
# Objectif : que ce vecteur soit le plus proche possible de A_FFNN_embeds de la bonne réponse


import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# Charger les données
df = pd.read_pickle("train_gpt_data.pkl")
X = tf.convert_to_tensor(df["Q_FFNN_embeds"].tolist(), dtype=tf.float32)
y = tf.convert_to_tensor(df["A_FFNN_embeds"].tolist(), dtype=tf.float32)

# Split en train/test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

# Définir le modèle (MLP simple)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(768,)),
    tf.keras.layers.Dense(768, activation='relu'),
    tf.keras.layers.Dense(768)  # sortie finale
])

# Perte = distance cosinus (à minimiser)
def cosine_distance_loss(y_true, y_pred):
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    return 1 - tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=1))

# Compilation
model.compile(optimizer='adam', loss=cosine_distance_loss)

# Entraînement
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Sauvegarder le modèle
model.save("question_extractor_model_2_11")
print("Modèle sauvegardé dans 'question_extractor_model_2_11'")
