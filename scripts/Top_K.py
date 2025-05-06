import numpy as np
import faiss

def evaluate_model(model, X_val, y_val, top_k_list=[1, 5, 10]):
    print("Évaluation du modèle...")

    # Projeter les questions avec le modèle
    q_proj = model.predict(X_val, batch_size=32)
    
    # Normaliser les vecteurs
    q_proj = q_proj.astype('float32')
    y_val = y_val.numpy().astype('float32')
    
    q_proj = q_proj / np.linalg.norm(q_proj, axis=1, keepdims=True)
    y_val = y_val / np.linalg.norm(y_val, axis=1, keepdims=True)
    
    # Construire l'index FAISS des réponses
    index = faiss.IndexFlatIP(768)
    index.add(y_val)

    # Chercher top-k réponses pour chaque question projetée
    topk = max(top_k_list)
    _, indices = index.search(q_proj, topk)
    
    # Pour chaque question, vérifier si la bonne réponse est dans le top-k
    topk_accuracies = {}
    for k in top_k_list:
        correct = 0
        for i in range(len(X_val)):
            if i in indices[i][:k]:
                correct += 1
        topk_accuracies[f"top_{k}"] = round(100 * correct / len(X_val), 2)
    
    print("Résultats top-k:")
    for k, v in topk_accuracies.items():
        print(f"{k} accuracy: {v}%") # si  top-10 est au-dessus de 80 %, le modèle est utilisable pour le chatbot médical.

# Appeler la fonction d'évaluation
evaluate_model(model, X_val, y_val)
