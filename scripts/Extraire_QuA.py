import os
import xml.etree.ElementTree as ET
import pandas as pd

data = []

folder_path = '../MedQuAD'  # Assure-toi que ce chemin est correct

for root_dir, _, files in os.walk(folder_path):  # Explore tous les sous-dossiers
    for filename in files:
        if filename.endswith('.xml'):
            file_path = os.path.join(root_dir, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            question = root.findtext('.//Question')  # Recherche récursive
            answer = root.findtext('.//Answer')

            if question and answer:
                data.append({'question': question.strip(), 'answer': answer.strip()})

# Convertir en DataFrame
df = pd.DataFrame(data)

# Sauvegarder en CSV ou pickle
df.to_pickle('medquad_qa.pkl')

print(f"{len(df)} questions-réponses extraites et sauvegardées.")
