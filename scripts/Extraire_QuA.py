import os
import xml.etree.ElementTree as ET
import pandas as pd

data = []


folder_path = '../MedQuAD'

for filename in os.listdir(folder_path):
    if filename.endswith('.xml'):
        file_path = os.path.join(folder_path, filename)
        tree = ET.parse(file_path)
        root = tree.getroot()

        question = root.findtext('Question')
        answer = root.findtext('Answer')

        if question and answer:
            data.append({'question': question.strip(), 'answer': answer.strip()})

# Convertir en DataFrame
df = pd.DataFrame(data)

# Sauvegarder en CSV ou pickle
df.to_pickle('medquad_qa.pkl')

print(f"{len(df)} questions-réponses extraites et sauvegardées.")
