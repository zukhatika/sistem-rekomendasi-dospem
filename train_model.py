import pandas as pd
import pickle
import torch
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("Load dataset...")
judul = pd.read_excel("data_judul_skiripsi.xlsx")
dosen = pd.read_excel("data dosen.xlsx")

judul = judul[["Judul Skiripsi", "Nama Dosen"]]
data = pd.merge(judul, dosen, on="Nama Dosen")

texts = data["Judul Skiripsi"].astype(str).tolist()
labels = data["Nama Dosen"].tolist()

print("Load IndoBERT...")
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
bert = BertModel.from_pretrained("indobenchmark/indobert-base-p1")

def embed(texts):
    emb = []
    for t in texts:
        inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            out = bert(**inputs)
        emb.append(out.last_hidden_state[:,0,:].numpy()[0])
    return emb

print("Embedding judul skripsi...")
X = embed(texts)

le = LabelEncoder()
y = le.fit_transform(labels)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X, y)

pickle.dump(rf, open("model_rf.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("MODEL BERHASIL DIBUAT!")
