# train_model.py
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# 1) Charge ton dataset
# Remplace 'data.csv' et 'churn' par tes vrais chemins/nom de cible (0=non résilié, 1=résilié)
df = pd.read_csv("C:/Users/amira/Downloads/datas/testappfinale.csv")

# 2) Aligne les features sur l'app Flask
FEATURES = [
    "MONTANT_PRIME_TOTALE_VALUE",
    "CODE_SEXE",
    "STATUT_MATRIMONIAL",
    "NATURE_DOSSIER_CONTRAT",
    "PRIME_ANORMALE",
    "ANCIENNETE",
    "PACK",
]
TARGET = "churn"  # <-- adapte selon ta colonne cible

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

# 3) Définis les types
num_cols = ["MONTANT_PRIME_TOTALE_VALUE", "ANCIENNETE"]
cat_cols = ["CODE_SEXE", "STATUT_MATRIMONIAL", "NATURE_DOSSIER_CONTRAT", "PRIME_ANORMALE", "PACK"]

# 4) Préprocesseur + modèle
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("rf", clf),
])

# 5) Split + fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe.fit(X_train, y_train)

# 6) Sauvegarde au bon endroit (là où app.py le cherche)
Path("models").mkdir(exist_ok=True)
joblib.dump(pipe, "models/random_forest_model.pkl")
print("✅ Modèle pipeline sauvegardé -> models/random_forest_model.pkl")
