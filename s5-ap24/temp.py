import pandas as pd

print("1. Chargement optimisé des métadonnées...")
df_meta = pd.read_csv("donnees/train-metadata.csv", low_memory=False)

# Identifier le nom exact de la colonne cible/target
col_cible = "target" if "target" in df_meta.columns else ("cible" if "cible" in df_meta.columns else None)

print("2. Chargement du manifeste de l'échantillon...")
man_ids = pd.read_csv("model/manifeste_echantillon.csv")["isic_id"]
man = df_meta[df_meta["isic_id"].isin(man_ids)]

# Définition des différentes populations
populations = [
    ("Manifeste (votre échantillon)", man),
    ("Positifs de l'archive Permissive", df_meta[df_meta[col_cible] == 1] if col_cible else pd.DataFrame()),
    ("Archive Permissive entière", df_meta),
    ("train-metadata.csv complet (S2)", df_meta)
]

print("\n" + "="*71)
print(f"{'Population':<35} | {'3D: white (%)':<15} | {'3D: XP (%)':<15}")
print("="*71)

for nom, d in populations:
    if not d.empty and "tbp_tile_type" in d.columns:
        props = (d.tbp_tile_type.value_counts(normalize=True) * 100).round(2).to_dict()
        white = props.get("3D: white", 0.0)
        xp = props.get("3D: XP", 0.0)
        print(f"{nom:<35} | {white:<15} | {xp:<15}")
    else:
        print(f"{nom:<35} | Données indisponibles")
print("="*71)