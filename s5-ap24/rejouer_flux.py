import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL, PHASE = sys.argv[1].rstrip("/"), sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 400
DEBIT = float(sys.argv[4]) if len(sys.argv) > 4 else 0
LEVIER = sys.argv[5] if len(sys.argv) > 5 else "eclairage"
DONNEES = Path(os.environ.get("DONNEES", "donnees"))
MANIFESTE = Path(os.environ.get("MANIFESTE", "model/manifeste_echantillon.csv"))
COLONNES = ["age_approx", "sex", "anatom_site_general", "tbp_tile_type"]

LEVIERS = {
    "eclairage": lambda d: d.tbp_tile_type == "3D: XP",
    "age": lambda d: pd.to_numeric(d.age_approx, errors="coerce") >= 70,
    "site": lambda d: d.anatom_site_general == "head/neck"
}
assert PHASE in ("conforme", "derive") and LEVIER in LEVIERS, __doc__

meta_files = [f for f in os.listdir(DONNEES) if f.endswith('.csv') and 'metadata' in f]
csv_path = DONNEES / meta_files[0] if meta_files else DONNEES / "train-metadata.csv"

archive = pd.read_csv(csv_path, usecols=["isic_id"], low_memory=False)
meta = pd.read_csv(csv_path, usecols=["isic_id", "patient_id"] + COLONNES, low_memory=False)
d = archive.merge(meta, on="isic_id")

if MANIFESTE.exists():
    d = d[~d.isic_id.isin(pd.read_csv(MANIFESTE).isic_id)]

if PHASE == "derive":
    d = d[LEVIERS[LEVIER](d)]

graine = {"conforme": 1, "derive": 2}[PHASE]
flux = d.sample(n=min(N, len(d)), random_state=graine)

codes, alertes = {}, 0
zip_path = DONNEES / "ISIC_2024_Permissive_Training_GroundTruth.zip"

with zipfile.ZipFile(zip_path) as z, requests.Session() as s:
    # Indexation des chemins dans le zip pour un accès direct instantané et robuste
    print("Indexation de l'archive ZIP en cours...")
    namelist = z.namelist()
    image_map = {os.path.basename(name): name for name in namelist if name.endswith('.jpg')}
    print(f"{len(image_map)} images indexées dans l'archive.")

    for k, ligne in enumerate(flux.itertuples(index=False), 1):
        debut = time.monotonic()
        champs = {c: str(getattr(ligne, c)) for c in COLONNES if pd.notna(getattr(ligne, c))}
        
        img_filename = f"{ligne.isic_id}.jpg"
        if img_filename not in image_map:
            # Si l'image n'est pas trouvée, on passe à la suivante
            continue
            
        internal_path = image_map[img_filename]
        try:
            r = s.post(f"{URL}/predict", data=champs, timeout=60,
                       files={"image": (img_filename, z.read(internal_path), "image/jpeg")},
                       headers={"X-Request-Id": f"{PHASE}-{LEVIER if PHASE == 'derive' else 'flux'}-{k:05d}",
                                "X-Patient-Ref": ligne.patient_id})
        except requests.RequestException as exc:
            codes[type(exc).__name__] = codes.get(type(exc).__name__, 0) + 1
            continue
        codes[r.status_code] = codes.get(r.status_code, 0) + 1
        alertes += r.ok and r.json().get("decision") == "a_examiner"
        if DEBIT:
            time.sleep(max(0.0, 1 / DEBIT - (time.monotonic() - debut)))

print(f"{PHASE} ({LEVIER if PHASE == 'derive' else 'archive'}) : {len(flux)} appels, codes {codes}, "
      f"a_examiner {alertes} ({alertes / max(codes.get(200, 0), 1):.1%})")