import json
import math
import os
import socket
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import torch
from torchvision.models import resnet18
from PIL import Image

MODEL_FILE = os.getenv("MODEL_PATH", "model/modele.pt")
SVC_NAME = os.getenv("SERVICE_NAME", "isic-vignettes-api")
LISTENING_PORT = int(os.getenv("PORT", 8000))
START_TIMESTAMP = time.time()

app = FastAPI()

def _load_inference_engine():
    try:
        path_obj = Path(MODEL_FILE)
        meta_data = json.loads(path_obj.with_name("modele_meta.json").read_text())
        
        net = resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, 1)
        net.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True))
        net.eval()
        
        threshold = float(os.getenv("SEUIL", meta_data["seuil"]))
        mean_tensor = torch.tensor(meta_data["normalisation"]["moyenne"]).view(3, 1, 1)
        std_tensor = torch.tensor(meta_data["normalisation"]["ecart_type"]).view(3, 1, 1)
        
        return net, meta_data, threshold, mean_tensor, std_tensor, None
    except Exception as err:
        error_msg = f"{type(err).__name__}: {err}"
        return None, None, None, None, None, error_msg

NET, META, THRESHOLD, MEAN, STD, LOAD_ERROR = _load_inference_engine()


@app.get("/health")
def health():
    is_ready = NET is not None
    payload = {
        "status": "ok" if is_ready else "degraded",
        "model_loaded": is_ready,
        "service": SVC_NAME,
        "hostname": socket.gethostname(),
        "uptime_seconds": round(time.time() - START_TIMESTAMP, 1)
    }
    if not is_ready:
        payload["error"] = LOAD_ERROR
    
    # En FastAPI, on lève une HTTPException pour renvoyer un code 503 propre
    if not is_ready:
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.get("/model/info")
def info():
    if NET is None:
        raise HTTPException(status_code=503, detail={"error": "modele non charge"})
        
    return {
        "version": META["version"],
        "version_deployee": os.getenv("MODEL_VERSION"),
        "seuil": THRESHOLD,
        "rappel_cible": META["rappel_cible"],
        "metriques": META["metriques"],
        "prevalence_entrainement": META["prevalence_entrainement"],
        "prevalence_reelle": META["prevalence_reelle"],
        "architecture": META["architecture"],
        "taille_entree": META["taille_entree"],
        "echantillonnage": META["echantillonnage"],
        "sha256_poids": META["sha256_poids"],
        "entraine_le": META["entraine_le"],
    }


@app.post("/predict")
async def predict(image: UploadFile = File(None)):
    if NET is None:
        raise HTTPException(status_code=503, detail={"error": "modele non charge"})

    if image is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "champ multipart 'image' attendu",
                "exemple": "curl -F image=@ISIC_xxx.jpg .../predict"
            }
        )

    try:
        t_size = META["taille_entree"]
        contents = await image.read()
        import io
        img = Image.open(io.BytesIO(contents)).convert("RGB").resize((t_size, t_size), Image.BILINEAR)
        tensor_img = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float().div(255)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "image illisible",
                "detail": f"{type(exc).__name__}: {exc}"
            }
        )

    with torch.no_grad():
        normalized_input = (tensor_img - MEAN) / STD
        logit_val = NET(normalized_input.unsqueeze(0)).item()

    probability = 1.0 / (1.0 + math.exp(-(logit_val + META["correction_logit"])))

    return {
        "probabilite_maligne": round(probability, 6),
        "seuil": THRESHOLD,
        "decision": "a_examiner" if probability >= THRESHOLD else "surveillance",
        "version_modele": META["version"],
        "rappel_cible": META["rappel_cible"],
    }