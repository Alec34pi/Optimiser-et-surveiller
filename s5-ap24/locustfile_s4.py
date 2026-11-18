import os
from locust import HttpUser, task, between

IMAGE_PATH = os.environ.get("IMAGE_CONTROLE_S4", "controle.jpg")
with open(IMAGE_PATH, "rb") as f:
    IMAGE_BYTES = f.read()

class DermaScanUser(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def get_health(self):
        self.client.get("/health", name="GET /health")

    @task(5)
    def predire(self):
        with self.client.post(
            "/predict",
            files={"image": ("controle.jpg", IMAGE_BYTES, "image/jpeg")},
            name="POST /predict",
            catch_response=True,
        ) as reponse:
            if reponse.status_code == 200:
                reponse.success()
            else:
                reponse.failure(f"Statut inattendu : {reponse.status_code}")