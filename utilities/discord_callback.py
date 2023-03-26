import json
import requests
from tensorflow.keras.callbacks import Callback


class DiscordCallback(Callback):
    def __init__(self, webhook_url):
        super().__init__()
        self.webhook_url = webhook_url

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        message = f"Epoch {epoch + 1} - "
        message += ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        data = {"content": message}
        requests.post(self.webhook_url, data=json.dumps(data), headers={"Content-Type": "application/json"})