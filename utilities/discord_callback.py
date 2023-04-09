import json
import requests
from tensorflow.keras.callbacks import Callback


class DiscordCallback(Callback):
    """
        A custom Keras Callback to send training progress updates to a Discord channel
        using a webhook URL.

        Usage:
            discord_callback = DiscordCallback(webhook_url)
            model.fit(..., callbacks=[discord_callback])
        """
    def __init__(self, webhook_url):
        """
        Initializes the DiscordCallback class.

        Args:
            webhook_url (str): The webhook URL for the Discord channel to send updates to.
        """
        super().__init__()
        self.webhook_url = webhook_url

    def on_epoch_end(self, epoch, logs=None):
        """
        This method is called at the end of each training epoch. It sends a message
        to the Discord channel containing the training progress for that epoch.

        Args:
            epoch (int): The current epoch number (0-indexed).
            logs (dict, optional): A dictionary containing the training metrics for this epoch.
        """
        logs = logs or {}
        # Format the message to include epoch number and training metrics
        message = f"Epoch {epoch + 1} - "
        message += ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])

        # Prepare the data to be sent as a JSON object
        data = {"content": message}

        # Send the message to the Discord channel using the webhook URL
        requests.post(self.webhook_url, data=json.dumps(data), headers={"Content-Type": "application/json"})