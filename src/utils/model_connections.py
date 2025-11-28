from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
 
class OpenAIConnection:
    def __init__(self, use_batch=False):
        load_dotenv()

        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = "2024-10-01-preview"

        self.model = os.getenv("OPENAI_MODEL_BATCH") if use_batch else os.getenv("OPENAI_MODEL")

        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )