from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openAI_api = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openAI_api)

def embed(text: str)-> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding

if __name__ == "__main__":
    result = embed("Nike sports deal")
    print(f"Length: {len(result)}")