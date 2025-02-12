
import os
from dotenv import load_dotenv
env = load_dotenv(override=True)
print(env)
print(os.getenv("OPENAI_API_KEY"))

from praisonaiagents import Agent
config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "praison",
            "path": ".praison"
        }
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "deepseek-r1:14b",
            "temperature": 0,
            "max_tokens": 8000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": "http://localhost:11434",
            "embedding_dims": 1536
        },
    },
}

agent = Agent(
    name="Knowledge Agent",
    instructions="You answer questions based on the provided knowledge.",
    knowledge=["elia_big.pdf"], # kag-research-paper.pdf
    knowledge_config=config,
    user_id="user1",
    llm="deepseek-r1:14b"
)

agent.start("Текстът, който ще анализираш е на български език. "
"Отговори на българския език с 2-3 реда как еколозите могат да бъдат полезни в случая.") # Retrieval