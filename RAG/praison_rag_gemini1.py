
import os
from dotenv import load_dotenv
env = load_dotenv(override=True)
print(env)
print(os.getenv("GEMINI_API_KEY"))

from praisonaiagents import Agent
config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "praison",
            "path": ".praison"
        }
    },
    # "llm": {
    #     "provider": "ollama",
    #     "config": {
    #         "model": "deepseek-r1:14b",
    #         "temperature": 0,
    #         "max_tokens": 8000,
    #         "ollama_base_url": "http://localhost:11434",
    #     },
    # },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": "http://localhost:11434",
            "embedding_dims": 1536
        },
    },
}
llm_config = {
  "model": "gemini/gemini-2.0-flash-thinking-exp-01-21",
  "response_format": {"type": "text"} # type is text, because json_object is not supported
}

agent = Agent(
    name="Knowledge Agent",
    instructions="You answer questions based on the provided knowledge.",
    knowledge=["elia_big.pdf"], # kag-research-paper.pdf
    knowledge_config=config,
    user_id="user1",
    llm=llm_config
)

agent.start("Текстът, който ще анализираш е на български език. "
"Отговори на българския език с 2-3 реда как еколозите могат да бъдат полезни в случая.") # Retrievalsw 