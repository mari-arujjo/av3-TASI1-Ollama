import requests
from transformers import AutoTokenizer
import json

OLLAMA_HOST = "http://localhost:11434"
MODELO = "nomic-embed-text"
TEXTO_EMBED = "Boa noite, tudo bem? Me chamo Mariana e sou estudante do IFRN."

# método pra gerar os embeddings
def GerarEmbedding(texto, modelo):
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": modelo, "prompt": texto}
        )
        response.raise_for_status()
        return response.json()
    
    except Exception as e:
        print(f"Erro ao gerar embeddings: {e}")
        return 

# gerando os tokens! adicionei isso para testare ficar completinho
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
tokens = tokenizer.tokenize(TEXTO_EMBED)

# executando!!
if __name__ == "__main__":
    embedding_data = GerarEmbedding(TEXTO_EMBED, MODELO)
    
    if embedding_data:
        print("==="*50)
        print("Embedding gerado!")
        print("---"*50)
        print(f"MODELO: {MODELO}")
        print(f"PROMPT: '{TEXTO_EMBED}'")
        print(f"- Tokens: {tokens}")
        print(f"- Tamanho do vetor: {len(embedding_data['embedding'])}")
        print(f"- Primeiros 6 valores do vetor: {embedding_data['embedding'][:6]}")
    else:
        print("==="*50)
        print("Erro ao gerar embeddings =()")