from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # --- LLM & IA ---
    LM_STUDIO_URL: str = "http://localhost:1234/v1"
    LLM_MODEL_NAME: str = "qwen/qwen3-4b-2507"
    # Agregamos esta para el ETL
    OLLAMA_BASE_URL: str = "http://ollama-service:11434/v1"

    # --- Metabase ---
    METABASE_URL_FOR_DOCKER: str = "http://host.docker.internal:32768"
    METABASE_USERNAME: str = "admin@prisma.com"
    METABASE_PASSWORD: str = "password"

    # --- Neo4j (LAS QUE FALTABAN) ---
    NEO4J_URI: str = "bolt://neo4j-db:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "prismafinance123"

    class Config:
        env_file = ".env"
        # Esto permite que haya variables extra en el .env sin dar error
        extra = "ignore" 

settings = Settings()