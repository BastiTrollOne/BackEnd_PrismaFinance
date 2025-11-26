import os
import re
import csv
import logging
import requests
from io import StringIO
from typing import Optional

# --- IMPORTS LANGCHAIN ---
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph

# --- CONFIGURACIÓN CENTRALIZADA ---
from app.core.config import settings

# Logger
logger = logging.getLogger("GraphETL")

# Configuración derivada de settings o variables de entorno
NEO4J_URI = settings.NEO4J_URI
NEO4J_USERNAME = settings.NEO4J_USERNAME
NEO4J_PASSWORD = settings.NEO4J_PASSWORD

# URL interna del servicio de ingesta (Docker network)
INGESTION_URL = os.getenv("INGESTION_URL", "http://ingestion-service:8000/upload")

# URL de Ollama (Backend)
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
MODEL_NAME = "qwen2.5:3b"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ==============================================================================
# 1. ETL CON LIMPIEZA (Lógica original preservada)
# ==============================================================================
def clean_text_content(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def csv_to_narrative(text_chunk):
    """Transforma CSV a narrativa ignorando líneas decorativas (Lógica Original)."""
    if text_chunk is None: return ""
    
    # 1. LIMPIEZA PREVIA
    lines = text_chunk.strip().split('\n')
    clean_lines = [line for line in lines if not line.startswith("===")]
    cleaned_text = "\n".join(clean_lines)

    if not cleaned_text: return text_chunk

    # 2. Detección rápida
    if "ID" not in cleaned_text and "Monto" not in cleaned_text:
        if "," not in cleaned_text and ";" not in cleaned_text:
            return text_chunk

    try:
        # 3. Separador
        first_line = cleaned_text.strip().split('\n')[0]
        delimiter = ";" if ";" in first_line and "," not in first_line else ","
        
        f = StringIO(cleaned_text)
        reader = csv.DictReader(f, delimiter=delimiter)
        
        # 4. Validación Cabeceras
        if not reader.fieldnames: 
            return text_chunk
            
        headers = [str(h).lower() for h in reader.fieldnames]
        
        # Filtro permisivo financiero
        if not any(x in str(headers) for x in ['monto', 'valor', 'costo', 'presupuesto', 'usd', 'precio', 'id']):
            return text_chunk

        logger.info(f"📊 [ETL] Generando narrativa financiera para chunk...")
        narrative = []
        
        for row in reader:
            row_norm = {str(k).lower().strip(): str(v).strip() for k, v in row.items() if k and v}
            
            def get_val(patterns):
                for p in patterns:
                    for k, v in row_norm.items():
                        if p in k: return v
                return "Desconocido"

            concepto = get_val(['concepto', 'glosa', 'proyecto', 'nombre', 'item', 'ítem'])
            categoria = get_val(['categor', 'tipo'])
            org = get_val(['organi', 'fuente', 'proveedor', 'empresa', 'responsable'])
            persona = get_val(['persona', 'sponsor', 'responsable', 'encargado'])
            monto_raw = get_val(['monto', 'valor', 'costo', 'usd'])
            
            monto = re.sub(r'[^\d]', '', monto_raw)
            
            if monto:
                oracion = f"Registro: La entidad '{org}' (Representante: {persona}) tiene una relación '{categoria}' con el proyecto '{concepto}' por un monto de {monto}."
                narrative.append(oracion)
            
        if narrative:
            return "\n".join(narrative)
            
    except Exception as e:
        logger.error(f"❌ DEBUG Parsing: {e}")
    
    return text_chunk

def connect_to_neo4j():
    try:
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        # graph.refresh_schema() # Opcional si es muy lento
        return graph
    except Exception as e:
        logger.error(f"❌ Error Neo4j: {e}")
        return None

# ==============================================================================
# 2. UNIFICACIÓN (Lógica original preservada)
# ==============================================================================
def unify_entities(graph):
    logger.info("🚀 [UNIFICACIÓN] Conectando nodos...")
    try:
        # Limpieza básica
        graph.query("MATCH (n) WHERE size(n.id) < 2 OR n.id CONTAINS 'copyright' DETACH DELETE n")
        
        # Fusión de nodos duplicados por nombre
        graph.query("""
        MATCH (n) WITH toLower(n.id) as name, collect(n) as nodes 
        WHERE size(nodes) > 1 
        CALL apoc.refactor.mergeNodes(nodes, {properties:'combine', mergeRels:true}) 
        YIELD node RETURN count(node)
        """)
        
        # Inferencia Contextual de Relaciones
        graph.query("MATCH (p:Persona)<-[:MENTIONS]-(d)-[:MENTIONS]->(o:Organizacion) MERGE (p)-[:PERTENECE_A]->(o)")
        graph.query("MATCH (o:Organizacion)<-[:MENTIONS]-(d)-[:MENTIONS]->(m) WHERE labels(m) IN [['Monto'],['Costo'],['Presupuesto']] MERGE (o)-[:TIENE_PRESUPUESTO]->(m)")
        
        logger.info("✨ Grafo optimizado.")
    except Exception as e:
        logger.warning(f"⚠️ Aviso unificación: {e}")

# ==============================================================================
# 3. PROCESO PRINCIPAL (Adaptado para ser llamado desde Main)
# ==============================================================================
def run_graph_extraction(file_path: str):
    """
    Función maestra que orquesta:
    Ingesta (API) -> Limpieza (CSV) -> Extracción (LLM) -> Persistencia (Neo4j) -> Unificación
    """
    logger.info(f"🎬 Iniciando ETL Graph para: {file_path}")
    
    # 1. Llamar al servicio de Ingesta (Musculo OCR/Parsing)
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                INGESTION_URL, 
                files={'file': f}, 
                params={"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
                timeout=120
            )
        if response.status_code != 200:
            logger.error(f"❌ Error API Ingesta: {response.text}")
            return
        raw_data = response.json()
    except Exception as e:
        logger.error(f"❌ Excepción conectando a Ingesta: {e}")
        return

    # 2. Convertir respuesta JSON a Documentos LangChain con limpieza
    def safe_doc(c, m): return Document(page_content=str(c) if c else "", metadata=m or {})
    
    lc_docs = []
    if "documents" in raw_data:
        for d in raw_data["documents"]:
            if isinstance(d, dict):
                # APLICAMOS TU MAGIA: csv_to_narrative
                clean_text = csv_to_narrative(d.get("page_content"))
                lc_docs.append(safe_doc(clean_text, d.get("metadata")))
    elif "page_content" in raw_data:
        # Caso single doc
        clean_text = csv_to_narrative(raw_data.get("page_content"))
        lc_docs.append(safe_doc(clean_text, raw_data.get("metadata")))

    if not lc_docs:
        logger.warning("⚠️ No se obtuvieron documentos válidos de la ingesta.")
        return

    logger.info(f"📄 Extrayendo grafo de {len(lc_docs)} fragmentos con LLM...")

    # 3. Extracción de Grafo con LLM Local
    try:
        # Usamos base_url directo para Ollama
        llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)
        
        transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=["Organizacion", "Persona", "Proyecto", "Monto", "Concepto", "Costo", "Presupuesto"],
            allowed_relationships=["FINANCIA", "DIRIGE", "TIENE_COSTO", "PERTENECE_A", "MENTIONS"],
            node_properties=False # Simplificado para velocidad
        )
    except Exception as e:
        logger.error(f"❌ Error inicializando LLM: {e}")
        return

    graph = connect_to_neo4j()
    if not graph: return

    # 4. Procesamiento por lotes
    success_count = 0
    for i, doc in enumerate(lc_docs):
        try:
            res = transformer.convert_to_graph_documents([doc])
            if res:
                graph.add_graph_documents(res, baseEntityLabel=True, include_source=True)
                success_count += 1
                logger.info(f"   ✅ Chunk {i+1}/{len(lc_docs)} procesado.")
            else:
                logger.debug(f"   ⚠️ Chunk {i+1} sin entidades relevantes.")
        except Exception as e:
            logger.error(f"   ❌ Error en Chunk {i+1}: {e}")

    # 5. Unificación Final
    if success_count > 0:
        unify_entities(graph)
        logger.info(f"🎉 Proceso FINALIZADO para {file_path}")
    else:
        logger.error("💀 No se guardaron datos en el grafo.")