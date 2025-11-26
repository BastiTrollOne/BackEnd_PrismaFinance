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
# Usamos settings para mantener la coherencia con el resto de la API
from app.core.config import settings

logger = logging.getLogger("GraphETL")

# Configuración derivada de settings
NEO4J_URI = settings.NEO4J_URI
NEO4J_USERNAME = settings.NEO4J_USERNAME
NEO4J_PASSWORD = settings.NEO4J_PASSWORD

# URL interna del servicio de ingesta (Docker network)
# Nota: Ajustamos la ruta para que coincida con tu estructura de endpoints
INGESTION_URL = os.getenv("INGESTION_URL", "http://ingestion-service:8000/upload")

# URL de Ollama (Backend)
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
# Tu modelo preferido
MODEL_NAME = "qwen2.5:3b"

# Chunking optimizado
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ==============================================================================
# 1. ETL CON LIMPIEZA (TU LÓGICA ACTUALIZADA)
# ==============================================================================
def clean_text_content(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def csv_to_narrative(text_chunk):
    """Transforma CSV a narrativa ignorando líneas decorativas."""
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
        logger.debug(f"   🔎 DEBUG: Separador: '{delimiter}'")

        f = StringIO(cleaned_text)
        reader = csv.DictReader(f, delimiter=delimiter)
        
        # 4. Validación Cabeceras
        if not reader.fieldnames: 
            logger.debug("   ⚠️ DEBUG: Sin cabeceras válidas.")
            return text_chunk
            
        headers = [str(h).lower() for h in reader.fieldnames]
        
        # Filtro permisivo
        if not any(x in str(headers) for x in ['monto', 'valor', 'costo', 'presupuesto', 'usd', 'precio', 'id']):
            logger.debug("   ⚠️ DEBUG: No parece financiero.")
            return text_chunk

        logger.info(f"   📊 [ETL] Generando narrativa...")
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
                # --- TU FRASE ACTUALIZADA ---
                oracion = f"Registro: El responsable '{persona}' GESTIONA un '{categoria}' de {monto} pagado a la entidad '{org}' para el proyecto '{concepto}'."
                narrative.append(oracion)
            
        if narrative:
            logger.info(f"   ✅ [ETL] {len(narrative)} historias creadas.")
            return "\n".join(narrative)
            
    except Exception as e:
        logger.error(f"   ❌ DEBUG Parsing: {e}")
    
    return text_chunk

def connect_to_neo4j():
    try:
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        # graph.refresh_schema()
        return graph
    except Exception as e:
        logger.error(f"   ❌ Error Neo4j: {e}")
        return None

# ==============================================================================
# 2. UNIFICACIÓN (TU LÓGICA ACTUALIZADA)
# ==============================================================================
def unify_entities(graph):
    logger.info("🚀 [UNIFICACIÓN] Conectando nodos...")
    try:
        graph.query("MATCH (n) WHERE size(n.id) < 2 OR n.id CONTAINS 'copyright' DETACH DELETE n")
        
        graph.query("""
        MATCH (n) WITH toLower(n.id) as name, collect(n) as nodes 
        WHERE size(nodes) > 1 
        CALL apoc.refactor.mergeNodes(nodes, {properties:'combine', mergeRels:true}) 
        YIELD node RETURN count(node)
        """)
        
        # Inferencia Contextual
        graph.query("MATCH (p:Persona)<-[:MENTIONS]-(d)-[:MENTIONS]->(o:Organizacion) MERGE (p)-[:PERTENECE_A]->(o)")
        graph.query("MATCH (o:Organizacion)<-[:MENTIONS]-(d)-[:MENTIONS]->(m) WHERE labels(m) IN [['Monto'],['Costo'],['Presupuesto']] MERGE (o)-[:TIENE_PRESUPUESTO]->(m)")
        
        logger.info("✨ Grafo optimizado.")
    except Exception as e:
        logger.warning(f"   ⚠️ Aviso unificación: {e}")

# ==============================================================================
# 3. PROCESO PRINCIPAL
# ==============================================================================
def run_graph_extraction(file_path: str):
    """Orquesta la ingesta y extracción."""
    logger.info(f"🎬 Iniciando ETL Graph para: {file_path}")
    
    # 1. Llamar a Ingesta
    try:
        with open(file_path, 'rb') as f:
            # Nota: Ajustamos params para que coincidan con tu script
            response = requests.post(
                INGESTION_URL, 
                files={'file': f}, 
                params={"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
                timeout=120
            )
        if response.status_code != 200:
            logger.error(f"   ❌ Error API Ingesta: {response.text}")
            return
        raw_data = response.json()
    except Exception as e:
        logger.error(f"   ❌ Excepción conectando a Ingesta: {e}")
        return

    # 2. Preparar documentos
    def safe_doc(c, m): return Document(page_content=str(c) if c else "", metadata=m or {})
    
    lc_docs = []
    if "page_content" in raw_data:
        clean = csv_to_narrative(raw_data.get("page_content"))
        lc_docs.append(safe_doc(clean, raw_data.get("metadata")))
    elif "documents" in raw_data:
        for d in raw_data["documents"]:
            if isinstance(d, dict):
                clean = csv_to_narrative(d.get("page_content"))
                lc_docs.append(safe_doc(clean, d.get("metadata")))

    logger.info(f"📄 Extrayendo grafo de {len(lc_docs)} fragmentos...")

    # 3. Configurar LLM y Transformer
    try:
        llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)
        
        # --- TUS RELACIONES ACTUALIZADAS ---
        transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=["Organizacion", "Persona", "Proyecto", "Monto", "Concepto", "Costo", "Presupuesto"],
            # Agregamos GESTIONA y PAGADO_A según tu script
            allowed_relationships=["FINANCIA", "DIRIGE", "TIENE_COSTO", "PERTENECE_A", "MENTIONS", "GESTIONA", "PAGADO_A"],
            node_properties=False
        )
    except Exception as e:
        logger.error(f"   ❌ Error LLM: {e}")
        return

    graph = connect_to_neo4j()
    if not graph: return

    # 4. Procesar
    success_count = 0
    for i, doc in enumerate(lc_docs):
        try:
            res = transformer.convert_to_graph_documents([doc])
            if res:
                graph.add_graph_documents(res, baseEntityLabel=True, include_source=True)
                success_count += 1
                logger.info(f"   ✅ Chunk {i+1} OK.")
            else:
                logger.debug(f"   ⚠️ Chunk {i+1} sin entidades.")
                
        except Exception as e:
            logger.error(f"   ❌ Error en Chunk {i+1}: {e}")

    # 5. Finalizar
    if success_count > 0:
        unify_entities(graph)
        logger.info("🎉 ¡Proceso FINALIZADO!")
    else:
        logger.error("💀 Error: No se guardaron datos.")