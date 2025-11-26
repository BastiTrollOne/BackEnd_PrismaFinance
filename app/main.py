import logging
import sys
from contextlib import AsyncExitStack
from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import shutil
import os
import time

# Importaciones MCP y Agentes
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from langchain_core.messages import HumanMessage

# Tus módulos
from app.api.v1 import agents as agents_router
from app.agents import brain_agent as brain_agent_module # Metabase
from app.agents import mcp_agent as mcp_agent_module     # OpenWebUI
from app.agents.orchestrator import build_orchestrator   # El nuevo orquestador
from app.services.graph_etl import run_graph_extraction  # El servicio ETL

# Configuración de Logs
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("PrismaFinanceAPI")

app = FastAPI(title="PrismaFinance API", version="2.0.0")
app.include_router(agents_router.router, prefix="/v1")

# URL del servidor p.py (Open WebUI)
MCP_OPENWEBUI_URL = "http://localhost:9001/sse"

# Modelo para recibir peticiones
class UserQuery(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Iniciando Sistema Multi-Agente...")
    app.state.exit_stack = AsyncExitStack()

    try:
        # --- 1. INICIAR AGENTE METABASE (Brain) ---
        logger.info("📊 Conectando a Metabase...")
        brain_agent_module.initialize_mcp_client() 
        metabase_tools = await brain_agent_module.mcp_client.get_tools(server_name="MCP_METABASE")
        brain_agent_instance = brain_agent_module.LangChainBrainAgent(tools=metabase_tools)
        logger.info(f"✅ Brain Agent listo con {len(metabase_tools)} herramientas.")

        # --- 2. INICIAR AGENTE OPEN WEBUI (MCP Worker) ---
        logger.info(f"🔧 Conectando a Open WebUI ({MCP_OPENWEBUI_URL})...")
        try:
            streams = await app.state.exit_stack.enter_async_context(sse_client(MCP_OPENWEBUI_URL))
            session = await app.state.exit_stack.enter_async_context(ClientSession(streams[0], streams[1]))
            await session.initialize()
            
            mcp_agent_instance = await mcp_agent_module.build_mcp_worker_agent(session)
            logger.info("✅ MCP OpenWebUI Agent listo.")
        except Exception as e:
            logger.error(f"⚠️ Fallo al conectar con p.py: {e}. El orquestador funcionará sin OpenWebUI.")
            mcp_agent_instance = None

        # --- 3. PREPARAR EL ORQUESTADOR ---
        app.state.orchestrator = build_orchestrator()
        app.state.agents_config = {
            "brain_agent": brain_agent_instance,
            "mcp_agent": mcp_agent_instance
        }
        logger.info("🤖 ORQUESTADOR OPERATIVO.")

        # --- 4. INICIALIZAR ÍNDICES DE NEO4J (Bloque Try interno) ---
        try:    
            from langchain_community.graphs import Neo4jGraph
            from app.core.config import settings
            
            logger.info("🔗 Verificando índices de Neo4j...")
            graph = Neo4jGraph(
                url=settings.NEO4J_URI,
                username=settings.NEO4J_USERNAME,
                password=settings.NEO4J_PASSWORD
            )
            # Crear restricciones de unicidad
            graph.query("CREATE CONSTRAINT unique_persona_id IF NOT EXISTS FOR (p:Persona) REQUIRE p.id IS UNIQUE")
            graph.query("CREATE CONSTRAINT unique_org_id IF NOT EXISTS FOR (o:Organizacion) REQUIRE o.id IS UNIQUE")
            graph.query("CREATE CONSTRAINT unique_proy_id IF NOT EXISTS FOR (pr:Proyecto) REQUIRE pr.id IS UNIQUE")
            graph.query("CREATE CONSTRAINT unique_doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
            
            logger.info("✅ Índices de Neo4j listos.")
        except Exception as e:
            # Advertencia suave: Si falla Neo4j (ej. container lento), la API sigue funcionando
            logger.warning(f"⚠️ No se pudieron crear índices en Neo4j (puede que el contenedor aún esté iniciando): {e}")

    except Exception as e:
        # Error fatal general
        logger.error(f"❌ Error fatal en inicio de la aplicación: {e}")
        raise RuntimeError("Startup failed") from e

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, "exit_stack"):
        await app.state.exit_stack.aclose()
    logger.info("🔌 API Apagada.")

@app.post("/upload", tags=["Ingestion"])
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Sube un archivo, lo guarda temporalmente y dispara el proceso ETL
    (OCR -> Limpieza CSV -> Grafo) en segundo plano.
    """
    upload_dir = "/app/backups"
    os.makedirs(upload_dir, exist_ok=True)
    
    safe_filename = file.filename.replace(" ", "_")
    file_path = os.path.join(upload_dir, f"{int(time.time())}_{safe_filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    logger.info(f"📥 Archivo recibido: {safe_filename}. Iniciando ETL...")

    background_tasks.add_task(run_graph_extraction, file_path)

    return {
        "status": "success",
        "message": "Archivo recibido. El procesamiento ETL (Grafo + OCR) ha comenzado en segundo plano.",
        "filename": safe_filename
    }

# --- ENDPOINT PRINCIPAL ---
@app.post("/chat", tags=["Orchestrator"])
async def chat_endpoint(request: UserQuery, fastapi_req: Request):
    """Endpoint único que recibe la pregunta y el orquestador decide."""
    
    orchestrator = fastapi_req.app.state.orchestrator
    config = fastapi_req.app.state.agents_config
    
    inputs = {"messages": [HumanMessage(content=request.query)]}
    
    # Ejecutar el grafo del orquestador
    result = await orchestrator.ainvoke(inputs, config={"configurable": config})
    
    return {
        "response": result["final_response"],
        "routed_to": result.get("next_agent", "unknown")
    }