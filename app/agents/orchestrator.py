from typing import TypedDict, Literal, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from app.core.config import settings

# Importamos la instancia singleton que creamos en el paso anterior
from app.agents.financial_agent import financial_agent

# --- Definición del Estado del Grafo ---
class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_agent: str
    final_response: str

# --- 1. El Clasificador (El "Portero") ---
async def classifier_node(state: AgentState):
    """Analiza la última pregunta y decide qué agente usar."""
    last_message = state["messages"][-1].content
    
    llm = ChatOpenAI(
        base_url=settings.LM_STUDIO_URL,
        api_key="not-needed",
        model=settings.LLM_MODEL_NAME,
        temperature=0
    )
    
    # Prompt de clasificación estricta actualizado con la nueva categoría FINANZAS
    prompt = f"""
    Eres un enrutador inteligente. Clasifica la intención del usuario en una de estas 4 categorías:
    
    1. 'METABASE': Preguntas sobre ESTADÍSTICAS, VENTAS GLOBALES, TABLAS o SQL (ej: "ventas totales", "tabla de clientes", "dashboard de KPIs").
    2. 'FINANZAS': Preguntas sobre ENTIDADES ESPECÍFICAS, CONTRATOS, AUDITORÍA o RELACIONES (ej: "qué pagos hizo Ana Rojas", "contrato con Candelaria", "gastos del proyecto X", "quién financia a Y").
    3. 'OPENWEBUI': Preguntas de SISTEMA (instalar modelos, crear usuarios, logs, configuración).
    4. 'GENERAL': Saludos, cháchara o preguntas fuera de contexto.
    
    Usuario: "{last_message}"
    
    Responde SOLAMENTE con una palabra: METABASE, FINANZAS, OPENWEBUI o GENERAL.
    """
    
    response = await llm.ainvoke(prompt)
    decision = response.content.strip().upper()
    
    # Lógica de enrutamiento
    if "METABASE" in decision: return {"next_agent": "metabase"}
    if "FINANZAS" in decision: return {"next_agent": "finanzas"}
    if "OPENWEBUI" in decision: return {"next_agent": "openwebui"}
    return {"next_agent": "general"}

# --- 2. Nodos Ejecutores (Los "Trabajadores") ---

async def metabase_node(state: AgentState, config):
    """Invoca al BrainAgent (Datos Tabulares / BI)"""
    query = state["messages"][-1].content
    # BrainAgent se pasa por config porque tiene estado complejo de herramientas
    brain_agent = config["configurable"]["brain_agent"]
    
    if not brain_agent:
        return {"final_response": "Error: El agente de Metabase no está inicializado."}

    result = await brain_agent.ainvoke({"input": query})
    return {"final_response": result["output"]}

async def financial_node(state: AgentState, config):
    """Invoca al FinancialAgent (Grafo / Auditoría)"""
    query = state["messages"][-1].content
    
    # FinancialAgent es singleton y stateless (usa Neo4j), lo llamamos directo
    response = await financial_agent.ainvoke(query)
    return {"final_response": response}

async def openwebui_node(state: AgentState, config):
    """Invoca al MCPAgent (Admin de Sistema)"""
    query = state["messages"][-1].content
    mcp_agent = config["configurable"]["mcp_agent"]
    
    if not mcp_agent:
        return {"final_response": "El agente de administración (OpenWebUI) no está disponible."}

    result = await mcp_agent.ainvoke({"messages": [HumanMessage(content=query)]})
    return {"final_response": result["messages"][-1].content}

async def general_node(state: AgentState):
    """Responde preguntas simples directamente"""
    llm = ChatOpenAI(
        base_url=settings.LM_STUDIO_URL, 
        api_key="not-needed", 
        model=settings.LLM_MODEL_NAME
    )
    response = await llm.ainvoke(state["messages"])
    return {"final_response": response.content}

# --- 3. Construcción del Grafo ---
def build_orchestrator():
    workflow = StateGraph(AgentState)
    
    # Añadir nodos
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("metabase_agent", metabase_node)
    workflow.add_node("financial_agent", financial_node) # <-- Nuevo nodo
    workflow.add_node("openwebui_agent", openwebui_node)
    workflow.add_node("general_chat", general_node)
    
    # Definir punto de entrada
    workflow.set_entry_point("classifier")
    
    # Definir aristas condicionales (Routing)
    def route(state):
        if state["next_agent"] == "metabase": return "metabase_agent"
        if state["next_agent"] == "finanzas": return "financial_agent"
        if state["next_agent"] == "openwebui": return "openwebui_agent"
        return "general_chat"

    workflow.add_conditional_edges(
        "classifier",
        route,
        {
            "metabase_agent": "metabase_agent",
            "financial_agent": "financial_agent",
            "openwebui_agent": "openwebui_agent",
            "general_chat": "general_chat"
        }
    )
    
    # Todos los agentes terminan el flujo en END
    workflow.add_edge("metabase_agent", END)
    workflow.add_edge("financial_agent", END)
    workflow.add_edge("openwebui_agent", END)
    workflow.add_edge("general_chat", END)
    
    return workflow.compile()