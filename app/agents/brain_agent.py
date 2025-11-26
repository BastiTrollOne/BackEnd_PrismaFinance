import logging
import re
import json
import os
from typing import List, Any, Tuple, Dict, Optional

from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.agents import AgentAction, AgentFinish
from langchain.tools import tool
from fastapi import HTTPException

# --- CORRECCIÓN DE IMPORTACIONES ---
# Importamos create_tool_calling_agent del paquete base
from langchain.agents import create_tool_calling_agent
# Importamos AgentExecutor directamente de su módulo para evitar errores de init
from langchain.agents.agent import AgentExecutor

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import settings
from app.agents.sql_agent import create_sql_agent_chain

brain_logger = logging.getLogger("LangChainBrainAgent")

# --- MCP Client Setup ---
mcp_client: MultiServerMCPClient = None
mcp_tools_cache: Dict[str, Any] = {}

def initialize_mcp_client() -> MultiServerMCPClient:
    """Initializes the MCP client."""
    global mcp_client
    # Sin bloque 'env' para compatibilidad Linux
    mcp_client = MultiServerMCPClient({
        "MCP_METABASE": {
            "command": "docker",
            "args": [
                "run", "-i", "--rm",
                "--add-host=host.docker.internal:host-gateway",
                "-e", f"METABASE_URL={settings.METABASE_URL_FOR_DOCKER}",
                "-e", f"METABASE_USERNAME={settings.METABASE_USERNAME}",
                "-e", f"METABASE_PASSWORD={settings.METABASE_PASSWORD}",
                "mcp/metabase"
            ],
            "transport": "stdio"
        }
    })
    return mcp_client

# --- Helper Functions ---
def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";" ))

def enforce_limit(sql: str, default_limit: int = 200) -> str:
    return sql if re.search(r"\blimit\s+\d+\b", sql, re.IGNORECASE) else f"{sql} LIMIT {default_limit}"

def format_markdown_table(rows: List[Dict[str, Any]], max_rows: int = 20) -> str:
    if not rows: return "La consulta no devolvió filas."
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    if len(rows) > max_rows: lines.append(f"\n_Mostrando las primeras {max_rows} filas de {len(rows)}._")
    return "\n".join(lines)

def convert_output_to_markdown(output: Any, max_rows: int = 20) -> str:
    if isinstance(output, str):
        try: output = json.loads(output)
        except: return output
    if isinstance(output, list) and output and isinstance(output[0], dict):
        return format_markdown_table(output, max_rows)
    if isinstance(output, dict):
        return "\n".join([f"**{k}:** {v}" for k, v in output.items()])
    if isinstance(output, list):
        return "\n".join([f"- {item}" for item in output])
    return str(output)

# --- Tools ---
async def get_default_database(agent_instance):
    brain_logger.info("Finding default database...")
    raw = await agent_instance.list_db_tool.ainvoke({})
    if isinstance(raw, str): raw = json.loads(raw)
    dbs = raw if isinstance(raw, list) else (raw.get("databases") or raw.get("data") or [raw])
    valid_dbs = [d for d in dbs if isinstance(d, dict) and str(d.get("engine", "")).lower() != "h2"]
    if not valid_dbs: raise ValueError("No valid databases found.")
    return valid_dbs[0].get("id"), valid_dbs[0].get("name", "db")

@tool
async def execute_sql(user_question: Optional[str] = None, query: Optional[str] = None) -> str:
    """Executes a SQL query or answers a data question."""
    input_for_sql = user_question or query or mcp_tools_cache.get("last_user_input", "")
    if not input_for_sql: return "Error: Query required."
    
    agent = mcp_tools_cache.get("agent_instance")
    db_id, _ = await get_default_database(agent)
    
    is_sql = bool(re.match(r"^\s*(SELECT|WITH)\b", input_for_sql, re.IGNORECASE))
    if is_sql:
        final_sql = input_for_sql
    else:
        gen = await agent.sql_agent_chain.ainvoke({"question": input_for_sql, "database_id": db_id})
        final_sql = gen.get("sql") if isinstance(gen, dict) else str(gen)
        
    res = await agent.ainvoke_sql_direct(db_id, final_sql)
    
    if isinstance(res, dict) and "data" in res:
        rows = res["data"].get("rows", [])
        cols = [c.get("display_name", c["name"]) for c in res["data"].get("cols", [])]
        if rows and cols: return format_markdown_table([dict(zip(cols, r)) for r in rows])
        
    return convert_output_to_markdown(res)

@tool
async def list_products(user_question: Optional[str] = None, query: Optional[str] = None) -> str:
    return await execute_sql.ainvoke({"user_question": user_question, "query": query})

@tool
async def list_databases_pretty() -> str:
    return str(await mcp_tools_cache.get("agent_instance").list_db_tool.ainvoke({}))

# --- Agent Class ---
class LangChainBrainAgent:
    def __init__(self, tools: list):
        self.llm = ChatOpenAI(base_url=settings.LM_STUDIO_URL, api_key="x", model=settings.LLM_MODEL_NAME, temperature=0)
        self.tools = [t for t in tools if t.name not in ["execute_query", "get_database_schema"]] + [execute_sql, list_products, list_databases_pretty]
        self.sql_agent_chain = create_sql_agent_chain(tools)
        self.mcp_tool = next((t for t in tools if t.name == "execute_query"), None)
        self.list_db_tool = next((t for t in tools if t.name == "list_databases"), None)
        
        global mcp_tools_cache
        mcp_tools_cache = {"agent_instance": self}
        
        prompt = ChatPromptTemplate.from_messages([("system", "Eres un experto en Metabase."), ("user", "{input}"), MessagesPlaceholder("agent_scratchpad")])
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False, handle_parsing_errors=True)

    async def ainvoke(self, input_data: dict):
        mcp_tools_cache["last_user_input"] = input_data.get('input', '')
        try:
            res = await self.agent_executor.ainvoke(input_data)
            return {"output": convert_output_to_markdown(res.get("output"))}
        except Exception as e:
            return {"output": f"Error: {e}"}

    async def ainvoke_sql_direct(self, db_id: int, sql: str):
        return await self.mcp_tool.ainvoke({"database_id": db_id, "query": normalize_sql(enforce_limit(sql))})