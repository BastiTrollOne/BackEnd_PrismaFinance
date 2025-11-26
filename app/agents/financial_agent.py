import logging
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import settings

logger = logging.getLogger("FinancialAgent")

# --- PROMPTS ESPECIALIZADOS (Cerebro Financiero) ---
EXTRACTION_PROMPT = """You are a Named Entity Extractor for a Mining Finance System.
Task: Extract the main Person, Organization, or Project name from the query.
Rules:
1. Output ONLY the name. No explanations.
2. Remove words like "gastos", "reporte", "cuánto", "pagos".
Example: "Gastos de Ana Rojas" -> Ana Rojas
"""

SYNTHESIS_PROMPT_TEMPLATE = """
Actúa como Auditor Financiero Senior.
Analiza los siguientes REGISTROS DE LA BASE DE DATOS DE GRAFOS:
{graph_data}

PREGUNTA DEL USUARIO: "{query}"

INSTRUCCIONES:
1. Identifica relaciones explícitas (quién pagó a quién, qué proyecto tiene qué presupuesto).
2. Si hay montos (nodos 'Monto', 'Costo'), menciónalos con precisión.
3. Si la información viene de un documento, menciona que existe evidencia ("Según los registros...").
4. Si no hay información suficiente, dilo claramente.

Responde en español profesional, directo y basado en evidencia.
"""

class FinancialAgent:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD
        )
        # Usamos el LLM configurado en settings (LM Studio / Ollama via API compatible)
        self.llm = ChatOpenAI(
            base_url=settings.LM_STUDIO_URL,
            api_key="not-needed",
            model=settings.LLM_MODEL_NAME,
            temperature=0
        )

    async def ainvoke(self, query: str) -> str:
        """
        Flujo de razonamiento:
        1. Entender qué entidad busca el usuario (Extracción).
        2. Buscar esa entidad y sus vecinos en el Grafo (Retrieval).
        3. Sintetizar una respuesta de auditoría (Generación).
        """
        logger.info(f"🕵️‍♂️ Analizando consulta financiera: {query}")
        
        try:
            # FASE 1: Extracción de Entidad
            extract_msg = [
                SystemMessage(content=EXTRACTION_PROMPT),
                HumanMessage(content=f"Input: {query}")
            ]
            entity_res = await self.llm.ainvoke(extract_msg)
            entity_name = entity_res.content.strip().replace('"', '').replace("'", "")
            
            if len(entity_name) < 2:
                return "No pude identificar una entidad específica (Persona, Empresa, Proyecto) en tu consulta."

            logger.info(f"   🎯 Entidad objetivo: '{entity_name}'")

            # FASE 2: Búsqueda en Grafo (Cypher Robusto)
            # Buscamos nodos que coincidan difusamente y traemos su vecindario (2 saltos)
            cypher_query = f"""
            MATCH (n)-[r*1..2]-(related)
            WHERE toLower(toString(n.id)) CONTAINS toLower('{entity_name}')
            RETURN n, r, related LIMIT 100
            """
            
            # Ejecución síncrona (Neo4j driver standard es sync, lo envolvemos si es necesario, pero suele ser rápido)
            results = self.graph.query(cypher_query)
            
            if not results:
                return f"Busqué información sobre '{entity_name}' en el grafo financiero, pero no encontré registros, contratos o relaciones vinculadas."

            logger.info(f"   🔎 Encontrados {len(results)} registros en el grafo.")

            # FASE 3: Síntesis Financiera
            final_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
                graph_data=str(results)[:8000], # Truncar para no explotar contexto
                query=query
            )
            
            final_response = await self.llm.ainvoke(final_prompt)
            return final_response.content

        except Exception as e:
            logger.error(f"❌ Error en FinancialAgent: {e}", exc_info=True)
            return f"Ocurrió un error técnico procesando la consulta financiera: {str(e)}"

# Instancia singleton para usar en el orquestador
financial_agent = FinancialAgent()