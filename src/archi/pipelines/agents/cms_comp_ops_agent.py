from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.documents import Document

from src.utils.logging import get_logger
from src.utils.env import read_secret
from src.archi.pipelines.agents.base_react import BaseReActAgent
from src.data_manager.vectorstore.retrievers import HybridRetriever
from src.archi.pipelines.agents.tools import (
    create_document_fetch_tool,
    create_file_search_tool,
    create_metadata_search_tool,
    create_metadata_schema_tool,
    create_retriever_tool,
    RemoteCatalogClient,
    MONITOpenSearchClient,
    create_monit_opensearch_search_tool,
    create_monit_opensearch_aggregation_tool,
)
from src.archi.pipelines.agents.utils.history_utils import infer_speaker

logger = get_logger(__name__)


def _load_skill(skill_name: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Load a skill markdown file by name from the config's skills directory.
    
    Skills are markdown files in the config directory's `skills/` subdirectory.
    Returns None if the skill file doesn't exist.
    
    Args:
        skill_name: Name of the skill file (without .md extension).
        config: Agent config dict containing 'config_path'.
        
    Returns:
        Skill content as string, or None if not found.
    """
    config_path = config.get("config_path")
    if not config_path:
        logger.warning("No config_path in config; cannot load skill '%s'", skill_name)
        return None
    
    skill_path = Path(config_path).parent / "skills" / f"{skill_name}.md"
    if not skill_path.exists():
        logger.warning("Skill file not found: %s", skill_path)
        return None
    
    try:
        content = skill_path.read_text(encoding="utf-8")
        logger.info("Loaded skill '%s' from %s (%d chars)", skill_name, skill_path, len(content))
        return content
    except Exception as e:
        logger.error("Failed to read skill file %s: %s", skill_path, e)
        return None


class CMSCompOpsAgent(BaseReActAgent):
    """Agent designed for CMS CompOps operations."""

    def __init__(
        self,
        config: Dict[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config, *args, **kwargs)

        self.catalog_service = RemoteCatalogClient.from_deployment_config(self.config)
        self._vector_retrievers = None
        self._vector_tool = None

        self.rebuild_static_tools()
        self.rebuild_static_middleware()
        self.refresh_agent()

    def _build_static_tools(self) -> List[Callable]:
        """Initialise static tools that are always available to the agent."""
        file_search_tool = create_file_search_tool(
            self.catalog_service,
            description= (
                "Grep-like search over file contents. Provide a distinctive phrase or regex; optionally use regex=true, "
                "case_sensitive=true, and context (before/after). Returns matching lines with hashes; "
                "use fetch_catalog_document for full text."
            ),
            store_docs=self._store_documents,
        )
        metadata_search_tool = create_metadata_search_tool(
            self.catalog_service,
            description=(
                "Query the files' metadata catalog (ticket IDs, source URLs, resource types, etc.). "
                "Supports key:value filters and OR (e.g., source_type:git OR url:https://... ticket_id:CMS-123). "
                "Returns matching files with metadata; use fetch_catalog_document to pull full text."
            ),
            store_docs=self._store_documents,
        )
        metadata_schema_tool = create_metadata_schema_tool(
            self.catalog_service,
            description=(
                "List metadata schema hints: supported keys, distinct source_type values, and suffixes. "
                "Use this to learn which key:value filters are available before searching."
            ),
        )

        fetch_tool = create_document_fetch_tool(
            self.catalog_service,
            description=(
                "Fetch full document text by resource hash after a search hit. "
                "Use this sparingly to pull only the most relevant files."
            ),
        )

        all_tools = [file_search_tool, metadata_search_tool, metadata_schema_tool, fetch_tool]

        # MONIT OpenSearch tools for querying various indices
        monit_token = read_secret("MONIT_GRAFANA_TOKEN")
        monit_url = "https://monit-grafana.cern.ch/api/datasources/proxy/9269/_msearch"
        if monit_token:
            try:
                monit_client = MONITOpenSearchClient(url=monit_url, token=monit_token)
                
                # Load skill for Rucio events monitoring
                rucio_events_skill = _load_skill("rucio_events", self.config)
                
                # Rucio events search tool (for fetching individual events)
                rucio_events_search_tool = create_monit_opensearch_search_tool(
                    monit_client,
                    name="search_rucio_events",
                    index_pattern="monit_prod_cms_rucio_raw_events*",
                    index_description="CMS Rucio events (transfers, deletions, rules, datasets). Use for fetching individual event details.",
                    skill=rucio_events_skill,
                )
                all_tools.append(rucio_events_search_tool)
                logger.info("Rucio events search tool initialized")
                
                # Rucio events aggregation tool (for counting, grouping, statistics)
                rucio_events_agg_tool = create_monit_opensearch_aggregation_tool(
                    monit_client,
                    name="aggregate_rucio_events",
                    index_pattern="monit_prod_cms_rucio_raw_events*",
                    index_description="Aggregate CMS Rucio events. Use for questions like 'top errors', 'count by RSE', 'total bytes'.",
                    skill=rucio_events_skill,
                )
                all_tools.append(rucio_events_agg_tool)
                logger.info("Rucio events aggregation tool initialized")
                
            except Exception as e:
                logger.warning(f"Failed to initialize MONIT OpenSearch tools: {e}")
        else:
            logger.info("MONIT_GRAFANA_TOKEN not found; MONIT OpenSearch tools not available")

        return all_tools

    def _store_documents(self, stage: str, docs: Sequence[Document]) -> None:
        """Centralised helper used by tools to record documents into the active memory."""
        memory = self.active_memory
        if not memory:
            return
        # Prefer memory convenience method if available
        try:
            logger.debug("Recording %d documents from stage '%s' via record_documents", len(docs), stage)
            memory.record_documents(stage, docs)
        except Exception:
            # fallback to explicit record + note
            memory.record(stage, docs)
            memory.note(f"{stage} returned {len(list(docs))} document(s).")

    def _prepare_inputs(self, history: Any, **kwargs) -> Dict[str, Any]:
        """Create list of messages using LangChain's formatting."""
        history = history or []
        history_messages = [infer_speaker(msg[0])(msg[1]) for msg in history]
        return {"history": history_messages}

    def _prepare_agent_inputs(self, **kwargs) -> Dict[str, Any]:
        """Prepare agent state and formatted inputs shared by invoke/stream."""

        # event-level memory (which documents were retrieved)
        memory = self.start_run_memory()

        # refresh vs connection
        vectorstore = kwargs.get("vectorstore")
        if vectorstore:
            self._update_vector_retrievers(vectorstore)
        else:
            self._vector_retrievers = None
            self._vector_tools = None
        extra_tools = self._vector_tools if self._vector_tools else None

        self.refresh_agent(extra_tools=extra_tools)

        inputs = self._prepare_inputs(history=kwargs.get("history"))
        history_messages = inputs["history"]
        if history_messages:
            memory.note(f"History contains {len(history_messages)} message(s).")
            last_message = history_messages[-1]
            content = self._message_content(last_message)
            if content:
                snippet = content if len(content) <= 200 else f"{content[:197]}..."
                memory.note(f"Latest user message: {snippet}")
        return {"messages": history_messages}

    def _update_vector_retrievers(self, vectorstore: Any) -> None:
        """Instantiate or refresh the vectorstore retriever tool using hybrid retrieval."""
        retrievers_cfg = self.dm_config.get("retrievers", {})
        hybrid_cfg = retrievers_cfg.get("hybrid_retriever", {})

        k = hybrid_cfg["num_documents_to_retrieve"]
        bm25_weight = hybrid_cfg["bm25_weight"]
        semantic_weight = hybrid_cfg["semantic_weight"]

        hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            k=k,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )

        hybrid_description = (
            "Hybrid search over the knowledge base that combines both lexical (BM25) and semantic (vector) search. "
            "This automatically finds documents matching exact keywords, error messages, ticket IDs, filenames, "
            "and function names (via BM25) as well as conceptually related content and paraphrased information "
            "(via semantic search). Use this for comprehensive retrieval - it handles both precise keyword matches "
            "and conceptual similarity automatically."
        )

        self._vector_retrievers = [hybrid_retriever]
        self._vector_tools = []
        self._vector_tools.append(
            create_retriever_tool(
                hybrid_retriever,
                name="search_vectorstore_hybrid",
                description=hybrid_description,
                store_docs=self._store_documents,
            )
        )
