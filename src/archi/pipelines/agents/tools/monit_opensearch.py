"""
Generic OpenSearch client and LangChain tool for querying any OpenSearch index.

This module provides a flexible, index-agnostic interface to CERN's MONIT Grafana API
for querying OpenSearch indices. It supports dynamic schema discovery and works with
any index pattern.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import requests
from langchain.tools import tool

from src.utils.logging import get_logger
from src.utils.env import read_secret

logger = get_logger(__name__)


class MONITOpenSearchClient:
    """
    HTTP client for querying OpenSearch via CERN's MONIT Grafana API.
    
    This client handles authentication and query formatting for the
    _msearch endpoint used by Grafana datasource proxies.
    """

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        url: str = "https://monit-grafana.cern.ch/api/datasources/proxy/9269/_msearch",
        default_index: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize the MONIT OpenSearch client.
        
        Args:
            token: Bearer token for MONIT Grafana API authentication.
                   If not provided, reads from MONIT_GRAFANA_TOKEN env var.
            url: Full URL to the _msearch endpoint.
            default_index: Default index pattern for queries (optional).
            timeout: Request timeout in seconds.
        """
        self.token = token or read_secret("MONIT_GRAFANA_TOKEN")
        if not self.token:
            raise ValueError(
                "MONIT Grafana token not provided. Set MONIT_GRAFANA_TOKEN environment variable."
            )
        
        self.url = url
        self.default_index = default_index
        self.timeout = timeout
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def query(
        self,
        es_query: Dict[str, Any],
        *,
        index: Optional[str] = None,
        search_type: str = "query_then_fetch",
    ) -> Dict[str, Any]:
        """
        Execute an Elasticsearch query against MONIT.
        
        Args:
            es_query: Elasticsearch Query DSL dictionary.
            index: Index pattern (required if no default_index set).
            search_type: Search type for meta query.
            
        Returns:
            Raw JSON response from OpenSearch.
            
        Raises:
            requests.HTTPError: On HTTP errors.
            requests.Timeout: On timeout.
            ValueError: If no index is provided.
        """
        index_pattern = index or self.default_index
        if not index_pattern:
            raise ValueError("Index pattern is required. Provide 'index' argument or set default_index.")
        
        meta_query = {
            "search_type": search_type,
            "ignore_unavailable": True,
            "index": [index_pattern],
        }
        
        # Format as NDJSON (newline-delimited JSON) for _msearch
        payload = "\n".join([json.dumps(meta_query), json.dumps(es_query)]) + "\n"
        
        response = requests.post(
            self.url,
            headers=self.headers,
            data=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def search_with_lucene(
        self,
        lucene_query: str,
        *,
        index: str,
        from_time: str = "now-24h",
        to_time: str = "now",
        time_field: str = "metadata.timestamp",
        size: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute a Lucene query with time range filtering.
        
        Args:
            lucene_query: Lucene query string (e.g., 'data.name="/store/..."').
            index: Index pattern to query (required).
            from_time: Start time in Elasticsearch date math.
            to_time: End time in Elasticsearch date math.
            time_field: Field to use for time range filtering.
            size: Maximum number of results to return.
            
        Returns:
            Raw JSON response from OpenSearch.
        """
        es_query = {
            "size": size,
            "_source": True,
            "query": {
                "bool": {
                    "must": [
                        {
                            "query_string": {
                                "query": lucene_query,
                                "analyze_wildcard": True,
                            }
                        }
                    ],
                    "filter": [
                        {
                            "range": {
                                time_field: {
                                    "gte": from_time,
                                    "lte": to_time,
                                    "format": "strict_date_optional_time||epoch_millis",
                                }
                            }
                        }
                    ],
                }
            },
            "sort": [
                {time_field: {"order": "desc"}}
            ],
        }
        
        return self.query(es_query, index=index)

    def get_index_fields(self, index: str, sample_size: int = 1) -> Dict[str, str]:
        """
        Discover available fields by fetching a sample document.
        
        Args:
            index: Index pattern to sample.
            sample_size: Number of documents to sample (default: 1).
            
        Returns:
            Dict mapping field paths to their types, e.g.:
            {"data.name": "str", "data.bytes": "int", "metadata.timestamp": "int"}
        """
        es_query = {
            "size": sample_size,
            "_source": True,
            "query": {"match_all": {}},
        }
        
        try:
            response = self.query(es_query, index=index)
            responses = response.get("responses", [response])
            
            if responses and responses[0].get("hits", {}).get("hits"):
                source = responses[0]["hits"]["hits"][0].get("_source", {})
                return self._extract_field_paths(source)
        except Exception as e:
            logger.warning("Failed to discover fields for index %s: %s", index, e)
        
        return {}

    def _extract_field_paths(self, obj: Any, prefix: str = "") -> Dict[str, str]:
        """Recursively extract field paths and types from a nested dict."""
        fields = {}
        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    fields.update(self._extract_field_paths(value, path))
                else:
                    fields[path] = type(value).__name__
        return fields


# =============================================================================
# Generic Formatting Utilities
# =============================================================================

def _flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """Flatten nested dict into dot-notation keys."""
    items = {}
    if not isinstance(d, dict):
        return items
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


def _format_value(value: Any, max_length: int = 80) -> str:
    """Format a value for display, truncating if necessary."""
    if value is None:
        return "null"
    
    str_value = str(value)
    if len(str_value) > max_length:
        return str_value[:max_length - 3] + "..."
    return str_value


def _format_hit_generic(
    hit: Dict[str, Any],
    idx: int,
    key_fields: Optional[List[str]] = None,
) -> str:
    """
    Format any OpenSearch hit generically.
    
    Args:
        hit: The OpenSearch hit document.
        idx: Index number for display.
        key_fields: Optional list of field paths to highlight at the top.
        
    Returns:
        Formatted string representation of the hit.
    """
    source = hit.get("_source", {})
    
    # Handle case where _source is a JSON string
    if isinstance(source, str):
        try:
            source = json.loads(source)
        except json.JSONDecodeError:
            logger.warning("Failed to parse _source as JSON")
            return f"[{idx}] Error: Could not parse document data"
    
    if not isinstance(source, dict):
        return f"[{idx}] Error: Unexpected document format"
    
    # Flatten the nested structure
    flat = _flatten_dict(source)
    
    lines = [f"[{idx}] Document (score: {hit.get('_score', 'N/A')})"]
    lines.append("    " + "─" * 50)
    
    # Show key fields first (if specified)
    shown_fields = set()
    if key_fields:
        lines.append("    Key fields:")
        for field in key_fields:
            if field in flat:
                value = _format_value(flat[field])
                lines.append(f"      {field}: {value}")
                shown_fields.add(field)
        lines.append("")
    
    # Show remaining fields
    remaining = {k: v for k, v in flat.items() if k not in shown_fields}
    if remaining:
        lines.append("    All fields:")
        for key in sorted(remaining.keys()):
            value = _format_value(remaining[key])
            lines.append(f"      {key}: {value}")
    
    return "\n".join(lines)


def _format_opensearch_response(
    response: Dict[str, Any],
    query: str,
    index_pattern: str,
    max_results: int,
    key_fields: Optional[List[str]] = None,
) -> str:
    """
    Format OpenSearch response for LLM consumption.
    
    Args:
        response: Raw OpenSearch response.
        query: The original query string.
        index_pattern: The index pattern queried.
        max_results: Maximum results to display.
        key_fields: Optional list of field paths to highlight.
        
    Returns:
        Formatted string suitable for LLM consumption.
    """
    # Handle _msearch response format (array of responses)
    responses = response.get("responses", [response])
    if not responses:
        return f"No results found for query: {query}"
    
    first_response = responses[0]
    
    # Check for errors
    if first_response.get("error"):
        error = first_response["error"]
        error_type = error.get("type", "unknown")
        error_reason = error.get("reason", str(error))
        return f"Query error ({error_type}): {error_reason}\n\nQuery was: {query}"
    
    hits_obj = first_response.get("hits", {})
    hits = hits_obj.get("hits", [])
    total = hits_obj.get("total", {})
    
    # Handle different total formats
    if isinstance(total, dict):
        total_count = total.get("value", 0)
        relation = total.get("relation", "eq")
        total_str = f"{total_count}+" if relation == "gte" else str(total_count)
    else:
        total_count = total
        total_str = str(total_count)
    
    if not hits:
        return f"No documents found in '{index_pattern}' matching query: {query}"
    
    # Format header
    lines = [
        f"Found {total_str} document(s) in '{index_pattern}' matching: {query}",
        f"Showing {min(len(hits), max_results)} result(s):",
        "",
    ]
    
    # Format each hit
    for idx, hit in enumerate(hits[:max_results], start=1):
        lines.append(_format_hit_generic(hit, idx, key_fields))
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Tool Description Builder
# =============================================================================

def _build_tool_description(
    index_pattern: str,
    index_description: str,
    key_fields: Optional[List[str]] = None,
) -> str:
    """
    Build tool description dynamically based on index configuration.
    
    Args:
        index_pattern: The OpenSearch index pattern.
        index_description: Human-readable description of what the index contains.
        key_fields: Optional list of key field paths for query hints.
        
    Returns:
        Tool description string.
    """
    lines = [
        f"Query OpenSearch index '{index_pattern}' using Lucene query syntax.",
    ]
    
    if index_description:
        lines.append(index_description)
    
    lines.append("")
    lines.append("Input parameters:")
    lines.append("- query: Lucene query string (required). Examples:")
    lines.append("    field_name:value  (exact match)")
    lines.append("    field_name:*wildcard*  (wildcard search)")
    lines.append("    field_a:value AND field_b:value  (boolean)")
    lines.append("- from_time: Start time (default: 'now-24h'). Supports ES date math.")
    lines.append("- to_time: End time (default: 'now'). Supports ES date math.")
    lines.append("- max_results: Max documents to return (default: 10).")
    
    if key_fields:
        lines.append("")
        lines.append(f"Key fields in this index: {', '.join(key_fields)}")
        lines.append("Use these in queries like: field_name:value")
    
    return "\n".join(lines)


# =============================================================================
# LangChain Tool Factory
# =============================================================================

def create_monit_opensearch_tool(
    client: MONITOpenSearchClient,
    *,
    index_pattern: str,
    name: str = "search_opensearch",
    index_description: str = "",
    key_fields: Optional[List[str]] = None,
    time_field: str = "metadata.timestamp",
    description: Optional[str] = None,
    max_results: int = 10,
    auto_discover_fields: bool = False,
) -> Callable[..., str]:
    """
    Create a LangChain tool for querying a specific OpenSearch index.
    
    The tool accepts Lucene query syntax and returns formatted results
    suitable for LLM consumption.
    
    Args:
        client: MONITOpenSearchClient instance.
        index_pattern: The OpenSearch index pattern to query (required).
        name: Tool name for LangChain.
        index_description: Human-readable description of what the index contains.
        key_fields: List of important field paths to highlight in output.
                   If None and auto_discover_fields is True, fields are discovered.
        time_field: Field to use for time range filtering.
        description: Full tool description (overrides auto-generated if provided).
        max_results: Default maximum results to return.
        auto_discover_fields: If True and key_fields is None, discover fields from index.
        
    Returns:
        LangChain tool function.
    """
    # Optionally discover fields from index
    effective_key_fields = key_fields
    if effective_key_fields is None and auto_discover_fields:
        try:
            discovered = client.get_index_fields(index_pattern)
            # Take first 10 fields as key fields
            effective_key_fields = list(discovered.keys())[:10]
            logger.info("Discovered %d fields from index %s", len(discovered), index_pattern)
        except Exception as e:
            logger.warning("Field discovery failed for %s: %s", index_pattern, e)
            effective_key_fields = None
    
    # Build tool description
    tool_description = description or _build_tool_description(
        index_pattern=index_pattern,
        index_description=index_description,
        key_fields=effective_key_fields,
    )
    
    @tool(name, description=tool_description)
    def _search_opensearch(
        query: str,
        from_time: str = "now-24h",
        to_time: str = "now",
        max_results_override: Optional[int] = None,
    ) -> str:
        """
        Query OpenSearch for documents matching the given Lucene query.
        
        Args:
            query: Lucene query string
            from_time: Start time in ES date math (default: now-24h)
            to_time: End time in ES date math (default: now)
            max_results_override: Override default max results
            
        Returns:
            Formatted string with matching documents.
        """
        if not query or not query.strip():
            return "Please provide a non-empty Lucene query."
        
        effective_max = max_results_override or max_results
        
        try:
            response = client.search_with_lucene(
                lucene_query=query.strip(),
                index=index_pattern,
                from_time=from_time,
                to_time=to_time,
                time_field=time_field,
                size=effective_max,
            )
            return _format_opensearch_response(
                response=response,
                query=query.strip(),
                index_pattern=index_pattern,
                max_results=effective_max,
                key_fields=effective_key_fields,
            )
            
        except requests.exceptions.Timeout:
            logger.warning("OpenSearch query timed out for query: %s", query)
            return (
                "Query timed out. The service may be slow or the query too broad. "
                "Try narrowing the time range or making the query more specific."
            )
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "unknown"
            logger.warning("OpenSearch HTTP error %s for query: %s", status_code, query)
            if status_code == 401 or status_code == 403:
                return "Authentication failed. The token may be invalid or expired."
            return f"Query failed with HTTP error {status_code}. Please try again."
        except Exception as e:
            logger.error("OpenSearch query error: %s", e, exc_info=True)
            return f"Error querying OpenSearch: {str(e)}"
    
    return _search_opensearch


__all__ = [
    "MONITOpenSearchClient",
    "create_monit_opensearch_tool",
]
