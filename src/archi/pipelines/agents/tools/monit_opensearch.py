"""
Generic OpenSearch client and LangChain tools for querying any OpenSearch index.

This module provides a flexible, index-agnostic interface to CERN's MONIT Grafana API
for querying OpenSearch indices. It supports search and aggregation queries and works
with any index pattern. Domain-specific knowledge is provided via skill files.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

import requests
from langchain.tools import tool

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Safety limits to prevent overwhelming the LLM context
MAX_RESULTS_HARD_LIMIT = 50  # Never return more than this many documents
MAX_OUTPUT_CHARS = 50000    # Truncate output if it exceeds this

# Time format accepted by OpenSearch range filters
_TIME_FORMAT = "strict_date_optional_time||epoch_millis"


class MONITOpenSearchClient:
    """
    HTTP client for querying OpenSearch via CERN's MONIT Grafana API.
    
    This client handles authentication and query formatting for the
    _msearch endpoint used by Grafana datasource proxies.
    """

    def __init__(
        self,
        *,
        url: str,
        token: str,
        timeout: float = 60.0,
    ):
        """
        Initialize the MONIT OpenSearch client.
        
        Args:
            url: Full URL to the _msearch endpoint (required).
            token: Bearer token for MONIT Grafana API authentication (required).
            timeout: Request timeout in seconds.
        """
        self.url = url
        self.token = token
        self.timeout = timeout
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def query(
        self,
        query_dsl: Dict[str, Any],
        *,
        index: str,
        search_type: str = "query_then_fetch",
    ) -> Dict[str, Any]:
        """
        Execute a query against MONIT OpenSearch.
        
        Args:
            query_dsl: Query DSL dictionary (OpenSearch compatible).
            index: Index pattern (required).
            search_type: Search type for meta query.
            
        Returns:
            Raw JSON response from OpenSearch.
            
        Raises:
            requests.HTTPError: On HTTP errors.
            requests.Timeout: On timeout.
        """
        
        meta_query = {
            "search_type": search_type,
            "ignore_unavailable": True,
            "index": [index],
        }
        
        # Format as NDJSON (newline-delimited JSON) for _msearch
        payload = "\n".join([json.dumps(meta_query), json.dumps(query_dsl)]) + "\n"
        
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
            from_time: Start time in OpenSearch date math.
            to_time: End time in OpenSearch date math.
            time_field: Field to use for time range filtering.
            size: Maximum number of results to return.
            
        Returns:
            Raw JSON response from OpenSearch.
        """
        query_dsl = {
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
                                    "format": _TIME_FORMAT,
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
        
        return self.query(query_dsl, index=index)

    def search_with_aggregation(
        self,
        lucene_query: str,
        agg_field: str,
        *,
        index: str,
        agg_type: str = "terms",
        agg_size: int = 10,
        from_time: str = "now-24h",
        to_time: str = "now",
        time_field: str = "metadata.timestamp",
    ) -> Dict[str, Any]:
        """
        Execute a Lucene query with aggregation.
        
        This method runs an aggregation query and returns only the aggregation
        results (no individual documents). Useful for counting, grouping, and
        statistical analysis.
        
        Args:
            lucene_query: Lucene query string to filter documents before aggregation.
                         Use "*" for all documents.
            agg_field: Field to aggregate on (e.g., 'data.reason', 'data.src_rse').
            index: Index pattern to query (required).
            agg_type: Type of aggregation. Supported types:
                     - 'terms': Count documents by unique values (default)
                     - 'sum': Sum numeric field values
                     - 'avg': Average of numeric field values
                     - 'min': Minimum value
                     - 'max': Maximum value
                     - 'cardinality': Count unique values
            agg_size: Maximum number of buckets to return for terms aggregation.
            from_time: Start time in OpenSearch date math.
            to_time: End time in OpenSearch date math.
            time_field: Field to use for time range filtering.
            
        Returns:
            Raw JSON response from OpenSearch with aggregation results.
        """
        # Build the aggregation based on type
        if agg_type == "terms":
            agg_body = {
                "terms": {
                    "field": agg_field,
                    "size": agg_size,
                }
            }
        elif agg_type in ("sum", "avg", "min", "max", "cardinality"):
            agg_body = {
                agg_type: {
                    "field": agg_field,
                }
            }
        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}. "
                           f"Supported: terms, sum, avg, min, max, cardinality")
        
        query_dsl = {
            "size": 0,  # Don't return documents, just aggregations
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
                                    "format": _TIME_FORMAT,
                                }
                            }
                        }
                    ],
                }
            },
            "aggs": {
                "result": agg_body
            },
        }
        
        return self.query(query_dsl, index=index)


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
) -> str:
    """
    Format any OpenSearch hit generically.
    
    Args:
        hit: The OpenSearch hit document.
        idx: Index number for display.
        
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
    
    # Show all fields sorted alphabetically
    if flat:
        lines.append("    Fields:")
        for key in sorted(flat.keys()):
            value = _format_value(flat[key])
            lines.append(f"      {key}: {value}")
    
    return "\n".join(lines)


def _format_opensearch_response(
    response: Dict[str, Any],
    query: str,
    index_pattern: str,
    max_results: int,
    from_time: str = "now-24h",
    to_time: str = "now",
) -> str:
    """
    Format OpenSearch response for LLM consumption.
    
    Args:
        response: Raw OpenSearch response.
        query: The original query string.
        index_pattern: The index pattern queried.
        max_results: Maximum results to display.
        from_time: Start time used for the query.
        to_time: End time used for the query.
        
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
        return f"No documents found in '{index_pattern}' matching query: {query} (time range: {from_time} to {to_time})"
    
    # Format header
    lines = [
        f"Found {total_str} document(s) in '{index_pattern}' matching: {query}",
        f"Time range: {from_time} to {to_time}",
        f"Showing {min(len(hits), max_results)} result(s):",
        "",
    ]
    
    # Format each hit
    for idx, hit in enumerate(hits[:max_results], start=1):
        lines.append(_format_hit_generic(hit, idx))
        lines.append("")
    
    return "\n".join(lines)


def _format_aggregation_response(
    response: Dict[str, Any],
    query: str,
    index_pattern: str,
    agg_field: str,
    agg_type: str,
    from_time: str = "now-24h",
    to_time: str = "now",
) -> str:
    """
    Format OpenSearch aggregation response for LLM consumption.
    
    Args:
        response: Raw OpenSearch response with aggregations.
        query: The original filter query string.
        index_pattern: The index pattern queried.
        agg_field: Field that was aggregated on.
        agg_type: Type of aggregation performed.
        from_time: Start time used for the query.
        to_time: End time used for the query.
        
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
        return f"Aggregation error ({error_type}): {error_reason}\n\nQuery was: {query}"
    
    # Get total document count
    hits_obj = first_response.get("hits", {})
    total = hits_obj.get("total", {})
    if isinstance(total, dict):
        total_count = total.get("value", 0)
    else:
        total_count = total
    
    # Get aggregation results
    aggs = first_response.get("aggregations", {})
    result_agg = aggs.get("result", {})
    
    lines = [
        f"Aggregation results for '{index_pattern}'",
        f"Filter query: {query}",
        f"Time range: {from_time} to {to_time}",
        f"Aggregation: {agg_type} on field '{agg_field}'",
        f"Total matching documents: {total_count}",
        "",
    ]
    
    if agg_type == "terms":
        # Terms aggregation - list of buckets
        buckets = result_agg.get("buckets", [])
        if not buckets:
            lines.append("No buckets found. The field may not exist or have no values.")
        else:
            lines.append(f"Top {len(buckets)} values:")
            lines.append("-" * 60)
            lines.append(f"{'Value':<45} {'Count':>10}")
            lines.append("-" * 60)
            for bucket in buckets:
                key = bucket.get("key", "N/A")
                doc_count = bucket.get("doc_count", 0)
                # Truncate long keys
                key_str = str(key)
                if len(key_str) > 42:
                    key_str = key_str[:39] + "..."
                lines.append(f"{key_str:<45} {doc_count:>10}")
            
            # Show sum of other docs not in top buckets
            sum_other = result_agg.get("sum_other_doc_count", 0)
            if sum_other > 0:
                lines.append("-" * 60)
                lines.append(f"{'(other values)':<45} {sum_other:>10}")
    else:
        # Metric aggregations (sum, avg, min, max, cardinality)
        value = result_agg.get("value")
        if value is not None:
            # Format large numbers with commas
            if isinstance(value, float):
                formatted_value = f"{value:,.2f}"
            else:
                formatted_value = f"{value:,}"
            lines.append(f"{agg_type.upper()}: {formatted_value}")
        else:
            lines.append(f"No value returned. The field may not exist or be non-numeric.")
    
    return "\n".join(lines)


# =============================================================================
# Tool Description Builder
# =============================================================================

def _build_tool_description(
    index_pattern: str,
    index_description: str,
) -> str:
    """
    Build tool description dynamically based on index configuration.
    
    Args:
        index_pattern: The OpenSearch index pattern.
        index_description: Human-readable description of what the index contains.
        
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
    lines.append("- from_time: Start time (default: 'now-24h'). Supports date math (e.g., now-7d, now-24h).")
    lines.append("- to_time: End time (default: 'now'). Supports date math (e.g., now-7d, now-24h).")
    lines.append(f"- max_results: Max documents to return (default: 10, max: {MAX_RESULTS_HARD_LIMIT}).")
    
    return "\n".join(lines)


# =============================================================================
# LangChain Tool Factory
# =============================================================================

def create_monit_opensearch_search_tool(
    client: MONITOpenSearchClient,
    *,
    index_pattern: str,
    name: str = "search_opensearch",
    index_description: str = "",
    time_field: str = "metadata.timestamp",
    description: Optional[str] = None,
    max_results: int = 10,
    skill: Optional[str] = None,
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
        time_field: Field to use for time range filtering.
        description: Full tool description (overrides auto-generated if provided).
        max_results: Default maximum results to return.
        skill: Optional markdown content with domain expertise for this tool.
               Provides field documentation and query guidance to the LLM.
        
    Returns:
        LangChain tool function.
    """
    # Build tool description
    tool_description = description or _build_tool_description(
        index_pattern=index_pattern,
        index_description=index_description,
    )
    
    # Append skill content if provided
    if skill:
        tool_description = f"{tool_description}\n\n---\n\n{skill}"
    
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
            from_time: Start time in date math (e.g., now-7d, now-24h) (default: now-24h)
            to_time: End time in date math (e.g., now-7d, now-24h) (default: now)
            max_results_override: Override default max results
            
        Returns:
            Formatted string with matching documents.
        """
        if not query or not query.strip():
            return "Please provide a non-empty Lucene query."
        
        # Apply hard limit to prevent overwhelming the LLM context
        requested_max = max_results_override or max_results
        effective_max = min(requested_max, MAX_RESULTS_HARD_LIMIT)
        if requested_max > MAX_RESULTS_HARD_LIMIT:
            logger.warning(
                "Requested %d results, capping to %d", 
                requested_max, MAX_RESULTS_HARD_LIMIT
            )
        
        try:
            response = client.search_with_lucene(
                lucene_query=query.strip(),
                index=index_pattern,
                from_time=from_time,
                to_time=to_time,
                time_field=time_field,
                size=effective_max,
            )
            output = _format_opensearch_response(
                response=response,
                query=query.strip(),
                index_pattern=index_pattern,
                max_results=effective_max,
                from_time=from_time,
                to_time=to_time,
            )
            
            # Safety truncation to prevent overwhelming the LLM context
            if len(output) > MAX_OUTPUT_CHARS:
                truncated = output[:MAX_OUTPUT_CHARS]
                # Find last complete document boundary
                last_doc_marker = truncated.rfind("\n[")
                if last_doc_marker > MAX_OUTPUT_CHARS // 2:
                    truncated = truncated[:last_doc_marker]
                output = truncated + f"\n\n... [Output truncated. Showing partial results. Use more specific queries or reduce max_results.]"
                logger.warning("Output truncated from %d to %d chars", len(output), len(truncated))
            
            return output
            
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


def _build_aggregation_tool_description(
    index_pattern: str,
    index_description: str,
) -> str:
    """
    Build tool description for the aggregation tool.
    
    Args:
        index_pattern: The OpenSearch index pattern.
        index_description: Human-readable description of what the index contains.
        
    Returns:
        Tool description string.
    """
    lines = [
        f"Aggregate and summarize data from OpenSearch index '{index_pattern}'.",
        "Use this tool for counting, grouping, and statistical analysis instead of fetching individual documents.",
    ]
    
    if index_description:
        lines.append(index_description)
    
    lines.append("")
    lines.append("Input parameters:")
    lines.append("- query: Lucene query string to filter documents before aggregation. Use '*' for all documents.")
    lines.append("- group_by: Field to aggregate on (e.g., 'data.reason', 'data.src_rse').")
    lines.append("- agg_type: Aggregation type (default: 'terms'). Options:")
    lines.append("    'terms' - Count documents by unique values (most common)")
    lines.append("    'sum' - Sum of numeric field values")
    lines.append("    'avg' - Average of numeric field values")
    lines.append("    'min' / 'max' - Minimum/maximum value")
    lines.append("    'cardinality' - Count unique values")
    lines.append("- top_n: Number of top buckets for 'terms' aggregation (default: 10, max: 100).")
    lines.append("- from_time: Start time (default: 'now-24h'). Supports date math (e.g., now-7d, now-24h).")
    lines.append("- to_time: End time (default: 'now'). Supports date math (e.g., now-7d, now-24h).")
    
    lines.append("")
    lines.append("Example queries:")
    lines.append("- 'What are the top transfer errors?' -> query='data.event_type:transfer-failed', group_by='data.reason', agg_type='terms'")
    lines.append("- 'How many transfers per RSE?' -> query='data.event_type:transfer-done', group_by='data.dst_rse', agg_type='terms'")
    lines.append("- 'Total bytes transferred?' -> query='data.event_type:transfer-done', group_by='data.bytes', agg_type='sum'")
    
    return "\n".join(lines)


# Maximum buckets for aggregation
MAX_AGG_BUCKETS = 100


def create_monit_opensearch_aggregation_tool(
    client: MONITOpenSearchClient,
    *,
    index_pattern: str,
    name: str = "aggregate_opensearch",
    index_description: str = "",
    time_field: str = "metadata.timestamp",
    description: Optional[str] = None,
    skill: Optional[str] = None,
) -> Callable[..., str]:
    """
    Create a LangChain tool for aggregating data from an OpenSearch index.
    
    This tool is designed for counting, grouping, and statistical analysis.
    It returns aggregation results instead of individual documents.
    
    Args:
        client: MONITOpenSearchClient instance.
        index_pattern: The OpenSearch index pattern to query (required).
        name: Tool name for LangChain.
        index_description: Human-readable description of what the index contains.
        time_field: Field to use for time range filtering.
        description: Full tool description (overrides auto-generated if provided).
        skill: Optional markdown content with domain expertise for this tool.
               Provides field documentation and query guidance to the LLM.
        
    Returns:
        LangChain tool function.
    """
    # Build tool description
    tool_description = description or _build_aggregation_tool_description(
        index_pattern=index_pattern,
        index_description=index_description,
    )
    
    # Append skill content if provided
    if skill:
        tool_description = f"{tool_description}\n\n---\n\n{skill}"
    
    @tool(name, description=tool_description)
    def _aggregate_opensearch(
        query: str,
        group_by: str,
        agg_type: str = "terms",
        top_n: int = 10,
        from_time: str = "now-24h",
        to_time: str = "now",
    ) -> str:
        """
        Aggregate OpenSearch data by grouping and counting/summing values.
        
        Args:
            query: Lucene query string to filter documents (use '*' for all)
            group_by: Field to aggregate on
            agg_type: Type of aggregation (terms, sum, avg, min, max, cardinality)
            top_n: Number of top buckets for terms aggregation
            from_time: Start time in date math (e.g., now-7d, now-24h) (default: now-24h)
            to_time: End time in date math (e.g., now-7d, now-24h) (default: now)
            
        Returns:
            Formatted string with aggregation results.
        """
        if not query or not query.strip():
            return "Please provide a query to filter documents (use '*' for all documents)."
        
        if not group_by or not group_by.strip():
            return "Please provide a field to aggregate on (group_by parameter)."
        
        # Validate aggregation type
        valid_agg_types = ("terms", "sum", "avg", "min", "max", "cardinality")
        if agg_type not in valid_agg_types:
            return f"Invalid aggregation type '{agg_type}'. Valid options: {', '.join(valid_agg_types)}"
        
        # Cap the number of buckets
        effective_top_n = min(top_n, MAX_AGG_BUCKETS)
        if top_n > MAX_AGG_BUCKETS:
            logger.warning("Requested %d buckets, capping to %d", top_n, MAX_AGG_BUCKETS)
        
        try:
            response = client.search_with_aggregation(
                lucene_query=query.strip(),
                agg_field=group_by.strip(),
                index=index_pattern,
                agg_type=agg_type,
                agg_size=effective_top_n,
                from_time=from_time,
                to_time=to_time,
                time_field=time_field,
            )
            
            output = _format_aggregation_response(
                response=response,
                query=query.strip(),
                index_pattern=index_pattern,
                agg_field=group_by.strip(),
                agg_type=agg_type,
                from_time=from_time,
                to_time=to_time,
            )
            
            return output
            
        except requests.exceptions.Timeout:
            logger.warning("OpenSearch aggregation timed out for query: %s", query)
            return (
                "Aggregation timed out. The service may be slow or the query too broad. "
                "Try narrowing the time range or making the query more specific."
            )
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "unknown"
            logger.warning("OpenSearch HTTP error %s for aggregation: %s", status_code, query)
            if status_code == 401 or status_code == 403:
                return "Authentication failed. The token may be invalid or expired."
            return f"Aggregation failed with HTTP error {status_code}. Please try again."
        except ValueError as e:
            return str(e)
        except Exception as e:
            logger.error("OpenSearch aggregation error: %s", e, exc_info=True)
            return f"Error performing aggregation: {str(e)}"
    
    return _aggregate_opensearch


__all__ = [
    "MONITOpenSearchClient",
    "create_monit_opensearch_search_tool",
    "create_monit_opensearch_aggregation_tool",
]
