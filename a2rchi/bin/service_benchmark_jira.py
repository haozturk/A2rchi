"""
Modified benchmarker for JIRA documents that checks document filenames directly
instead of requiring a sources.yml mapping file.

This is a simplified version for JIRA-based benchmarking where "links" are
actually document filenames (e.g., "jira_CMSTRANSF-1078.txt").
"""

from a2rchi.bin.service_benchmark import Benchmarker as BaseBenchmarker
from typing import Dict
import os


class JIRABenchmarker(BaseBenchmarker):
    """
    Extended benchmarker that can handle JIRA document filename matching
    without requiring a sources.yml file.
    """
    
    def get_link_results(self, result: Dict, expected_doc):
        """
        Modified version that checks if the expected document filename
        appears in the retrieved sources.
        
        Args:
            result: The chain result containing documents
            expected_doc: Expected document filename (e.g., "jira_CMSTRANSF-1078.txt")
        
        Returns:
            Tuple of (result_dict, match_bool)
        """
        res = {}
        sources = result['documents']
        
        num_sources = len(sources)
        match = False
        sources_found = []
        
        for k in range(num_sources):
            document = sources[k]
            # Get the source from metadata
            document_source = document.metadata.get('source', '')
            
            # Extract just the filename if it's a path
            if '/' in document_source:
                document_source = document_source.split('/')[-1]
            
            # Add to list of sources found
            if document_source and document_source not in sources_found:
                sources_found.append(document_source)
            
            # Check if this matches the expected document
            # Support both exact match and partial match (e.g., with/without .txt)
            if expected_doc in document_source or document_source in expected_doc:
                match = True
        
        result_dict = {
            True: "DOCUMENT FOUND",
            False: "NOT FOUND"
        }
        
        res["expected_document"] = expected_doc
        res['documents_retrieved'] = sources_found
        res['document_result'] = result_dict[match]
        
        return res, match


# For backwards compatibility when importing
if __name__ == "__main__":
    from pathlib import Path
    import json
    
    query_file = Path("QandA.txt")
    configs_folder = Path('configs')
    
    question_to_answer = {}
    
    with open(Path(query_file), "r") as f:
        obj = json.load(f)
    
    for d in obj:
        # Support both 'link' (original) and 'expected_doc' (JIRA version)
        link_or_doc = d.get('link', d.get('expected_doc', 'unknown'))
        question_to_answer[d['question']] = (link_or_doc, d.get('answer', ''))
    
    benchmarker = JIRABenchmarker(configs_folder, question_to_answer)
    benchmarker.run()
    
    from a2rchi.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("\n\nFINISHED RUNNING\n\n")







