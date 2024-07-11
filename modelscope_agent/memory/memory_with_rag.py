from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import json
from modelscope_agent.agent import Agent
from modelscope_agent.llm.base import BaseChatModel
from modelscope_agent.rag.knowledge import BaseKnowledge
from modelscope_agent.storage import KnowledgeVector
from modelscope_agent.utils.logger import agent_logger as logger

from .base import Memory, enable_rag_callback


class MemoryWithRag(Memory, Agent):

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 storage_path: Optional[Union[str, Path]] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 use_knowledge_cache: bool = True,
                 urls: List[str] = [],
                 **kwargs):
        Memory.__init__(self, path=kwargs.get('memory_path', ''))
        Agent.__init__(
            self,
            function_list=function_list,
            llm=llm,
            name=name,
            description=description,
            stream=False,
            **kwargs)

        # allow vector storage to save knowledge
        self.store_knowledge = BaseKnowledge(
            urls,
            llm=llm,
            cache_dir=storage_path,
            use_cache=use_knowledge_cache,
            **kwargs)

    @enable_rag_callback
    def _run(self,
             query: str = None,
             url: str = None,
             max_token: int = 4000,
             **kwargs) -> Union[str, Iterator[str]]:
        if isinstance(url, str):
            url = [url]
        if url and len(url):
            self.store_knowledge.add(files=url)
        if query:
            summary_result = self.store_knowledge.run(
                query, files=url, **kwargs)
            # limit length
            if isinstance(summary_result, list):
                if len(summary_result) == 0:
                    return ''
                single_max_token = int(max_token / len(summary_result))
                concatenated_records = '\n'.join([
                    record[0:single_max_token - 1] for record in summary_result
                ])

                return concatenated_records
            else:
                return summary_result[0:max_token - 1]
