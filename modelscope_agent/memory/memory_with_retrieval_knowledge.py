from typing import Dict, Iterator, List, Optional, Union

import json
from modelscope_agent.agent import Agent
from modelscope_agent.llm.base import BaseChatModel
from modelscope_agent.storage import KnowledgeVector

from .base import Memory


class MemoryWithRetrievalKnowledge(Memory, Agent):

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 storage_path: Optional[str] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 **kwargs):
        Memory.__init__(self, path=kwargs.get('memory_path', ''))
        Agent.__init__(
            self,
            function_list=function_list,
            llm=llm,
            name=name,
            description=description)

        # allow vector storage to save knowledge
        self.store_knowledge = KnowledgeVector(storage_path, name)

    def _run(self,
             query: str = None,
             url: str = None,
             max_token: int = 4000,
             **kwargs) -> Union[str, Iterator[str]]:
        # no need for llm in this agent yet, all the operation could be handled by simple logic
        if url:
            try:
                url = json.loads(url)
            except json.JSONDecodeError:
                pass

            # add file to index
            self.store_knowledge.add(url)

            # save store knowledge
            self.store_knowledge.save()

        # no query then return None
        if not query:
            return None

        # search records
        records = self.store_knowledge.search(query)

        # limit length
        concatenated_records = '\n'.join(records)
        if len(concatenated_records) > max_token:
            single_max_token = int(max_token / len(records))
            concatenated_records = '\n'.join(
                [record[0:single_max_token - 1] for record in records])

        return concatenated_records
