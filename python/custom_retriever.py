from typing import List, Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever, Field


class CustomRetriever(VectorStoreRetriever):
    vectorstore: VectorStoreRetriever
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.vectorstore.get_relevant_documents(query=query)
        filter_value = self.search_kwargs['filter']['id']
        return [doc for doc in results if doc.metadata['id'] == int(filter_value)]
