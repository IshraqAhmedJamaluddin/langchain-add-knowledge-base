from typing import Any, Dict, List
from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
from langchain_core.documents import Document


class DeDuplicationRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query: str) -> List[Document]:
        emb = self.embeddings.embed_query(query)

        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8,
        )

    async def aget_relevant_documents(self) -> List[Document]:
        return []
