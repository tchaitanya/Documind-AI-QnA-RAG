import os
import time
from typing import List, Optional, AsyncGenerator

from langchain_community.vectorstores import AzureSearch as AzureSearchVS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import PromptTemplate


class RAGPipeline:
    def __init__(
        self,
        *,
        azure_openai_endpoint: str,
        azure_openai_api_key: str,
        chat_deployment: str,
        embed_deployment: str,
        search_endpoint: str,
        search_key: str,
        search_index: str,
        prompt_template: str,
        api_version: str = "2024-08-01-preview",
    ) -> None:
        """Initialize RAG pipeline clients and prompt template."""
        self.azure_openai_endpoint = azure_openai_endpoint
        self.azure_openai_api_key = azure_openai_api_key
        self.chat_deployment = chat_deployment
        self.embed_deployment = embed_deployment
        self.search_endpoint = search_endpoint
        self.search_key = search_key
        self.search_index = search_index
        self.api_version = api_version

        self.prompt_template = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=self.azure_openai_endpoint,
            api_key=self.azure_openai_api_key,
            azure_deployment=self.embed_deployment,
            api_version=self.api_version,
        )
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.azure_openai_endpoint,
            api_key=self.azure_openai_api_key,
            azure_deployment=self.chat_deployment,
            api_version=self.api_version,
        )

        self._vectorstore = None

    def get_vectorstore(self):
        """Return (or create) the Azure AI Search vector store using the embedding client."""
        if self.embeddings is None:
            raise ValueError("Embeddings client is not initialized; check AZURE_OPENAI_EMBED_DEPLOYMENT and API key")
        if self._vectorstore is None:
            self._vectorstore = AzureSearchVS(
                azure_search_endpoint=self.search_endpoint,
                azure_search_key=self.search_key,
                index_name=self.search_index,
                embedding_function=self.embeddings.embed_query,
            )
            print(f"✅ Vectorstore initialized: {self.search_index}")
        return self._vectorstore

    def _search(self, query: str, top_k: int):
        """Run hybrid search with vector+keyword, falling back to vector-only if needed."""
        vs = self.get_vectorstore()
        try:
            return vs.similarity_search(query, k=top_k, search_type="hybrid")
        except Exception:
            return vs.similarity_search(query, k=top_k)

    @staticmethod
    def _build_context(docs) -> str:
        """Concatenate retrieved docs into a single context string with source labels."""
        return "\n\n".join(
            [f"Document: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs]
        )

    @staticmethod
    def _dedupe_sources(docs) -> List[dict]:
        """Merge chunks by source name to keep a single entry per document."""
        sources_dict = {}
        for d in docs:
            source_name = d.metadata.get("source", "")
            if source_name not in sources_dict:
                sources_dict[source_name] = d.page_content
            else:
                sources_dict[source_name] += f"\n\n---\n\n{d.page_content}"
        return [{"source": name, "content": content} for name, content in sources_dict.items()]

    def chat(self, query: str, top_k: int):
        start = time.time()
        docs = self._search(query, top_k)
        search_time = time.time() - start

        context = self._build_context(docs)
        prompt_text = self.prompt_template.format(context=context, question=query)

        llm_start = time.time()
        answer = self.llm.invoke(prompt_text).content
        llm_time = time.time() - llm_start

        sources = self._dedupe_sources(docs)
        ungrounded_phrases = (
            "i don't have",
            "not in the",
            "no information",
            "cannot find",
            "not available",
            "not mentioned",
            "not provided",
        )
        is_grounded = not any(p in answer.lower() for p in ungrounded_phrases)

        total_time = time.time() - start
        reasoning_log = [
            {
                "phase": "Retrieval",
                "details": f"Retrieved {len(docs)} documents using hybrid search",
                "duration": f"{search_time:.2f}s",
            },
            {
                "phase": "Generation",
                """Retrieve, build context, generate an answer, and return answer + sources + reasoning."""
                "details": f"Model: {self.chat_deployment} | Context size: {len(context)} chars",
                "duration": f"{llm_time:.2f}s",
            },
            {
                "phase": "Grounding",
                "details": f"Answer is {'grounded in documents' if is_grounded else 'missing information'}",
                "duration": "N/A",
            },
            {
                "phase": "Total", 
                "details": "End-to-end response time", 
                "duration": f"{total_time:.2f}s"
            },
        ]

        return {
            "answer": answer,
            "sources": sources,
            "agentic": is_grounded,
            "reasoning_log": reasoning_log,
        }

    async def stream_chat(self, query: str, top_k: int) -> AsyncGenerator[bytes, None]:
        """Stream the LLM answer text while using the same retrieval + prompt context."""
        docs = self._search(query, top_k)
        context = self._build_context(docs)
        prompt_text = self.prompt_template.format(context=context, question=query)

        streaming_llm = AzureChatOpenAI(
            azure_endpoint=self.azure_openai_endpoint,
            api_key=self.azure_openai_api_key,
            azure_deployment=self.chat_deployment,
            streaming=True,
            api_version=self.api_version,
        )

        try:
            async for chunk in streaming_llm.astream(prompt_text):
                if chunk.content:
                    yield chunk.content.encode("utf-8")
        except Exception as e:
            yield f"\n\nError: {str(e)}".encode("utf-8")
