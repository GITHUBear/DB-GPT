import os
import logging
import threading
from typing import Any, List

from pydantic import Field

from dbgpt._private.config import Config
from dbgpt.rag.chunk import Chunk
from dbgpt.storage.vector_store.base import VectorStoreBase, VectorStoreConfig

logger = logging.getLogger(__name__)
sql_logger = None
sql_dbg_log_path = os.getenv("OB_SQL_DBG_LOG_PATH", "")
if sql_dbg_log_path != "":
    sql_logger = logging.getLogger('ob_sql_dbg')
    sql_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(sql_dbg_log_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    sql_logger.addHandler(file_handler)

CFG = Config()

class OceanBaseConfig(VectorStoreConfig):
    """OceanBase config"""
    connection_string: str = Field(
        default=None,
        description="the connection string of vector store, if not set, will use the default connection string.",
    )

ob_collection_stats_lock = threading.Lock()
ob_collection_stats = {}

class OceanBaseStore(VectorStoreBase):
    def __init__(self, vector_store_config: OceanBaseConfig) -> None:
        import langchain.vectorstores
        from langchain.vectorstores import OceanBase
        self.OB_HOST = os.getenv("OB_HOST", "127.0.0.1")
        self.OB_PORT = os.getenv("OB_PORT", 2881)
        self.OB_USER = os.getenv("OB_USER", "root")
        self.OB_PASSWORD = os.getenv("OB_PASSWORD", "")
        self.OB_DATABASE = os.getenv("OB_DATABASE", "test")
        self.connection_string = OceanBase.connection_string_from_db_params(self.OB_HOST, self.OB_PORT, self.OB_DATABASE, self.OB_USER, self.OB_PASSWORD)
        self.embeddings = vector_store_config.embedding_fn
        self.collection_name = vector_store_config.name
        self.logger = logger
        with ob_collection_stats_lock:
            if ob_collection_stats.get(self.collection_name) is None:
                ob_collection_stats[self.collection_name] = langchain.vectorstores.oceanbase.OceanBaseCollectionStat()
            self.collection_stat = ob_collection_stats[self.collection_name]                
        
        self.vector_store_client = OceanBase(
            connection_string = self.connection_string,
            embedding_function = self.embeddings,
            collection_name = self.collection_name,
            logger = self.logger,
            sql_logger = sql_logger,
            enable_index = False,
            collection_stat = self.collection_stat
        )

    def similar_search(self, text, topk, **kwargs: Any) -> List[Chunk]:
        self.logger.info("OceanBase: similar_search..")
        lc_documents = self.vector_store_client.similarity_search(text, topk, enable_subtitle=True)
        return [
            Chunk(content=doc.page_content, metadata=doc.metadata)
            for doc in lc_documents
        ]

    def similar_search_with_scores(self, text, topk, score_threshold) -> List[Chunk]:
        self.logger.info("OceanBase: similar_search_with_scores..")
        docs_and_scores = (
            self.vector_store_client.similarity_search_with_score(text, topk, enable_subtitle=True)
        )
        return [
            Chunk(content=doc.page_content, metadata=doc.metadata, score=score)
            for doc, score in docs_and_scores
        ]

    def vector_name_exists(self):
        self.logger.info("OceanBase: vector_name_exists..")
        try:
            self.vector_store_client.create_collection()
            return True
        except Exception as e:
            logger.error("vector_name_exists error", e.message)
            return False

    def load_document(self, chunks: List[Chunk]) -> List[str]:
        self.logger.info("OceanBase: load_document..")
        lc_documents = [Chunk.chunk2langchain(chunk) for chunk in chunks]
        subtitles = ['-'.join((list(chunk.metadata.values()))[:-1]) for chunk in chunks]
        texts = [d.page_content for d in lc_documents]
        metadatas = [d.metadata for d in lc_documents]
        ids = self.vector_store_client.add_texts(texts=texts, metadatas=metadatas, subtitles=subtitles)
        return ids

    def delete_vector_name(self, vector_name):
        self.logger.info("OceanBase: delete_vector_name..")
        return self.vector_store_client.delete_collection()

    def delete_by_ids(self, ids):
        self.logger.info("OceanBase: delete_by_ids..")
        ids = ids.split(",")
        if len(ids) > 0:
            self.vector_store_client.delete(ids, enable_subtitle=True)