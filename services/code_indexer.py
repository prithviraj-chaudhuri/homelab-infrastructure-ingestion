import logging
from pathlib import Path
from typing import List, Dict

from pymongo import MongoClient

from langfuse.openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

from utils.index import make_id, build_embedding_text, extract_chunks

logger = logging.getLogger(__name__)


class CodeIndexer:
    def __init__(
        self,
        repo_path: str,
        mongo_uri: str,
        mongo_db: str,
        mongo_collection: str,
        qdrant_url: str,
        qdrant_collection: str,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.repo_path = repo_path
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.mongo_collection = mongo_collection
        self.qdrant_url = qdrant_url
        self.qdrant_collection = qdrant_collection
        self.embedding_model = embedding_model

        self.mongo = MongoClient(self.mongo_uri)
        self.collection = self.mongo[self.mongo_db][self.mongo_collection]

        logger.info(
            "Initializing CodeIndexer repo=%s qdrant=%s mongo=%s",
            repo_path,
            qdrant_collection,
            mongo_db,
        )

        self.client = OpenAI()
        
        self.qdrant_client = QdrantClient(url=self.qdrant_url)
        self._ensure_qdrant_collection(self.qdrant_collection)
        self._ensure_mongo_indexes()


    def _ensure_qdrant_collection(self, collection_name: str) -> None:
        try:
            self.qdrant_client.get_collection(collection_name)
        except Exception:
            logger.info("Creating Qdrant collection %s", collection_name)
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,  # text-embedding-3-small
                    distance=Distance.COSINE,
                ),
            )

    def _ensure_mongo_indexes(self) -> None:
        """Create indexes on the MongoDB collection."""
        try:
            self.collection.create_index("file_path")
            logger.info("Created index on file_path for collection %s", self.mongo_collection)
        except Exception as e:
            logger.warning("Failed to create index on file_path: %s", e)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Langfuse-traced embeddings via OpenAI SDK wrapper.
        Fully avoids LangChain callback issues.
        """

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )

        return [item.embedding for item in response.data]

    def index_file(self, file_path: str) -> None:
        chunks = extract_chunks(file_path)

        if not chunks:
            return

        file_path = file_path.lstrip("files/code/")
        texts = [build_embedding_text(file_path, c) for c in chunks]

        logger.info("Indexing %s (%d chunks)", file_path, len(chunks))

        vectors = self.embed_texts(texts)

        mongo_docs = []

        for chunk, vector, text in zip(chunks, vectors, texts):

            chunk_id = make_id(
                f"{file_path}:{chunk['symbol']}:{chunk['start_line']}"
            )
            mongo_doc = {
                "_id": chunk_id,
                "file_path": file_path,
                "symbol": chunk["symbol"],
                "chunk_type": chunk["type"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "content": chunk["code"],
            }

            mongo_docs.append(mongo_doc)

            self.qdrant_client.upsert(
                collection_name=self.qdrant_collection,
                points=[
                    PointStruct(
                        id=chunk_id,
                        vector=vector,
                        payload={
                            "file_path": file_path,
                            "symbol": chunk["symbol"],
                            "chunk_type": chunk["type"],
                            "mongo_id": chunk_id,
                        },
                    )
                ],
            )

        for doc in mongo_docs:
            self.collection.replace_one(
                {"_id": doc["_id"]},
                doc,
                upsert=True,
            )

        logger.info("Indexed %s (%d chunks)", file_path, len(chunks))

    def index_repo(self) -> None:
        logger.info("Starting repo indexing: %s", self.repo_path)

        for path in Path(self.repo_path).rglob("*"):
            try:
                if ".git" in path.parts:
                    continue

                if path.is_file():
                    self.index_file(str(path))
            except Exception:
                logger.exception("Failed file %s", path)

        logger.info("Completed repo indexing: %s", self.repo_path)