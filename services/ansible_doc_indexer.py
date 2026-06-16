import logging
from pathlib import Path
from typing import List

from langfuse.openai import OpenAI
from qdrant_client.models import PointStruct

from services.mongo_client import MongoStorage
from services.qdrant_client import QdrantStorage
from utils.index import (
    make_id,
    build_embedding_text,
    extract_chunks,
    split_chunk_for_embedding,
)

logger = logging.getLogger(__name__)


class AnsibleDocIndexer:
    def __init__(
        self,
        repo_path: str,
        mongo_storage: MongoStorage,
        qdrant_storage: QdrantStorage,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.repo_path = repo_path
        self.mongo = mongo_storage
        self.qdrant = qdrant_storage
        self.embedding_model = embedding_model

        logger.info(
            "Initializing AnsibleDocIndexer repo=%s qdrant=%s mongo=%s",
            repo_path,
            qdrant_storage.collection_name,
            mongo_storage.collection_name,
        )

        self.client = OpenAI()
        self.mongo.ensure_indexes()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def index_file(self, file_path: str) -> None:
        chunks = extract_chunks(file_path)
        if not chunks:
            return

        try:
            relative_path = str(Path(file_path).relative_to(self.repo_path))
        except ValueError:
            relative_path = file_path

        segment_chunks = []
        mongo_docs = []
        for chunk in chunks:
            chunk_id = make_id(f"{relative_path}:{chunk['symbol']}:{chunk['start_line']}")
            mongo_docs.append({
                "_id": chunk_id,
                "file_path": relative_path,
                "symbol": chunk["symbol"],
                "chunk_type": chunk["type"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "content": chunk["code"],
            })

            split_segments = split_chunk_for_embedding(chunk)
            for segment_index, segment in enumerate(split_segments, start=1):
                segment_chunks.append({
                    **segment,
                    "chunk_id": chunk_id,
                    "segment_index": segment_index,
                })

        texts = [build_embedding_text(relative_path, segment) for segment in segment_chunks]
        logger.info("Indexing ansible doc %s (%d segments)", relative_path, len(texts))

        vectors = self.embed_texts(texts)

        points = []
        for segment, vector in zip(segment_chunks, vectors):
            point_id = make_id(
                f"{segment['chunk_id']}:seg{segment['segment_index']}"
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "file_path": relative_path,
                        "symbol": segment["symbol"],
                        "chunk_type": segment["type"],
                        "mongo_id": segment["chunk_id"],
                    },
                )
            )

        self.qdrant.upsert_points(points)
        self.mongo.upsert_documents(mongo_docs)
        logger.info("Indexed ansible doc %s (%d chunks, %d segments)", relative_path, len(chunks), len(texts))

    def index_repo(self) -> None:
        logger.info("Starting ansible docs indexing: %s", self.repo_path)

        for path in Path(self.repo_path).rglob("*.rst"):
            try:
                if ".git" in path.parts:
                    continue

                if path.is_file():
                    self.index_file(str(path))
            except Exception:
                logger.exception("Failed file %s", path)

        logger.info("Completed ansible docs indexing: %s", self.repo_path)
