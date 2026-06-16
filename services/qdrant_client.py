import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

logger = logging.getLogger(__name__)


class QdrantStorage:
    def __init__(self, url: str, collection_name: str, vector_size: int):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            logger.info("Creating Qdrant collection %s", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_points(self, points: list[PointStruct]) -> None:
        if not points:
            return
        self.client.upsert(collection_name=self.collection_name, points=points)
