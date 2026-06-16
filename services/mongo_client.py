import logging
from typing import Iterable, Mapping

from pymongo import MongoClient

logger = logging.getLogger(__name__)


class MongoStorage:
    def __init__(self, uri: str, db_name: str, collection_name: str):
        self.client = MongoClient(uri)
        self.collection = self.client[db_name][collection_name]
        self.collection_name = collection_name

    def ensure_indexes(self, keys: list[str] | None = None) -> None:
        if keys is None:
            keys = ["file_path"]

        for key in keys:
            try:
                self.collection.create_index(key)
                logger.info("Created index on %s for collection %s", key, self.collection_name)
            except Exception as exc:
                logger.warning("Failed to create index on %s for collection %s: %s", key, self.collection_name, exc)

    def upsert_documents(self, documents: Iterable[Mapping]) -> None:
        for document in documents:
            self.collection.replace_one({"_id": document["_id"]}, document, upsert=True)
