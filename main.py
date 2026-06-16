import os
from dotenv import load_dotenv
import logging
import utils.code
from services.code_indexer import CodeIndexer
from services.ansible_doc_indexer import AnsibleDocIndexer
from services.mongo_client import MongoStorage
from services.qdrant_client import QdrantStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

def main():
    git_url = os.getenv("HOMELAB_GIT_URL")
    target_dir = os.getenv("LOCAL_GIT_PATH")

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    mongo_db = os.getenv("MONGO_DB", "rag")
    mongo_collection = os.getenv("MONGO_COLLECTION", "code_chunks")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "codebase")

    if git_url and target_dir:
        utils.code.clone_repo(git_url, target_dir)
        code_mongo = MongoStorage(mongo_uri, mongo_db, mongo_collection)
        code_qdrant = QdrantStorage(qdrant_url, qdrant_collection, 1536)

        indexer = CodeIndexer(
            repo_path=target_dir,
            mongo_storage=code_mongo,
            qdrant_storage=code_qdrant,
            embedding_model=embedding_model,
        )
        indexer.index_repo()
    else:
        logger.warning("HOMELAB_GIT_URL or LOCAL_GIT_PATH not set; skipping code repository indexing.")

    ansible_docs_git_url = os.getenv("ANSIBLE_DOCS_GIT_URL")
    ansible_docs_local_path = os.getenv("ANSIBLE_DOCS_LOCAL_PATH")
    ansible_docs_mongo_collection = os.getenv("ANSIBLE_DOCS_MONGO_COLLECTION", "ansible_docs")
    ansible_docs_qdrant_collection = os.getenv("ANSIBLE_DOCS_QDRANT_COLLECTION", "ansible_docs")

    if ansible_docs_git_url and ansible_docs_local_path:
        utils.code.clone_repo(ansible_docs_git_url, ansible_docs_local_path)
        ansible_mongo = MongoStorage(mongo_uri, mongo_db, ansible_docs_mongo_collection)
        ansible_qdrant = QdrantStorage(qdrant_url, ansible_docs_qdrant_collection, 1536)

        ansible_indexer = AnsibleDocIndexer(
            repo_path=ansible_docs_local_path,
            mongo_storage=ansible_mongo,
            qdrant_storage=ansible_qdrant,
            embedding_model=embedding_model,
        )
        ansible_indexer.index_repo()
    else:
        logger.info("ANSIBLE_DOCS_GIT_URL or ANSIBLE_DOCS_LOCAL_PATH not configured; skipping ansible docs indexing.")

if __name__ == "__main__":
    main()
