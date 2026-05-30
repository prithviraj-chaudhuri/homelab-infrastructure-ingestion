import os
import shutil
from dotenv import load_dotenv
import logging
from git import Repo
import utils.code
from services.code_indexer import CodeIndexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

def main():
    git_url = os.getenv("HOMELAB_GIT_URL")
    target_dir = os.getenv("LOCAL_GIT_PATH")

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    utils.code.clone_repo(git_url, target_dir)

    indexer = CodeIndexer(
        repo_path=target_dir,
        mongo_uri=mongo_uri,
        mongo_db="rag",
        mongo_collection="code_chunks",
        qdrant_url=qdrant_url,
        qdrant_collection="codebase",
        embedding_model=embedding_model,
    )
    indexer.index_repo()

if __name__ == "__main__":
    main()
