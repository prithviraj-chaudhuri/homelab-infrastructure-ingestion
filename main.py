import os
import shutil
from dotenv import load_dotenv
import logging
from git import Repo
import utils.code

logger = logging.getLogger(__name__)
load_dotenv()

def main():
    git_url = os.getenv("HOMELAB_GIT_URL")
    target_dir = os.getenv("LOCAL_GIT_PATH")
    if not git_url or not target_dir:
        logger.error("HOMELAB_GIT_URL and LOCAL_GIT_PATH must be set in the environment variables.")
        return
    utils.code.clone_repo(git_url, target_dir)

if __name__ == "__main__":
    main()
