from git import Repo
import os
import shutil
import logging

logger = logging.getLogger(__name__)

def clone_repo(git_url, target_dir):
    logger.info(f"Cloning repository from {git_url} to {target_dir}")

    if os.path.exists(target_dir):
        logger.warning(f"Target directory {target_dir} already exists. It will be removed before cloning.")
        shutil.rmtree(target_dir)
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    cloned_repo = Repo.clone_from(git_url, target_dir)
    logger.info(f"Repository cloned successfully to {target_dir}")