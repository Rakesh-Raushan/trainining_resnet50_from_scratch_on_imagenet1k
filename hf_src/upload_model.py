from huggingface_hub import HfApi, login
import os

# Login to Hugging Face
login()  # This will ask for your token

# Initialize the API
api = HfApi()

# Create a new model repository
repo_id = "Perpetualquest/resnet50_explorer_v1"  # e.g., "johndoe/resnet50-custom"

# Create the repo (set private=True if you want it private)
api.create_repo(repo_id, repo_type="model", private=False)

# Upload the model file
api.upload_file(
    path_or_fileobj="./checkpoint.pth",
    path_in_repo="checkpoint.pth",
    repo_id=repo_id
) 