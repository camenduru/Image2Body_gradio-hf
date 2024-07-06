import os
from huggingface_hub import hf_hub_download

def download_file(filename, subfolder=None):
    print(f'Downloading {filename} from Hugging Face Hub...')
    return hf_hub_download(
        repo_id=os.environ['REPO_ID'],
        filename=filename,
        subfolder=subfolder,
        token=os.environ['HF_TOKEN'],
        cache_dir=os.environ['CACHE_DIR']
    )
