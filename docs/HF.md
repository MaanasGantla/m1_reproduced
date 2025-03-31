# HF

## Set up

```shell
pip install -U huggingface_hub hf_transfer

git config --global credential.helper store

# Login huggingface hub
# huggingface-cli login
# export HF_TOKEN=hf_YOUR_TOKEN

# Check login
set -a
source .env
set +a
huggingface-cli whoami
```


## Upload to Huggingface Hub

```shell
# Speed up with hf_transfer
# export HF_HUB_ENABLE_HF_TRANSFER=1 

REPO_TYPE=dataset # model, dataset
NUM_WORKERS=8

LOCAL_DIR=
REPO_URL=

huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}"
```

If you encounter `Failed to preupload LFS: Error while uploading huggingface`, then disable `hf_transfer` by `HF_HUB_ENABLE_HF_TRANSFER=0`.

```shell
# Single file
REPO_TYPE=dataset # model, dataset
NUM_WORKERS=1

FILE_PATH=
REPO_URL=xk-huang/$(date +%y%m%d-%H%M%S)-$(basename FILE_PATH)
huggingface-cli upload "$REPO_URL" --repo-type="${REPO_TYPE}" ${FILE_PATH}
```

## Download from HF

```shell
REPO_TYPE=dataset # model, dataset
LOCAL_DIR=outputs/ckpts

REPO_URL=

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
```
