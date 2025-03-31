# ENV

create a `.env` file by `touch .env`, add the following content, and change `??????` variables by changing to your own ones.

```bash
WANDB_API_KEY==?????????
WANDB_PROJECT=m1
WANDB_MODE=online
WANDB_ENTITY==?????????

HF_HOME=cache/

OPENAI_API_KEY=?????????
CURATOR_CACHE_DIR=cache/curator
```

## Conda

If you do not have conda, install mini conda by the following commands. Change `??????` to your target path:

```shell
BASE_DIR=??????

cd $BASE_DIR
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -p $BASE_DIR/miniconda3

# source ~/.bashrc
source activate base
rm Miniconda3-latest-Linux-x86_64.sh
```



Install pytorch and other libraries. We assume the cuda is 12.4.


```bash
conda create -y -n m1 python=3.10
source activate
conda activate m1

pip install --upgrade pip
pip install uv
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

uv pip install -r requirements.m1.txt

# bug of 0.4.4.post1: https://github.com/sgl-project/sglang/issues/4404
uv pip install "sglang[all]==0.4.3.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python


uv pip install packaging ninja
ninja --version
if [[ $? = 0 ]]; then
    echo "Installing flash-attn"
    uv pip install flash-attn --no-build-isolation
else
    echo "Ninja Error, not installing"
fi
```


### Download Eval data

```bash
REPO_TYPE=dataset # model, dataset
LOCAL_DIR=misc/

# https://huggingface.co/datasets/UCSC-VLAA/m1_eval_data
REPO_URL=UCSC-VLAA/m1_eval_data

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
```