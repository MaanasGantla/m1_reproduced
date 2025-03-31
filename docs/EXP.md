# EXP

## Remove unfinished exp

```bash
find outputs -type d -exec sh -c '[ -f "{}/config.yaml" ] && [ ! -f "{}/metrics.json" ] && echo "{}"' \;

find outputs -type d -exec sh -c '[ -f "{}/config.yaml" ] && [ ! -f "{}/metrics.json" ] && rm -r "{}"' \;

find outputs -type d -empty 

find outputs -type d -empty -exec rmdir {} \;
```

## Sync outputs to hf

https://huggingface.co/docs/huggingface_hub/v0.17.1/en/guides/upload

Only upload `seed*temperature*/` folder.

```bash
REPO_TYPE=dataset # model, dataset
NUM_WORKERS=8

LOCAL_DIR=outputs
REPO_URL=UCSC-VLAA/m1-results

huggingface-cli upload-large-folder \
--repo-type="${REPO_TYPE}" \
--num-workers="${NUM_WORKERS}" \
--include='seed*temperature*/' \
"$REPO_URL" "$LOCAL_DIR"
```

delete files on hf: https://github.com/huggingface/huggingface_hub/issues/2235


Download results from hf

```bash
REPO_TYPE=dataset # model, dataset
NUM_WORKERS=8

LOCAL_DIR=outputs
REPO_URL=UCSC-VLAA/m1-results

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
```

Get tree:

```bash
tree outputs -I 'version_*'
```