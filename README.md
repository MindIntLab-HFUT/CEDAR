# <img src="figs/cedar.png" alt="cedar logo" style="height: 1.2em; vertical-align: middle;"> CEDAR Benchmark

<a href="https://arxiv.org/abs/2601.13024"><img src="https://img.shields.io/badge/arXiv-2601.13024-red"></a>
<a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>

Official repository for the ACL 2026 paper **[Tears or Cheers? Benchmarking LLMs via Culturally Elicited Distinct Affective Responses](https://arxiv.org/abs/2601.13024)**.

🔥 **[2026.04]** Accepted to **ACL 2026** as an **SAC Highlight**!

## Overview

CEDAR tests whether models can predict how people from different cultures emotionally respond to the same scenario. It has 10,962 instances in 7 languages (Arabic, Chinese, English, Hindi, Japanese, Spanish, Swahili), labeled with 14 emotions.

| Subset | Input | Per language | Total |
| --- | --- | ---: | ---: |
| Text-only | Narrative + Question | 1,166 | 8,162 |
| Multimodal | Image + Narrative + Question | 400 | 2,800 |

```text
data/
├── text-only/{original, multilingual}/
└── multimodal/{original, multilingual, images}/
```

`original/` holds the English scenarios with ground-truth labels for all 7 languages. `multilingual/` holds one file per language, which is what models are evaluated on.

## Evaluation

```bash
pip install -r requirements.txt

# Text-only
python evaluation/run_benchmark_textonly.py --mode api \
  --base_url https://api.openai.com/v1 --api_key $OPENAI_API_KEY --model_name gpt-4o

# Multimodal
python evaluation/run_benchmark_multimodal.py --mode api \
  --base_url https://api.openai.com/v1 --api_key $OPENAI_API_KEY --model_name gpt-4o
```

- `--mode api` works with any OpenAI-compatible endpoint, including a local `vllm serve` (see [`scripts/`](scripts/)). For text-only, `--mode vllm --model_path /path/to/model` runs vLLM offline.
- Results go to `results/` and include per-language accuracy and per-item outputs. Re-running the same command resumes an interrupted run.

## Citation

```bibtex
@misc{dai2026tearscheersbenchmarkingllms,
      title={Tears or Cheers? Benchmarking LLMs via Culturally Elicited Distinct Affective Responses},
      author={Chongyuan Dai and Yaling Shen and Jinpeng Hu and Zihan Gao and Jia Li and Yishun Jiang and Yaxiong Wang and Liu Liu and Zongyuan Ge},
      year={2026},
      eprint={2601.13024},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.13024},
}
```
