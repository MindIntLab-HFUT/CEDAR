# <img src="figs/cedar.png" alt="cedar logo" style="height: 1.2em; vertical-align: middle;"> CEDAR Benchmark

<a href="https://arxiv.org/abs/2601.13024"><img src="https://img.shields.io/badge/arXiv-2601.13024-red"></a>
<a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>

Official repository for the ACL 2026 paper **[Tears or Cheers? Benchmarking LLMs via Culturally Elicited Distinct Affective Responses](https://aclanthology.org/2026.acl-long.1769.pdf)**.

🔥 **[2026.07]** Our paper was selected as **ACL 2026 SAC Highlight**!

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

## Citation

```bibtex
@inproceedings{dai-etal-2026-tears,
    title = "Tears or Cheers? Benchmarking {LLM}s via Culturally Elicited Distinct Affective Responses",
    author = "Dai, Chongyuan  and
      Shen, Yaling  and
      Gao, Zihan  and
      Li, Jia  and
      Jiang, Yishun  and
      Wang, Yaxiong  and
      Liu, Liu  and
      Ge, Zongyuan  and
      Hu, Jinpeng",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1769/",
    doi = "10.18653/v1/2026.acl-long.1769",
    pages = "38171--38196",
    ISBN = "979-8-89176-390-6",
    abstract = "Culture serves as a fundamental determinant of human affective processing and profoundly shapes how individuals perceive and interpret emotional stimuli. Despite this intrinsic link extant evaluations regarding cultural alignment within Large Language Models primarily prioritize declarative knowledge such as geographical facts or established societal customs. These benchmarks remain insufficient to capture the subjective interpretative variance inherent to diverse sociocultural lenses. To address this limitation, we introduce CEDAR, a multimodal benchmark constructed entirely from scenarios capturing \textbf{C}ulturally \textbf{E}licited \textbf{D}istinct \textbf{A}ffective \textbf{R}esponses. To construct CEDAR, we implement a novel pipeline that leverages LLM-generated provisional labels to isolate instances yielding cross-cultural emotional distinctions, and subsequently derives reliable ground-truth annotations through rigorous human evaluation. The resulting benchmark comprises 10,962 instances across seven languages and 14 fine-grained emotion categories, with each language including 400 multimodal and 1,166 text-only samples. Comprehensive evaluations of 17 representative multilingual models reveal a dissociation between language consistency and cultural alignment, demonstrating that culturally grounded affective understanding remains a significant challenge for current models."
}
```
