
This is the repository for ACL 2026 main paper "Attention Weights as an Indicator: Analyzing and Improving Document Utilization in Retrieval-Augmented Generation".

## Data Preparation
Datasets we used in the experiments are from <https://github.com/panruotong/CAG>.
We primarily used the open-domain datasets included in the collection: HotpotQA, Musique, and 2wikiMultiHopQA.

## Inference
```
python aw.py --model_path meta-llama/Llama-3.1-8B --model_type llama3_8b --data_path datasets_processed --setting_type rerank --temperature 0.01 --hotpot --musique --wikimulti --processed --zero_shot --inference_mode vanilla
python aw.py --model_path meta-llama/Llama-3.1-8B --model_type llama3_8b --data_path datasets_processed --setting_type rerank --temperature 0.01 --hotpot --musique --wikimulti --processed --zero_shot --inference_mode dr
```

## Evaluation
The final stage of the inference process includes evaluation. If additional evaluation is required on the generated result file:
```
python metrics.py
```

## Citation
```
@inproceedings{jin-etal-2026-attention,
    title = "Attention Weights as an Indicator: Analyzing and Improving Document Utilization in Retrieval-Augmented Generation",
    author = "Jin, Jing  and
      Song, Yuhan  and
      Luo, Wen  and
      Wang, Houfeng",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1245/",
    pages = "27034--27052",
    ISBN = "979-8-89176-390-6"
}
```