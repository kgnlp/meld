<div align="center">
<img src="docs/logo.svg" width="275ch">
<h1>MELD: Melding Diverse Multilingual and Multi-Domain Datasets for Named Entity Recognition Evaluation</h1>

<nav>
    <a href="#installation">Installation</a> •
    <a href="#download-meld">Download MELD</a> •
    <a href="#convert-to-huggingface-datasets-format">Convert to HuggingFace Datasets Format</a> •
    <a href="#included-datasets">Included Datasets</a> •
    <a href="#citation">Citation</a>
<nav>
</div>

----------

MELD is a multilingual and multi-domain dataset for Named Entity Recognition (NER) constructed from **60 existing datasets**. Built with reproducibility and extensibility in mind, MELD currently provides **gold-standard annotations for 60 languages** across up to **14 domains** with a total of **601 normalized entity labels**. MELD was primarily designed for diverse mutlilingual and multi-domain evaluation but also includes all training and validation sets from its source datasets where available.

# Key Features

- **Standardized Formats**: All datasets are converted to a consistent parquet format, preserving nested and discontinuous annotations, and document boundaries where available.
- **Highly Multilingual**: Gold-standard annotations for 60 languages and silver-standard annotations derived from Wikipedia for 134 additional languages
- **Multi-domain**: 14 diverse domains including legal, biomedical, financial, and social media text. Domain diversity is more limited for languages other than English.
- **Structural Validation**: Several structural issues in source datasets are identified and automatically resolved during processing, such as misaligned span indices and inconsistent IOB labels.
- **Reproducible**: Fully end-to-end reproducible from published source.
- **Extensible**: Designed to be extended further through its modular data processing framework. If a data format is already supported, adding new datasets can be as simple as defining a single JSON file.
- **Zero-Shot Ready**: Provides a normalized entity label mapping specifically designed for zero-shot NER evaluation

# Installation

To start working with our dataset, install MELD using pip:

```bash
pip install meld-data
```

For [reproducing sentence level tokenization from source](#reproducing-meld-from-source),  the `sentence-segmentation` extra needs to be enabled:

```
pip install 'meld-data[sentence-segmentation]'
```

For **development**, we recommend managing your environment with [`uv`](https://docs.astral.sh/uv):

```
git clone https://github.com/kgnlp/meld.git
cd meld
uv sync
```

# Listing Available Datasets

To list all datasets available for download:
```bash
meld-data list
```

# Download MELD

> [!NOTE]
> It is recommended to log into a HuggingFace account with `huggingface-cli login` before downloading datasets to avoid running into API rate limits, particularly when reproducing MELD from source.

To get started, the preprocessed, redistributable subset of MELD can be downloaded using:

```bash
meld-data download -v info path/to/download_directory
```

By default, this downloads the `meld:open` profile, which includes all dataset available in preprocessed form on the HuggingFace Hub. MELD Open can also be used independently of the `meld` package. Versions with original [kgnlp/meld-open](https://huggingface.co/datasets/kgnlp/meld-open) and normalized entity labels [kgnlp/meld-open-normalized](https://huggingface.co/datasets/kgnlp/meld-open-normalized) are available. Datasets not includes in `meld:open` will be automatically downloaded from their original source and processed locally due to licensing restrictions. To download all datasets including CoNLL-2003:

```bash
meld-data download -v info --datasets meld:full path/to/download_directory
```

> [!NOTE]
> Currently, the initially downloaded data will contain the original unnormalized entity labels from each dataset. To apply our label normalization, the [`meld-data hf`](#convert-to-huggingface-datasets-format) command can be used.

**Notice regarding CoNLL-2003:**

Because of copyright restrictions, we cannot redistribute the Reuters Corpus data itself, on which CoNLL-2003 is based. Please refer to the [Reuters Corpus licensing information](https://trec.nist.gov/data/reuters/reuters.html) for specific terms and conditions before downloading or using this dataset. This restriction applies only to the CoNLL-2003 dataset and does not affect other datasets in MELD.

## MELD Directory Structure

The `download_directory` passed to `meld-data download` will contain a `downloads` subdirectory for source datasets processed by MELD and a `meld` subdirectory containing the final processed data. The `downloads` subdirectory can be deleted once data processing is complete. Each dataset in `meld` contains a `meld_metadata.json` including additional metadata, statistics, and paths for each subset and split. The NER data itself will be stored in `parquet` format, optionally in subdirectories for each subset if a dataset includes more than one subset:

```
download_directory/
├─ downloads/
│  └─ ... # Source datasets processed by MELD
└─ meld/
   ├─ CrossNER/
   │  ├─ meld_metadata.json
   │  ├─ literature/
   │  │  ├─ train.parquet
   │  │  ├─ test.parquet
   │  │  └─ validation.parquet
   │  └─ ...
   ├─ AnatEM/
   │  ├─ meld_metadata.json
   │  ├─ train.parquet
   │  ├─ test.parquet
   │  └─ validation.parquet
   └─ ...
```

## Download Groups of Datasets

Download all redistributable datasets (default):
```bash
meld-data download -v info --datasets meld:open path/to/download_directory
```

Download all non-proprietary datasets that include a test set for evaluation:

```bash
meld-data download -v info --datasets meld:non-proprietary-eval path/to/download_directory
```

Download all non-proprietary datasets including Polyglot-NER:
```bash
meld-data download -v info --datasets meld:non-proprietary path/to/download_directory
```

Download all datasets supported by MELD including CoNLL-2003:
```bash
meld-data download -v info --datasets meld:full path/to/download_directory
```

Profiles and individual dataset names can also be mixed. E.g., for downloading MELD Open and CoNLL-2003:
```bash
meld-data download -v info --datasets 'meld:open,CoNLL-2003' path/to/download_directory
```

## Download Specific Datasets

Individual datasets can be downloaded by passing their names as a comma separated list. Dataset names are case-sensitive corresponding to the output of the [list command](###listing-available-datasets).

```bash
meld-data download -v info --datasets "conll-2003,scierc,few-nerd" path/to/download_directory
```

## Reproducing MELD from Source

By default, `meld download` downloads the already processed version of datasets contained in `meld-open` to save bandwidth and processing time. To process all datasets from their original source data, add the `-r/--reproduce` flag. E.g. for reproducing `meld:open` from source:
```
meld download -v info -r path/to/download_directory
```

We use the [SAT sentence tokenizer](https://github.com/segment-any-text/wtpsplit) introduced by [Frohmann et al. (2024)](https://aclanthology.org/2024.emnlp-main.665/) to tokenize long documents into sentences where no canonical sentence tokenization is available. To avoid slightly different boundaries being generated, e.g., due to GPU non-determinism, sentence boundaries bundled with the MELD package are used by default even when `-r/--reproduce` is set. To also reproduce the sentence boundaries from scratch, use:

```
meld download -v info -r --sentence-span-path path/to/new/segmentations path/to/download_directory
```

Where the directory passed as `--sentence-span-path` will contain parquet files in the same format as those [bundled with the MELD package](/meld/package_data/sentence_spans).

# Convert to HuggingFace Datasets Format

The `meld-data hf` subcommand can be used to convert locally processed MELD data to a format compatible with the HuggingFace datasets library and optionally apply our normalized entity label mapping. For instance, for converting processed datasets belonging to the `meld:open` subset with normalized entity labels:

```
meld-data hf -d meld:open --normalize-labels /path/to/processed/meld/data /path/to/converted/data
```

Note that datasets converted in this way should not be uploaded to the HuggingFace Hub unless the constituent dataset's licensing requirements are fulfilled. See `meld-data hf --help` for additional options.

# Included Datasets

MELD integrates **60 NER datasets** spanning **194 languages** (**60 with gold standard test sets**), **14 domains**, and **601 normalized entity labels**. The table below provides a general overview of the included datasets:

| Name | Primary Domain | Languages | Annotation Type | License |
|------|----------------|-----------|-----------------|---------|
| AgCNER | Agriculture | zho | gold-standard | CC 0 |
| AgriNER | Agriculture | eng | gold-standard | CC BY-SA 4.0 |
| AnatEM | Biomedical | eng | gold-standard | CC BY-SA 3.0 |
| BC2GM | Biomedical | eng | gold-standard | CC BY 4.0 |
| BC4CHEMD | Biomedical | eng | gold-standard | Unspecified |
| BC5CDR | Biomedical | eng | gold-standard | Public Domain |
| BioRED | Biomedical | eng | gold-standard | Public Domain |
| JNLPBA | Biomedical | eng | gold-standard | GENIA Project License (CC BY 3.0 annotations) |
| NCBI-Disease | Biomedical | eng | gold-standard | Public Domain |
| CANTEMIST | Clinical | spa | gold-standard | CC BY 4.0 |
| EBM-NLP | Clinical | eng | gold-standard | Unspecified |
| RaTE-NER | Clinical | eng | gold-standard, silver-standard | CC BY-NC 4.0 |
| FiNER-139 | Finance | eng | gold-standard | CC BY-SA 4.0 |
| TASTEset | Food | eng | gold-standard | MIT |
| Arabic-Cross-Dialectal-NER | General | apc, ary, arz | gold-standard | Unspecified |
| Naamapadam | General | asm, ben, guj, hin, kan, mal, mar, ori, pan, tam, tel | gold-standard, silver-standard | CC 0 |
| Thai-NER | General | tha | gold-standard | CC BY 4.0 |
| Turku-NER-corpus | General | fin | gold-standard | CC BY-SA 4.0 |
| TurkuONE | General | fin | gold-standard | CC BY-ND-NC 1.0, CC BY-SA 3.0, CC BY-SA 4.0 |
| NYTK-NerKor | General, Law, Literature, News, Wikipedia | hun | gold-standard | CC BY-SA 4.0 |
| UniversalNER | General, Literature, News, Wikipedia | 15 languages | gold-standard | CC BY-SA 4.0 |
| E-NER | Law | eng | gold-standard | CC BY-NC-SA 4.0 |
| German-LER | Law | deu | gold-standard | CC BY 4.0 |
| LegalNERo | Law | ron | gold-standard | CC BY-NC-ND 4.0 |
| Herodotos-Project-NER | Literature | lat | gold-standard | AGPL-3.0 license |
| CLEANANERCorp | News | ara | gold-standard | GPL 3.0 |
| CoNLL-2003 | News | eng | gold-standard | Proprietary text (See [Download MELD](#download-meld) for details) |
| EverestNER | News | nep | gold-standard | Non-commercial |
| FiNER-ORD | News | eng | gold-standard | CC BY-NC 4.0 |
| FoNE | News | fao | gold-standard | CC BY 4.0 |
| idner-news-2k | News | ind | gold-standard | MIT |
| MasakhaNER-X | News | 20 languages | gold-standard | CC BY-NC 4.0 |
| PhoNER-COVID19 | News | vie | gold-standard | Research and Education Purposes Only |
| pioNER | News, Wikipedia | hye | gold-standard, silver-standard | Apache 2.0 |
| FabNER | Science | eng | gold-standard | CC BY 4.0 |
| SciER | Science | eng | gold-standard | GPL 3.0 |
| SCIERC | Science | eng | gold-standard | Unspecified |
| SciREX | Science | eng | gold-standard | Apache 2.0 |
| SOFC-Exp | Science | eng | gold-standard | CC BY 4.0 |
| SoMeSci | Science | eng | gold-standard | CC BY 4.0 |
| WIESP2022 | Science | eng | gold-standard | CC BY 4.0 |
| WLP | Science | eng | gold-standard | MIT |
| DanfeNER | Social Media | nep | gold-standard | Non-commercial |
| HarveyNER | Social Media | eng | gold-standard | Unspecified |
| MIT-Movie | Social Media | eng | gold-standard | Unspecified |
| MIT-Restaurant | Social Media | eng | gold-standard | Unspecified |
| Tweebank-NER | Social Media | eng | gold-standard | Apache 2.0 |
| TweetNER7 | Social Media | eng | gold-standard | Non-commercial |
| Weibo-NER | Social Media | zho | gold-standard | CC BY-SA 3.0 |
| WNUT2017 | Social Media | eng | gold-standard | CC BY 4.0 |
| StackOverflowNER | Software | eng | gold-standard | MIT |
| FindVehicle | Transportation | eng | gold-standard | Unspecified |
| CrossNER | Wikipedia | eng | gold-standard | MIT |
| Few-NERD | Wikipedia | eng | gold-standard | CC BY-SA 4.0 |
| Japanese-Wikipedia | Wikipedia | jpn | gold-standard | CC BY-SA 3.0 |
| MultiCoNER | Wikipedia | MULTI, ben, deu, eng, fas, fra, hin, ita, por, spa, swe, ukr, zho | silver-standard | CC BY 4.0 |
| MultiNERd | Wikipedia | deu, eng, fra, ita, nld, pol, por, rus, spa, zho | silver-standard | CC BY-NC-SA 4.0 |
| Polyglot-NER | Wikipedia | 40 languages | silver-standard | Unspecified |
| WikiANN | Wikipedia | 175 languages | silver-standard | Unspecified |
| WikiNEuRal | Wikipedia | deu, eng, fra, ita, nld, pol, por, rus, spa | silver-standard | CC BY-NC-SA 4.0 |

# BibTeX Citations

Get citation for MELD:
```bash
meld-data cite
```

Get citations for specific datasets:
```bash
meld-data cite conll-2003,scierc
```

Get citations for all datasets:
```bash
meld-data cite --all
```

# Citation

When using MELD, please cite our paper:

```bibtex
@inproceedings{glocker2026meld,
  title = {MELD: Melding Diverse Multilingual and Multi-Domain Datasets for
           Named Entity Recognition Evaluation},
  author = {Glocker, Kevin and Kuhlmann, Marco},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation
               Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {1889--1903},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide,
            Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/32qrd24xac2e},
}
```

When using the PhoNER COVID19 subset, also cite the following article in accordance with its [terms of use](https://github.com/VinAIResearch/PhoNER_COVID19):

```bibtex
@inproceedings{PhoNER_COVID19,
  title = {{COVID-19 Named Entity Recognition for Vietnamese}},
  author = {Thinh Hung Truong and Mai Hoang Dao and Dat Quoc Nguyen},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter
               of the Association for Computational Linguistics: Human Language
               Technologies},
  year = {2021},
}
```

To retrieve citations for other datasets in MELD, see [BibTex Citations](#bibtex-citations).

# Contributing

We welcome contributions to expand the dataset! Documentation and guidelines for adding new datasets to MELD are coming soon.

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
