import shutil
from os import PathLike
from pathlib import Path

import polars as pl
from huggingface_hub import DatasetCard, DatasetCardData
from pydantic import TypeAdapter

from meld.data_filters import SplitFilter
from meld.formats import DatasetMetadata, Split, Subset, local_datasets
from meld.manifest import LabelMap

_IGNORED_LANGUAGES = {
    "MULTI"  # Subset of MultiCoNER II without language metadata
}

_DEFAULT_TEMPLATE = """---
{{ card_data }}
---

# {{ pretty_name }}
"""

_LOWER_BOUNDS = [
    (0, "n<1K"),
    (1_000, "1K<n<10K"),
    (10_000, "10K<n<100K"),
    (100_000, "100K<n<1M"),
    (1_000_000, "1M<n<10M"),
    (10_000_000, "10M<n<100M"),
    (100_000_000, "100M<n<1B"),
    (1_000_000_000, "1B<n<10B"),
    (10_000_000_000, "10B<n<100B"),
    (100_000_000_000, "100B<n<1T"),
    (1_000_000_000_000, "n>1T"),
]


def hf_config_name(dataset_name: str, subset_hierarchy: list[str]) -> str:
    """
    Generate a Hugging Face configuration name from a dataset name and subset hierarchy.

    Args:
        dataset_name: Name of the dataset.
        subset_hierarchy: List of subset names forming the hierarchy.

    Returns:
        A single identifier for a Hugging Face dataset config,
        consisting of the dataset name and subset hierarchy joined by double hyphens.
    """
    if subset_hierarchy:
        return f"{dataset_name}--{'--'.join(subset_hierarchy)}"

    return dataset_name


def convert_to_hf(
    meld_data_path: PathLike | str,
    output_data_path: PathLike | str,
    data_filter: SplitFilter | None = None,
    label_mapping: LabelMap | None = None,
    dataset_card_template: str | None = None,
    pretty_name: str | None = None,
    extra_files: dict[str, list[str]] | None = None,
) -> None:
    """
    Convert MELD datasets to Hugging Face format.

    Args:
        meld_data_path: Path to the MELD dataset directory.
        output_data_path: Output directory for the converted HF dataset.
        data_filter: Optional filter to include only specific splits.
        label_mapping: Optional mapping to normalize entity type labels.
        dataset_card_template: Optional custom template for the dataset card.
            If not provided, a minimal default template will be used.
            The template should contain:
                - {{ card_data }} within YAML frontmatter at the top of the template
                - {{ pretty_name }}: Will be replaced with the human-readable dataset name
        pretty_name: Optional pretty name for the dataset card.
            If not provided, the output directory name will be used as the pretty name.
        extra_files: Lists of additional files per dataset to include in their output directories, such as licenses
    """
    metadata_adapter = TypeAdapter(DatasetMetadata)
    output_data_path = Path(output_data_path)
    output_data_path.mkdir()

    if extra_files is None:
        extra_files = {}

    dataset_meta = {}
    languages = set()
    configs = []
    total_sequences = 0

    for dataset in local_datasets(meld_data_path):
        all_splits_included = True
        dataset_configs = []
        input_files: list[tuple[Subset, list[str]]] = []
        dataset_sequences = 0

        for subset in dataset:
            subset_metadata = subset.metadata
            subset_name = Path(hf_config_name(dataset.metadata.name, subset.hierarchy))

            split_data_files = []
            split_input_files = []
            for split_name, metadata in subset.splits.items():
                split = Split(
                    split_name,
                    dataset.metadata.name,
                    subset_metadata.language,
                    subset_metadata.tagsets,
                    metadata,
                    subset_metadata.labels,
                    subset_metadata.bio_labels,
                )
                if data_filter is not None and not data_filter(split):
                    all_splits_included = False
                    break

                dataset_sequences += split.metadata.sequence_count
                split_data_files.append({"split": split_name, "path": [str(subset_name / metadata.path.name)]})
                split_input_files.append(dataset.path / metadata.path)

            dataset_configs.append({"config_name": str(subset_name), "data_files": split_data_files})
            input_files.append((subset, split_input_files))

        # Only include datasets if all subsets match the filter
        if not all_splits_included:
            continue

        total_sequences += dataset_sequences

        for config, (subset, subset_input_files) in zip(dataset_configs, input_files):
            config_dir = output_data_path / config["config_name"]
            config_dir.mkdir()
            subset_hierarchy = tuple(subset.hierarchy)

            if label_mapping is not None and subset.metadata.pre_tokenized:
                iob_label_map = {
                    tagset: {
                        prefix + label: prefix + target
                        for label, target in label_map.items()
                        for prefix in ("B-", "I-")
                    }
                    | {"O": "O"}
                    for tagset, label_map in label_mapping[dataset.metadata.name][subset_hierarchy].items()
                }
            else:
                iob_label_map = None

            for split_output_file, split_input_file in zip(config["data_files"], subset_input_files):
                # Format binary UUIDs as UUID hex strings for datasets library compatibility
                id_string = pl.col("sequence_id").str
                dash = pl.lit("-")
                data_frame = (
                    pl.scan_parquet(split_input_file)
                    .with_columns(pl.col("sequence_id").bin.encode("hex"))
                    .with_columns(
                        pl.concat_str(
                            [
                                id_string.slice(0, 8),
                                dash,
                                id_string.slice(8, 4),
                                dash,
                                id_string.slice(12, 4),
                                dash,
                                id_string.slice(16, 4),
                                dash,
                                id_string.slice(20),
                            ]
                        )
                    )
                )

                if label_mapping is not None:
                    tag_maps = label_mapping[dataset.metadata.name][subset_hierarchy]

                    data_frame = data_frame.with_columns(
                        *(
                            pl.col(tagset).list.eval(
                                pl.struct(
                                    label=pl.element().struct["label"].replace_strict(tag_map),
                                    spans=pl.element().struct["spans"],
                                )
                            )
                            for tagset, tag_map in tag_maps.items()
                        )
                    )

                    if iob_label_map is not None:
                        data_frame = data_frame.with_columns(
                            *(
                                pl.col(tagset + "_iob").list.eval(pl.element().replace_strict(tag_map))
                                for tagset, tag_map in iob_label_map.items()
                            )
                        )

                data_frame.sink_parquet(output_data_path / split_output_file["path"][0])

                # Copy any additional files to the config directory
                if (additional_files := extra_files.get(dataset.metadata.name)) is not None:
                    for file in additional_files:
                        shutil.copyfile(file, config_dir / Path(file).name)

        dataset_meta[dataset.metadata.name] = metadata_adapter.dump_python(dataset.metadata, mode="json")
        languages.update(language.split("-")[0] for language in dataset.metadata.languages)
        configs.extend(dataset_configs)

    # Fall back to directory name
    if pretty_name is None:
        pretty_name = output_data_path.name

    size_categories = _LOWER_BOUNDS[0][1]
    for lower_bound, category in _LOWER_BOUNDS:
        if total_sequences <= lower_bound:
            break

        size_categories = category

    card_metadata = DatasetCardData(
        pretty_name=pretty_name,
        language=sorted(languages - _IGNORED_LANGUAGES),
        multilinguality="multilingual",
        size_categories=size_categories,
        task_categories=["token-classification"],
        task_ids=["named-entity-recognition"],
        meld_metadata=dataset_meta,
        configs=configs,
    )

    if dataset_card_template is None:
        dataset_card_template = _DEFAULT_TEMPLATE

    card = DatasetCard.from_template(card_metadata, template_str=dataset_card_template, pretty_name=pretty_name)
    card.save(output_data_path / "README.md")


def dataset_filter(keep_datasets: set[str]) -> SplitFilter:
    """
    Create a filter function that keeps only splits from the specified datasets.

    Args:
        keep_datasets: Set of dataset names to keep.

    Returns:
        A SplitFilter function that returns True for splits belonging to the specified datasets.
    """

    def filter(split: Split) -> bool:
        return split.dataset_name in keep_datasets

    return filter
