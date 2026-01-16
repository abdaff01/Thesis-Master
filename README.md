# Thesis: Developing an OCR System for Automated Metadata Extraction

## Abstract

This repository contains the implementation, experiments, and documentation for a Master's thesis project focused on developing an Optical Character Recognition (OCR) system for automated metadata extraction from scholarly documents. The project combines image preprocessing, layout analysis, OCR transcription, and natural language processing to extract structured metadata (e.g., title, authors, affiliations, abstract, keywords, references) from scanned or digital-born research papers.

## Authors and Supervisors

- Author: Abdallah Daff (GitHub: @abdaff01)
- Supervisor: [Supervisor Name]
- Affiliation: [Department, University]
- Thesis year: 2026

Please replace bracketed fields with the correct information for final submission.

## Table of Contents

- [Background and Motivation](#background-and-motivation)
- [Objectives](#objectives)
- [Key Contributions](#key-contributions)
- [Methodology Overview](#methodology-overview)
- [Datasets](#datasets)
- [Implementation and Usage](#implementation-and-usage)
- [Evaluation and Results](#evaluation-and-results)
- [Repository Structure](#repository-structure)
- [Reproducibility](#reproducibility)
- [How to Cite](#how-to-cite)
- [License and Acknowledgements](#license-and-acknowledgements)
- [Contact](#contact)

## Background and Motivation

Accurate and scalable extraction of metadata from scholarly documents enables better indexing, search, citation analysis, and digital library services. Traditional OCR pipelines often struggle with heterogeneous document layouts, multi-column formats, and non-textual elements. This thesis aims to build a robust pipeline that addresses layout variability and improves extraction accuracy for downstream bibliographic and information retrieval tasks.

## Objectives

- Design and implement an end-to-end pipeline for extracting structured metadata from research articles.
- Combine classical layout analysis and modern OCR models to improve transcription quality.
- Apply named-entity recognition and rule-based post-processing to map raw text to structured metadata fields.
- Evaluate performance on public datasets and provide reproducible experiments.

## Key Contributions

- A modular, reproducible OCR pipeline tailored for scholarly documents.
- Comparative evaluation of OCR and layout-analysis models for metadata extraction.
- Open-source code, trained model artifacts, and evaluation scripts to facilitate further research.

## Methodology Overview

The pipeline is organized into the following stages:

1. Data ingestion and normalization: collect PDF or image inputs, convert to high-resolution images where necessary, and normalize orientation.
2. Image preprocessing: denoising, binarization, deskewing, and contrast enhancement to improve OCR performance.
3. Layout analysis and segmentation: detect columns, headers, figures, tables, and text blocks using heuristic rules and learned models.
4. OCR transcription: apply an OCR engine (configurable; e.g., Tesseract or a neural OCR such as CRNN/Transformer-based models) to segmented text regions.
5. Post-processing and metadata extraction: apply NLP (tokenization, NER, pattern matching) and heuristics to identify titles, authors, affiliations, abstracts, keywords, and references.
6. Evaluation: measure transcription quality (Character Error Rate, Word Error Rate) and metadata extraction performance (precision, recall, F1) against annotated ground truth.

## Datasets

The experiments in this repository use one or more public datasets of scholarly documents. Where applicable, dataset download and preparation scripts are provided under `data/`. Examples of dataset sources used in the literature include:

- PubLayNet (layout annotations)
- DocBank (document structure annotations)
- GROTOAP (metadata extraction)

Ensure you comply with each dataset's license when downloading and using the data.

## Implementation and Usage

Prerequisites

- Python 3.8+
- Git
- Optional: NVIDIA GPU with CUDA for model training

Setup (recommended)

1. Clone the repository:

   git clone https://github.com/abdaff01/Thesis-Master.git
   cd Thesis-Master

2. Create and activate a virtual environment (venv or conda):

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .\.venv\Scripts\activate  # Windows

3. Install required Python packages:

   pip install -r requirements.txt

Common scripts

- data/prepare_dataset.sh  — prepare and preprocess datasets
- src/train.py             — training entry point for OCR/layout/NER models
- src/evaluate.py          — evaluation scripts and metrics
- src/infer.py             — inference script to process input PDF/images and output JSON metadata

Example: run inference on a PDF

   python src/infer.py --input examples/sample.pdf --output examples/sample_metadata.json --model-path models/best_model.pth

Output format

Metadata outputs are JSON files with fields such as:

{  
  "title": "...",  
  "authors": ["..."],  
  "affiliations": ["..."],  
  "abstract": "...",  
  "keywords": ["..."],  
  "references": [ {"raw": "...", "parsed": {...} } ]
}

## Evaluation and Results

Evaluation scripts compute standard OCR and information extraction metrics:

- Character Error Rate (CER)
- Word Error Rate (WER)
- Precision, Recall, F1 for extracted metadata fields

Key results and figures from experiments should be included in `docs/` or the thesis document. Replace or extend the `results/` directory with plots, confusion matrices, and sample outputs.

## Repository Structure

A suggested project layout:

- README.md
- requirements.txt
- data/                 # dataset downloads and preprocessing scripts
- docs/                 # thesis chapters, figures, and results
- src/                  # source code (training, inference, utilities)
- notebooks/            # exploratory notebooks
- models/               # trained model checkpoints
- experiments/          # experiment logs and configs
- examples/             # example inputs and outputs

## Reproducibility

To reproduce experiments exactly, use the `experiments/` configuration files and the recorded random seeds. Recommended steps:

1. Create the same Python environment (use `requirements.txt` or `environment.yml`).
2. Prepare datasets using the provided scripts in `data/`.
3. Launch training with the saved config: `python src/train.py --config experiments/config_example.yaml`.
4. Run evaluation: `python src/evaluate.py --predictions output/preds.json --ground-truth data/ground_truth.json`.

For full reproducibility, a Dockerfile and/or conda environment may be provided. If available, follow the instructions in `docs/REPRODUCIBILITY.md`.

## How to Cite

If you use code or models from this repository in academic work, please cite the thesis and repository. Example BibTeX entry:

@misc{daff2026thesis,  
  author = {Daff, Abdallah},  
  title = {Developing an OCR System for Automated Metadata Extraction},  
  year = {2026},  
  howpublished = {Master's thesis, [University Name]},  
  note = {Code available at https://github.com/abdaff01/Thesis-Master}
}

## License and Acknowledgements

This repository is distributed under the MIT License. See the LICENSE file for details.

Acknowledgements:

- Supervisor and committee
- Project collaborators and lab members
- Datasets and open-source projects (Tesseract, PyTorch, etc.)

## Contact

For questions or collaboration, contact: abdallah.daff@example.edu (replace with your institutional email) or open an issue on the repository.


---

This README is intended as a professional, academic introduction to the project. Please review and update author/supervisor details, dataset references, and the evaluation/results sections with concrete numbers and figures from your thesis before final submission.