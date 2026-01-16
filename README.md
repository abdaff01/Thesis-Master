# Thesis: Developing an OCR System for Automated Metadata Extraction

## Abstract

This repository contains the implementation, experiments, and documentation for a Master's thesis that develops a robust Optical Character Recognition (OCR) pipeline for automated metadata extraction from scholarly documents. The system combines image preprocessing, layout analysis, OCR transcription, and natural language processing to extract structured metadata fields (title, authors, affiliations, abstract, keywords, references) from scanned and digital-born research papers.

## Author, Supervisor & Affiliation

- Author: Abdallah Daff (GitHub: @abdaff01)  
- Supervisor: [Supervisor Name]  
- Affiliation: [Department, University]  
- Thesis year: 2026

Replace bracketed fields with the correct information for final submission.

## Table of Contents

- [Background and Motivation](#background-and-motivation)  
- [Objectives](#objectives)  
- [Contributions](#contributions)  
- [Methodology Overview](#methodology-overview)  
- [Datasets](#datasets)  
- [Installation & Requirements](#installation--requirements)  
- [Usage](#usage)  
- [Evaluation & Results](#evaluation--results)  
- [Repository Structure](#repository-structure)  
- [Reproducibility](#reproducibility)  
- [How to Cite](#how-to-cite)  
- [License & Acknowledgements](#license--acknowledgements)  
- [Contact](#contact)

## Background and Motivation

Accurate, scalable metadata extraction from scholarly documents enables improved indexing, searching, citation analysis, and management of digital libraries. Heterogeneous document layouts, multi-column formats, and non-textual elements make reliable extraction a challenging task. This thesis builds a modular pipeline to address layout variability and improve extraction accuracy for downstream bibliographic tasks.

## Objectives

- Design and implement an end-to-end pipeline to extract structured metadata from research articles.
- Combine classical layout-analysis techniques with modern OCR and NLP models to improve transcription and field recognition.
- Provide evaluation scripts, reproducible experiments, and open-source artifacts to support further research.

## Contributions

- A modular, extensible OCR pipeline optimized for scholarly documents.
- Comparative evaluation of layout-analysis and OCR strategies for metadata extraction.
- Open-source code, model checkpoints, and evaluation scripts to reproduce and extend experiments.

## Methodology Overview

The pipeline is organized into the following stages:

1. Data ingestion and normalization: convert PDF or image inputs to high-resolution images; normalize orientation.
2. Image preprocessing: denoising, binarization, deskewing, and contrast enhancement.
3. Layout analysis and segmentation: detect text blocks, headers, titles, authors, figures, tables, and references using heuristic and learned methods.
4. OCR transcription: apply a configurable OCR engine (e.g., Tesseract or neural OCR models) to segmented text regions.
5. Post-processing and metadata extraction: use NLP (tokenization, NER, pattern matching) and rule-based mapping to populate structured metadata fields.
6. Evaluation: compute transcription and metadata extraction metrics against annotated ground truth.

## Datasets

Experiments were conducted using public document/layout datasets. Preparation scripts are provided in `data/`. Typical sources used in the literature:

- PubLayNet — layout annotations  
- DocBank — document structure annotations  
- GROTOAP / other metadata extraction corpora

Comply with dataset licenses and usage terms before download and use.

## Installation & Requirements

Prerequisites:

- Python 3.8+  
- Git  
- Optional: NVIDIA GPU with CUDA for model training

Recommended setup:

1. Clone the repository:
   git clone https://github.com/abdaff01/Thesis-Master.git
   cd Thesis-Master

2. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .\.venv\Scripts\activate    # Windows

3. Install dependencies:
   pip install -r requirements.txt

If available, a Dockerfile or environment.yml can be used for fully reproducible environments.

## Usage

Common scripts and entry points:

- data/prepare_dataset.sh — download and preprocess datasets  
- src/train.py — train OCR, layout, or NER models  
- src/evaluate.py — compute metrics and generate evaluation reports  
- src/infer.py — run inference on PDF/images and produce structured metadata

Example: run inference on a PDF
python src/infer.py --input examples/sample.pdf --output examples/sample_metadata.json --model-path models/best_model.pth

Example output (JSON):
{
  "title": "Example Paper Title",
  "authors": ["Author A", "Author B"],
  "affiliations": ["Institution X", "Institution Y"],
  "abstract": "This paper presents...",
  "keywords": ["OCR", "metadata extraction"],
  "references": [
    {"raw": "Author, Title, Venue, Year", "parsed": {"authors": ["Author"], "title": "Title", "year": "Year"}}
  ]
}

Adjust CLI arguments and config files in `experiments/` for alternate behavior.

## Evaluation & Results

Evaluation metrics provided:

- Character Error Rate (CER)  
- Word Error Rate (WER)  
- Precision, Recall, F1 for metadata fields (title, authors, affiliations, abstract, references)

Include experiment results, plots, and sample outputs in `docs/` or `results/`. Replace placeholder results with concrete tables and figures from your experiments.

## Repository Structure

Suggested layout:

- README.md  
- requirements.txt  
- data/                 — dataset download and preprocessing scripts  
- docs/                 — thesis chapters, figures, and evaluation artifacts  
- src/                  — source code (training, inference, utilities)  
- notebooks/            — exploratory notebooks and analyses  
- models/               — trained model checkpoints (large files may be stored externally)  
- experiments/          — experiment configs and logs  
- examples/             — sample inputs and outputs  
- results/              — evaluation outputs and plots

## Reproducibility

To reproduce experiments:

1. Recreate the environment (use `requirements.txt` or `environment.yml`).  
2. Run dataset preparation scripts in `data/`.  
3. Train with a saved config: python src/train.py --config experiments/config_example.yaml  
4. Evaluate: python src/evaluate.py --predictions output/preds.json --ground-truth data/ground_truth.json

For exact reproducibility, include random seeds, dependency versions, and (optionally) a Docker image.

## How to Cite

If you use code or models from this repository, please cite the thesis and repository. Example BibTeX:

@misc{daff2026thesis,
  author = {Daff, Abdallah},
  title = {Developing an OCR System for Automated Metadata Extraction},
  year = {2026},
  howpublished = {Master's thesis, [University Name]},
  note = {Code available at https://github.com/abdaff01/Thesis-Master}
}

## License & Acknowledgements

This repository is provided under the MIT License. See LICENSE for details.

Acknowledgements:
- Supervisor, committee members, and collaborators  
- Open-source tools and libraries used (e.g., Tesseract, PyTorch)  
- Dataset providers and maintainers

## Contact

For questions, contributions, or collaborations:
- Author: Abdallah Daff — abdallah.daff@example.edu (replace with your institutional email)  
- GitHub: https://github.com/abdaff01  
- Issues and pull requests are welcome.

---

This README is intended as a professional, academic introduction to the project. Please review and update author/supervisor information, dataset citations, and the results section with concrete numbers and figures from your thesis prior to final submission.
