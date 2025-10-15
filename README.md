Identification of Key Risk Factors in Construction Quality Defects Using Unsupervised Deep Learning and Bayesian Networks

⚠️ Since the related paper is under review, none of the materials in this repository are permitted for reuse until this notice is removed.

Project Overview

This project leverages unsupervised deep learning and Bayesian networks to identify construction quality defects and their key risk factors. By integrating textual data, experimental parameters, and expert-coded information, the project enables structured representation of construction issues, causal relationship modeling, and risk visualization.

Key Features:

Identification of construction quality defect phenomena from textual data

Unsupervised semantic clustering to discover hidden problem patterns

Bayesian network modeling of key risk factor causality

Interpretive Structural Modeling (ISM) to analyze problem hierarchy and reachability

Visualization of defect distribution and risk levels

project-root/
├── data/                               # Data files
│   ├── Boolean Matrix.csv
│   ├── Causes of quality defects.xlsx
│   ├── Experimental Parameter and Environment Configuration.xlsx
│   ├── Grounded Theory Codebook.xlsx
│   ├── ISM Adjacency Matrix.xlsx
│   ├── ISM reachability matrix.xlsx
│   ├── Quality defect phenomenon.xlsx
│   └── Quality defect prevention and control.xlsx
├── src/                                # Python scripts
│   ├── Regular Expressions.py          # Keyword extraction & text preprocessing
│   ├── aprior.py                       # Association rule mining
│   ├── clustering.py                   # Unsupervised clustering
│   └── ism.py                          # ISM analysis
├── pseudocode.docx                      # Pseudocode of methods
├── README.md                            # This document


Installation

Recommended: Python 3.8+. Install dependencies via pip:

pip install pandas numpy scikit-learn tensorflow pgmpy matplotlib seaborn openpyxl

Usage
1. Data Preprocessing
from Regular_Expressions import preprocess_texts

data = preprocess_texts("data/Quality defect phenomenon.xlsx")

2. Unsupervised Clustering
from clustering import semantic_clustering

clusters = semantic_clustering(data)

3. Association Rule Mining
from aprior import mine_association_rules

rules = mine_association_rules(data)

4. ISM Hierarchy Analysis
from ism import build_ism, get_levels

ism_model = build_ism("data/ISM Adjacency Matrix.xlsx")
levels = get_levels(ism_model)

5. Bayesian Network Causal Inference
from bayesian_network import build_bn, infer_defects

bn_model = build_bn(clusters)
results = infer_defects(bn_model, data)
