# Chest X-ray Pneumonia Detection

## Overview
A deep learning project to detect Pneumonia from Chest X-rays using a ResNet18 backbone.

## Structure
- `data/`: Raw and processed data (Not included in git).
- `src/`: Source code.
- `notebooks/`: EDA.
- `configs/`: Parameters.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download Data:
   ```bash
   python src/download_data.py
   ```
3. Prepare Splits:
   ```bash
   python src/prepare_splits.py
   ```
4. Train:
   ```bash
   python src/train.py
   ```
