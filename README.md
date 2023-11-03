## **Practical Machine Learning and Deep Learning - Assignment 1 - Text De-toxification**

Dinara Yaryeva\
d.yaryeva@innopolis.university\
BS20-RO

## Task:
The aim of the assignment is to create a solution for detoxing text with high level of toxicity. It can be a model or set of models, or any algorithm that would work. 

## Basic usage

### Setup

```bash
git clone https://github.com/dinarayaryeva/pml-dl.git
```
### Install requirements
```
pip install -r requirements.txt
```
### Transform data
```bash
python src/data/make_dataset.py
```
### Train model

```bash
python src/models/train_model.py
```

### Make predictions
```bash
python src/models/predict_model.py
```
You can download weights from [GoogleDrive](https://drive.google.com/drive/folders/1gHOxODVGO1xua27KlHhRuAf6_kR_zl-R?usp=sharing) into models folder

### Visualization
```bash
python src/visualization/visualize.py
``````