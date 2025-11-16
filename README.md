# Human detection on CCTV footage
Fine-tuning a pretrained neural network to detect human presence on cctv footages

## Getting started
Don't forget the Python virtual environment `.venv`!
### Prerequisites
- Install the requirements
```sh
pip install -r requirements.txt
```
- Download the dataset from [here](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset/data)
  and place them into a `dataset` folder.

## Model Architecture
The MobileNetV2 model and its pretrained weights are used in this model.
* The last layer of the model is modified to classify the cctv footage as 0 (no human presence) and 1 (human presence).
* The critirion (loss) used is Cross Entropy Loss.

## Results
The dataset is split into train, test and validation set. The validation set is not involved in the fine-tuning of the model.
- The model takes around 5 minutes to train (on GPU)
- The results on the validation set: a loss of ~0.2 and accuracy of ~0.94.
- The low loss value indicates the model is confident on the result.
