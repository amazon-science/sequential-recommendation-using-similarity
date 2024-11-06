# SimRec: Mitigating the Cold-Start Problem in Sequential Recommendation by Integrating Item Similarity
This repository is the official implementation of the paper "SimRec: Mitigating the Cold-Start Problem in Sequential Recommendation by Integrating Item Similarity".
Access the full paper here: https://www.amazon.science/publications/simrec-mitigating-the-cold-start-problem-in-sequential-recommendation-by-integrating-item-similarity 

## Datasets

[data_preprocessing](/data_preprocessing/) contains the code for generating the datasets.

This includes 2 steps:
1. Generating the dataset, by using the jupyter notebook `preprocessing_data.ipynb` that can be found in each sub-directory.
2. Calculating the similarity scores using the jupyter notebook [`calculate_similarity_scores.ipynb`](/data_preprocessing/calculate_similarity_scores.ipynb)

After you create the dataset(s) you can move to training the model.

## Model Training

[SimRec](/SimRec/) contains the code for training SimRec on the generated datasets.

To run SimRec:
```bash 
cd SimRec
```

Train the model on Beauty dataset:
```
bash SimRec/scripts/Beauty/train.sh
```
Train the model on Tools dataset:
```
bash SimRec/scripts/Tools/train.sh
```
Train the model on Pet Supplies dataset:
```
bash SimRec/scripts/PetSupplies/train.sh
```
Train the model on Home & Kitchen dataset:
```
bash SimRec/scripts/HomeKitchen/train.sh
```
Train the model on ML-1M dataset:
```
bash SimRec/scripts/ML-1M/train.sh
```
Train the model on Steam dataset:
```
bash SimRec/scripts/Steam/train.sh
```

Note that the implementation of SimRec is based on [SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch) repo.

## Requirements

```
pip install -r requirements
```

## Hardware
In general, all experiments can run on either GPU or CPU.

## License
This project is licensed under the Apache-2.0 License.
