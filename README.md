# Image classification using PyTorch

- This repo is a part for following project 
[Web Scraping with product search relevance using NLP, rules and image classification](https://github.com/jithinanievarghese/product-search-relevance/blob/main/README.md)

- Here we train a binary image classification model for e-commerce product image classification using PyTorch
and make inference on notebook [inference.ipynb](https://github.com/jithinanievarghese/image_classification_pytorch/blob/main/inference.ipynb)

- Model architecture used is convolutional neural networks (CNNs)
- Training data was collected using the following [Scrapy Spider](https://github.com/jithinanievarghese/flipkart_scraper_scrapy) and open source kaggle datasets:
    1. https://www.kaggle.com/datasets/sunnykusawa/tshirts
    2. https://www.kaggle.com/datasets/dqmonn/zalando-store-crawl
    3. https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings
- If local system dont have GPU, then we can make use of the following [Kaggle Notebook](https://www.kaggle.com/code/jithinanievarghese/image-classification-pytorch) for training
- Detailed insights regarding the data gathering, data labeling, model perfomance are available [here](https://github.com/jithinanievarghese/product-search-relevance/tree/main)


# Usage

1. Install Requirements `pip3 install -r requirements.txt`
2. `python3 train.py`
