# STSbenchmark

## Introduction

This repository contains an implementation of paper "HCTI at SemEval-2017 Task 1: Use convolutional neural network to evaluate semantic textual similarity." 
A simple CNN approach to get semantic textual similarity scores is proposed in the paper, which got the 3rd place in SemEval-2017 task 1. 
This implementation is based on python 3.5/Tensorflow 1.7.0/Keras 1.2.2 and can achieve over 0.77 on STSbenchmark datasets. 

## Citation

If you use these codes for your research, please cite:

    @inproceedings{Shao:2017,
    	author = {Yang Shao},
    	title  = {{HCTI} at SemEval-2017 Task 1: Use convolutional neural network to evaluate Semantic Textual Similarity},
    	booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval 2017)},
    	year   = {2017},
    	address = {Vancouver, Canada},
    	publisher = {Association for Computational Linguistics},
    }

Yang Shao. 2017. HCTI at SemEval-2017 Task 1: Use convolutional neural network to evaluate semantic textual similarity. In Proceedings of SemEval-2017. https://github.com/celarex/Semantic-Textual-Similarity-STSbenchmark-HCTI-CNN

## Training

To train your own model, you need to download GloVe pre-trained word vectors from http://nlp.stanford.edu/data/glove.840B.300d.zip first and unzip it in your folder.  
To train your CNN model on STSbenchmark datasets:

    python sts.py
