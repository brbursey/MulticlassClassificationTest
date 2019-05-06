# MulticlassClassificationTest
Learning how to create a Classification model using ML.NET.

# What is it?
This is a model that can predict an input's value based on its features when it can be of more than two types.

# How to use
To predict your own datasets, you will need to write up a few classes just like the Iris data classes are setup. 
Further documentation will be provided once I have finished refactoring.

# What needs to be changed?
## Input Data
Currently, The model is grabbing data from text files. This includes the training data, predicting data, and the data model.
The Classification model needs a way of loading data into the pipeline that is from a database and not a text file.

## Classfication Model Dependency
The model is dependent on IrisData and IrisPrediction. Need to refactor this into interfaces so that the model is completely independent from the data it consumes.

## Statistics
Now that the model works successfully, it needs to have some statistics to prove that the categories were reasonable to predict.
This includes the probabilities of the other categories and other statistical data.