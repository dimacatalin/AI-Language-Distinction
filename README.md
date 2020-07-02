# IA-Proiect

In the Romanian Dialect Identification (RDI) shared task, participants have to train a model on tweets. Therefore, participants have to build a model for a in-genre binary 
classification by dialect task, in which a classification model is required to discriminate between the Moldavian (MD) and the Romanian (RO) dialects.

<h2>Submission Format</h2>
For every sample in the dataset, submission files should contain two columns: id and label. The id is the identifier associated to a data sample. The label should be the class 
label, 1 or 2, predicted for the corresponding data sample.

<h2>The documentation should include:</h2>

The description of your machine learning approach including the chosen feature set (words, character n-grams, etc.), model (SVM, neural networks, etc.). Details should also 
include hyperparameter choices (learning rate, performance function, regularization, etc.). A minimum of two pages (excluding tables and figures) is expected
The macro F1 scores for each of the provided validation set and the corresponding confusion matrices.
The python code of your model should include explanatory comments.
