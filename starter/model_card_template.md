# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Megan McGee created this model. It is a gradient boosting classifier using scikit-learn 0.24.2 with all default hyperparameters except subsample=0.8 and max_features=0.8.

## Intended Use
This model should be used to predict whether salary is more than $50,000 or not based off of census data. The intended users for this model are researchers investigating bias.

## Training Data
The data was obtained from https://github.com/udacity/nd0821-c3-starter-code/tree/master/starter/data as census.csv, though it is originally from archive.ics.uci.edu/ml/datasets/census+income.

Spaces were removed from the file in preprocessing. A total of 32,561 rows are included in the data. A random sample of 80% of this data was used for training. The training data was processed using one hot encoding and a label binarizer.

## Evaluation Data
The remaining 20% of the data was used for evaluation. The same encoder and label binarizer were used on the evaluation data.

## Metrics
On the evaluation data, the model performance is as follows:
- Precision: 0.7528455284552845
- Recall: 0.6227303295225286
- F1 Score: 0.6816341553183658

## Ethical Considerations
The data is not evenly distributed across subpopulations. Performance for groups that are not well-represented may suffer. For example, for Amer-Indian-Eskimo, precision, recall, and F1 Score are all well below the overall metrics. Please see slice_output.txt for model performance metrics by subgroup.

## Caveats and Recommendations
While this model is interesting to use to investigate performance differences by subgroup, this model should not be used by any end users to predict salary. This model is produced for investigatory purposes, but due to the ethical considerations above, it should not be relied upon for predictive purposes.

