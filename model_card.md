# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier trained to predict whether an individual's annual income exceeds $50K 
based on census demographic data. The model uses 100 decision trees with a fixed random state of 42 for reproducibility.
It was implemented using scikit-learn and trained on preprocessed census data with categorical fatures encoded using 
one-hot encoding.

## Intended Use
This model is intended for educational purposes to demonstrate machine learning pipeline development and deployment. 
It predicts binary income classification (<=50K or >50K) based on demographic features such as age, education, occupation, 
and work class. The model should not be used for making actual hiring, lending, or other consequential decisions about 
individuals.

## Training Data
The model was trained on the Adult Census Income dataset containing 32,561 records from the 1994 Census Bureau database. 
The dataset includes 14 features: age, workclass, education, marital status, occupation, relationship, race, sex, capital 
gain/loss, hours per week, and native country. Missing values encoded as '?' were replaced with the most frequent value for 
each categorical feature. The data was split 80/20 for training and testing.

## Evaluation Data
The model was evaluated on a held-out test set containing 20% of the original data (approximately 6,513 samples). 
The same preprocessing steps applied to training data were used on the test set, including one-hot encoding of categorical 
features and missing value imputation.

## Metrics
The model performance was evaluated using precision, recall, and F1-score. Overall performance on the test set: 
Precision: 0.7444, 
Recall: 0.6359, 
F1-score: 0.6859. 
These metrics indicate the model correctly identifies high-income individuals 74% of the time when it predicts them 
(precision), but only captures 64% of all actual high-income individuals (recall).

_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations
The slice analysis revealed significant performance disparities across demographic groups. Some concerning findings 
include very low performance for certain racial and national origin groups, and perfect scores on very small sample 
sizes that may not be reliable. The model shows potential bias that could lead to unfair treatment of underrepresented 
groups. Users should be aware that historical census data may perpetuate existing societal biases.

## Caveats and Recommendations
This model should not be used for real-world income prediction or decision-making due to potential bias and the age of the 
training data (1994). Performance varies significantly across demographic slices, with some groups having insufficient 
representation for reliable predictions. The model assumes the same socioeconomic patterns from 1994 apply today, which is 
likely outdated. Future work should include bias mitigation techniques and more recent, representative data.
