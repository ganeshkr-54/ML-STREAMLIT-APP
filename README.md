📌 Problem Statement
The objective of this project is to build and compare multiple machine learning classification models to predict whether an individual earns more than $50,000 per year based on demographic and employment-related attributes.
This is a supervised binary classification problem where the target variable represents income level (>50K or <=50K). The goal is not only to build models but also to evaluate their performance using multiple evaluation metrics and deploy them through an interactive Streamlit web application.

📊 Dataset Description
Dataset Name: Adult Income Dataset
 Source: UCI Machine Learning Repository
 Total Instances: 48,842
 Total Features: 14 input features
 Target Variable: Income (>50K or <=50K)
 Problem Type: Binary Classification
The dataset contains a mix of numerical and categorical features including:
Age


Workclass


Education


Marital Status


Occupation


Relationship


Race


Sex


Capital Gain


Capital Loss


Hours per Week


Native Country


Missing values were handled and categorical features were encoded before model training. Numerical features were scaled using StandardScaler.

🤖 Models Implemented
The following six machine learning models were implemented and evaluated on the same dataset:
Logistic Regression


Decision Tree Classifier


K-Nearest Neighbors (KNN)


Naive Bayes (Gaussian)


Random Forest (Ensemble)


XGBoost (Ensemble Boosting Model)



📈 Model Evaluation Metrics
Each model was evaluated using the following metrics:
Accuracy


AUC (Area Under ROC Curve)


Precision


Recall


F1 Score


Matthews Correlation Coefficient (MCC)







📊 Model Comparison Table


index	Model	Accuracy	AUC	Precision	Recall	F1	MCC
0	Logistic Regression	0.8231394	0.8597735134	0.7444561774	0.4607843137	0.5692369802	0.4868252923
1	Decision Tree	0.8547986077	0.8993235447	0.7924865832	0.5790849673	0.66918429	0.5907299366
2	KNN	0.8257914802	0.8584562361	0.6767527675	0.5993464052	0.6357019064	0.5234413811
3	Naive Bayes	0.7984419029	0.8594766162	0.7098930481	0.3470588235	0.4661984197	0.394551655
4	Random Forest	0.8538040776	0.9137093064	0.8214285714	0.5411764706	0.6524822695	0.5845372003
5	XGBoost	0.8717056191	0.9282842666	0.7939346812	0.6673202614	0.7251420455	0.6464072097



🔍 Observations and Performance Analysis
Logistic Regression
Logistic Regression provides a strong baseline performance due to the relatively linear separability present in several numerical features. It achieves balanced precision and recall while maintaining stable generalization performance.
Decision Tree
The Decision Tree model captures non-linear relationships effectively but tends to overfit the training data when not depth-limited. While accuracy is competitive, its generalization performance may slightly drop compared to ensemble models.
K-Nearest Neighbors (KNN)
KNN performs reasonably well after feature scaling. However, its performance is sensitive to feature normalization and the choice of k. It may struggle with high-dimensional data and large datasets due to computational complexity.
Naive Bayes
Naive Bayes assumes conditional independence between features, which does not fully hold in this dataset. As a result, its performance is slightly lower compared to more flexible models, though it remains computationally efficient.
Random Forest (Ensemble)
Random Forest improves generalization by aggregating multiple decision trees. It reduces overfitting and achieves higher robustness and improved MCC compared to single-tree models.
XGBoost (Ensemble Boosting)
XGBoost delivers the best overall performance in terms of AUC and balanced metrics. Boosting sequentially corrects errors from previous trees, leading to superior predictive performance and better handling of complex feature interactions.

🚀 Deployment
The models were deployed using Streamlit and hosted on Streamlit Community Cloud. The web application allows:
Uploading a CSV file for prediction


Selecting different classification models


Displaying evaluation metrics


Visualizing the confusion matrix


