# Data analysis
- Document here the project: project_fraud
- Description: Project Description
- Data Source:
The dataset from Kaggle, provided by leading payment service company, Vesta Corporation, seeking the best solutions for the fraud prevention industry (https://www.kaggle.com/c/ieee-fraud-detection)
- Description of the project:
This repository contains the procedure we followed to deploy a machine learning model with the highest score as a web app of Credit Card Fraud detection on Heroku.

# 1 st step (data preprocessing)
As in the beginning, we have 4 CSV files ( train_identity, test_identity, train_transactions, text_transactions), we merge them to make data cleaning more efficient. We get rid of rows that contained more than 60% of missing values.

# 2nd step (feature engineering)
We engineered features, that we fought may be useful for our model:
- Weekday
- Hour of the day 
- Credit-card specific features 
- Distance of current transaction from a median value of transactions of one credit card vs. fraud
also, we engineered features connected to mails of users (like mail bin(f.e. Gmail)), but it never improved the performance of the final model, so we get rid of them. 

# 3rd step (Column transformer)
Next, we separated numerical and categorical values. And created a pipeline using Column transformer to the rows that we defined as our features. 
For rows with numerical values we used Simple imputer, where the number of missing values was between 15% - 60%, we used strategy 'mean', for rows the number of missing values was less than 15% - strategy 'median' was used.
As for categorical data, we try not to impute missing values, as it can jeopardize the results of the predictions. that is why we labeled missing values as "Unknown". 

# 4th step Model Training
We trained different models and compare their recall and f1:

1) Svc - recall = 0.65, f1 = 0.09
2) XGB - recall = 0.26, f1 = 0.39 
3) Logistic - recall = 0.71, f1 = 0
4) Lgb: - recall = 0.17, f1 = 0.28
5) Knn - recall = 0.074, f1 = 0.074
6) Random forest - recall = 0.32, f1 = 0.48

Random forest showed the best scores we used it as our final model. 

# 5th step Web App Production

We uploaded our model on Heroku through Flask. The idea of a web app consists of a company/person uploading a CSV file with information about transactions. And our model predicts if the transaction is fraud or not. Weblink: https://dash-fraud.herokuapp.com










