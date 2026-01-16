**Assessment rules**

Mihai Toma and Piotr Wójcik
academic year 2025/2026
General information
Your will work on a practical machine learning project related to classification. The data is exactly the same for all students.
_
dataset: job_change_
Your task is to apply various ML algorithms (see the rules below) to build a model explaining whether a particular person is willing to change job based on the training sample and generate predictions for all observations from the test sample.

The dataset includes 12427 observations in the training sample and 3308 in the test sample and the following columns:

id – unique observation identifier
gender – gender of a person
age – age of a person in years
education – highest formal education level of a person attained so far
field_of_studies – field of studies of a person
is_studying – information whether a person is currently studying
county – code of the county in which the person currently lives and works
relative_wage – relative wage in the county of residence (as percentage of country average)
years_since_job_change – years since a person last changed job
years_of_experience – total number of years of professional experience of a person
hours_of_training – total number of training hours completed by a person
is_certified – does a person have any formal certificate of completed trainings
size_of_company – size of a company in which a person currently works
type_of_company – type of a company in which a person currently works
willing_to_change_job – is a person willing to change job (outcome variable, only in the training sample)
Various algorithms
Please consider and compare at least 5 different ML algorithms discussed in the course, e.g. logistic regression, KNN, LASSO, ridge, elastic net, SVM with various kernel functions, decision tree, random forest, different boosting algorithms (e.g. gbm, xgboost, adaboost, catboost), neural network. You can also apply any combination of the above mentioned models (melting or stacking).
