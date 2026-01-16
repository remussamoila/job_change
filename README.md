**AI Strategy and Digital transformation**

Assessment rules

Mihai Toma and Piotr Wójcik

academic year 2025/2026

General information

Your will work on a practical machine learning project related to **classification**. The data is **exactly the same for all students**.

dataset: job_change

Your task is to apply various ML algorithms (see the rules below) to build a model explaining whether a particular person **is willing to change job** based on the **training sample** and generate predictions for **all observations** from the **test sample**.

The dataset includes 12427 observations in the training sample and 3308 in the test sample and the following columns:

- id - unique observation identifier
- gender - gender of a person
- age - age of a person in years
- education - highest formal education level of a person attained so far
- field_of_studies - field of studies of a person
- is_studying - information whether a person is currently studying
- county - code of the county in which the person currently lives and works
- relative_wage - relative wage in the county of residence (as percentage of country average)
- years_since_job_change - years since a person last changed job
- years_of_experience - total number of years of professional experience of a person
- hours_of_training - total number of training hours completed by a person
- is_certified - does a person have any formal certificate of completed trainings
- size_of_company - size of a company in which a person currently works
- type_of_company - type of a company in which a person currently works
- willing_to_change_job - is a person willing to change job (**outcome variable**, only in the training sample)

Various algorithms

Please consider and compare **at least 5 different ML algorithms discussed in the course**, e.g. logistic regression, KNN, LASSO, ridge, elastic net, SVM with various kernel functions, decision tree, random forest, different boosting algorithms (e.g. gbm, xgboost, adaboost, catboost), neural network. You can also apply any combination of the above mentioned models (melting or stacking).

Selection of the best algorithm

The choice of the final algorithm applied to generate predictions should be **clearly explained** in the presentation.

HINT !!!!! Use cross-validation to make sure that you correcly assess the performance of your models on the new data and find the best variants of the applied algorithms (hyperparameter tuning).

Performance measure

The performance of predictions will be based on the balanced accuracy (non-weighted arithmetic average of recall for 0s and recall for 1s) - in python called "macro avg accuracy".

Please report the **expected** value of a particular performance measure (expectation for the test sample) in your presentation.

Points

In total **50 points** can be collected:

- **presentation** - its structure, way of presenting, etc. (**20 pts**)
- **presentation contents** - assessed by the lecturer after you present in class (**20 pts**)
- **test sample** performance (**10 pts**):
  - 10 if predictive performance in top quartile group (best),
  - 7.5 if predictive performance in the 2nd quartile group (good),
  - 5 if predictive performance in the 3rd quartile group (below average),
  - 2.5 if predictive performance in the 4th quartile group (unlucky),

Presentations

The **presentation** together with the **best model python codes** have to be submitted by email to the lecturer [mihai.toma@fabiz.ase.ro](file:///C:\Users\Remus\OneDrive\Master%20Business%20Analytics%202024\2025-2026\S1\AI%20strategy%20and%20digital%20transformation\Piotr%20Wojcik\2026-01_Bucharest\_assessment_rules_and_data\mihai.toma@fabiz.ase.ro) or [pwojcik@wne.uw.edu.pl](mailto:pwojcik@wne.uw.edu.pl) **until midnight eod 2026-01-18**. The provided codes should **load the training data**, **train** the single best algorithm, **apply** this model on the **test data** and save test data predictions in the csv file, which **should be also attached**. The file with predictions should **only include** the observation **id** and the **predicted value of the outcome variable**. In addition, you should also provide the link to the repository, where you store your full codes, that were used to

In the best model codes do NOT include all the codes which you applied to find the best algorithm, parameter search, etc. ONLY a simple code for a FINALLY selected algorithm i.e. with the selected set of best performing parameters. The FULL codes with all the algos and attempts you applied should be in the repository (google drive or Github) that you share with the lecturers.

All students will give presentations (not more than **10 minutes**) informing about the algorithms considered, selection process and their **expected results**.

Presentations will take place on 2026-01-19.

Students that do NOT present their results in class **will not be graded**.

Important dates again

- 2026-01-18 by 23:59 - submission of **presentations**, **codes** and **test sample predictions**,
- 2026-01-19 - **presentations**.

Each submission should be done **via email** to [mihai.toma@fabiz.ase.ro](file:///C:\Users\Remus\OneDrive\Master%20Business%20Analytics%202024\2025-2026\S1\AI%20strategy%20and%20digital%20transformation\Piotr%20Wojcik\2026-01_Bucharest\_assessment_rules_and_data\mihai.toma@fabiz.ase.ro) or [pwojcik@wne.uw.edu.pl](mailto:pwojcik@wne.uw.edu.pl) **before midnight** of the deadline day **if not stated otherwise**.
