## Topic: Predict whether income exceeds $50K/yr based on census data
##### Introduction: 
###### The level of salary can determine what kind of life a person can live as lower salaries come with more budgeting by the person. We are interested in the relation between people’s attributes (education, race, age, etc) and their salary. After analyzing the data, we will then create a machine learning model to predict whether a person's salary will be greater than $50k a year.

##### Process:
###### 1. Data cleaning
###### 2. Variable selection
###### &nbsp;&nbsp;   - Correlation coefficient
###### &nbsp;&nbsp;   - Information Value
###### 3. Split data into training and testing data set
###### 4. Train models (We pick logistics regression, C5.0 decision tree, and SVM)
###### 5. Use K-fold to get the overall performance of models

<br>

##### Data source:
##### Data set from Kaggle (https://www.kaggle.com/datasets/uciml/adult-census-income). This data was extracted from the 1994 Census bureau and it provides information like age, work class, education, marital status, occupation, race, sex, hours per week, native country, etc.
