---
caption: #what displays in the portfolio grid:
  title: Analyzing Survivability In The Titanic
  subtitle: Data Science
  thumbnail: https://i.imgur.com/wJjLuZx.jpg
  
#what displays when the item is clicked:
title: Titanic - Machine Learning from Disaster
subtitle: Predicting survival on the Titanic with Machine Learning
image: https://i.imgur.com/wJjLuZx.jpg #main image, can be a link or a file in assets/img/portfolio
alt: image alt text

---
They say that a data scientist's journey starts with the ever popular Titanic dataset. In this dataset, we are given several features to work with such as age, fare, embark location, ticket class, and many more. [Here](https://www.kaggle.com/c/titanic/data) is the dataset.

## Let's begin!

Let's start by importing the necessary modules first.

```py
import pandas as pd
import random # <-- This we will need later
```
Use pandas to store the dataframe
```py
titanic_train = pd.read_csv('train.csv')
titanic_train.head()
```
![titanic_train.head()](https://i.imgur.com/fNSPIDc.png)
As expected, there are several columns that are not needed such as PassengerId, Ticket, Name, and perhaps even Cabin. We will drop these later on.

Looking at the Sex and Embarked column, it would be much easier if these are converted to numerical order instead. So male/female becomes 1/0 and C/Q/S becomes 1/2/3. Another feature that we could add is Family Count. Looking at SibSp (Sibling and Spouse) and Parch (Parents and Children), we can combine these two features into one big family feature instead.
```py
titanic_train['Sex'] = titanic_train['Sex'].replace("male", 1)
titanic_train['Sex'] = titanic_train['Sex'].replace("female", 0)
titanic_train['Embarked'] = titanic_train['Embarked'].replace("C", 1)
titanic_train['Embarked'] = titanic_train['Embarked'].replace("Q", 2)
titanic_train['Embarked'] = titanic_train['Embarked'].replace("S", 3)
titanic_train['Family Count'] = titanic_train['SibSp'] + titanic_train['Parch'] + 1
```
Now that we have the features we wanted, let's see if there's NaN values in our features.
```py
titanic_train.isna().sum()
```
![titanic_train.isna().sum()](https://i.imgur.com/GgKu11w.png)
Wow! Age has 177 missing values, Cabin has a remarkable 687 missing values and Embarked missing only 2. 

Well, nothing could be done about it. Let's fill these missing values. And since we won't be using Cabin, we will drop them instead.
```py
titanic_train['Age'] = titanic_train['Age'].fillna(titanic_train.Age.median())
titanic_train['Embarked'] = titanic_train['Embarked'].fillna(random.randint(1,3))
```
Notice how I used random to fill Embarked.

Now that everything is filled, let's drop the irrelevant features.
```py
titanic_train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'], axis=1, inplace=True)
```
How does our dataframe look like now?
```py
titanic_train.head()
```
![titanic_train.head()](https://i.imgur.com/EFFoVTs.png)

Much cleaner! Now, do the same thing but to the test.csv which we will use after this.
*Assuming that test.csv is saved at titanic_test.csv and have already been cleaned just like titanic_train, the dataframe should look like this.
```py
titanic_test.head()
```
![titanic_test.head()](https://i.imgur.com/uW0hNZm.png)
Notice the missing Survived column. That is because we will be using this for the submission.

## Time to model!
Fortunately, the creator of the dataset is kind enough to provide different csv for training and testing. Hence, we don't need to split them.
```py
X_train = titanic_train.drop(['Survived'], axis=1)
y_train = titanic_train['Survived']
X_test = titanic_test
```
We will be using these models
```py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
```
*Disclaimer, I don't actually know how 90% of these models work. I just use them because apparently every other person on Kaggle does it.

Now let's use a for loop to model our data and append the result to a list.
```py
classifiers=[['Logistic Regression :',LogisticRegression()],
       ['Decision Tree Classification :',DecisionTreeClassifier()],
       ['Random Forest Classification :',RandomForestClassifier()],
       ['Gradient Boosting Classification :', GradientBoostingClassifier()],
       ['Ada Boosting Classification :',AdaBoostClassifier()],
       ['Extra Tree Classification :', ExtraTreesClassifier()],
       ['K-Neighbors Classification :',KNeighborsClassifier()],
       ['Support Vector Classification :',SVC()],
       ['Gausian Naive Bayes :',GaussianNB()]]
       
cla_pred=[]
for name,model in classifiers:
    model=model
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    cla_pred.append(round(model.score(X_train, y_train) * 100, 2))
```
We could plot each model's result to a barplot to see which model is better.
```py
y_ax=['Logistic Regression' ,
      'Decision Tree Classifier',
      'Random Forest Classifier',
      'Gradient Boosting Classifier',
      'Ada Boosting Classifier',
      'Extra Tree Classifier' ,
      'K-Neighbors Classifier',
      'Support Vector Classifier',
       'Gaussian Naive Bayes']
x_ax=cla_pred

import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x=x_ax,y=y_ax,linewidth=1.5,edgecolor="0.8")
plt.xlabel('Accuracy')
```
![Model results](https://i.imgur.com/kznxoSE.png)
Decision Tree, Random Forest, and Extra Tree got the exact same results as the highest scoring model.

To save you some time, I submitted each model's guesses and found that Random Forest achieved the highest score in the test dataset with 74.401% accuracy! Hey, that's better than a coin flip.

Lastly, I wanted to see the importance of each features.
```py
mod = RandomForestClassifier()
mod.fit(X_train, y_train)

plt.figure(figsize=(12, 5))
plt.bar(range(len(mod.feature_importances_)), mod.feature_importances_)
plt.xticks(range(len(mod.feature_importances_)), X_train.columns)
plt.show()
```
![Feature importances](https://i.imgur.com/Hhn4Yfe.png)
Sex, Age, and Fare appears to be the most relevant feature in our model.

Thanks for reading!

{:.list-inline} 
- Date: April 2021
- Client: Personal
- Category: Data Science

