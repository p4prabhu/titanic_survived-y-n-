#!/usr/bin/env python
# coding: utf-8

# ## Titanic Data

# In[60]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import seaborn as sns
import os


# In[61]:


pwd


# In[62]:


# Read CSV train data file into DataFrame

train_df = pd.read_csv(r"C:\\Users\\Prabhat Singh\desktop\dataset\train.csv")

# Read CSV test data file into DataFrame

test_df = pd.read_csv(r"C:\\Users\\Prabhat Singh\desktop\dataset\test.csv")




# In[63]:


train_df.head()


# In[64]:


print("Number of sample in trained data is {}".format(train_df.shape[0]))


# In[65]:


test_df.head()


# In[66]:


print("Number of sample in test data is {}".format(test_df.shape[0]))


# ## Dealing with missing value

# In[67]:


train_df.isnull().sum()


# In[68]:


train_df['Age'].isnull().sum()


# ### lot of missing age and cabin !!
# 

# In[69]:


#percentage of missing age
#note
print('percent of missing Age is %.2f%%'  %((train_df['Age'].isnull().sum()/train_df.shape[0])*100))


# #### approx 20% of age data is missing

# In[70]:


ax=train_df["Age"].hist(bins=15,density=True,stacked=True,color='green',alpha=.7)
train_df['Age'].plot(kind='density',color="green")
ax.set(xlabel="Age")
plt.xlim(-10,85)
plt.show()


# #### since age is right skewed,using mean will give biased result so we will use median for missing value

# In[71]:


# mean age
print('The mean age is %.2f'  %(train_df["Age"].mean(skipna="True")))

# median age
print ("The median age is %.2f" %(train_df["Age"].median()))


# ### cabin missing values

# In[72]:


print("The percentage of missing values in cabin is %.2f%%" %((train_df["Cabin"].isnull().sum()/train_df.shape[0])*100))


# #### approx 77% data is missing which is not good because finding mean or median will give an appropriate solution

# ### missing values in Embarked

# In[73]:


#perentage of missing values in embarked
print("percent of missing Embarked is %.2f%%" %((train_df["Embarked"].isnull().sum()/train_df.shape[0])*100))


# #### so in this case  missing values is less , so we can impute the value which has most frequency  

# In[74]:


print("passengers in the port C = Cherbourg, Q = Queenstown, S = Southampton):")
print(train_df["Embarked"].value_counts())
sns.countplot(x='Embarked',data=train_df, palette='Set2')
plt.show()


# In[75]:


print("most common boarding port is %s" %train_df['Embarked'].value_counts().idxmax())


# By far the most passengers boarded in Southhampton, so we'll impute those 2 NaN's w/ "S".

# #### now
# --If "Age" is missing for a given row, I'll impute with   28 (median age).
# 
# --If "Embarked" is missing for a riven row, I'll impute   with "S" (the most common boarding port).
# 
# --I'll ignore "Cabin" as a variable. There are too many   missing values for imputation. Based on the             information available, it appears that this value is   associated with the passenger's class and fare paid.

# In[76]:


train_data = train_df.copy()
train_data["Age"].fillna(train_df["Age"].median(skipna=True),inplace=True)
train_data["Embarked"].fillna(train_df["Embarked"].value_counts().idxmax(), inplace=True)
train_data.drop("Cabin",axis=1,inplace=True)


# In[77]:


train_data.isnull().sum()


# In[78]:


train_data.head()


# In[79]:


plt.figure(figsize=(15,8))
ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_df["Age"].plot(kind='density', color='teal')
ax = train_data["Age"].hist(bins=15, density=True, stacked=True, color='orange', alpha=0.5)
train_data["Age"].plot(kind='density', color='orange')
ax.legend(['Raw Age', 'Adjusted Age'])
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# #### both SibSp and Parch relate to traveling with family. For simplicity's sake (and to account for possible multicollinearity), I'll combine the effect of these variables into one categorical predictor: whether or not that individual was traveling alone.

# In[80]:


train_data['TravelAlone']= np.where((train_data['SibSp']+train_data["Parch"])>0,0,1)
train_data.drop('SibSp',axis=1,inplace=True)
train_data.drop("Parch",axis=1,inplace=True)


# #### create categorical variables for Passenger Class ("Pclass"), Gender ("Sex"), and Port Embarked ("Embarked").
# 

# In[81]:


#create categorical variables and drop some variables
training=pd.get_dummies(train_data, columns =['Pclass','Embarked','Sex'])
training.drop( 'Sex_female',axis=1,inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)
final_train = training
final_train.head()








# ### Now, apply the same changes to the test data.
# I will apply to same imputation for "Age" in the Test data as I did for my Training data (if missing, Age = 28).
# I'll also remove the "Cabin" variable from the test data, as I've decided not to include it in my analysis.
# There were no missing values in the "Embarked" port variable.
# I'll add the dummy variables to finalize the test set.
# Finally, I'll impute the 1 missing value for "Fare" with the median, 14.45.

# In[82]:


test_df.isnull().sum()


# In[83]:


test_data =test_df.copy()
test_data['Age'].fillna(train_df['Age'].median(skipna=True),inplace=True)
test_data["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)
test_data['TravelAlone']= np.where((test_data['SibSp']+test_data['Parch'])>0,0,1)
test_data.drop('SibSp', axis=1,inplace=True)
test_data.drop('Parch',axis=1,inplace=True)


# In[84]:


testing =pd.get_dummies(test_data,columns=['Sex','Pclass','Embarked'])
testing.drop("Sex_female",axis=1,inplace=True)
testing.drop("Name",axis=1,inplace=True)
testing.drop("PassengerId",axis=1,inplace=True)
testing.drop("Ticket",axis=1,inplace=True)
final_test=testing
final_test.head()


# ## Exploratory Data Analysis

# In[85]:


plt.figure(figsize=(15,8))
ax=sns.kdeplot(final_train['Age'][final_train.Survived==1],color='orange',shade=True)
sns.kdeplot(final_train["Age"][final_train.Survived==0],color='green',shade=True)
plt.legend(['survived'],['Died'])
plt.title("density plot of age for survival")
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# #### The age distribution for survivors and deceased is actually very similar. One notable difference is that, of the survivors, a larger proportion were children. The passengers evidently made an attempt to save children by giving them a place on the life rafts.

# In[86]:


plt.figure(figsize=(15,8))
avg_survival_byage= final_train[['Age','Survived']].groupby(['Age'], as_index =False).mean()
g=sns.barplot(x="Age",y='Survived',data=avg_survival_byage,color='brown')
plt.show()


# #### Considering the survival rate of passengers under 16, I'll also include another categorical variable in my dataset: "Minor"

# In[87]:


final_train['IsMinor']=np.where(final_train['Age']<=16,1,0)
final_train['IsMinor']=np.where(final_train['Age']<=16,1,0)


# ### Exploration of Fare

# In[88]:


plt.figure(figsize=(15,8))
ax= sns.kdeplot(final_train['Fare'][final_train.Survived==1],color='orange',shade=True)
sns.kdeplot(final_train['Fare'][final_train.Survived==0],color='green',shade=True)
plt.legend(['survived','died'])
plt.title("density plot of fare")
ax.set(xlabel='Fare')
plt.xlim(-20,200)
plt.show()


# #### As the distributions are clearly different for the fares of survivors vs. deceased, it's likely that this would be a significant predictor in our final model. Passengers who paid lower fare appear to have been less likely to survive. This is probably strongly correlated with Passenger Class, which we'll look at next.

# ### Exploration of Passenger Class

# In[89]:


sns.barplot('Pclass','Survived',data=train_df,color="blue")
plt.show()


# #### being a first class passenger was safest.

# ### Exploration of Embarked Port

# In[90]:


sns.barplot('Embarked','Survived',data=train_df, color='green')
plt.show()


# #### Passengers who boarded in Cherbourg, France, appear to have the highest survival rate. Passengers who boarded in Southhampton were marginally less likely to survive than those who boarded in Queenstown. This is probably related to passenger class, or maybe even the order of room assignments (e.g. maybe earlier passengers were more likely to have rooms closer to deck).
# #### It's also worth noting the size of the whiskers in these plots. Because the number of passengers who boarded at Southhampton was highest, the confidence around the survival rate is the highest. The whisker of the Queenstown plot includes the Southhampton average, as well as the lower bound of its whisker. It's possible that Queenstown passengers were equally, or even more, ill-fated than their Southhampton counterparts.

# ### Exploration of Traveling Alone vs. With Family

# In[91]:


sns.barplot('TravelAlone','Survived',data=final_train,color='red')
plt.show()


# #### Individuals traveling without family were more likely to die in the disaster than those with family aboard. Given the era, it's likely that individuals traveling alone were likely male.

# ### Exploration of Gender Variable

# In[92]:


sns.barplot('Sex','Survived',data=train_df,color='green')
plt.show()


# #### This is a very obvious difference. Clearly being female greatly increased your chances of survival.

# ### Logistic Regression and Results

# #### Feature selection
# 

# #### Recursive feature elimination
# Given an external estimator that assigns weights to features, recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features.That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

# In[93]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
cols=['Age','Fare','TravelAlone','Pclass_1',"Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"]
X=final_train[cols]
y=final_train['Survived']
# build logreg and compute
model =LogisticRegression()
#build RFE model with 8 attribute
rfe=RFE(model,8)
rfe=rfe.fit(X,y)
print("selected feature : %s" %list(X.columns[rfe.support_]))


# #### Feature ranking with recursive feature elimination and cross-validation
# RFECV performs RFE in a cross-validation loop to find the optimal number or the best number of features. Hereafter a recursive feature elimination applied on logistic regression with automatic tuning of the number of features selected with cross-validation.

# In[94]:


from sklearn.feature_selection import RFECV
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv=RFECV(estimator=LogisticRegression() , step=1, cv=10, scoring='accuracy')
rfecv.fit(X,y)
print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))
 
#plot
plt.figure(figsize=(10,6))
plt.xlabel("number of feature selected")
plt.ylabel("cross validation score(number of correct classification)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# As we see, eight variables were kept.

# In[95]:


Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C', 
                     'Embarked_S', 'Sex_male', 'IsMinor']
X = final_train[Selected_features]
plt.subplots(figsize=(8, 5))
sns.heatmap(X.corr() ,annot=True, cmap="RdYlGn")
plt.show()
    


# ### Review of model evaluation procedures

# #### Motivation: Need a way to choose between machine learning models
# 
# Goal is to estimate likely performance of a model on out-of-sample data
# Initial idea: Train and test on the same data
# 
# But, maximizing training accuracy rewards overly complex models which overfit the training data
# Alternative idea: Train/test split
# 
# Split the dataset into two pieces, so that the model can be trained and tested on different data
# 
# Testing accuracy is a better estimate than training accuracy of out-of-sample performance
# 
# Problem with train/test split
# It provides a high variance estimate since changing which observations happen to be in the testing set can significantly change testing accuracy
# Testing accuracy can change a lot depending on a which observation happen to be in the testing set
# 

# ### Model evaluation based on simple train/test split using train_test_split() function
# 

# In[96]:


from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score,recall_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
# create X (features) and y (response)
X = final_train[Selected_features]
y = final_train['Survived']
# use train/test split with different random_state values
# we can change the random_state values that changes the accuracy scores
# the scores change a lot, this is why testing scores is a high-variance estimate
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=2)
# check classification scores of logistic regression
logreg= LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
y_pred_proba=logreg.predict_proba(X_test)[:,1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95
#plot

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))



# #### Model evaluation based on K-fold cross-validation using cross_val_score() function

# In[97]:


# 10-fold cross-validation logistic regression
logreg = LogisticRegression()
# Use cross_val_score function
# # We are passing the entirety of X and y, not X_train or y_train, it takes care of splitting the data
# cv=10 for 10 folds
# scoring = {'accuracy', 'neg_log_loss', 'roc_auc'} for evaluation metric - althought they are many
scores_accuracy = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
scores_log_loss = cross_val_score(logreg, X, y, cv=10, scoring='neg_log_loss')
scores_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc')
print('K-fold cross-validation results:')
print(logreg.__class__.__name__+" average accuracy is %2.3f" % scores_accuracy.mean())
print(logreg.__class__.__name__+" average log_loss is %2.3f" % -scores_log_loss.mean())
print(logreg.__class__.__name__+" average auc is %2.3f" % scores_auc.mean())


# In[98]:


from sklearn.model_selection import cross_validate
scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}
modelCV = LogisticRegression()
results = cross_validate(modelCV, final_train[cols], y, cv=10, scoring=list(scoring.values()), 
                         return_train_score=False)
print('K-fold cross-validation results:')
for sc in range(len(scoring)):
    print(modelCV.__class__.__name__+" average %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values())[sc]].mean()
                               if list(scoring.values())[sc]=='neg_log_loss' 
                               else results['test_%s' % list(scoring.values())[sc]].mean(), 
                               results['test_%s' % list(scoring.values())[sc]].std()))


# We notice that the model is slightly deteriorated. The "Fare" variable does not carry any useful information. Its presence is just a noise for the logistic regression model.

# #### GridSearchCV evaluating using multiple scorers simultaneously

# In[99]:


from sklearn.model_selection import GridSearchCV

X = final_train[Selected_features]

param_grid = {'C': np.arange(1e-05, 3, 0.1)}
scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}
gs = GridSearchCV(LogisticRegression(), return_train_score=True,
                  param_grid=param_grid, scoring=scoring, cv=10, refit='Accuracy')
gs.fit(X, y)
results = gs.cv_results_

print('='*20)
print("best params: " + str(gs.best_estimator_))
print("best params: " + str(gs.best_params_))
print('best score:', gs.best_score_)
print('='*20)

plt.figure(figsize=(10, 10))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",fontsize=16)

plt.xlabel("Inverse of regularization strength: C")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
ax.set_xlim(0, param_grid['C'].max()) 
ax.set_ylim(0.35, 0.95)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_C'].data, dtype=float)

for scorer, color in zip(list(scoring.keys()), ['g', 'k', 'b']): 
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = -results['mean_%s_%s' % (sample, scorer)] if scoring[scorer]=='neg_log_loss' else results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = -results['mean_test_%s' % scorer][best_index] if scoring[scorer]=='neg_log_loss' else results['mean_test_%s' % scorer][best_index]
        
        # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()


# ### GridSearchCV evaluating using multiple scorers, RepeatedStratifiedKFold and pipeline for preprocessing simultaneously
# We can applied many tasks together for more in-depth evaluation like gridsearch using cross-validation based on k-folds repeated many times, that can be scaled or no with respect to many scorers and tunning on parameter for a given estimator!

# In[100]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
#Define simple model
###############################################################################
C = np.arange(1e-05, 5.5, 0.1)
scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}
log_reg = LogisticRegression()

#Simple pre-processing estimators
###############################################################################
std_scale = StandardScaler(with_mean=False, with_std=False)
#std_scale = StandardScaler()

#Defining the CV method: Using the Repeated Stratified K Fold
###############################################################################

n_folds=5
n_repeats=5

rskfold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=2)

#Creating simple pipeline and defining the gridsearch
###############################################################################

log_clf_pipe = Pipeline(steps=[('scale',std_scale), ('clf',log_reg)])

log_clf = GridSearchCV(estimator=log_clf_pipe, cv=rskfold,
              scoring=scoring, return_train_score=True,
              param_grid=dict(clf__C=C), refit='Accuracy')

log_clf.fit(X, y)
results = log_clf.cv_results_
print('='*20)
print("best params: " + str(log_clf.best_estimator_))
print("best params: " + str(log_clf.best_params_))
print('best score:', log_clf.best_score_)
print('='*20)

plt.figure(figsize=(10, 10))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",fontsize=16)

plt.xlabel("Inverse of regularization strength: C")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
ax.set_xlim(0, C.max()) 
ax.set_ylim(0.35, 0.95)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_clf__C'].data, dtype=float)

for scorer, color in zip(list(scoring.keys()), ['g', 'k', 'b']): 
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = -results['mean_%s_%s' % (sample, scorer)] if scoring[scorer]=='neg_log_loss' else results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = -results['mean_test_%s' % scorer][best_index] if scoring[scorer]=='neg_log_loss' else results['mean_test_%s' % scorer][best_index]
    
     # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()


# In[148]:


final_train['Survived']= log_clf.predict(final_train[Selected_features])
final_test['PassengerId'] = test_df['PassengerId']
submission = final_train[['PassengerId','Survived']]
submission.to_csv("submission.csv", index=False)
submission.tail(10)



# In[ ]:




