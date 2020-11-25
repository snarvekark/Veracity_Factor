import pandas as pd
import numpy as np
import csv
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from string import punctuation
import seaborn as sns
import pandas as pd
import numpy as np
import nltk
import re
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import matplotlib.pyplot as plt
from scipy import sparse

from google.colab import drive
drive.mount("/content/drive")

class Context_Veracity():
  def __init__(self):
    # Reading the tsv file into DataFrame
    test_filename = 'Veracity/test2.tsv'
    train_filename = 'Veracity/train2.tsv'
    valid_filename = 'Veracity/val2.tsv'
    colnames = ['jsonid', 'label', 'headline_text', 'subject', 'speaker', 'speakerjobtitle', 'stateinfo','partyaffiliation', 'barelytruecounts', 'falsecounts','halftruecounts','mostlytruecounts','pantsonfirecounts','context', 'text']

    self.train_news = pd.read_csv(train_filename, sep='\t', names = colnames, error_bad_lines=False)
    self.test_news = pd.read_csv(test_filename, sep='\t', names = colnames, error_bad_lines=False)
    self.valid_news = pd.read_csv(valid_filename, sep='\t', names = colnames, error_bad_lines=False)

    self.news_text = ""

#####################################################

  def get_veracity_scores(self):
    self.train_news.drop('jsonid', axis = 1, inplace = True)
    self.test_news.drop('jsonid', axis = 1, inplace = True)
    self.valid_news.drop('jsonid', axis = 1, inplace = True)

    self.train_news['label'] = self.train_news['label'].astype('category')
  # Assigning numerical values and storing in another column
    self.train_news['Label_cat'] = self.train_news['label'].cat.codes

    self.test_news['label'] = self.test_news['label'].astype('category')
  # Assigning numerical values and storing in another column
    self.test_news['Label_cat'] = self.test_news['label'].cat.codes

    self.valid_news['label'] = self.valid_news['label'].astype('category')
  # Assigning numerical values and storing in another column
    self.valid_news['Label_cat'] = self.valid_news['label'].cat.codes

    cat_to_nums_train = {"label":     {"true":1, "false":0, "barely-true": 0, "half-true": 0, "mostly-true": 1,"pants-fire": 0} }
    self.train_news.replace(cat_to_nums_train, inplace=True)
    self.train_news['label']

    cat_to_nums_test = {"label":     {"true":1, "false":0, "barely-true": 0, "half-true": 0, "mostly-true": 1,"pants-fire": 0} }
    self.test_news.replace(cat_to_nums_test, inplace=True)
    self.test_news['label']

    cat_to_nums_valid = {"label":     {"true":1, "false":0, "barely-true": 0, "half-true": 0, "mostly-true": 1,"pants-fire": 0} }
    self.valid_news.replace(cat_to_nums_valid, inplace=True)
    self.valid_news['label']

    self.train_news['veracity'] = 0
    self.test_news['veracity'] = 0
    self.valid_news['veracity'] = 0

    falseNegative = 0
    falsePositive = 0

    trueNegative = 0
    truePositive = 0


    for index, row in self.train_news.iterrows():
      if (row.label == 0.0):
        if ((row.barelytruecounts < 4) | (row.falsecounts > 2) | (row.pantsonfirecounts > 1)):
          trueNegative += 1
          self.train_news.loc[index,'veracity'] = 1
        else:
          falseNegative += 1
          self.train_news.loc[index,'veracity'] = 0
      elif (row.label == 1.0):
        if ((row.halftruecounts > 4) | (row.mostlytruecounts > 4)):
          truePositive += 1
          self.train_news.loc[index,'veracity'] = 1
        else:
          falsePositive += 1
          self.train_news.loc[index,'veracity'] = 0
    print("trueNegative=", trueNegative)
    print("falseNegative=", falseNegative)
    print("truePositive=", truePositive)
    print("falsePositive=", falsePositive)

    #Drop any Null values with no source information
    self.train_news = self.train_news.dropna(how='any',axis=0) 

    for index, row in self.test_news.iterrows():
      if (row.label == 0.0):
        if ((row.barelytruecounts < 4) | (row.falsecounts > 2) | (row.pantsonfirecounts > 1)):
          self.test_news.loc[index,'veracity'] = 1
        else:
          self.test_news.loc[index,'veracity'] = 0
      elif (row.label == 1.0):
        if ((row.halftruecounts > 4) | (row.mostlytruecounts > 4)):
          self.test_news.loc[index,'veracity'] = 1
        else:
          falsePositive += 1
          self.test_news.loc[index,'veracity'] = 0

    for index, row in self.valid_news.iterrows():
      if (row.label == 0.0):
        if ((row.barelytruecounts < 4) | (row.falsecounts > 2) | (row.pantsonfirecounts > 1)):
          self.valid_news.loc[index,'veracity'] = 1
        else:
          self.valid_news.loc[index,'veracity'] = 0
      elif (row.label == 1.0):
        if ((row.halftruecounts > 4) | (row.mostlytruecounts > 4)):
          self.valid_news.loc[index,'veracity'] = 1
        else:
          falsePositive += 1
          self.valid_news.loc[index,'veracity'] = 0

    score = self.train_data()
    return score

####################################################################################################
  def train_data(self):
    col_names = ['barelytruecounts', 'falsecounts', 'halftruecounts', 'mostlytruecounts', 'pantsonfirecounts', 'veracity']
    X_train = self.train_news[col_names]
    Y_train = self.train_news['label']
    X_test = self.test_news[col_names]
    Y_test = self.test_news['label']

    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train,Y_train)
    pred = model.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print("Confusion Matrix: \n")
    print(confusion_matrix(Y_test,pred))
    print("Classification Report:\n")
    print(classification_report(Y_test,pred))
    print('Accuracy score for Naive Bayes:\n', accuracy_score(Y_test, pred))
    return accuracy_score(Y_test, pred)

####################################################################################

  def predict_veracity(self, text, c1, c2, c3, c4, c5):
    df = pd.DataFrame(columns = ['barelytruecounts', 'falsecounts', 'halftruecounts', 'mostlytruecounts', 'pantsonfirecounts', 'veracity', 'label'])
    #df.columns = ['barelytruecounts', 'falsecounts', 'halftruecounts', 'mostlytruecounts', 'pantsonfirecounts', 'veracity', 'label']
    if ((c1 < 4) | (c2 > 2) | (c5 > 1)):
      veracity = 0
    else:
      veracity = 1
    data = {"barelytruecounts": c1, "falsecounts": c2, "halftruecounts": c3, "mostlytruecounts": c4, "pantsonfirecounts": c5, "veracity": veracity, "label": 2}
    df = df.append(data, ignore_index=True)
    
    col_names = ['barelytruecounts', 'falsecounts', 'halftruecounts', 'mostlytruecounts', 'pantsonfirecounts', 'veracity']
    X_train = self.train_news[col_names]
    Y_train = self.train_news['label']
    X_test = self.test_news[col_names]
    Y_test = self.test_news['label']

    from sklearn.svm import SVC
    model = SVC()
    model.fit(X_train,Y_train)
    pred = model.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print("Confusion Matrix: \n")
    print(confusion_matrix(Y_test,pred))
    print("Classification Report:\n")
    print(classification_report(Y_test,pred))
    print('Accuracy score for SVM:\n', accuracy_score(Y_test, pred))
    return accuracy_score(Y_test, pred)

    ############# Test Custom input ##################
    #print(df)
    #X1_test = df[col_names]
    #Y1_test = df['label']
    #pred = model.predict(X_test)
    #from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    #print('Accuracy score for SVM:\n', accuracy_score(Y1_test, pred))
    #return accuracy_score(Y1_test, pred)


#####################################################################################

cv = Context_Veracity()
score = cv.get_veracity_scores()
print(score)
#return score
#custom_score = cv.predict_veracity("Says the Annies List political group supports third-trimester abortions on demand.", 1, 4, 2, 0, 0)
