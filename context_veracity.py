import pandas as pd
import numpy as np
import csv
import gensim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np
import nltk
import re
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import matplotlib.pyplot as plt
from scipy import sparse
from google_drive_downloader import GoogleDriveDownloader as gdd


from zipfile import ZipFile
import pickle

class Context_Veracity():
  def __init__(self):
    gdd.download_file_from_google_drive(file_id='17YjVfE6SH_x-iqDvihhov-2N_3o-KJCC',
                                  dest_path='./context_veracity_model.zip',
                                  unzip=False)
    self.model = None 
    colnames = ['jsonid', 'label', 'headline_text', 'subject', 'speaker', 'speakerjobtitle', 'stateinfo','partyaffiliation', 'barelytruecounts', 'falsecounts','halftruecounts','mostlytruecounts','pantsonfirecounts','context', 'text']

    # unpickling models
    names = ["Linear SVM"]
    with ZipFile('context_veracity_model.zip', 'r') as myzip:
        for name in names:
            self.model = pickle.load(myzip.open(f'{name}_model.pickle'))
            #print(clf_reload)


  def get_veracity_scores(self, title):
    #calculate title_count on veracity
    return self.get_veracity(0, 2)

  def get_veracity(self, title_count, veracity):
    df = pd.DataFrame(columns=['veracity', 'title_count'])
    df.loc[0]=[veracity, title_count]
    result = self.model.predict(df)
    return result
