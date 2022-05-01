"""
Author: Cullen Fitzgerald

Goal: Convert the narrative data to vectors and use these as features
Result: My machine just wasn't powerful enough for this to be practical 
"""




# Dependencies

import pandas as pd
import seaborn as sns
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# Load our data
train = pd.read_csv(r'/Users/cullenfitzgerald/Downloads/classify-reports/train.csv')
test = pd.read_csv(r'/Users/cullenfitzgerald/Downloads/classify-reports/test.csv')

# Check for null values
#print(train.isnull().sum())     # 9029 null values in NARRATIVE

# drop  null values
train = train.dropna()
train.drop('BEGDATE',inplace=True, axis=1)   # Drop the dates

# Best approach will probably be to vectorize the NARRATIVE values. We'll use SBERT
narr = train['NARRATIVE']   # Extract column
narr = narr.dropna()    # Let's try removing nulls

senVecModel = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Our prebuilt SBERT model

"""
narr = narr.tolist()

narrVec = senVecModel.encode(narr,show_progress_bar=True)

#Store sentences & embeddings on disc
with open('narrVec.pkl', "wb") as fOut:
    pickle.dump(narrVec, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    
testNarr = test['NARRATIVE']   # Extract column
testNarr = testNarr.dropna()    # just in case...

testNarr = testNarr.tolist()

testNarrVec = senVecModel.encode(testNarr,show_progress_bar=True)

#Store sentences & embeddings on disc
with open('testNarrVec.pkl', "wb") as fOut:
    pickle.dump(narrVec, fOut, protocol=pickle.HIGHEST_PROTOCOL)
"""

#Load sentences & embeddings from disc
with open('narrVec.pkl', "rb") as fIn:
    narrVec = pickle.load(fIn)

narrVec = pd.Series(list(narrVec))  # Convert back into a series

train['ENCODINGS']=narrVec  # Add a new encoding column to data
train.drop('NARRATIVE', inplace=True, axis=1)   # No longer need the narratives
train.drop('id', inplace=True, axis=1)
test.drop('BEGDATE', inplace=True, axis=1)

# Now train our model
X = train.drop('CRIMETYPE', axis=1)
y = train['CRIMETYPE']
pd.set_option('display.max_columns', None)
print(X['ENCODINGS'][99994])
print(y.head(3))

X, y = make_classification(n_samples=1000, n_features=3,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

# Now build encodings for test data and predict
"""
with open('testNarrVec.pkl', "rb") as fIn:
    testNarrVec = pickle.load(fIn)

testNarrVec = pd.Series(list(testNarrVec))  # Convert back into a series
test['ENCODINGS']=testNarrVec  # Add a new encoding column to data
test.drop('NARRATIVE', inplace=True, axis=1)   # No longer need the narratives

pred = clf.predict(test)
"""
















"""
for i in narr:  # Iterate through and vectorize each item
    if type(i) == float:    # error handling -- NULLs are represented as floats
        narrEnc = 0
    else:
        narrEnc = senVecModel.encode(i,show_progress_bar=True)     # Ok, encode the narrative
    narrVec.append(narrEnc)  # Add encoded vector
"""






# Iterate through and vectorize each item
# Replace nulls with average
# Replace original column with new vectorized column
