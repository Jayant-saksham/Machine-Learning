import streamlit as st 
from sklearn import datasets
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Machine Learning - Jayant Saksham")
st.write(
    '''
    # Explore different classifier and datasets
    Which one is the best ?
    '''
)

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN', 'SVM', 'Random Forest')
)

def get_dataset(name = None):
    '''Getting the features and labels from the datasets
        Input
            - name: Name of the dataset
    '''
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    
    x = data.data 
    y = data.target 
    return (x,y)


def get_classifier():
    pass


x,y = get_dataset(dataset_name)
st.write(f'Shape of dataset {x.shape}')
st.write(f'Number of classes {len(np.unique(y))}')



def add_parameter_ui(clf_name):
    '''Getting a dictionary 
        Input
            - clf_name: Classifier name
    '''
    params = dict()
    if clf_name == 'SVM':
        c = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = c 
    elif clf_name == 'KNN':
        k = st.sidebar.slider('K', 1,15)
        params["K"] = k
    else:
        n_estimator = st.sidebar.slider('n_estimators', 1,100)
        params['n_estimators'] = n_estimator
    return params

params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C = params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors = params['K'])
    else:
        clf = RandomForestClassifier(n_estimators = params['n_estimators'])
    return clf 

clf = get_classifier(classifier_name, params)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
clf.fit(x_train, y_train)
y_predtion = clf.predict(x_test)
acc = accuracy_score(y_true = y_test, y_pred = y_predtion)
st.write(f"Classifier - {classifier_name}")
st.write(f"Accuracy - {acc}")
pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:,0]
x2 = x_projected[:,1]
fig = plt.figure()
plt.scatter(x1, x2, c = y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)

