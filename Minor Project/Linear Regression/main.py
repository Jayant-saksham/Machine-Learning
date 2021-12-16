import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sn 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

st.title('Machine Learning Model - Logistic Regression')

breastCancer = load_breast_cancer()

df = pd.DataFrame(breastCancer.data, breastCancer.target)

x_train, x_test, y_train, y_test = train_test_split(breastCancer.data, breastCancer.target, test_size=0.1)

model = LogisticRegression()
model.fit(x_train, y_train)

sc = model.score(x_test, y_test)
st.write(f'Percentage Accuracy of model is {sc*100}%')

y_predicted = model.predict(x_test)

cm = confusion_matrix(y_pred = y_predicted, y_true = y_test)
plt.figure(figsize=(14,8))
c_map = plt.figure()
sn.heatmap(cm, annot = True)
plt.xlabel("Truth")
plt.ylabel("Predicted")
st.pyplot(c_map)

pca = PCA(2)
x_projected = pca.fit_transform(breastCancer.data)
x1 = x_projected[:,0]
x2 = x_projected[:,1]
fig = plt.figure()
plt.scatter(x1, x2, c=breastCancer.target)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
st.pyplot(fig)