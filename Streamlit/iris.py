import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.title("""
Simple Iris Flower Prediction App """)

st.markdown('This app predicts the **Iris flower** type!')

img = Image.open('Image1.png')
st.image(img,width=800)




st.header('User Input Parameters')

def user_input_features():
    sepal_length = st.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    # print(features)
    return features

df = user_input_features()

st.subheader('User Input parameters, from the slider above')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df).tolist()[0]


st.subheader('Class labels and their corresponding index number')
st.write(pd.DataFrame(list(iris.target_names)))


st.subheader('Final Prediction')

st.success(iris.target_names[prediction][0])

st.subheader('Prediction Probability')
df1= pd.DataFrame({'Probabilities':prediction_proba,'Type of flower':iris.target_names})
print(df1)
st.write(df1)




#### PLOT DATASET ####
# Project the data onto the 2 primary principal components

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=Y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=4)
plt.colorbar()


st.header(' The 2 dimensional representation of the Data Set')
#plt.show()
st.pyplot()
