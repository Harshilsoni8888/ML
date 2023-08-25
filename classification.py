

# Import libraries and classes as required:
import pandas as pd  ## To load the data and create DataFrame
import matplotlib.pyplot as plt ## For plotting of data
import seaborn as sns ## For plotting of data
from sklearn.model_selection import train_test_split ## Holdout - splitting data to train and test
from sklearn.preprocessing import StandardScaler   ## To scale the numeric values 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score ## To display the output

#1. CLASSIFICATION USING KNN
from sklearn.neighbors import KNeighborsClassifier  ## For KNN Classification method
# Load Dataset
df = pd.read_csv('/content/sample_data/Iris.csv')

#Dividing Data Into Features and Labels
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Assign values to the X and y variables:
X = df[feature_columns].values
y = df['Species'].values

# Assign values to the X and y variables: Alternative method
# X= df.iloc[:, [1,5]].values  
# y= df.iloc[:, 5].values  

# Split dataset into random train and test subsets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

# Standardize features by removing mean and scaling to unit variance:
# scaler = StandardScaler()
# scaler.fit(X_train)

# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test) 

# Use the KNN classifier to fit data:
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train) 

# Predict y data with classifier: 
y_predict = classifier.predict(X_test)

# Print results: 
print("CONFUSION MATRIX : ") 
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict)) 


#------------------ PLOTTING USING HEATMAP--------------------------

# cm = confusion_matrix(y_test, y_predict) 

# # # Transform to df for easier plotting
# cm_df = pd.DataFrame(cm,
#                       index = ['setosa','versicolor','virginica'], 
#                       columns = ['setosa','versicolor','virginica'])

# plt.figure(figsize=(5.5,4))
# sns.heatmap(cm_df, annot=True)
# plt.title('KNN \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_predict)))
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()


#2. CLASSIFICATION USING SVM

#Import SVM model
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv('/content/sample_data/Iris.csv')

#Dividing Data Into Features and Labels
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Assign values to the X and y variables:
X = df[feature_columns].values
y = df['Species'].values
 

# label_encoder_y= LabelEncoder()
# y= label_encoder_y.fit_transform(y)

# Split dataset into random train and test subsets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# Use the SVM classifier to fit data:
classifier1 = svm.SVC(kernel='sigmoid') # linear, sigmoid, rbf
classifier1.fit(X_train, y_train) 

# # Predict y data with classifier: 
y_predict = classifier1.predict(X_test)

# Print results: 
print("CONFUSION MATRIX : ") 
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

#3. CLASSIFICATION USING DECISION TREE

#Import Decision Tree model
from sklearn.tree import DecisionTreeClassifier

# Load Dataset
df = pd.read_csv('/content/sample_data/Iris.csv')


#Dividing Data Into Features and Labels
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Assign values to the X and y variables:
X = df[feature_columns].values
y = df['Species'].values

# Split dataset into random train and test subsets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44) 
 
# Use the Decision Tree classifier to fit data:
classifier2 = DecisionTreeClassifier(criterion="gini")
# train the model
classifier2.fit(X_train, y_train)

# Predict y data with classifier: 
y_predict = classifier2.predict(X_test)

# Print results: 
print("CONFUSION MATRIX : ") 
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict)) 
print("Accuracy:",accuracy_score(y_test, y_predict))

#4. CLASSIFICATION USING GAUSSIAN NAIVE BAYES

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

# Load Dataset
df = pd.read_csv('/content/sample_data/Iris.csv')

#Dividing Data Into Features and Labels
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Assign values to the X and y variables:
X = df[feature_columns].values
y = df['Species'].values

# Split dataset into random train and test subsets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

# Use the Decision Tree classifier to fit data:
classifier3 = GaussianNB()
# train the model
classifier3.fit(X_train, y_train)
# Predict y data with classifier: 
y_predict = classifier3.predict(X_test)

# Print results: 
print("CONFUSION MATRIX : ") 
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

#5. CLASSIFICATION USING RANDOM FOREST

#Import RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv('/content/sample_data/Iris.csv')

#Dividing Data Into Features and Labels
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Assign values to the X and y variables:
X = df[feature_columns].values
y = df['Species'].values

# Split dataset into random train and test subsets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 

# Use the Decision Tree classifier to fit data:
classifier4 = RandomForestClassifier()
# train the model
classifier4.fit(X_train, y_train)
# Predict y data with classifier: 
y_predict = classifier4.predict(X_test)

# Print results: 
print("CONFUSION MATRIX : ") 
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

#5. REGRESSION USING LOGISTIC REGRESSION

#Import LOGISTIC REGRESSION CLASSIFIER
from sklearn.linear_model import LogisticRegression

# Load Dataset
df = pd.read_csv('/content/sample_data/Iris.csv')

#Dividing Data Into Features and Labels
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Assign values to the X and y variables:
X = df[feature_columns].values
y = df['Species'].values

# Split dataset into random train and test subsets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

# Use the Decision Tree classifier to fit data:
classifier5 = LogisticRegression()
# train the model
classifier5.fit(X_train, y_train)
# Predict y data with classifier: 
y_predict = classifier5.predict(X_test)

# Print results: 
print("CONFUSION MATRIX : ") 
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))