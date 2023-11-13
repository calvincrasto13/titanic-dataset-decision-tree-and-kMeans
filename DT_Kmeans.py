#!/usr/bin/env python
# coding: utf-8

# In[114]:


import pandas as pd
titanic_data = pd.read_csv("titanic_train.csv")


# In[115]:


# remove irrelevant colums


# In[116]:


irrelevant_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']
titanic_data = titanic_data.drop(columns=irrelevant_columns)



# In[117]:


titanic_data


# In[118]:


#Create new columns


# In[119]:


# a. AgeGroup


# In[120]:


titanic_data['AgeGroup'] = pd.cut(titanic_data['Age'],
                                  bins=[0, 2, 14, 24, 65, float('inf')],
                                  labels=['Baby', 'Child', 'Youth', 'Adult', 'Senior'],
                                  right=False,
                                  include_lowest=True)
titanic_data['AgeGroup'] = titanic_data['AgeGroup'].cat.add_categories('NK').fillna('NK')
# Drop the original 'Age' column
titanic_data = titanic_data.drop(columns=['Age'])

# Rest of the code remains the same



# In[121]:


# b. Relatives 


# In[122]:


titanic_data['Relatives'] = titanic_data['SibSp'] + titanic_data['Parch']
titanic_data['Relatives'] = pd.cut(titanic_data['Relatives'],
                                   bins=[-1, 0, 3, float('inf')],
                                   labels=['None', 'Few', 'Many'],
                                   right=False,
                                   include_lowest=True)


# In[123]:


# c Faqre


# In[124]:


titanic_data['FareCategory'] = pd.cut(titanic_data['Fare'],
                                      bins=[-1, 0, 50, 100, float('inf')],
                                      labels=['Free', 'Low', 'Average', 'High'],
                                      right=False,
                                      include_lowest=True)


# In[125]:


# Step 4: Apply one-hot encoding
titanic_data


# In[126]:


# titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'AgeGroup', 'Relatives', 'FareCategory'], drop_first=True)
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'AgeGroup', 'Relatives', 'FareCategory', ], drop_first=True)

# Step 4: Apply one-hot encoding

titanic_data = titanic_data.drop(columns=['Age Group'])
titanic_data


# In[132]:


# Step 5: Split data into train and test set
titanic_data = titanic_data.drop(columns=['Age Group'])
titanic_data


# In[133]:


from sklearn.model_selection import train_test_split

X = titanic_data.drop(columns=['Survived'])
y = titanic_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[134]:


# Step 6: Fit model using Decision Trees
# Check data types of columns
titanic_data


# In[135]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# In[136]:


# Step 7: Predict survival for the test set
y_pred = model.predict(X_test)


# In[137]:


# Step 8: Print accuracy and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)


# In[138]:


# Step 9: Plot the decision tree
plt.figure(figsize=(15, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'])
plt.show()


# In[139]:


# Step 10: Perform clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np


# In[143]:


# Select relevant features for clustering
cluster_features = ['Pclass', 'AgeGroup_Child', 'AgeGroup_Youth', 'AgeGroup_Adult', 'AgeGroup_Senior',
       'AgeGroup_NK', 'Relatives_Few', 'Relatives_Many', 'FareCategory_Low',
       'FareCategory_Average', 'FareCategory_High', 'AgeGroup_Child',
       'AgeGroup_Youth', 'AgeGroup_Adult', 'AgeGroup_Senior', 'AgeGroup_NK']
X_cluster = titanic_data[cluster_features].dropna()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Elbow method to find the best k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# Choose the best k using the silhouette score
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

best_k = np.argmax(silhouette_scores) + 2  # Adding 2 because the loop starts from 2 clusters
print(f'Best k value: {best_k}')

# Fit KMeans with the best k
kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=42)
titanic_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Optional: Visualize the clusters (2D plot)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=titanic_data['Cluster'], cmap='viridis')
plt.title('Clustering Visualization')
plt.xlabel(cluster_features[0])
plt.ylabel(cluster_features[1])
plt.show()


# In[142]:


titanic_data.columns


# In[ ]:




