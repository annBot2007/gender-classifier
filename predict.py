from sklearn import tree

# [height, weight, shoe size]
# Sample data:
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
     [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
     [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female',
     'male', 'female', 'male']

clf = tree.DecisionTreeClassifier() # clf = classifier short form
clf = clf.fit(X, Y) # updates the clf variable, fit() is a way of training the data

# Example of a prediction of whether a person of height 190, weight 70, and shoe size 43 is male or female
prediction = clf.predict([[190, 70, 43]])
print(prediction)
