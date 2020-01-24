
import pandas as pd
import keras


df = pd.read_csv('Churn_Modelling.csv')

x = df.iloc[:, 3:13]

y = df['Exited'].values


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer


ct = ColumnTransformer([('Geography', OneHotEncoder(), [1])], remainder='passthrough')

x = ct.fit_transform(x)

le1 = LabelEncoder()

x[:, 4] = le1.fit_transform(x[:, 4])

# Drop first column created for geography
x = x[:, 1:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x = sc.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units= 6, activation='relu', input_dim = 11))
classifier.add(Dense(units= 6, activation='relu'))
classifier.add(Dense(units= 1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(x_train, y_train, batch_size=10, epochs=100)


ypred = classifier.predict(x_test)
ypred = ypred>0.5

from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test, ypred)
