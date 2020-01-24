import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_iris
iris = load_iris()
iris.keys()

df = pd.DataFrame(iris['data'], columns = iris['feature_names'])

y = iris['target']

# Removing the spaces and (cm) from the column names since our classfifier reuqired it

newcolumns = []
for i in df.columns:
  print(i)
  i = i[:-5]
  i = i.replace(' ', '_')
  newcolumns.append(i)
  
df.columns = newcolumns
  
feat_cols = []

for i in df.columns:
  feat_cols.append(tf.feature_column.numeric_column(i))
  
# convert array to series
y = pd.Series(data = y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3)

input_func = tf.estimator.inputs.pandas_input_fn(x = x_train, y = y_train, batch_size=10, shuffle=True)

classifier = tf.estimator.DNNClassifier(hidden_units=[10,20], n_classes= 3, feature_columns=feat_cols)

classifier.train(input_func, steps=50)

output_func = tf.estimator.inputs.pandas_input_fn(x = x_test, batch_size=len(x_test), shuffle=False)

predictions = list(classifier.predict(output_func))
