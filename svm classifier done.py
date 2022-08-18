import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

d1 = pd.read_csv('nonscaled1_varibles.csv',  index_col = 0)
d1.insert(7, 'Target', 0)
d1

d2 = pd.read_csv('nonscaled2_varibles.csv',  index_col = 0)
d2.insert(7, 'Target', 1)
d2

d2.rename(columns = {'effluent_Ph':'PH', 'Effluent_COD':'COD', 'Effluent_Bod':'BOD', 'Effluent_DO':'DO', 'Effluent_Nitrogen':'NITROGEN', 'Efffluent_phos':'PHOSPHOROUS', 'ETDS':'TDS'}, inplace = True)
d2

d1.rename(columns = {'Influent_pH':'PH', 'Influent_COD':'COD', 'Influent_BOD':'BOD', 'Influent_DO':'DO', 'Influent_Nitrogen':'NITROGEN', 'Influent_Phosphorous':'PHOSPHOROUS', 'Influent_TDS':'TDS'}, inplace = True)
d1

frames = [d1, d2]
new_frame = pd.concat(frames, axis = 0)

from sklearn.utils import shuffle
new_frame = shuffle(new_frame)

new_frame.head()
new_frame.describe()

conditions = [(6.5 >= (new_frame['PH'] <= 9.0)) & (new_frame['COD'] <= 90) & (new_frame['BOD'] <= 50) & (new_frame['DO'] <= 10) & (new_frame['NITROGEN'] <= 10) & (new_frame['PHOSPHOROUS'] <= 5) & (new_frame['TDS'] <= 500)]
value = ['safe']

new_frame['Target'] = np.select(conditions, value)
new_frame['Target'] = new_frame['Target'].replace(['0'], 'Not safe')
new_frame
new_frame['Target'].value_counts()

test = [12.07999824, 525.00012093,  62.99999876,   4.58057252, 2.1999998 ,   1.30000006, 510.99992726]

7.77000018, 38.00021553, 17.50000525,  4.29999993,  2.2999997, 1.20000009, 12.99989542

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(new_frame.drop('Target',axis=1))
scaled_features

X_data= pd.DataFrame(scaled_features,columns=new_frame.columns[: -1])
#X_data.loc[229]
X_data = new_frame.drop('Target',axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_data,new_frame['Target'],
                                                   test_size=0.30, random_state = 0)

x_train
x_test.head(60)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

test_set = [[11.88, 420.0, 63.0, 3.8, 2.90, 1.70, 405.0]]
y_pred1 = classifier.predict(test_set)
y_pred1

test_set2 = [[6.60, 60.0, 20.5, 20.0, 1.50, 2.00, 250.0]]
y_pred2 = classifier.predict(test_set2)
y_pred2

test_set3 = [[12, 250.0, 20, 7.0, 9.0, 4.0, 500.0]]

y_pred3 = classifier.predict(test_set3)
y_pred3

test_set4 = [[6.78, 50.0, 20, 7.0, 9.0, 4.0, 50.0]]
y_pred4 = classifier.predict(test_set4)
y_pred4

from sklearn.metrics import classification_report,confusion_matrix
pd.DataFrame(confusion_matrix(y_test, y_pred),
             index=['Actual: Unsafe', 'Actual: Safe'],
             columns = ['pred: Unsafe', 'pred: Safe'])
print(classification_report(y_test, y_pred))

y_pred1 = scaler.inverse_transform(test_set)
y_pred1
