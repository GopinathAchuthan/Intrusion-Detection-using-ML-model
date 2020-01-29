import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def main():

	a = pd.read_csv("dataset.csv")

	col_str = ['Source','Destination','Protocol','Info']

	#Converting non-numeric data into numeric data
	for i in col_str:
		label_enc = LabelEncoder()
		label_enc.fit(a[i])
		a[i] = label_enc.transform(a[i])

	X = a.drop(["Notified"] , axis=1)
	y = a["Notified"]
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	#Random Forest Classifier
	print("Random Forest Classifier")
	rfc = RandomForestClassifier(n_estimators=200)
	rfc.fit(X_train,y_train)
	pred_rfc = rfc.predict(X_test)
	print(classification_report(y_test,pred_rfc))
	print(confusion_matrix(y_test,pred_rfc))

	#SVM Classifier
	print("SVM Classifier")
	svmc = svm.SVC()
	svmc.fit(X_train,y_train)
	pred_svmc =  svmc.predict(X_test)
	print(classification_report(y_test,pred_svmc))
	print(confusion_matrix(y_test,pred_svmc))	

	#Neural Network
	print("Neural Network")
	mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
	mlpc.fit(X_train,y_train)
	pred_mlpc =  mlpc.predict(X_test)
	print(classification_report(y_test,pred_mlpc))
	print(confusion_matrix(y_test,pred_mlpc))


if __name__ == '__main__':
	main()




'''
Output:
Random Forest Classifier
              precision    recall  f1-score   support

           0       0.94      0.97      0.95        32
           1       1.00      1.00      1.00       954

   micro avg       1.00      1.00      1.00       986
   macro avg       0.97      0.98      0.98       986
weighted avg       1.00      1.00      1.00       986

[[ 31   1]
 [  2 952]]
SVM Classifier
              precision    recall  f1-score   support

           0       0.94      0.94      0.94        32
           1       1.00      1.00      1.00       954

   micro avg       1.00      1.00      1.00       986
   macro avg       0.97      0.97      0.97       986
weighted avg       1.00      1.00      1.00       986

[[ 30   2]
 [  2 952]]
Neural Network
              precision    recall  f1-score   support

           0       0.94      1.00      0.97        32
           1       1.00      1.00      1.00       954

   micro avg       1.00      1.00      1.00       986
   macro avg       0.97      1.00      0.98       986
weighted avg       1.00      1.00      1.00       986

[[ 32   0]
 [  2 952]]
'''
