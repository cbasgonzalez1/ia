# Dependencias
import pandas as pd
import os

#Variables Globales
pathDeTrabajo = "./"
pathModelos = "modelos/"
pathDatos = "datosIA8.csv"
separadorcsv= ','

#leer datos
if not os.path.exists(pathDeTrabajo+pathDatos):
    print('No existe el documento de datos')
    exit

datos = pd.read_csv(pathDeTrabajo+pathDatos, sep=separadorcsv, engine='python')

#Descartar columna innecesarias
columnasInecesarias = ['Paciente']
datos = datos.drop(columnasInecesarias, axis=1)
 
#Rellenar datos nulos con 0
datos = datos.fillna(0)

#---------Transformar datos a numericos---------------------------------------

datos[datos == 'si'] = 1
datos[datos == 'no'] = 0

datos[datos == 'femenino'] = 1
datos[datos == 'masculino'] = 0

datos[datos == 'leve'] = 1
datos[datos == 'moderado'] = 2
datos[datos == 'fuerte'] = 3

datos[datos == 'seca'] = 1
datos[datos == 'con flema'] = 2

#--------Separar Variable dependiente y independiente-----------------
x=datos.drop(columns='Diagnostico')
y = datos['Diagnostico']

#-------------------Balanceo y division datos de entrenamiento 70%  y prueba 30%------------------
from sklearn.model_selection import train_test_split
xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(x,y,test_size=0.3,stratify=y,random_state=44)

print("----------------------------------------------------------------------")
print("----------------------Entrenando pathModelos de IA------------------------")
print("----------------------------------------------------------------------")

# ARBOL DE DECISIONES--------------------------------------------------
from sklearn import tree, ensemble
dt = tree.DecisionTreeClassifier(max_depth=5) # numero de tipos
dt.fit(xEntrenamiento, yEntrenamiento)

# RANDOM FOREST--------------------------------------------------------
rf = ensemble.RandomForestClassifier(n_estimators=20)
rf.fit(xEntrenamiento, yEntrenamiento)

# GRADIENT BOOSTING----------------------------------------------------
gb = ensemble.GradientBoostingClassifier(n_estimators=40)
gb.fit(xEntrenamiento, yEntrenamiento)

# NAIVE BAYES----------------------------------------------------------
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(xEntrenamiento, yEntrenamiento)

# K-NEAREST NEIGHBOR---------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xEntrenamiento, yEntrenamiento)

# LOGISTIC REGRESSION--------------------------------------------------
#from sklearn.linear_model import LogisticRegression
#lr = LogisticRegression()
#lr.fit(xEntrenamiento, yEntrenamiento)

# SUPPORT VECTOR MACHINE-----------------------------------------------
from sklearn import svm
vm = svm.SVC(probability=False)
vm.fit(xEntrenamiento, yEntrenamiento)

print("----------------------------------------------------------------------")
print("---------------------Evaluacion del algoritmos de IA------------------")
print("----------------------------------------------------------------------")

# Validacion Cruzada
from sklearn.model_selection import cross_val_score
print("----------------------------------------------------------------------")

print("Validacion Cruzada - DESITION TREE")
print("Precicion de entrenamiento: " + f'{dt.score(xEntrenamiento, yEntrenamiento):.3f}%')
print("Precicion de prueba:" + f'{dt.score(xPrueba, yPrueba):.3f}%')
#print("Precision de entrenamiento: " + f'{dt.score(dt, xEntrenamiento, yEntrenamiento:.3f}%'))
#print("Precision de prueba: " + f'{dt.score(dt, xPrueba, yPrueba:.3f}%'))

print("----------------------------------------------------------------------")

print("Validacion Cruzada - RANDOM FOREST")
print("Precision de entrenamiento: " + f'{cross_val_score(rf, xEntrenamiento, yEntrenamiento, cv=5).mean():.3f}%')
print("Precision de prueba: " + f'{cross_val_score(rf, xPrueba, yPrueba, cv=5).mean():.3f}%')

print("----------------------------------------------------------------------")

print("Validacion Cruzada - GRADIENT BOOSTING")
print("Precision de entrenamiento: " + f'{cross_val_score(gb, xEntrenamiento, yEntrenamiento, cv=5).mean():.3f}%')
print("Precision de prueba: " + f'{cross_val_score(gb, xPrueba, yPrueba, cv=5).mean():.3f}%')

print("----------------------------------------------------------------------")

print("Validacion Cruzada - NAIVE BAYES")
print("Precision de entrenamiento: " + f'{cross_val_score(nb, xEntrenamiento, yEntrenamiento, cv=10).mean():.3f}%')
print("Precision de prueba: " + f'{cross_val_score(nb, xPrueba, yPrueba, cv=10).mean():.3f}%')

print("----------------------------------------------------------------------")

print("Validacion Cruzada - K-NEAREST NEIGHBOR")
print("Precision de entrenamiento: " + f'{cross_val_score(knn, xEntrenamiento, yEntrenamiento, cv=5).mean():.3f}%')
print("Precision de prueba: " + f'{cross_val_score(knn, xPrueba, yPrueba, cv=5).mean():.3f}%')

#print("----------------------------------------------------------------------")

#print("Validacion Cruzada - LOGISTIC REGRESSION")
#print("Precision de entrenamiento: " + f'{cross_val_score(lr, xEntrenamiento, yEntrenamiento, cv=5).mean():.3f}%')
#print("Precision de prueba: " + f'{cross_val_score(lr, xPrueba, yPrueba, cv=5).mean():.3f}%')

print("----------------------------------------------------------------------")

print("Validacion Cruzada - SUPPORT VECTOR MACHINE")
print("Precision de entrenamiento: " + f'{cross_val_score(vm, xEntrenamiento, yEntrenamiento, cv=5).mean():.3f}%')
print("Precision de prueba: " + f'{cross_val_score(vm, xPrueba, yPrueba, cv=5).mean():.3f}%')


# Matriz de confusion 
from sklearn.metrics import confusion_matrix, classification_report
print("----------------------------------------------------------------------")
print("------------------------MATRIZ DE CONFUSION---------------------------")
print("----------------------------------------------------------------------")

print("----------DESITION TREE----------")
y_pred = dt.predict(xPrueba)
print("----------------------------------------------------------------------")

print(confusion_matrix(yPrueba, y_pred))
print("----------------------------------------------------------------------")

print(classification_report(yPrueba, y_pred))
print("----------------------------------------------------------------------")


print("----------RANDOM FOREST----------")
y_pred = rf.predict(xPrueba)
print("----------------------------------------------------------------------")

print(confusion_matrix(yPrueba, y_pred))
print("----------------------------------------------------------------------")

print(classification_report(yPrueba, y_pred))
print("----------------------------------------------------------------------")


print("----------GRADIENT BOOSTING----------")
y_pred = gb.predict(xPrueba)
print("----------------------------------------------------------------------")

print(confusion_matrix(yPrueba, y_pred))
print("----------------------------------------------------------------------")

print(classification_report(yPrueba, y_pred))
print("----------------------------------------------------------------------")


print("----------NAIVE BAYES----------")
y_pred = nb.predict(xPrueba)
print("----------------------------------------------------------------------")

print(confusion_matrix(yPrueba, y_pred))
print("----------------------------------------------------------------------")

print(classification_report(yPrueba, y_pred))
print("----------------------------------------------------------------------")


print("----------K-NEAREST NEIGHBOR----------")
y_pred = knn.predict(xPrueba)
print("----------------------------------------------------------------------")

print(confusion_matrix(yPrueba, y_pred))
print("----------------------------------------------------------------------")

print(classification_report(yPrueba, y_pred))
print("----------------------------------------------------------------------")


#print("----------LOGISTIC REGRESSION----------")
#y_pred = lr.predict(xPrueba)
#print(confusion_matrix(yPrueba, y_pred))
#print("----------------------------------------------------------------------")

#print(classification_report(yPrueba, y_pred))
#print("----------------------------------------------------------------------")


print("----------SUPPORT VECTOR MACHINE----------")
y_pred = vm.predict(xPrueba)
print("----------------------------------------------------------------------")

print(confusion_matrix(yPrueba, y_pred))
print("----------------------------------------------------------------------")

print(classification_report(yPrueba, y_pred))
print("----------------------------------------------------------------------")


print("----------------------------------------------------------------------")
print("--------------------------Guardando pathModelos---------------------------")
print("----------------------------------------------------------------------")

import joblib

if not os.path.exists(pathDeTrabajo+pathModelos):
    os.makedirs(pathDeTrabajo+pathModelos)


print("DESITION TREE")
archivo_joblib = pathDeTrabajo+pathModelos+"DESITION-TREE.pkl"
joblib.dump(dt,archivo_joblib)

print("RANDOM FOREST")
archivo_joblib = pathDeTrabajo+pathModelos+"Random-Forest.pkl"
joblib.dump(rf,archivo_joblib)

print("GRADIENT BOOSTING")
archivo_joblib = pathDeTrabajo+pathModelos+"GRADIENT-BOOSTING.pkl"
joblib.dump(gb,archivo_joblib)

print("NAIVE BAYES")
archivo_joblib = pathDeTrabajo+pathModelos+"NAIVE-BAYES.pkl"
joblib.dump(nb,archivo_joblib)

print("K-NEAREST NEIGHBOR")
archivo_joblib = pathDeTrabajo+pathModelos+"K-NEAREST-NEIGHBOR.pkl"
joblib.dump(knn,archivo_joblib)

print("SUPPORT VECTOR MACHINE")
archivo_joblib = pathDeTrabajo+pathModelos+"SUPPORT-VECTOR-MACHINE.pkl"
joblib.dump(knn,archivo_joblib)