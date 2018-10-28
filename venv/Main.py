import numpy as np
import pandas as pd

from KNN import KNN

listaUczaca = np.array(pd.read_csv("data-learning.csv", header=None))
listaTestujaca = np.array(pd.read_csv("data-test.csv", header=None))

listaTestujacaBezEtykiet = np.delete(listaTestujaca, len(listaTestujaca[0])-1, axis=1)
etykietyListyTestujacej = listaTestujaca[:,len(listaTestujaca[0])-1]

ai = KNN(3, listaUczaca)
#print(ai.predict(listaTestujacaBezEtykiet))
#print(str(ai.score(listaTestujacaBezEtykiet, etykietyListyTestujacej))+"% poprawnych wyników")

listaUczaca = np.array([[3, 4, 1, 2, 'A'], [6, 8, 5, 4, 'B'], [6, 9, 7, 5, 'B'], [2, 3, 1, 2, 'A'], [5, 4, 8, 9, 'B'], [8, 8, 8, 9, 'B'], [9, 7, 5, 9, 'B'], [1, 2, 3, 4, 'A']],  dtype=np.dtype(object))
listaTestujacaBezEtykiet = np.array([[1, 9, 2, 8], [5, 5, 5, 5], [0, 0, 0, 0], [9, 9, 9, 9]], dtype=np.dtype(object))
etykietyListyTestujacej = ['A', 'B', 'A', 'B']
poprawnyWynik = ['B', 'B', 'A', 'B']

ai = KNN(3, listaUczaca)
print(ai.predict(listaTestujacaBezEtykiet))
print(ai.score(listaTestujacaBezEtykiet, etykietyListyTestujacej))
skutecznosc = 75.0
print(str(skutecznosc))