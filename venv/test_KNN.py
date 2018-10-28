import unittest
import numpy as np

from KNN import KNN

listaUczaca = np.array([[3, 4, 1, 2, 'A'], [6, 8, 5, 4, 'B'], [6, 9, 7, 5, 'B'], [2, 3, 1, 2, 'A'], [5, 4, 8, 9, 'B'], [8, 8, 8, 9, 'B'], [9, 7, 5, 9, 'B'], [1, 2, 3, 4, 'A']],  dtype=np.dtype(object))
listaTestujacaBezEtykiet = np.array([[1, 9, 2, 8], [5, 5, 5, 5], [0, 0, 0, 0], [9, 9, 9, 9]], dtype=np.dtype(object))
etykietyListyTestujacej = ['A', 'B', 'A', 'B']

poprawnyWynikAlgorytmu = ['B', 'B', 'A', 'B']
skutecznoscAlgorytmu = 75.0

ai = KNN(3, listaUczaca)


class TestKNN(unittest.TestCase):
    def test_predict(self):
        self.assertEqual(ai.predict(listaTestujacaBezEtykiet), poprawnyWynikAlgorytmu)
    def test_score(self):
        self.assertEqual(ai.score(listaTestujacaBezEtykiet, etykietyListyTestujacej), str(skutecznoscAlgorytmu))


if __name__ == '__main__':
    unittest.main()