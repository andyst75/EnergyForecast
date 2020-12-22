"""
Работа с моделью и данными по электроэнергетике ОЭС Средняя Волга
"""
import pickle

import numpy as np


class Energy:

    def __init__(self, filename='model_data.pkl'):
        """Загружаем заранее предобработанные данные и предобученную модель"""

        with open(filename, 'rb') as handle:
            self.data = pickle.load(handle)
            self.df = self.data['data']
            self.model = self.data['model']

    def get_data(self, date='1979-01-01'):
        """
        Датафрейм, начиная с указанной даты
        """

        return np.expm1(self.df[self.df.DATE >= date][["USE_FACT"]])

    def get_data_with_consumption(self, date):
        """
        Датафрейм, содержащий прогнозы, начиная с указанной даты, на последующие 5 дней
        """
        drop_columns = ["DATE", "USE_PRED1", "USE_PRED2", "USE_PRED3", "USE_PRED4", "USE_PRED5"]
        filtered_data = self.df[self.df.DATE >= date].drop(columns=drop_columns)
        filtered_data['consumption'] = np.expm1(self.data['model'].predict(filtered_data)[:, 2])
        return filtered_data
