"""
Работа с моделью и данными по электроэнергетике ОЭС Средняя Волга
"""
import pickle
import pandas as pd
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

    def get_predict(self, date):
        """
        Датафрейм, содержащий прогнозы, начиная с указанной даты, на последующие 5 дней
        """
        
        dataframe = data['data']
        X_data = self.df[self.df.DATE >= date] \
            .drop(columns=["DATE", "USE_PRED1", "USE_PRED2", "USE_PRED3", "USE_PRED4", "USE_PRED5"])
        return np.expm1(data['model'].predict(X_data))
