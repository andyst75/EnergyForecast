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
            self.df['fact'] = np.expm1(self.df['USE_FACT'])
            self.model = self.data['model']

    def get_data(self, date='1979-01-01'):
        """
        Датафрейм, начиная с указанной даты
        """

        return np.expm1(self.df[self.df.DATE >= date][['fact']])

    def get_data_with_consumption(self, date,
                                  predict_days=2,
                                  temperature_delta=0,
                                  consumption_index_delta=0,
                                  isolation_index_delta=0
                                  ):
        """
        Датафрейм, содержащий прогнозы, начиная с указанной даты, на последующие 5 дней
        """
        drop_columns = ['DATE', 'USE_PRED1', 'USE_PRED2', 'USE_PRED3', 'USE_PRED4', 'USE_PRED5']
        filtered_data = self.df[self.df.DATE >= date].drop(columns=drop_columns)
        filtered_data['TEMP'] = filtered_data['TEMP'] + temperature_delta
        filtered_data['TEMP1'] = filtered_data['TEMP1'] + temperature_delta
        filtered_data['TEMP2'] = filtered_data['TEMP2'] + temperature_delta
        filtered_data['TEMP3'] = filtered_data['TEMP3'] + temperature_delta
        filtered_data['TEMP5'] = filtered_data['TEMP5'] + temperature_delta
        filtered_data['COMS'] = filtered_data['COMS'] + consumption_index_delta
        filtered_data['II'] = filtered_data['II'] + isolation_index_delta
        filtered_data['II3'] = filtered_data['II3'] + isolation_index_delta
        filtered_data['consumption'] = np.expm1(
            self.data['model'].predict(filtered_data.drop(columns=['fact']))[:, predict_days])
        return filtered_data
