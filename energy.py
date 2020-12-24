"""
Работа с моделью и данными по электроэнергетике ОЭС Средняя Волга
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

class Energy:

    def __init__(self, filename='model_data.pkl'):
        """Загружаем заранее предобработанные данные и предобученную модель"""

        with open(filename, 'rb') as handle:
            self.data = pickle.load(handle)
            self.df = self.data['data']
            self.df['fact'] = np.expm1(self.df['USE_FACT'])
            self.model = self.data['model']

    def get_data_with_consumption(self, date,
                                  predict_days=2,
                                  temperature_delta=0,
                                  consumption_index_delta=0,
                                  isolation_index_delta=0
                                  ):
        """
        Датафрейм, содержащий прогноз, через день в будущем
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

    def what_if_predict(self, date_from, date_to, 
                                  temperature_delta=0,
                                  consumption_index_delta=0,
                                  isolation_index_delta=0
                                  ):
        """
        Датафрейм, содержащий прогнозы, начиная с указанной даты, на последующие 5 дней
        """
        drop_columns = ['USE_PRED1', 'USE_PRED2', 'USE_PRED3', 'USE_PRED4', 'USE_PRED5']
        result_columns = ['DATE', 'fact', 'TEMP', 'II', 'COMS', 'MSP', 'PRED_1', 'PRED_2', 'PRED_3', 'PRED_4', 'PRED_5']
        filtered_data = self.df[(self.df.DATE >= date_from) & (self.df.DATE <= date_to)].drop(columns=drop_columns)
        filtered_data['TEMP'] = filtered_data['TEMP'] + temperature_delta
        filtered_data['TEMP1'] = filtered_data['TEMP1'] + temperature_delta
        filtered_data['TEMP2'] = filtered_data['TEMP2'] + temperature_delta
        filtered_data['TEMP3'] = filtered_data['TEMP3'] + temperature_delta
        filtered_data['TEMP5'] = filtered_data['TEMP5'] + temperature_delta
        filtered_data['COMS'] = filtered_data['COMS'] + consumption_index_delta
        filtered_data['II'] = filtered_data['II'] + isolation_index_delta
        filtered_data['II3'] = filtered_data['II3'] + isolation_index_delta
        predicted_array = np.expm1(self.data['model'].predict(filtered_data.drop(columns=['DATE', 'fact'])))
        predicted_df = pd.DataFrame(predicted_array, columns=['PRED_1', 'PRED_2', 'PRED_3', 'PRED_4', 'PRED_5'])
        predicted_df = pd.concat([filtered_data.reset_index(drop=True), predicted_df], axis=1)
        metric = {}
        for i in range(1, 6):
            mape = np.mean(np.abs((predicted_df['fact'] - predicted_df[f'PRED_{i}']) / predicted_df['fact'])) * 100
            mae = mean_absolute_error(predicted_df['fact'], predicted_df[f'PRED_{i}'])
            r2 = r2_score(predicted_df['fact'], predicted_df[f'PRED_{i}'])
            metric.update({f'mape_{i}': mape})
            metric.update({f'mae_{i}': mae})
            metric.update({f'r2_{i}': r2})
        return (predicted_df[result_columns], metric)
