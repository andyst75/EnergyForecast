"""
Работа с моделью и данными по электроэнергетике ОЭС Средняя Волга
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

TEMP_FEATURES = ['TEMP', 'TEMP1', 'TEMP2', 'TEMP3', 'TEMP5']
METRIC_COLUMNS = ["MAPE", "MAE", "R2"]

class Energy:

    def __init__(self, filename='model_data.pkl'):
        """Загружаем заранее предобработанные данные и предобученную модель"""

        with open(filename, 'rb') as handle:
            self.data = pickle.load(handle)
            self.df = self.data['data']
            self.df['fact'] = np.expm1(self.df['USE_FACT'])
            self.model = self.data['model']
            self.date_begin = self.df['DATE'].min()
            self.date_end = self.df['DATE'].max()

    def get_period(self):
        """Доступные для запросов периоды"""
        return self.date_begin, self.date_end

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
        for feature in TEMP_FEATURES:
            filtered_data[feature] = filtered_data[feature] + temperature_delta
        
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
        drop_columns = ['DATE', 'fact', 'USE_PRED1', 'USE_PRED2', 'USE_PRED3', 'USE_PRED4', 'USE_PRED5']
        result_columns = ['DATE', 'fact', 'TEMP', 'II', 'COMS', 'MSP', 'PRED_1', 'PRED_2', 'PRED_3', 'PRED_4', 'PRED_5']
        mask = self.df['DATE'].between(pd.to_datetime(date_from), pd.to_datetime(date_to))
        filtered_data = self.df[mask][:]

        predicted_array = np.expm1(self.data['model'].predict(filtered_data.drop(columns=drop_columns)))
        predicted_df = pd.DataFrame(predicted_array, columns=['PRED_1', 'PRED_2', 'PRED_3', 'PRED_4', 'PRED_5'])
        predicted_df = pd.concat([filtered_data.reset_index(drop=True), predicted_df], axis=1)
        metric_df = pd.DataFrame(columns=METRIC_COLUMNS)

        for i in range(1, 6):
            consumption = np.expm1(predicted_df[f'USE_PRED{i}'])
            mape = np.mean(np.abs((consumption - predicted_df[f'PRED_{i}']) / predicted_df['fact'])) * 100
            mae = mean_absolute_error(consumption, predicted_df[f'PRED_{i}'])
            r2 = r2_score(consumption, predicted_df[f'PRED_{i}'])
            metric_df.loc[i, METRIC_COLUMNS] =  round(mape, 2), round(mae, 1), round(r2, 3)

        original_pred, original_metric = predicted_df.copy(), metric_df.copy()

        for feature in TEMP_FEATURES:
            filtered_data[feature] = filtered_data[feature] + temperature_delta
        filtered_data['COMS'] = filtered_data['COMS'].add(consumption_index_delta)
        filtered_data['II'] = filtered_data['II'].add(isolation_index_delta)
        filtered_data['II3'] = filtered_data['II3'].add(isolation_index_delta)

        predicted_array = np.expm1(self.data['model'].predict(filtered_data.drop(columns=drop_columns)))
        predicted_df = pd.DataFrame(predicted_array, columns=['PRED_1', 'PRED_2', 'PRED_3', 'PRED_4', 'PRED_5'])
        predicted_df = pd.concat([filtered_data.reset_index(drop=True), predicted_df], axis=1)
        metric_df = pd.DataFrame(columns=METRIC_COLUMNS)

        for i in range(1, 6):
            consumption = np.expm1(predicted_df[f'USE_PRED{i}'])
            mape = np.mean(np.abs((consumption - predicted_df[f'PRED_{i}']) / predicted_df['fact'])) * 100
            mae = mean_absolute_error(consumption, predicted_df[f'PRED_{i}'])
            r2 = r2_score(consumption, predicted_df[f'PRED_{i}'])
            metric_df.loc[i, METRIC_COLUMNS] =  round(mape, 2), round(mae, 1), round(r2, 3)

        return predicted_df[result_columns].set_index('DATE'), metric_df, \
            original_pred.set_index('DATE'), original_metric
