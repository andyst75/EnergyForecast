# Домашнее задание по курсу Машинное обучение Академии больших данных MADE

Команда разработчиков:
|                |Группа |Гитхаб                          |Роль в проекте |
|----------------|-------|--------------------------------|---------------|
|Авилов Илья     |DS-11  |https://github.com/Ilya2567     |DS, Frontend   |
|Дякин Николай   |ML-11  |https://github.com/nickdndev    |DS, Frontend   |
|Мунин Евгений   |ML-12  |https://github.com/EvgeniiMunin |ML, Frontend   |
|Орхан Гаджилы   |DS-12  |https://github.com/Fianketto    |DS, PM         |
|Стариков Андрей |ML-12  |https://github.com/andyst75     |DS, ML, DevOps |

[Демоверсия проекта](https://share.streamlit.io/andyst75/energyforecast/main/energy_app.py)

## Тема проекта: составление прогноза потребления электроэнергии в ОЭС Средней Волги

Вопрос краткосрочного / среднесрочного и долгосрочного прогнозирования постебления электроэнергии стоит очень остро перед электрогенерацией. От правильности долгосрочных прогнозов зависит выбор при проектировании и строительстве новых энергетических мощностей. От точности среднесрочных прогнозов зависит график вывода мощностей на профилактический ремонт и объем закупаемого топлива.

Перед нами стяла задача составления краткосрочного прогноза потребления электроэнергии.

Выбрано две метрики качества: MAE и MAPE, т.к. хорошо оценивают точность прогноза

В качестве возможных моделей будет рассмотренно несколько вариантов:

-  Константная модель
-  Наивная модель
-  Простая модель линейной регрессии
-  Сложная модель линейной регрессии (учет дополнительных факторов и лаговых значений показателей)
-  Сложная модель линейной регрессии с регуляризацией

В процессе работы над проектом были получены и обработаны данные со следующих ресурсов:
-  [https://br.so-ups.ru/BR/GenConsum](https://br.so-ups.ru/BR/GenConsum)
-  [https://sberindex.ru/ru/dashboards/indeks-potrebitelskoi-aktivnosti](https://sberindex.ru/ru/dashboards/indeks-potrebitelskoi-aktivnosti)
-  [https://sberindex.ru/ru/dashboards/izmenenie-aktivnosti-msp-po-regionam](https://sberindex.ru/ru/dashboards/izmenenie-aktivnosti-msp-po-regionam)
-  [https://datalens.yandex/7o7is1q6ikh23?tab=q6](https://datalens.yandex/7o7is1q6ikh23?tab=q6)
-  [ftp://ftp.ncdc.noaa.gov/pub/data/gsod/](ftp://ftp.ncdc.noaa.gov/pub/data/gsod/)

## Краткие выводы по работе над моделью

-  Константная модель дает огромную погрешность в прогнозе
-  Точность наивной модели по MAPE равна 1.65, это является хорошим ориентиром
-  Простая линейная модель хорошо улучшает прогноз наивой модели, например при прогнозе на три дня  
   значение метрики MAPE уменьшилось с 2,88 до 2,16
-  Сложная модель линейной регрессии значительно превосходит простую модель линейной регрессии, например  
   при прогнозе на три дня значение метрики MAPE уменьшилось с 2,16 до 0,97, а при прогнозе на 5 дней — с 2,44 до 1,22
-  Сложная модель с l2 регуляризацией не дала качественного улучшения прогноза, по сравнению с моделью без регуляризации

## Визуализация результатов работы модели
В качестве фреймворка для демонстрации результатов работы был выбран [Streamlit](https://www.streamlit.io/) который предоставляет широкие возможности по визуализации табличных данных.

Одной из отличительных особенностей данного проекта является реализация сценария **what-if**, багодаря которой предоставляется возможность изучения качественных изменений по потреблению электроэнергии в зависимости от изменений внешних условий. Изменения по потреблению электроэнергии можно также оценить визуально с помощью представленных на главной странице графиков. В частности, доступными для управления параметрами, отражающими внешние факторы, являются:
- Внешняя температура
- Индекс активности потребления
- Индекс изоляции

Доступный период прогнозирования - от 1 до 5 дней.
