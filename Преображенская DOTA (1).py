#!/usr/bin/env python
# coding: utf-8

# # Обработка данных

# In[1]:


import pandas as pd
import numpy as np
features = pd.read_csv(r'S:\Coursera\Введение в машинное обучение\Итоговое задание\features.csv', index_col='match_id')


# In[2]:


pd.set_option('display.max_columns', None)
features


# Выделим в списки категориальные и числовые признаки:
# (порядковые признаки - уровни героев - я отнесла к числовым).

# In[3]:


#колонки с категориальными признаками
cat_cols = ['lobby_type',
           'r1_hero',
           'r2_hero',
           'r3_hero',
           'r4_hero',
           'r5_hero',
           'd1_hero',
           'd2_hero',
           'd3_hero',
           'd4_hero',
           'd5_hero',
           'first_blood_player1',
           'first_blood_player2'
           ]


# In[4]:


#колонки с числовыми признаками
num_cols = ['r1_level',
           'r1_xp',
           'r1_gold',
           'r1_lh',
           'r1_kills',
           'r1_deaths',
           'r1_items',
           'r2_level',
           'r2_xp',
           'r2_gold',
           'r2_lh',
           'r2_kills',
           'r2_deaths',
           'r2_items',
           'r3_level',
           'r3_xp',
           'r3_gold',
           'r3_lh',
           'r3_kills',
           'r3_deaths',
           'r3_items',
           'r4_level',
           'r4_xp',
           'r4_gold',
           'r4_lh',
           'r4_kills',
           'r4_deaths',
           'r4_items',
           'r5_level',
           'r5_xp',
           'r5_gold',
           'r5_lh',
           'r5_kills',
           'r5_deaths',
           'r5_items',
           'd1_level',
           'd1_xp',
           'd1_gold',
           'd1_lh',
           'd1_kills',
           'd1_deaths',
           'd1_items',
           'd2_level',
           'd2_xp',
           'd2_gold',
           'd2_lh',
           'd2_kills',
           'd2_deaths',
           'd2_items',
           'd3_level',
           'd3_xp',
           'd3_gold',
           'd3_lh',
           'd3_kills',
           'd3_deaths',
           'd3_items',
           'd4_level',
           'd4_xp',
           'd4_gold',
           'd4_lh',
           'd4_kills',
           'd4_deaths',
           'd4_items',
           'd5_level',
           'd5_xp',
           'd5_gold',
           'd5_lh',
           'd5_kills',
           'd5_deaths',
           'd5_items',
           'first_blood_time',
           'first_blood_team',
           'radiant_bottle_time',
           'radiant_courier_time',
           'radiant_flying_courier_time',
           'radiant_tpscroll_count',
           'radiant_boots_count',
           'radiant_ward_observer_count',
           'radiant_ward_sentry_count',
           'radiant_first_ward_time',
           'dire_bottle_time',
           'dire_courier_time',
           'dire_flying_courier_time',
           'dire_tpscroll_count',
           'dire_boots_count',
           'dire_ward_observer_count',
           'dire_ward_sentry_count',
           'dire_first_ward_time'
           ]

#колонки с категориальными и числовыми признаками
feature_cols = num_cols + cat_cols

#целевая переменная
target_col = ['radiant_win']


# 
# 2. Как называется столбец, содержащий целевую переменную?
# 
# 'radiant_win'
# 

# In[5]:


#функция выделяет из массива данных X и y

def create_y_X(data):
    y = data[target_col]
    X = data[feature_cols]

    return y, X


# In[6]:


y, X = create_y_X(features)


# In[7]:


X.corr().style.set_precision(2)

#В матрице корреляций можно увидеть достаточно большую корреляцию между признаками уровня, 
#опыта, ценности героя и числом убитых юнитов. В дальнейшем это можно использовать для
#создания новых признаков.


# 
# 1. Какие признаки имеют пропуски среди своих значений? Что могут означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?
# 
# Пропуски связаны с тем, что событие (приобретение предмета или первая кровь) не произошло.
# 

# In[8]:


#Выявление столбцов с пропусками в данных. 

print('Пропуски данных в столбцах:')
for i in feature_cols:
    if X[i].count() != 97230:
        print(i, X[i].count())


# In[9]:


#Выделяем колонки, имеющие пропуски в данных и относящиеся ко времени 

time_gap_cols = ['first_blood_time', 
                 'radiant_bottle_time', 'radiant_courier_time', 'radiant_flying_courier_time', 'radiant_first_ward_time',
                 'dire_bottle_time', 'dire_courier_time', 'dire_flying_courier_time', 'dire_first_ward_time']


# In[10]:


#Данные есть о первых 5 минутах матча, значит максимально возможное значение в таких колонках будет 5*60 = 300 сек. Проверим:

features['dire_first_ward_time'].max()


# Обработка пропусков: 
# 1. first_blood_player заменяем на 0, так как кодировка игроков начинается с 1. 
# 2. first_blood_team заменяем на 0.5, что будет значить, что ни 0 ни 1 команда не совершила first blood 
# 3. для колонок time_gap_cols выбираем значение, сильно больше максимального, что будет значить, что событие произошло позже данного для исследования времени (300 сек)

# In[11]:


X['first_blood_player1'] = X['first_blood_player1'].fillna(0)
X['first_blood_player2'] = X['first_blood_player2'].fillna(0)
X['first_blood_team'] = X['first_blood_team'].fillna(.5)
X[time_gap_cols] = X[time_gap_cols].fillna(1000)


# In[12]:


X


# # Градиентный бустинг 
# ## 1. Градиентный бустинг в лоб
# Игнорируем категориальные фичи

# In[13]:


from sklearn.ensemble import GradientBoostingClassifier


# In[16]:


from sklearn.model_selection import train_test_split


# In[23]:


from sklearn.metrics import roc_auc_score


# In[14]:


y = np.array(y)
y = y.ravel()


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=241)
gbc = GradientBoostingClassifier(random_state=241)


# In[20]:


gbc.fit(X_train, y_train)


# In[24]:


Pred_gbc_test = gbc.predict_proba(X_test)
Pred_gbc_train = gbc.predict_proba(X_train)
roc_auc_test = roc_auc_score(y_test, Pred_gbc_test[:,1])
roc_auc_train = roc_auc_score(y_train, Pred_gbc_train[:,1])
print('значение roc_auc на test: ', roc_auc_test)
print('значение roc_auc на train: ', roc_auc_train)
print('среднее значение roc_auc: ', (roc_auc_test + roc_auc_train)/2)


# Вывод: качество низкое, подход неправильный (нельзя игнорировать категориальные фичи)

# ## 2. Улучшенный градиентный бустинг

# In[36]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import roc_auc_score


# In[27]:


from sklearn.model_selection import cross_val_score


# In[26]:


#one-hot-encoding: обработка кат. признаков
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
X


# In[32]:


gbc = GradientBoostingClassifier(random_state=241)


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=241)


# In[34]:


#сетка гиперпараметров для поиска лучших
params = {'n_estimators': (10, 20, 30, 50),
             'max_depth': (3, 5, 7, 10, 20)}


# In[37]:


#ищем по сетке
grid = GridSearchCV(gbc, params, cv=5)
grid.fit(X_train, y_train)


# In[38]:


grid.best_params_


# 4. Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге? Что бы вы предложили делать, чтобы ускорить его обучение при увеличении количества деревьев?
# 
# Есть смысл использовать более 30 деревьев. Как видно ниже, точность при 30 деревьях 0.7029, при 50 деревьях 0.7135.
# Чтобы ускорить обучение, можно использовать для обучения и кросс-валидации не всю выборку, а некоторое ее подмножество — например, половину объектов.

# In[39]:


#Обучаем модель с найденными параметрами:

gbc_best = GradientBoostingClassifier(n_estimators=50, max_depth=7, random_state=241)
gbc_best.fit(X_train, y_train)


# In[44]:


Pred_gbc_best_test = gbc_best.predict_proba(X_test)
Pred_gbc_best_train = gbc_best.predict_proba(X_train)
roc_auc_test = roc_auc_score(y_test, Pred_gbc_best_test[:,1])
roc_auc_train = roc_auc_score(y_train, Pred_gbc_best_train[:,1])
print('значение roc_auc на test: ', round(roc_auc_test, 4))
print('значение roc_auc на train: ', round(roc_auc_train, 4))
print('среднее значение roc_auc: ', round((roc_auc_test + roc_auc_train)/2, 4))


# Результаты gbc_best на каггле: 0.71048 (не густо)

# Наблюдается значительное прееобучение

# 3. Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями? Инструкцию по измерению времени можно найти ниже по тексту. Какое качество при этом получилось? Напомним, что в данном задании мы используем метрику качества AUC-ROC.

# In[46]:


#кросс-валидация для градиентного бустинга с 30 деревьями:

import time
import datetime

start_time = datetime.datetime.now()

gbc_30 = GradientBoostingClassifier(n_estimators=30, max_depth=7, random_state=241)
kf = KFold(n_splits=5, shuffle=True)

quality = cross_val_score(gbc_30, X, y, cv=kf, scoring='roc_auc') #качество на кросс-валидации
quality_mean = np.mean(quality) #среднее качество на кросс-валидации

print('значения roc_auc на кросс-валидации:', quality)
print('среднее значение roc_auc:', quality_mean)

print('Time elapsed:', datetime.datetime.now() - start_time)


# ## 3. CatBoost

# In[64]:


get_ipython().system('pip install catboost')
import catboost


# In[65]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score


# In[66]:


y, X = create_y_X(features)
X['first_blood_player1'] = X['first_blood_player1'].fillna(0)
X['first_blood_player2'] = X['first_blood_player2'].fillna(0)
X['first_blood_team'] = X['first_blood_team'].fillna(.5)
X[time_gap_cols] = X[time_gap_cols].fillna(1000)


# In[67]:


#Сейчас некоторые кат. признаки типа float. Напишем функцию, преобразующую их в int

def cat_to_int(data, cat_cols):
    data[cat_cols] = data[cat_cols].astype('int64')
    return data


# In[68]:


#Создадим матрицу признаков для кэтбустинга

X_catboost = cat_to_int(X, cat_cols)


# In[69]:


#дефолтный кэтбустинг
cat = catboost.CatBoostClassifier(eval_metric='AUC', silent=True, random_state=241, cat_features=cat_cols)

X_catboost_train, X_catboost_test, y_train, y_test = train_test_split(X_catboost, y, train_size=0.8, random_state=241)


# In[72]:


#обучение
import time
import datetime

start_time = datetime.datetime.now()

cat.fit(X_catboost_train, y_train)

print('Time elapsed:', datetime.datetime.now() - start_time)


# In[73]:


#предсказание

Pred_catboost_test = cat.predict_proba(X_catboost_test)
Pred_catboost_train = cat.predict_proba(X_catboost_train)
roc_auc_test = roc_auc_score(y_test, Pred_catboost_test[:,1])
roc_auc_train = roc_auc_score(y_train, Pred_catboost_train[:,1])
print('значение roc_auc на test: ', roc_auc_test)
print('значение roc_auc на train: ', roc_auc_train)
print('среднее значение roc_auc: ', (roc_auc_test + roc_auc_train)/2)


# Вывод: 
# 1. Наблюдается значительное переобучение 
# 2. Результат на каггле 0.73055
# 3. Время исполнения кода значительно ниже, чем при использовании GradientBoostingClassifier (1 мин 39 сек   vs   14 мин 33 сек)
# 4. Качество выше на 2.5%, чем при использовании GradientBoostingClassifier (0.7351    vs    0.7029)

# # ПРЕДСКАЗАНИЯ ДЛЯ ГРАДИЕНТНЫХ БУСТИНГОВ

# In[58]:


best_model = gbc_best


# In[55]:


X_test_test = pd.read_csv(r'S:\Coursera\Введение в машинное обучение\Итоговое задание\features_test.csv', index_col='match_id')
X_test_test = X_test_test[feature_cols]
X_test_test['first_blood_player1'] = X_test_test['first_blood_player1'].fillna(0)
X_test_test['first_blood_player2'] = X_test_test['first_blood_player2'].fillna(0)
X_test_test['first_blood_team'] = X_test_test['first_blood_team'].fillna(.5)
X_test_test[time_gap_cols] = X_test_test[time_gap_cols].fillna(1000)
X_test_test = pd.get_dummies(X_test_test, columns=cat_cols, drop_first=True) #для случайного леса
#X_test_test = cat_to_int(X_test_test, cat_cols) #для кэтбуста

X_test_test


# In[59]:


submission = pd.read_csv(r'S:\Coursera\Введение в машинное обучение\Итоговое задание\submission.csv')


# In[60]:


submission['match_id'] = X_test_test.index


# In[61]:


submission['radiant_win'] = best_model.predict_proba(X_test_test)[:,1]


# In[63]:


submission.to_csv(r'S:\Coursera\Введение в машинное обучение\Итоговое задание\my_submission_gbc_best.csv', index=False)


# # Логистическая регрессия

# In[74]:


#Готовим данные для лог. регрессии заново и немного по-другому
y, X = create_y_X(features)
X['first_blood_player1'] = X['first_blood_player1'].fillna(0)
X['first_blood_player2'] = X['first_blood_player2'].fillna(0)
X['first_blood_team'] = X['first_blood_team'].fillna(.5)
X[time_gap_cols] = X[time_gap_cols].fillna(1000)
#one-hot-encoding:
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
X


# In[94]:


y = np.array(y)
y = y.ravel()


# In[95]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score


# In[96]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=241)


# In[97]:


#масштабируем признаки

pca = StandardScaler()
X_train_ = pca.fit_transform(X_train)
X_test_ = pca.fit_transform(X_test)


# In[107]:


X_ = pca.fit_transform(X)


# In[98]:


#дефолтная лог. регрессия

logreg = LogisticRegression(penalty='l2', random_state=241, max_iter=500)
logreg.fit(X_train_, y_train)


# In[99]:


Pred_logreg_test = logreg.predict_proba(X_test_)
Pred_logreg_train = logreg.predict_proba(X_train_)
roc_auc_test = roc_auc_score(y_test, Pred_logreg_test[:,1])
roc_auc_train = roc_auc_score(y_train, Pred_logreg_train[:,1])
print('значение roc_auc на test: ', round(roc_auc_test, 4))
print('значение roc_auc на train: ', round(roc_auc_train, 4))
print('среднее значение roc_auc: ', round((roc_auc_test + roc_auc_train)/2, 4))


# Вывод: 
#     1. Переобучение незначительно 
#     2. roc_auc выше, чем у кэтбустинга (0.7461 vs 0.7351)

# # Логистическая регрессия с поиском лучших параметров

# In[101]:


# Значения коэффициента С: 100 значений в промежутке от 0.001 до 1000

a = np.linspace(0.001, 1000, 100)
logregcv = LogisticRegressionCV(Cs=a , fit_intercept=True, cv=5, scoring='roc_auc', random_state=241, refit=True, max_iter=500)


# In[102]:


logregcv.fit(X_train_, y_train)


# In[103]:


Pred_logregcv_test = logregcv.predict_proba(X_test_)
Pred_logregcv_train = logregcv.predict_proba(X_train_)
roc_auc_test = roc_auc_score(y_test, Pred_logregcv_test[:,1])
roc_auc_train = roc_auc_score(y_train, Pred_logregcv_train[:,1])
print('значение roc_auc на test: ', round(roc_auc_test, 4))
print('значение roc_auc на train: ', round(roc_auc_train, 4))
print('среднее значение roc_auc: ', round((roc_auc_test + roc_auc_train)/2, 4))


# Качество не изменилось

# In[104]:


#Лучший коэффициент с

c_best = logregcv.C_
c_best = np.max(c_best)
c_best


# In[108]:


#кросс-валидация для лог. регрессии, С = 60.607000000000006:

import time
import datetime

start_time = datetime.datetime.now()

logreg_best = LogisticRegression(C=c_best, penalty='l2', max_iter=500, random_state=241)
kf = KFold(n_splits=5, shuffle=True)

quality = cross_val_score(logreg_best, X_, y, cv=kf, scoring='roc_auc') #качество на кросс-валидации
quality_mean = np.mean(quality) #среднее качество на кросс-валидации

print('значения roc_auc на кросс-валидации:', quality)
print('среднее значение roc_auc:', quality_mean)

print('Time elapsed:', datetime.datetime.now() - start_time)


# In[109]:


logreg_best.fit(X_, y)


# Вывод: Результат на каггле 0.75053 - лучшее качество из всех моделей

# 1. Какое качество получилось у логистической регрессии над всеми исходными признаками? Как оно соотносится с качеством градиентного бустинга? Чем вы можете объяснить эту разницу? Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?
# Качество у логистической регрессии 0.75053, у град. бустинга 0.7351.
# Логистическая регрессия работает быстрее по сравнению с градиентным бустингом (52 сек   vs   1 мин 39 сек).
# Причиной, по которой лог. регрессия работает на этом датасете лучше, является разреженность матрицы признаков.

# 2. Как влияет на качество логистической регрессии удаление категориальных признаков (укажите новое значение метрики качества)? Чем вы можете объяснить это изменение?
# См. код ниже
# удаление категориальных признаков снижает значение roc_auc = 0.7211

# In[114]:


y, X = create_y_X(features)
X['first_blood_player1'] = X['first_blood_player1'].fillna(0)
X['first_blood_player2'] = X['first_blood_player2'].fillna(0)
X['first_blood_team'] = X['first_blood_team'].fillna(.5)
X[time_gap_cols] = X[time_gap_cols].fillna(1000)
X = X[num_cols]
X


# In[115]:


y = np.array(y)
y = y.ravel()


# In[116]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=241)


# In[117]:


pca = StandardScaler()
X_train_ = pca.fit_transform(X_train)
X_test_ = pca.fit_transform(X_test)


# In[118]:


logreg_num = LogisticRegression(C=c_best, penalty='l2', max_iter=500, random_state=241)
logreg.fit(X_train_, y_train)


# In[119]:


Pred_logreg_test = logreg.predict_proba(X_test_)
Pred_logreg_train = logreg.predict_proba(X_train_)
roc_auc_test = roc_auc_score(y_test, Pred_logreg_test[:,1])
roc_auc_train = roc_auc_score(y_train, Pred_logreg_train[:,1])
print('значение roc_auc на test: ', round(roc_auc_test, 4))
print('значение roc_auc на train: ', round(roc_auc_train, 4))
print('среднее значение roc_auc: ', round((roc_auc_test + roc_auc_train)/2, 4))


# 3. Сколько различных идентификаторов героев существует в данной игре?
# См. ниже
# Всего 112 id

# In[120]:


heroes = pd.read_csv(r'S:\Coursera\Введение в машинное обучение\Итоговое задание\data\data\dictionaries\heroes.csv', index_col='id')
heroes


# 4. Какое получилось качество при добавлении "мешка слов" по героям? Улучшилось ли оно по сравнению с предыдущим вариантом? Чем вы можете это объяснить?
# 
# Удаление категориальных признаков снижает значение roc_auc с 0.745 до 0.7211. 
# Выбор героев значительно влияет на исход игры, поэтому исключение этих признаков из матрицы снижает качество модели.

# 5. Какое минимальное и максимальное значение прогноза на тестовой выборке получилось у лучшего из алгоритмов?
# 
# Результат на каггле 0.75053 - лучшее качество из всех моделей

# In[ ]:





# # ПРЕДСКАЗАНИЯ ДЛЯ ЛОГ. РЕГРЕССИЙ

# In[110]:


best_model = logreg_best


# In[111]:


X_test_test = pd.read_csv(r'S:\Coursera\Введение в машинное обучение\Итоговое задание\features_test.csv', index_col='match_id')
X_test_test = X_test_test[feature_cols]
X_test_test['first_blood_player1'] = X_test_test['first_blood_player1'].fillna(0)
X_test_test['first_blood_player2'] = X_test_test['first_blood_player2'].fillna(0)
X_test_test['first_blood_team'] = X_test_test['first_blood_team'].fillna(.5)
X_test_test[time_gap_cols] = X_test_test[time_gap_cols].fillna(1000)
X_test_test = cat_to_int(X_test_test, cat_cols)
X_test_test = pd.get_dummies(X_test_test, columns=cat_cols, drop_first=True)
X_test_test_ = pca.fit_transform(X_test_test)
X_test_test_


# In[112]:


submission = pd.read_csv(r'S:\Coursera\Введение в машинное обучение\Итоговое задание\submission.csv')
submission['match_id'] = X_test_test.index
submission['radiant_win'] = best_model.predict_proba(X_test_test_)[:,1]


# In[113]:


submission.to_csv(r'S:\Coursera\Введение в машинное обучение\Итоговое задание\my_submission_logreg.csv', index=False)


# In[ ]:




