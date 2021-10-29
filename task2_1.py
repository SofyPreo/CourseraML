import pandas
import numpy as np
import os

from pandas.core.frame import DataFrame
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



def save_clean_data(cleaner, X_train, y_train, X_test, name='simple'):
    path = './data/clean/' + name
    if not os.path.exists(path):
        os.makedirs(path)

    y_train.to_csv(path + '/y_train.csv')
    cleaner(X_train).to_csv(path + '/X_train.csv')
    cleaner(X_test).to_csv(path + '/X_test.csv')

def get_clean_data(cleaner_name='simple'):
	path = './data/clean/' + cleaner_name
	X_train = pandas.read_csv(path + '/X_train.csv', index_col='match_id')
	y_train = pandas.read_csv(path + '/y_train.csv', index_col='match_id')
	X_test = pandas.read_csv(path + '/X_test.csv', index_col='match_id')
	return X_train, y_train['radiant_win'], X_test

def kaggle_save(name, model, X_test):
	y_test = model.predict_proba(X_test)[:, 1]
	result = pandas.DataFrame({'radiant_win': y_test}, index=X_test.index)
	result.index.name = 'match_id'
	result.to_csv('./data/kaggle/{}.csv'.format(name))


pandas.set_option('display.max_columns', None)

df = pandas.read_csv('D:\Магистратура\data\data\\features.csv', index_col='match_id')
desc = df.describe()
print(desc)

print(df.head(10))

rows = len(df)
counts = desc.T['count']
counts_na = counts[counts < rows]
print(counts_na.sort_values().apply(lambda c: (rows - c) / rows))



# Загружаем данные
train = pandas.read_csv('D:\Магистратура\data\data\\features.csv', index_col='match_id')
test = pandas.read_csv('D:\Магистратура\data\data\\features_test.csv', index_col='match_id')

# Удаляем признаки, связанные с итогами матча
train.drop(['duration', 
         'tower_status_radiant', 
         'tower_status_dire', 
         'barracks_status_radiant', 
         'barracks_status_dire'
        ], axis=1, inplace=True)

# И разделяем датасет на признаки и целевую переменную
X = train
y = train['radiant_win'].to_frame()
del train['radiant_win']

# Заменяем пропуски на 0
def clean(X):
    return X.fillna(0)

    # В данных присутствует 11 категориальных признаков, удаляем их
def clean_category(X):
    X = clean(X)
    del X['lobby_type']
    for n in range(1, 6):
        del X['r{}_hero'.format(n)]
        del X['d{}_hero'.format(n)]

    return X


heroes = pandas.read_csv('D:\Магистратура\data\data\dictionaries\\heroes.csv')
print('Всего героев в игре:', len(heroes))

#DataFrame.i

# Формируем "мешок слов" по героям
def hero_bag(X):
    X_pick = np.zeros((X.shape[0], len(heroes)))
    for i, match_id in enumerate(X.index):
        for p in range(5):
            
            X_pick[i, X.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, X.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1

    return pandas.DataFrame(X_pick, index=X.index)


save_clean_data(clean, X, y, test)
save_clean_data(clean_category, X, y, test, name='clean_category')
save_clean_data(hero_bag, X, y, test, name='hero_bag')

X, y, X_kaggle = get_clean_data()
kf = KFold(n_splits=5, shuffle=True, random_state=42)



scores = []
nums = [10, 20, 30, 50, 100, 250]
for n in nums:
    print('#', str(n))
    model = GradientBoostingClassifier(n_estimators=n, random_state=42)
    start_time = datetime.datetime.now()
    model_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc', n_jobs=-1)
    print('Time elapsed:', datetime.datetime.now() - start_time)
    print(model_scores)
    scores.append(np.mean(model_scores))


plt.plot(nums, scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()


'''
X, y, X_kaggle = get_clean_data()
scaler = StandardScaler()
X = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
def plot_C_scores(C_pow_range, scores):
    plt.plot(C_pow_range, scores)
    plt.xlabel('log(C)')
    plt.ylabel('score')
    plt.show()

def test_model_C(X, y, C):
    print('C =', str(C))
    model = LogisticRegression(C=C, random_state=42, n_jobs=-1)
    return cross_val_score(model, X, y, cv=kf, scoring='roc_auc', n_jobs=-1)
    
def test_model(X, y):
    scores = []
    C_pow_range = range(-5, 6)
    C_range = [10.0 ** i for i in C_pow_range]
    for C in C_range:
        start_time = datetime.datetime.now()
        model_scores = test_model_C(X, y, C)
        print(model_scores)
        print('Time elapsed:', datetime.datetime.now() - start_time)       
        scores.append(np.mean(model_scores))

    plot_C_scores(C_pow_range, scores)
    
    max_score = max(scores)
    max_score_index = scores.index(max_score)
    return C_range[max_score_index], max_score

C, score = test_model(X, y)

print(C)
print(score)

X, y, X_kaggle = get_clean_data('clean_category')
scaler = StandardScaler()
X = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

C, score = test_model(X, y)
print(C)
print(score)

X, y, X_kaggle = get_clean_data('clean_category')
X_hero, _y, X_kaggle_hero = get_clean_data('hero_bag')

scaler = StandardScaler()
X = pandas.DataFrame(scaler.fit_transform(X), index = X.index)
X_kaggle = pandas.DataFrame(scaler.transform(X_kaggle), index = X_kaggle.index)

X = pandas.concat([X, X_hero], axis=1)
X_kaggle = pandas.concat([X_kaggle, X_kaggle_hero], axis=1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
C, score = test_model(X, y)
print(C)
print(score)'''
