import pandas as pd
import numpy as np
import math as mth
import scipy.stats as st
import sklearn.linear_model as lm
import statsmodels.api as sm
import statsmodels.stats.api as sms
import sklearn.preprocessing as pp
from statsmodels.stats.api import anova_lm
from statsmodels.compat import lzip
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict

data = pd.read_csv('Hospitals.csv', index_col='No')

#Графики
==============================================================
#Гистограммы для каждого признака
for i in data.columns.values[:-1]:
    sns.distplot(data[i])
    plt.savefig('hist/'+str(i)+'.png', format='png')
    plt.clf()
==============================================================
#ящики с усами
for i in data.columns.values[:-1]:
    sns.boxplot(data[i], orient='v', color='grey')
    plt.savefig('box/'+str(i)+'.png', format='png')
    plt.clf()
#Эмпирические функции распределений для количественных признаков
for i in data.columns.values[:-1]:
    plt.title('ECDF for'+str(i))
    plt.ylim(0, 1)
    plt.xlim(0, np.max(data[i]))
    plt.xlabel(str(i))
    plt.ylabel('Fn('+str(i)+')')
    x = np.sort(data[i])
    num_bins = len(x)
    n, bins, patches = plt.hist(x, num_bins, normed=1, histtype='step', cumulative=True, label="Emperical cdf")
    plt.savefig('ecdf/'+str(i)+'.png', format='png')
    plt.clf()

#ящики с усами
for i in data.columns.values[0:5]:
    sns.boxplot(data['Place'], data[i], orient='v', color='grey')
    plt.savefig('box/Plase_'+str(i)+'.png', format='png')
    plt.clf()

sns.pairplot(data)
plt.savefig('pairpl/P.png', format='png')

#Проверка гипотез
ct = pd.crosstab(pd.qcut(data.Income, 2), pd.qcut(data.Beds,3))
res = st.chi2_contingency(ct)
ct = pd.crosstab(pd.qcut(data.Income, 2), pd.qcut(data.Healing_days,3))
res = st.chi2_contingency(ct)
ct = pd.crosstab(pd.qcut(data.Salary, 3), pd.qcut(data.Beds,3))
res = st.chi2_contingency(ct)
ct2 = pd.crosstab(pd.qcut(data.Salary, 3), pd.qcut(data.Healing_days,3))
res2 = st.chi2_contingency(ct2)
ct = pd.crosstab(pd.qcut(data.Costs, 3), pd.qcut(data.Beds,3))
res = st.chi2_contingency(ct)
ct3 = pd.crosstab(pd.qcut(data.Costs, 4), pd.qcut(data.Healing_days,3))
res3 = st.chi2_contingency(ct3)
print(ct)
print(ct2)
print(ct3)


dv1 = data[['Income', 'Place']].loc[lambda x: x.Place == 'urban', :]
dv2 = data[['Income', 'Place']].loc[lambda x: x.Place != 'urban', :]
res1 = st.mstats.normaltest(dv1.Income)
res2 = st.mstats.normaltest(dv2.Income)
res = st.mannwhitneyu(dv1.Income, dv2.Income)
dv1 = data[['Salary', 'Place']].loc[lambda x: x.Place == 'urban', :]
dv2 = data[['Salary', 'Place']].loc[lambda x: x.Place != 'urban', :]
res1 = st.mstats.normaltest(dv1.Salary)
res2 = st.mstats.normaltest(dv2.Salary)
res = st.ttest_ind(dv1.Salary, dv2.Salary)
dv1 = data[['Costs', 'Place']].loc[lambda x: x.Place == 'urban', :]
dv2 = data[['Costs', 'Place']].loc[lambda x: x.Place != 'urban', :]
res1 = st.mstats.normaltest(dv1.Costs)
res2 = st.mstats.normaltest(dv2.Costs)
res = st.mannwhitneyu(dv1.Costs, dv2.Costs)

dv1 = data[['Beds', 'Salary']].loc[lambda x: x.Salary > 3000, :]
dv2 = data[['Beds', 'Salary']].loc[lambda x: x.Salary < 3000, :]
res = st.mstats.normaltest(dw.Beds)
dv1 = data[['Beds', 'Place']].loc[lambda x: x.Place == 'urban', :]
dv2 = data[['Beds', 'Place']].loc[lambda x: x.Place != 'urban', :]
res = st.mstats.normaltest(dv1.Beds)
res2 = st.mstats.normaltest(dv2.Beds)
res = st.mannwhitneyu(dv2.Beds, dv1.Beds)
print(res)

res = st.mstats.normaltest(data['Healing_days'])
print(res)

res = st.pearsonr(data.Beds, data.Income)
res1 = st.pearsonr(data.Beds, data.Salary)
res2 = st.pearsonr(data.Beds, data.Costs)
res = st.pearsonr(data.Healing_days, data.Income)
res1 = st.pearsonr(data.Healing_days, data.Salary)
res2 = st.pearsonr(data.Healing_days, data.Costs)

#Выбор множественной регрессии
def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by AIC
    """
    remaining = set({ 'Healing_days:Income',\
        'Healing_days:Beds', 'Healing_days:Costs', 'Income:Beds', 'Income:Costs', 'Beds:Costs'})
    selected = []
    current_score, best_new_score = 1000., 1000.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ Healing_days+Income+Beds+Costs+{}".format(response,
                                           ' + '.join(selected + [candidate]))
            score = sm.OLS.from_formula(formula=formula, data=data).fit().aic
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]
        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ Healing_days+Income+Beds+Costs+{}".format(response,
                                   ' + '.join(selected))
    model = sm.OLS.from_formula(formula=formula, data=data).fit()
    return model
model = forward_selected(data, 'Salary')

print(model.model.formula)

print(model.summary())
data = data.drop(12)
data = data.drop(22)
data = data.drop(27)
model_new = sm.OLS.from_formula(formula=model.model.formula, data=data)
data_new = pd.DataFrame(pp.scale(data.values[:,:-1]), columns=['Beds','Healing_days','Income','Salary','Costs'])
model_new1 = sm.OLS.from_formula(formula=model.model.formula, data=data_new).fit()
print(model_new1.summary())
import  patsy
y, X = patsy.dmatrices(model.model.formula, data, return_type='dataframe')
model_n = lm.LinearRegression()
#Кросс-Валидация
k_fold = KFold(n_splits=10)
scores = cross_val_score(model_n, X, y, cv=k_fold, scoring='r2')
predicted = cross_val_predict(model_n,X,y,cv=k_fold)

slope, intercept, r_value, p_value, std_err = st.linregress(y.values[:,0],predicted[:,0])
print(r_value*r_value)  
#Гомоскедастичность (Бреуш-Паган, Голдфильд-Квандт)
test = sms.het_breushpagan(res11.resid, res11.model.exog)
name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
print(lzip(name, test))

name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(res11.resid, res11.model.exog)
print(lzip(name, test))

#Q-Q
st.probplot(res4.resid,plot=plt)
sm.qqplot(res11.resid, line='s')
plt.show()
#Дарбин-Уотсон
dw = sms.stattools.durbin_watson(res11.resid)
print(dw)
