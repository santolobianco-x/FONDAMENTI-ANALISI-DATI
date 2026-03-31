import pandas as pd
titanic = pd.read_csv("https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv",index_col='PassengerId')
titanic.drop(columns='Ticket',inplace=True)
titanic.rename(columns={'Pclass':'Class'}, inplace=True)
titanic['Adult'] = titanic['Age'] >= 18
print(titanic.info())



from matplotlib import pyplot as plt
#FREQUENZA RELATIVA E ASSOLUTA
age_count = titanic['Age'].dropna().astype(int).value_counts().sort_index()
plt.figure(figsize=(18,6))
plt.subplot(2,1,1)
age_count.plot.bar(color=['black'])
plt.title('Frequenza assoluta')
age_count = titanic['Age'].dropna().astype(int).value_counts(normalize=True).sort_index()
plt.subplot(2,1,2)
age_count.plot.bar()
plt.title('Frequenza relativa')
plt.show()

#FREQUENZA RELATIVA PER COMPARAZIONE
age_count_m = titanic[titanic['Sex'] == 'male']['Age'].dropna().astype(int).value_counts(normalize=True).sort_index()
age_count_f = titanic[titanic['Sex'] == 'female']['Age'].dropna().astype(int).value_counts(normalize=True).sort_index()
plt.figure(figsize=(18,6))
plt.bar(age_count_m.index+0.2,age_count_m.values,width=0.5,alpha=0.9)
plt.bar(age_count_f.index-0.2,age_count_f.values,width=0.5,alpha=0.9)
plt.xticks(titanic['Age'].dropna().unique().astype(int),rotation='vertical')
plt.legend(['M','F'])
plt.grid()
plt.show()


#FREQUENZA IMPILATA PER CLASSI

sex_counts = titanic.groupby(lambda _: 'All')['Sex'].value_counts(normalize=True).unstack()
sex_counts.plot(kind='bar', stacked=True, color=['yellow','green'])

#versione più rognosa
#plt.bar(sex_counts.index,sex_counts['male'],bottom=sex_counts['female'],color='yellow',label='male',width=0.2)
#plt.bar(sex_counts.index,sex_counts['female'],color='green',label='female',width=0.2)


plt.ylim(0,1)
plt.title("PROPORTION OF MALE AND FEMALE")
plt.xlabel("")

plt.legend(title='Sex', loc='upper center', bbox_to_anchor=(0.5,0), ncol=2)
plt.grid(axis='y',linestyle=':',alpha=0.7)
plt.show()


#GRAFICI A TORTA
mycolors = ['#AA0055','#BB06CC','#665544']
class_counts = titanic['Class'].value_counts(normalize=True).sort_index().plot.pie(colors=mycolors)
plt.ylabel("")
plt.legend(title='Class',loc='upper center',bbox_to_anchor=(0.5,0),ncol=3)
plt.show()

#GRAFICO A TORTA CON PERCENTUALE
mycolors = ['#AA0055','#BB06CC','#665544']
class_counts = titanic['Class'].value_counts(normalize=True).sort_index()
class_counts.plot(kind='pie', colors=mycolors, autopct='%1.1f%%')
plt.ylabel("")
plt.legend(title='Class',loc='upper center',bbox_to_anchor=(0.5,0),ncol=3)
plt.show()


#ECDF
link = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wines = pd.read_csv(link, sep=';')

wines_relative_frequencies = wines['volatile acidity'].value_counts(normalize=True).sort_index()
wines_relative_frequencies.plot(kind='bar',figsize=(18,6))
plt.show()

ecdf = wines['volatile acidity'].value_counts(normalize=True).sort_index().cumsum()
ecdf.plot(kind='bar', figsize=(18,6))
plt.show()
ecdf.plot(figsize=(18,6))
plt.show()


residual_sugar = wines['residual sugar']
residual_sugar = pd.cut(residual_sugar,bins=10)
residual_sugar = residual_sugar.value_counts(normalize=True).sort_index()
residual_sugar.plot.bar(figsize=(18,6))
plt.grid()
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.show()



ecdf_men_age = titanic[titanic['Sex'] == 'male']['Age'].value_counts(normalize=True).sort_index().cumsum()
ecdf_women_age = titanic[titanic['Sex'] == 'female']['Age'].value_counts(normalize=True).sort_index().cumsum()
plt.figure(figsize=(18,6))
plt.plot(ecdf_men_age.index,ecdf_men_age.values)
plt.plot(ecdf_women_age.index,ecdf_women_age.values)
plt.legend(['M','F'])
plt.show()



