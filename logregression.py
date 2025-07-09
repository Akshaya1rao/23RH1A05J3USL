import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

data={
    'maths':[78,89,50,45,15,33,37,85,22,66],
    'physics':[45,67,89,97,16,16,44,83,10,88],
    'chemistry':[42,98,77,63,28,25,82,51,31,99],
    'result':['pass','pass','pass','pass','fail','fail','pass','pass','fail','pass']
}
df=pd.DataFrame(data)
df['result']=df['result'].map({'pass':1,'fail':0})
x=df[['maths','physics','chemistry']]
y=df['result']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy:",accuracy_score(y_test,y_pred))
new_stu=pd.DataFrame([[31,20,17]],columns=['maths','physics','chemistry'])
prediction=model.predict(new_stu)
print(prediction[0])