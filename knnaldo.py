import pandas as pd
from sklearn.neighbours import KNeighoursClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data={
    'weight':[150,170,140,130,120,180,110,90,100],
    'size':[7.0,7.5,6.8,6.5,5.5,5.7,7.8,5.2,5.0,5.3],
    'fruit':["Apple","Apple","Apple","Apple","orange","Apple","orange","banana","banana"]


}
df=pd.DataFrame(data)
df['encode']