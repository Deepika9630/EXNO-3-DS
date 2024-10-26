## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```Developed  By: Sakthi Navaneetha M.R
Register Number : 212222040138
````
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df

![image](https://github.com/user-attachments/assets/2048d9c3-a724-41b0-a06e-d6f01074051c)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/user-attachments/assets/f0b9518e-c268-4a86-b55a-45fe02467b00)
```df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/user-attachments/assets/dddab473-f7b0-4608-a871-8705007ed2b3)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![image](https://github.com/user-attachments/assets/be603242-8fad-404e-8d9e-f8d904a6e893)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

```

![image](https://github.com/user-attachments/assets/229c866b-feed-4fec-ba12-25549c802df6)
```df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/user-attachments/assets/5c08737f-1172-4373-a407-3b01e75483c3)
```
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/user-attachments/assets/5cd59a49-9cbd-4bc0-8b3a-e79e2a339dc3)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/0a2817da-3b30-4e18-a628-0aa5cbd41819)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df

```

![image](https://github.com/user-attachments/assets/607a7c45-9d79-426d-b2e4-dc5d90eb3ec7)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb 
```

![image](https://github.com/user-attachments/assets/553ed341-191c-4812-99e8-d98d4f11889a)

```from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![image](https://github.com/user-attachments/assets/97e75e3b-3723-4e15-bf2e-c49c651c6f6f)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![image](https://github.com/user-attachments/assets/d54aef54-0d8d-441a-9b90-98bde7310818)

df.skew()
![image](https://github.com/user-attachments/assets/4448cd23-48ea-4444-82c0-fa1ce05adf73)


np.log(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/622dc0fd-4af2-4512-a2bc-1108edc9287c)


np.reciprocal(df["Moderate Positive Skew"])
![image](https://github.com/user-attachments/assets/f7b5dc92-2b4c-4d5f-b648-dc646018e3dc)


np.sqrt(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/439b3b61-b364-4bcb-b067-181c654ae7cc)


np.square(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/3ed2682d-bfe6-49f7-99b9-b4189169862c)


df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
![image](https://github.com/user-attachments/assets/a01f4e5b-bec2-4080-a062-5a374b3cbc9e)


df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
![image](https://github.com/user-attachments/assets/dc2378be-4e82-4070-9724-3c10ef255c68)


import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
![image](https://github.com/user-attachments/assets/2a1e3427-eb08-4c45-9852-befa4a9e9397)


sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
![image](https://github.com/user-attachments/assets/f7b8cddf-914a-4c4f-b729-d6b4976e6d2b)


from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
![image](https://github.com/user-attachments/assets/4d3575a0-cb3f-46e1-adfa-cd032fefff60)


df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
![image](https://github.com/user-attachments/assets/d2edda94-1c76-43c4-b0f0-8d72e9994ded)


sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
![image](https://github.com/user-attachments/assets/2fb4fbe7-15bb-4daa-b5de-8781192a8270)


sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()


![image](https://github.com/user-attachments/assets/6c7695a2-1acc-4a75-a782-58b49eee729d)


# RESULT:
      Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
