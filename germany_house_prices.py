import pandas as pd
import numpy as np

# visuals
import seaborn as sns
import matplotlib.pyplot as plt


# Support Functions
def get_df_null_info(df):
    # analzyse sparse columns (= many NaNs/Null-Values)
    nulls:pd.Series =df.isnull().sum(axis = 0).sort_values(ascending = False)
    df_null_info:pd.DataFrame = pd.DataFrame(data={
        'nulls_amount': nulls,
        'nulls_percentage': nulls.apply(lambda row: round((row/df.shape[0])*100,2))
    })
    return df_null_info


# Get to know the data
# import
df_source = pd.read_csv("germany_housing_data_14.07.2020.csv", sep=",")

df_null_info = get_df_null_info(df_source)
df_null_info

df_source.describe(include="all")


# Pre-Processing
### Data Reduction

# Drop all columns:
#  (a) with more than 26% null values
#  (b) 'City' & 'Place' because  'State', 'City', and 'Place' have same information but with different granularity
drop_col_percentage_treshold=26
cols_to_drop = list(df_null_info[(df_null_info['nulls_percentage'] > drop_col_percentage_treshold)].index.values) 
cols_to_drop.append('City')
cols_to_drop.append('Place')
print(f'Drop following columns: {cols_to_drop}')

df_source.drop(columns=cols_to_drop, inplace=True)



### Data Cleaning (handle noisy & missing data)

df_null_info = get_df_null_info(df_source)
df_null_info

##### Start to clean each column.
# for column 'City' drop all rows with no city
df_source.drop(df_source[df_source['State'].isnull()].index, inplace=True)
df_source.loc[df_source['State'].isnull()]

# for all other columns: fill NaNs with mode value
for col in df_source:
    df_source[col].fillna(df_source[col].mode()[0], inplace=True)






#plt.figure (figsize=(16,5))  
#plt.subplot(1,2,1)
#sns.distplot(df_source['Price'])
#plt.subplot(1,2,2)
#sns.distplot(df_source['Bathrooms'])
#plt.show()
sns.boxplot(df_source['Floors'])


# categorical data
# df_source['Garagetype'].value_counts().plot(kind='bar')




### Data Transformation (How to handle outliers?)
# - one hot encoding   (categorical data)
# - how to handle outliers (numerical data)

df_source.dtypes



## remove outlier: z_score transformation


# get all numerical columns
cols_to_zscore = list(df_source.select_dtypes(include='float64'))
cols_to_zscore.remove("Price")
cols_zscore = []

# create df with zscore-columns 
for col in cols_to_zscore:
    col_zscore = col + '_zscore'
    cols_zscore.append(col_zscore)
    
    # zscore of x in column :    (x- col.mean)-col.std
    # -> add an extra column for zscore
    df_source[col_zscore] = (df_source[col] - df_source[col].mean())/df_source[col].std(ddof=0)

cols_zscore


treshhold = 3
treshhold_range = [-treshhold, treshhold]

df_source["max_zscore"] = df_source[cols_zscore].max(axis=1) 
df_source['min_zscore'] = df_source[cols_zscore].min(axis=1) 


df_source['is_outlier'] = (
    (treshhold <= df_source["max_zscore"]) | 
    (df_source["max_zscore"] <= -treshhold ) | 

    (treshhold  <= df_source["min_zscore"]) | 
    (df_source["min_zscore"]  <= -treshhold )
)
print(len(df_source[df_source['is_outlier'] == True]))



df_new = df_source[df_source['is_outlier'] == False]
fig = plt.figure( figsize=(15, 20))
id=1
for i in range(0,len(cols_to_zscore)):
    col = cols_to_zscore[i] 

    # old one
    ax = fig.add_subplot(7,2,id)
    ax.boxplot(df_source[col])
    ax.set_title(f"{col} (before)")
    id=id+1

    # new one
    ax = fig.add_subplot(7,2,id)
    ax.boxplot(df_new[col])
    ax.set_title(f"{col} (after)")
    id=id+1



