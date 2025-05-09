#!/usr/bin/env python
# coding: utf-8

# In[1]:


# installlation required
get_ipython().system('pip install Lifetimes')
get_ipython().system('pip install openpyxl')


# In[2]:


get_ipython().system('pip install squarify')


# In[3]:


# libraries
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
import squarify  # treemap 
import warnings 
warnings.filterwarnings("ignore")


# In[6]:


df_2010_2011 = pd.read_csv('C:/Users/user/Desktop/Projects/CRM_Analytics/Year 2010-2011.csv', encoding='ISO-8859-1')


# In[7]:


df_2010_2011.head()


# In[8]:


df_2010_2011.shape


# In[9]:


# Invoice: The unique number of each transaction, namely the invoice. Aborted operation if it starts with C.
# We deal with purchases in our analysis. Therefore, we have excluded returns from the data.
df_2010_2011 = df_2010_2011[~df_2010_2011["Invoice"].str.contains("C", na=False)]


# In[10]:


df_2010_2011.shape


# In[11]:


# Creating a function for reading the imported dataset.
def check_df(dataframe):
    print(dataframe.shape)
    print(dataframe.columns)
    print(dataframe.dtypes)
    print(dataframe.head())
    print(dataframe.tail())
    print(dataframe.describe().T)

check_df(df_2010_2011)


# In[12]:


df_2010_2011.isnull().sum()


# In[13]:


df_2010_2011.dropna(inplace = True)


# In[14]:


df_2010_2011.isnull().sum()


# In[15]:


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df_2010_2011, "Quantity")
replace_with_thresholds(df_2010_2011, "Price")


# In[16]:


# How much sales has been recorded for each product

df_product = df_2010_2011.groupby("Description").agg({"Quantity":"count"})
df_product.reset_index(inplace=True)
df_product


# In[17]:


# Plotting top 10 products by quantity

top_pr= df_product.sort_values(by="Quantity",ascending=False).head(10)

sns.barplot(x="Description", y="Quantity", data=top_pr)
plt.xticks(rotation=90)
plt.show()


# In[18]:


# total price per invoice
df_2010_2011["TotalPrice"] = df_2010_2011["Price"] * df_2010_2011["Quantity"]


# In[19]:


df_2010_2011.head()


# In[20]:


# About RFM
# The RFM method is a tool for assessing consumer value. It's frequently utilized in database marketing and direct marketing, as well as retail and professional services.
# RFM stands for the three dimensions:
# Recency: How recently did the customer purchase, the difference between today and the customer's last purchase date, in days
# Frequency: How often do they purchase, customer's shopping frequency
# Monetary Value: How much do they spend?


# In[21]:


# Determining the analysis date for the recency

df_2010_2011["InvoiceDate"] = pd.to_datetime(df_2010_2011["InvoiceDate"])
df_2010_2011["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)


# In[22]:


# Generating RFM metrics

rfm = df_2010_2011.groupby("Customer ID").agg({"InvoiceDate": lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                    "Invoice": lambda Invoice: Invoice.nunique(),
                                    "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

rfm.columns = ["recency","frequency","monetary"]
rfm.describe().T


# In[23]:


# monetary, the min value of the total money paid can't be 0
# let's remove them from the data

rfm = rfm[rfm["monetary"] > 0]
rfm.describe().T


# In[24]:


# Generating RFM Score

# recency_score
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
# frequency_score
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
# monetary_score
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

#  RFM Score
#rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))
rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str) + rfm["monetary_score"].astype(str))
rfm.head(10)


# In[25]:


import re

seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Can\'t Lose',
    r'3[1-2]': 'About to Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

# Remove numbers from the 'segment' column using regular expressions
rfm['segment'] = rfm['segment'].apply(lambda x: re.sub(r'\d', '', x))

rfm.head(10)


# In[26]:


# Grouping RFM mean and frequency values according to segments
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])


# In[27]:


segment= rfm["segment"].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=segment.index,y=segment.values)
plt.xticks(rotation=45)
plt.title('Customer Segments',color = 'blue',fontsize=15)
plt.show()


# In[28]:


# Treemap Visualization
df_treemap = rfm.groupby('segment').agg('count').reset_index()
df_treemap.head()


# In[29]:


fig, ax = plt.subplots(1, figsize = (10,10))

squarify.plot(sizes=df_treemap['RFM_SCORE'], 
              label=df_treemap['segment'], 
              alpha=.8,
              color=['tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
             )
plt.axis('off')
plt.show()


# In[30]:


# Preparing Data for CLTV

# Determining the analysis date for the recency
df_2010_2011["InvoiceDate"] = pd.to_datetime(df_2010_2011["InvoiceDate"])
df_2010_2011["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)


# In[31]:


cltv_df = df_2010_2011.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df.head()


# In[32]:


cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[cltv_df["monetary"] > 0]

cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df.head()


# In[33]:


bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])


# In[34]:


# 1 week expected purchase
cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

cltv_df.sort_values("expected_purc_1_week", ascending=False).head(10)


# In[35]:


# 1 month expected purchase
cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

cltv_df.sort_values("expected_purc_1_month", ascending=False).head(10)


# In[36]:


# Using Gamma Gamma Model

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])


# In[37]:


cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values("expected_average_profit", ascending=False).head(20)


# In[38]:


cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  
                                   freq="W",  
                                   discount_rate=0.01)


# In[39]:


# Reset index
cltv = cltv.reset_index()
# Merging the main table and the forecast values table
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
# sorting
cltv_final.sort_values(by="clv", ascending=False).head(10)


# In[40]:


# 1 Month CLTV:
cltv_1 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # 1 month
                                   freq="W",  # frequency of T
                                   discount_rate=0.01)

cltv_1.head()
cltv_1= cltv_1.reset_index()
cltv_1 = cltv_df.merge(cltv_1, on="Customer ID", how="left")
cltv_1.sort_values(by="clv", ascending=False).head(10)


# In[41]:


# 12 Month CLTV Forecast:

cltv_12 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  
                                   freq="W", 
                                   discount_rate=0.01)

cltv_12.head()
cltv_12 = cltv_12.reset_index()
cltv_12 = cltv_df.merge(cltv_12, on="Customer ID", how="left")
cltv_12.sort_values(by="clv", ascending=False).head(10)


# In[42]:


# Normalization 0-1 Range For CLV Values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

cltv_final.sort_values(by="scaled_clv", ascending=False).head()


# In[43]:


# Segmentation of Customers
cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()

cltv_final.head()


# In[44]:


# Examination of Segments
cltv_final.groupby("segment").agg({"count", "mean", "sum"})


# In[45]:


# Examination of Segments
cltv_final.groupby("segment").agg({"count", "mean"})


# In[ ]:




