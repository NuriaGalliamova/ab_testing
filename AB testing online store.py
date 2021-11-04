#!/usr/bin/env python
# coding: utf-8

# # A/B testing of hypotheses to determine strategies to increase online store revenue
# 
# 1. Prioritizing hypotheses and selecting the most promising ones.
# 2. Analyzing the results of A / B testing and choosing the most effective group of test takers.

# ## Data preprocessing

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import scipy.stats as stats
from scipy import stats as st
import matplotlib.pyplot as plt


# In[2]:


#loading the table with hypotheses
hypothesis = pd.read_csv('/datasets/hypothesis.csv')


# In[3]:


#lowercasing the column names of the hypothesis table
hypothesis.columns = hypothesis.columns.str.lower()


# In[4]:


#loading the table with users
visitors = pd.read_csv('/datasets/visitors.csv')
visitors.info()
display(visitors.sample(10))


# In[5]:


#converting the date column to the desired data type
visitors['date'] = pd.to_datetime(visitors['date'],format='%Y-%m-%d')
visitors.info()


# In[6]:


#loading the table with orders
orders = pd.read_csv('/datasets/orders.csv')
orders.info()
display(orders.sample(10))


# In[7]:


#converting the orders to the desired data type
orders['transactionId'] = orders['transactionId'].astype('object')
orders['visitorId'] = orders['visitorId'].astype('object')
orders['date'] = pd.to_datetime(orders['date'],format='%Y-%m-%d')
orders.info()


# In[8]:


#building a function to check data for missing values and duplicates
def function(df):
    d = df.duplicated().sum()
    m = df.isna().sum()
    n = df.isnull().sum()
    
    print("Duplicated data: {}".format(d))
    print("Missing data: {}".format(m))
    print("Zero value data: {}".format(n))


# In[9]:


#checking the orders table for duplicates, missing and null values
function(orders)


# In[10]:


#checking the visitors table for duplicates, missing and null values
function(visitors)


# In[11]:


#checking if the orders table contains users that are included into both study groups

#saving in separate variables the unique customers of each group
unique_visitors_A = orders[orders['group'] == "A"]['visitorId'].unique()
unique_visitors_B = orders[orders['group'] == "B"]['visitorId'].unique()

#checking the intersection of user IDs in two data sets
len(np.intersect1d(unique_visitors_A, unique_visitors_B))


# ## Prioritizing hypotheses

# ### ICE framework for prioritizing hypotheses

# <div class="alert alert-block alert-warning">
# <br>
# $$ICE = \frac{Impact * Confidence}{Efforts}$$
# </div>

# In[12]:


#applying the ICE framework and sorting the table with hypotheses in descending order of priority
hypothesis['ICE'] = hypothesis['impact'] * hypothesis['confidence'] / hypothesis['efforts']
pd.set_option('display.max_colwidth',1000)
hypothesis[['hypothesis', 'ICE']].sort_values(by='ICE', ascending=False)


# ### RICE framework for prioritizing hypotheses

# <div class="alert alert-block alert-warning">
# <br>
# $$RICE = \frac{Rearch * Impact * Confidence}{Efforts}$$
# </div>

# In[13]:


#applying the RICE framework and sorting the table with hypotheses in descending order of priority
hypothesis['RICE'] = hypothesis['reach'] * hypothesis['impact'] * hypothesis['confidence'] / hypothesis['efforts']
hypothesis[['hypothesis', 'RICE']].sort_values(by='RICE', ascending=False)


# ### Output

# The most promising hypotheses for ICE assessment: 8, 0, 7, 6.
# 
# Most promising hypotheses for RICE assessment: 7, 2, 0, 6, 8.
# 
# When finally choosing a hypothesis to be applied, it is necessary to take into account how many people it can capture during testing.

# ## A/B test analysis

# ### Plotting the cumulative revenue by groups

# In[14]:


#creating an array of unique pairs of date values and test groups
datesGroups = orders[['date','group']].drop_duplicates() 


# In[15]:


#declaring the variable ordersAggregated, where we will save the data: date, A/B-test group, number of unique
#orders in the test group up to and including the specified date, the number of unique users who made at least 1 order
#in the test group up to the specified date inclusive, 
#the total revenue of orders in the test group up to the specified date inclusive

ordersAggregated = datesGroups.apply(
    lambda x: orders[
        np.logical_and(
            orders['date'] <= x['date'], orders['group'] == x['group']
        )
    ].agg(
        {
            'date': 'max',
            'group': 'max',
            'transactionId': pd.Series.nunique,
            'visitorId': pd.Series.nunique,
            'revenue': 'sum',
        }
    ),
    axis=1,
).sort_values(by=['date', 'group'])

#declaring the variable visitorsAggregated, where we will save the date, the A / B-test group, the number of unique
#visitors in the test group up to and including the specified date

visitorsAggregated = datesGroups.apply(
    lambda x: visitors[
        np.logical_and(
            visitors['date'] <= x['date'], visitors['group'] == x['group']
        )
    ].agg({'date': 'max', 'group': 'max', 'visitors': 'sum'}),
    axis=1,
).sort_values(by=['date', 'group'])


# In[16]:


#combining a cumulative data in one table and assigning clear names to its columns
cumulativeData = ordersAggregated.merge(visitorsAggregated, left_on=['date', 'group'], right_on=['date', 'group'])
cumulativeData.columns = ['date', 'group', 'orders', 'buyers', 'revenue', 'visitors']
print(cumulativeData.head(5)) 


# In[17]:


#dataframe with cumulative number of orders and cumulative revenue by day in group A
cumulativeRevenueA = cumulativeData[cumulativeData['group']=='A'][['date','revenue', 'orders']]

#dataframe with cumulative number of orders and cumulative revenue by day in group B
cumulativeRevenueB = cumulativeData[cumulativeData['group']=='B'][['date','revenue', 'orders']]

import warnings
warnings.filterwarnings('ignore')

#plotting the revenue of group A
plt.figure(num=1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(cumulativeRevenueA['date'], cumulativeRevenueA['revenue'], label='A')

#plotting the revenue of group B
plt.plot(cumulativeRevenueB['date'], cumulativeRevenueB['revenue'], label='B')
plt.xticks(rotation='vertical')
plt.legend()
plt.title('Cumulative revenue chart by groups',  fontdict={'size':15})
plt.ylabel('Amount of proceeds, USD')
plt.grid(axis = 'both', alpha = 0.3)
plt.show()


# #### Output

# Throughout the test, revenue for both groups increases almost equally. However, the revenue charts for Group B grows sharply between 08/17/2019 and 08/21/2019. This could be the result of a large number of orders on these dates, or very expensive orders in a sample.

# ### Plotting the cumulative average bill by groups

# In[18]:


#plotting the cumulative average check of group A
plt.figure(num=1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(cumulativeRevenueA['date'], cumulativeRevenueA['revenue']/cumulativeRevenueA['orders'], label='A')

#plotting the cumulative average check of group B
plt.plot(cumulativeRevenueB['date'], cumulativeRevenueB['revenue']/cumulativeRevenueB['orders'], label='B')

plt.xticks(rotation='vertical')
plt.title('Cumulative average check chart by group',  fontdict={'size':15})
plt.legend()
plt.ylabel('Average check amount, USD')
plt.grid(axis = 'both', alpha = 0.3)
plt.show()


# #### Output

# The average check chart has a lot of sharp up and down movements. For group B, this is the period of August 17-21, 2019, when the graph curve goes up. Perhaps there are large orders on these dates for the group. Then more data is needed for this group in order to come to the real average check and settle at its level. There are also small spikes up on August 2-3 and 8-9, 2019. For group A, in August 1-5 on August 2-19, the average bill sharply falls, and by August 13 reaches its peak value and further maintains approximately the same level. Large orders could have been on these dates for group A.

# ### Plotting the relative change in the cumulative average check of group B to group A

# In[19]:


#collecting data in one dataframe
mergedCumulativeRevenue = cumulativeRevenueA.merge(
    cumulativeRevenueB, left_on='date', right_on='date', how='left', suffixes=['A', 'B'])

#building the ratio of average checks
plt.figure(num=1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(
    mergedCumulativeRevenue['date'], 
    (mergedCumulativeRevenue['revenueB'] / mergedCumulativeRevenue['ordersB']) /
    (mergedCumulativeRevenue['revenueA'] / mergedCumulativeRevenue['ordersA']) - 1)

#adding the X axis
plt.axhline(y=0, color='black', linestyle='--')
plt.xticks(rotation='vertical')
plt.grid(axis = 'both', alpha = 0.3)
plt.title('Graph of the relative change in the cumulative average check of group B to group A',  fontdict={'size':17})
plt.ylabel('The ratio of the average check of group B to A')
plt.show()


# #### Output

# Orders from 4 orders and more than 30,000 USD will be taken as abnormal. 
# 
# Judging by the sharp jumps in the schedule, there are large orders and picks in the sample, then this will be checked.

# ### Plotting a cumulative conversion chart by groups

# In[20]:


#counting the cumulative conversion
cumulativeData['conversion'] = cumulativeData['orders']/cumulativeData['visitors']

#separating data for group A
cumulativeDataA = cumulativeData[cumulativeData['group']=='A']

#separating data for group B
cumulativeDataB = cumulativeData[cumulativeData['group']=='B']

#plotting
plt.figure(num=1, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(cumulativeDataA['date'], cumulativeDataA['conversion'], label='A')
plt.plot(cumulativeDataB['date'], cumulativeDataB['conversion'], label='B')
plt.xticks(rotation='vertical')
plt.title('Cumulative conversion chart by group', fontdict={'size':22})
plt.legend()
plt.ylabel('The ratio of the amount of orders to the number of buyers')
plt.grid(axis = 'both', alpha = 0.3)
plt.axis(["2019-08-01", '2019-09-01', 0.02, 0.04])
plt.show()


# #### Output

# The groups fluctuated around one value, but then the conversion of group B pulled ahead and fixed, while the conversion of group A sank and also fixed. In this case, it is necessary that the test must be continued. After 18.08, the metric stabilized relatively.

# ### Plotting the relative change in the cumulative conversion of group B to group A

# In[21]:


mergedCumulativeConversions = cumulativeDataA[['date','conversion']].merge(
             cumulativeDataB
             [['date','conversion']], 
             left_on='date', 
             right_on='date', 
             how='left', 
             suffixes=['A', 'B'])

plt.plot(
    mergedCumulativeConversions['date'], 
    mergedCumulativeConversions['conversionB']/mergedCumulativeConversions['conversionA']-1, 
    label="Relative increase in conversion of group B relative to group A")

plt.xticks(rotation='vertical')
plt.title('Graph of the relative change in the cumulative conversion of group B to group A')
plt.legend()
plt.grid(axis = 'both', alpha = 0.3)

plt.axhline(y=0, color='black', linestyle='--')
plt.axhline(y=-0.1, color='grey', linestyle='--')
plt.show()


# #### Output

# At the beginning of the test, Group B lost significantly to Group A, then pulled ahead and even reached a difference of more than 20% in its favor.

# ### Plotting a dot plot of the number of orders by users

# In[22]:


#selecting a separate table with data on orders per customer
ordersByUsers = (
    orders.drop(['group', 'revenue', 'date'], axis=1)
    .groupby('visitorId', as_index=False)
    .agg({'transactionId': pd.Series.nunique})
)
ordersByUsers.columns = ['userId', 'orders']


# In[23]:


#building a dot graph of the number of orders by users
x_values = pd.Series(range(0,len(ordersByUsers)))
plt.scatter(x_values, ordersByUsers['orders']) 
plt.ylabel('Number of orders per user')
plt.title('Chart of the number of orders for each user')
plt.show()


# #### Output

# The exact number of users in each interval is not yet clear, but a gap of 2-4 orders will most likely be considered anomalous.

# ### Counting the 95th and 99th percentile of the number of orders per user

# In[24]:


print("95th percentile of the number of orders per user:", np.percentile(ordersByUsers['orders'], [95])) 
print("99th percentile of the number of orders per user:", np.percentile(ordersByUsers['orders'], [99])) 


# #### Output

# No more than 5% of users have placed an order 2 times, and 1% or less have made 4 orders. The interval from 4 orders or more will be considered abnormal.

# ### Plotting a scatter plot of order values

# In[25]:


x_values = pd.Series(range(0,len(orders['revenue'])))
plt.scatter(x_values, orders['revenue']) 
plt.ylabel('Cost of one order, USD')
plt.show()


# #### Output

# Most of the orders are in the zone up to 100,000 USD. There are single outliers with very large orders.

# ### Border to identify abnormal orders

# In[26]:


print("95th percentile of the cost of one order:", np.percentile(orders['revenue'], [95]))  
print("99th percentile of the cost of one order:", np.percentile(orders['revenue'], [99])) 


# #### Output

# 5 percent or less of buyers made an order for 28,000 USD, and less than 1 percent of buyers made a purchase for 58,000 USD. 
# 
# The lower limit of the anomalous revenue values will be considered as 28,000 USD.

# ### The statistical significance of differences in conversion between groups based on raw data

# In[27]:


#selecting data on buyers for each day of the analyzed period in group A
visitorsADaily = visitors[visitors['group'] == 'A'][['date', 'visitors']]
visitorsADaily.columns = ['date', 'visitorsPerDateA']
 
visitorsACummulative = visitorsADaily.apply(
    lambda x: visitorsADaily[visitorsADaily['date'] <= x['date']].agg(
        {'date': 'max', 'visitorsPerDateA': 'sum'}
    ),
    axis=1,
)
visitorsACummulative.columns = ['date', 'visitorsCummulativeA']

#selecting data on buyers for each day of the analyzed period in group B
visitorsBDaily = visitors[visitors['group'] == 'B'][['date', 'visitors']]
visitorsBDaily.columns = ['date', 'visitorsPerDateB']
 
visitorsBCummulative = visitorsBDaily.apply(
    lambda x: visitorsBDaily[visitorsBDaily['date'] <= x['date']].agg(
        {'date': 'max', 'visitorsPerDateB': 'sum'}
    ),
    axis=1,
)
visitorsBCummulative.columns = ['date', 'visitorsCummulativeB']

#highlighting the data on the purchases made on each day of the analyzed period in group A
ordersADaily = (
    orders[orders['group'] == 'A'][['date', 'transactionId', 'visitorId', 'revenue']]
    .groupby('date', as_index=False)
    .agg({'transactionId': pd.Series.nunique, 'revenue': 'sum'})
)
ordersADaily.columns = ['date', 'ordersPerDateA', 'revenuePerDateA']
 
ordersACummulative = ordersADaily.apply(
    lambda x: ordersADaily[ordersADaily['date'] <= x['date']].agg(
        {'date': 'max', 'ordersPerDateA': 'sum', 'revenuePerDateA': 'sum'}
    ),
    axis=1,
).sort_values(by=['date'])
ordersACummulative.columns = [
    'date',
    'ordersCummulativeA',
    'revenueCummulativeA',
]

#highlighting the data on the purchases made on each day of the analyzed period in group B
ordersBDaily = (
    orders[orders['group'] == 'B'][['date', 'transactionId', 'visitorId', 'revenue']]
    .groupby('date', as_index=False)
    .agg({'transactionId': pd.Series.nunique, 'revenue': 'sum'})
)
ordersBDaily.columns = ['date', 'ordersPerDateB', 'revenuePerDateB']
 
ordersBCummulative = ordersBDaily.apply(
    lambda x: ordersBDaily[ordersBDaily['date'] <= x['date']].agg(
        {'date': 'max', 'ordersPerDateB': 'sum', 'revenuePerDateB': 'sum'}
    ),
    axis=1,
).sort_values(by=['date'])
ordersBCummulative.columns = [
    'date',
    'ordersCummulativeB',
    'revenueCummulativeB',
]


# In[28]:


#combininig all the received data into one dataframe
data_merged = (
    ordersADaily.merge(
        ordersBDaily, left_on='date', right_on='date', how='left'
    )
    .merge(ordersACummulative, left_on='date', right_on='date', how='left')
    .merge(ordersBCummulative, left_on='date', right_on='date', how='left')
    .merge(visitorsADaily, left_on='date', right_on='date', how='left')
    .merge(visitorsBDaily, left_on='date', right_on='date', how='left')
    .merge(visitorsACummulative, left_on='date', right_on='date', how='left')
    .merge(visitorsBCummulative, left_on='date', right_on='date', how='left')
)
 
display(data_merged.head(5))


# In[29]:


#creating variables ordersByUsersA and ordersByUsersB with columns ['userId', 'orders'].
#for users who ordered at least 1 time, we indicate the number of completed orders

ordersByUsersA = (
    orders[orders['group'] == 'A']
    .groupby('visitorId', as_index=False)
    .agg({'transactionId': pd.Series.nunique})
)
ordersByUsersA.columns = ['userId', 'orders']
 
ordersByUsersB = (
    orders[orders['group'] == 'B']
    .groupby('visitorId', as_index=False)
    .agg({'transactionId': pd.Series.nunique})
)
ordersByUsersB.columns = ['userId', 'orders'] 


# In[30]:


#declaring variables sampleA and sampleB, in which the number of orders will correspond to users from different groups
#for those who did not order anything, zeros will be matched

sampleA = pd.concat(
    [ordersByUsersA['orders'],
     pd.Series(0, index=
               np.arange(data_merged['visitorsPerDateA'].sum() - len(ordersByUsersA['orders'])), 
               name='orders')],axis=0)
 
sampleB = pd.concat(
    [ordersByUsersB['orders'],
     pd.Series(0, index=
               np.arange(data_merged['visitorsPerDateB'].sum() - len(ordersByUsersB['orders'])), 
               name='orders')],axis=0) 


# In[ ]:


#finding the statistical significance of the differences in conversion between groups A and B
print("P-value in conversion for groups A and B:","{0:.3f}".format(
    stats.mannwhitneyu(sampleA, sampleB,alternative='two-sided')[1]))

#finding the relative increase in the conversion of group B
print("Relative increase in conversion of group B:","{0:.3f}".format(
    (data_merged['ordersPerDateB'].sum() / data_merged['visitorsPerDateB'].sum())
    / (data_merged['ordersPerDateA'].sum() / data_merged['visitorsPerDateA'].sum())
    - 1))


# #### Output

# Formulating hypotheses:
# 
# - H0: There are no statistically significant differences in the two groups
# - H1: There are statistically significant differences in the two groups

# - P-value in conversion for groups A and B: 0.017
# - Relative increase in conversion of group B: 0.138
# 
# According to the "raw" data, the conversions of groups A and B do not match. The first number is p-value = 0.017 less than the critical value of 0.05. 
# We reject the null hypothesis that there are no statistically significant differences in conversion rates between groups. Group B has a relative gain of 13.8%.

# ### Statistical significance of differences in the average order receipt between groups according to raw data

# In[ ]:


#finding the statistical significance of the differences in the average order receipt between groups A and B
print("The p-value of the average check of groups A and B:",'{0:.3f}'.format(
    stats.mannwhitneyu(orders[orders['group']=='A']['revenue'], orders[orders['group']=='B']['revenue'], 
                       alternative='two-sided')[1]))
#finding the relative differences in the average bill between groups A and B
print("Relative difference in average bill between groups A and B:", '{0:.3f}'.format(
    orders[orders['group']=='B']['revenue'].mean()/orders[orders['group']=='A']['revenue'].mean()-1))


# #### Output

# Formulating hypotheses:
# - Н0: there are no statistically significant differences in the average check of the two groups
# - Н1: statistically significant differences in the average check of the two groups are present

# The p-value is significantly greater than 0.05. This means that there is no reason to reject the null hypothesis and assume that there are differences in the average check. But the average check of group B is significantly higher than the average check of group A. The difference in the average check of 25.9 percent appeared due to the strong dependence of the average on emissions.

# ### Statistical significance of differences in conversion between groups according to cleaned data

# In[ ]:


#users with 4 orders made or more
usersWithManyOrders = pd.concat(
    [
        ordersByUsersA[ordersByUsersA['orders'] >= 4]['userId'],
        ordersByUsersB[ordersByUsersB['orders'] >= 4]['userId'],
    ],
    axis=0,
)


# In[ ]:


#users with an order from 28,000 USD and more
usersWithExpensiveOrders = orders[orders['revenue'] >= 28000]['visitorId']
usersWithExpensiveOrders = usersWithExpensiveOrders.rename(columns={'visitorId': 'userId'})


# In[ ]:


#combining data on abnormal users into one table
abnormalUsers = (
    pd.concat([usersWithManyOrders, usersWithExpensiveOrders], axis=0)
    .drop_duplicates()
    .sort_values()
)
print(abnormalUsers.head(5))
print("Number of abnormal users:", abnormalUsers.shape) 


# In[ ]:


#preparing samples of the number of orders by users and by groups

sampleAFiltered = pd.concat(
    [
        ordersByUsersA[
            np.logical_not(ordersByUsersA['userId'].isin(abnormalUsers))
        ]['orders'],
        pd.Series(
            0,
            index=np.arange(
                data_merged['visitorsPerDateA'].sum() - len(ordersByUsersA['orders'])
            ),
            name='orders',
        ),
    ],
    axis=0,
)

sampleBFiltered = pd.concat(
    [
        ordersByUsersB[
            np.logical_not(ordersByUsersB['userId'].isin(abnormalUsers))
        ]['orders'],
        pd.Series(
            0,
            index=np.arange(
                data_merged['visitorsPerDateB'].sum() - len(ordersByUsersB['orders'])
            ),
            name='orders',
        ),
    ],
    axis=0,
) 


# In[ ]:


#applying the statistical Mann-Whitney test to the obtained samples
print("P-value of two groups:",'{0:.3f}'.format(stats.mannwhitneyu(sampleAFiltered, sampleBFiltered, alternative='two-sided')[1]))
print("Relative increase in conversion of group B:",'{0:.3f}'.format(sampleBFiltered.mean()/sampleAFiltered.mean()-1))


# #### Output

# After clearing the data from anomalies, the conversion results remained practically unchanged.

# ### Statistical significance of differences in the average order receipt between groups according to cleaned data

# In[ ]:


#renaming the column with customers in the orders table
orders = orders.rename(columns={'visitorId': 'userId'})


# In[ ]:


#calculating the statistical significance of the differences in the average order receipt between the two groups

print("The p-value of the average check of the two groups:",
    '{0:.3f}'.format(
        stats.mannwhitneyu(
            orders[
                np.logical_and(
                    orders['group'] == 'A',
                    np.logical_not(orders['userId'].isin(abnormalUsers)),
                )
            ]['revenue'],
            orders[
                np.logical_and(
                    orders['group'] == 'B',
                    np.logical_not(orders['userId'].isin(abnormalUsers)),
                )
            ]['revenue'], alternative='two-sided'
        )[1]
    )
)

print("Relative difference between groups by average order receipt:",
    "{0:.3f}".format(
        orders[
            np.logical_and(
                orders['group'] == 'B',
                np.logical_not(orders['userId'].isin(abnormalUsers)),
            )
        ]['revenue'].mean()
        / orders[
            np.logical_and(
                orders['group'] == 'A',
                np.logical_not(orders['userId'].isin(abnormalUsers)),
            )
        ]['revenue'].mean()
        - 1
    )
)


# #### Output

# After removing the anomalous values, the P-value remained well above 0.05. This means that there are no differences in the average check of the two groups.

# ## General conclusions and decision making on the A / B test

# Let's list the main points of the past testing:
# 
# - Throughout the test, revenue for both groups increased almost evenly, as confirmed by the graph of revenue and cumulative conversion for both groups. However, the revenue charts for Group B grow sharply between 08/17/2019 and 08/21/2019. This could be the result of a large number of orders on those dates, or very expensive orders in a sample, i.e. a single outlier. This is also confirmed by the chart of the average check.
# 
# - Before removing the anomalous values in the conversion testing of the two groups, the result was that the two groups were not the same, and for the most part, group B was better than group A. After removing the anomalous values, the result remained the same.
# 
# - Before removing the anomalous values in testing the average check of the two groups, the result turned out that the two groups were the same, but for the most part group B had a higher than group A. After removing the anomalous values, the result remained the same.
# 
# In connection with the above conclusions, I propose to stop testing, and recognize the revenue of group B as more than in group A, and that this difference is statistically significant.
