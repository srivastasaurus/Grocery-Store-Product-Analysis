# importing the libraries
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from mlxtend.preprocessing import TransactionEncoder
"""Using TransactionEncoder object, we can transform this dataset into an array format suitable for typical machine learning APIs. Via the fit method, the TransactionEncoder learns the unique labels in the dataset, and via the transform method, it transforms the input dataset (a Python list of lists) into a one-hot encoded NumPy boolean array:"""
# APRIORI FUNCTION 
from mlxtend.frequent_patterns import apriori, association_rules


# ----------------------------
# ITERTOOLS 
import itertools

# ----------------------------
# CONFIGURATION
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

""" this will help to display the columns wuth maximum width"""
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format

df = pd.read_csv(r"C:\Users\prern\Documents\GitHub\web_scrape\GroceryStoreDataSet.csv",names=['products'],header=None)
"""print(df)"""

# a value "cock" replaced by muffin
df.replace(to_replace = "COCK", value = "MUFFIN" , inplace = True , regex = True , method = "pad" )

df.replace(to_replace = "SUGER", value = "SUGAR" , inplace = True , regex = True , method = "pad" )

""" tidy data for association rules """

# taking out the list of items after separating with the commas 
data = list(df["products"].apply(lambda x:x.split(',')))

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data,columns=te.columns_).astype(int)

print(df)
p = df.copy()
for i in range(1, len(p.columns)+1):
    p["Cat"] = np.where(p[p.columns[i]] == 1, 1, 0)
    p[p.columns[i]]= i
    g =sns.scatterplot(p.index, p[p.columns[i]], hue = p.Cat, legend = False)
    g.yaxis.set_label_text('Products')
    g.set_yticks(np.arange(1, len(p.columns)))
    g.set_xticks(df.index)
    g.set_yticklabels(df.columns)
    plt.title("Data Structure")


# first iteration : find support values for each product 
# Find Frequency of Items
df.sum()

""" 
If we divide all items with row number, we can find Support value. Our threshold value is 0.2 for Support value.""" 

# Product Frequency / Total Sales
first = pd.DataFrame(df.sum() / df.shape[0], columns = ["Support"]).sort_values("Support", ascending = False)
first

# Elimination by Support Value
first[first.Support >= 0.15]

# creating the pie chart 

explode = np.linspace(0, 1 , 11) # Shift the slices away from the centre of the pie 
plt.figure(dpi = 108)

plt.pie(first["Support"] , labels = list(first.index) , wedgeprops = {"edgecolor" :"black"} ,autopct='%1.2f%%',pctdistance=1.2, labeldistance=1.5)
plt.show()


# creating the donut chart 
index_list = list(first.index)

fig1, ax1 = plt.subplots()
explode = np.linspace(0, 0.09, 11)
#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', "red", "orange", "pink","green", "blue", "yellow", "purple"]
plt.pie(first["Support"] , labels= list(first.index), autopct='%1.1f%%',colors = colors, startangle=90, pctdistance=0.85 , explode = explode)

#draw circle

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle) 
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.show()


# a new graph consisting only the pie chart with the names of the products 


plt.figure(dpi = 108)

fig1, ax1 = plt.subplots()
patches, texts = ax1.pie(first["Support"], labels=first.index , colors = colors, 
                        startangle=10, labeldistance=0.8, explode = explode )
for t in texts:
    t.set_horizontalalignment('center')

plt.show()








#  Second Iteration: Find support values for pair product combinations.

second = list(itertools.combinations(first.index, 2))   # this create a list of pair of combinations of products 
second = [list(i) for i in second]  # earlier it was a tuple , converting that into a list 
# Sample of combinations
second[:10]

# Finding support values
value = []
for i in range(0, len(second)):
    temp = df.T.loc[second[i]].sum() 
    temp = len(temp[temp == df.T.loc[second[i]].shape[0]]) / df.shape[0]
    value.append(temp)
# Create a data frame            
secondIteration = pd.DataFrame(value, columns = ["Support"])
secondIteration["index"] = [tuple(i) for i in second]
secondIteration['length'] = secondIteration['index'].apply(lambda x:len(x))
secondIteration = secondIteration.set_index("index").sort_values("Support", ascending = False)
# Elimination by Support Value
secondIteration = secondIteration[secondIteration.Support > 0.1]
secondIteration 





 # Shift the slices away from the centre of the pie 
plt.figure(dpi = 108)

plt.pie(secondIteration["Support"] , labels = list(secondIteration.index), wedgeprops = {"edgecolor" :"black"} ,autopct='%1.2f%%')
plt.show()





def ar_iterations(data, num_iter = 1, support_value = 0.1, iterationIndex = None):
    
    # Next Iterations
    def ar_calculation(iterationIndex = iterationIndex): 
        # Calculation of support value
        value = []
        for i in range(0, len(iterationIndex)):
            result = data.T.loc[iterationIndex[i]].sum() 
            result = len(result[result == data.T.loc[iterationIndex[i]].shape[0]]) / data.shape[0]
            value.append(result)
        # Bind results
        result = pd.DataFrame(value, columns = ["Support"])
        result["index"] = [tuple(i) for i in iterationIndex]
        result['length'] = result['index'].apply(lambda x:len(x))
        result = result.set_index("index").sort_values("Support", ascending = False)
        # Elimination by Support Value
        result = result[result.Support > support_value]
        return result    
    
    # First Iteration
    first = pd.DataFrame(df.T.sum(axis = 1) / df.shape[0], columns = ["Support"]).sort_values("Support", ascending = False)
    first = first[first.Support > support_value]
    first["length"] = 1
    
    if num_iter == 1:
        res = first.copy()
        
    # Second Iteration
    elif num_iter == 2:
        
        second = list(itertools.combinations(first.index, 2))
        second = [list(i) for i in second]
        res = ar_calculation(second)
        
    # All Iterations > 2
    else:
        nth = list(itertools.combinations(set(list(itertools.chain(*iterationIndex))), num_iter))
        nth = [list(i) for i in nth]
        res = ar_calculation(nth)
    
    return res


iteration1 = ar_iterations(df, num_iter=1, support_value=0.1)
iteration1

iteration2 = ar_iterations(df, num_iter=2, support_value=0.1)
iteration2

iteration3 = ar_iterations(df, num_iter=3, support_value=0.01,
              iterationIndex=iteration2.index)
iteration3

iteration4 = ar_iterations(df, num_iter=4, support_value=0.01,
              iterationIndex=iteration3.index)
iteration4

plt.figure(dpi = 108)

plt.pie(iteration4["Support"] , labels = list(iteration4.index), wedgeprops = {"edgecolor" :"black"} ,autopct='%1.2f%%')
plt.show()
# 6. Association Rules
"""There are two main functions here.

apriori() function evaluate support value for each product.
association_rules() function help us to understand relationship between antecedents and consequences products. It gives some remarkable information about products."""

# Apriori
freq_items = apriori(df, min_support = 0.1, use_colnames = True )
freq_items.sort_values("support", ascending = False).head()

freq_items.sort_values("support", ascending = False ).tail()

# Association Rules & Info
df_ar = association_rules(freq_items, metric = "confidence", min_threshold = 0.5)
df_ar

df_ar[(df_ar.support > 0.15) & (df_ar.confidence > 0.5)].sort_values("confidence", ascending = False)


