# Python implementation of the Apriori algorithm 
<br />

### The apriori algorithm uncovers hidden structures in categorical data. We would like to uncover association rules such as {bread, eggs} -> {bacon} from the data. This is the goal of [association rule](https://en.wikipedia.org/wiki/Association_rule_learning) learning, and the [Apriori algorithm](https://machinelearningknowledge.ai/best-explanation-of-apriori-algorithm-for-association-rule-mining/) is arguably the most famous algorithm for this problem.


# Example 

<br />

```
from efficient_apriori import apriori
transactions = [('eggs', 'bacon', 'soup'),
                ('eggs', 'bacon', 'apple'),
                ('soup', 'bacon', 'banana')]
itemsets, rules = apriori(transactions, min_support=0.5, min_confidence=1)
print(rules)  # [{eggs} -> {bacon}, {soup} -> {bacon}]

```

# Installation 

## The software is available through GitHub, and through [PyPI](https://pypi.org/project/efficient-apriori/). You may install the software using pip.

# Running the tests
<br />

* The program takes data source , Minimum Support in percentage and Minimum Confidence in perecentage as input .
* Every purchase has a number of items associated with it. 
* Data Source : This is to select where the input is coming from. For this test , the data is coming from one of the competitions from kaggle .
* Minimum Support : A minimum Support is applied to find all frequent itemsets in a database .
* Minimum Confidence : A minimum confidence is applied to these frequent itemsets in order to form rules. 
* Result : The result will show the association rules in teh given dataset with the given minimum support and minimum confidence if there are any. If there are no association rules in the set with the given support and confidence conditions , try to plug in some different ( if  you didn't get any results, try feeding some lower values) values of them .

<img src="https://github.com/srivastasaurus/Grocery-Store-Product-Analysis/tree/main/images/donut_chart" alt="Your image title" width="250"/>
