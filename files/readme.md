# Introduction to Data Science (WS25/26) — Assignment Part 2


## Deliverables
- Report PDF: `report.pdf` (max 20 pages)
- Notebooks:
    - `Q1.ipynb`
    - `Q2.ipynb`
    - `Q3.ipynb`
    - `Q4.ipynb`
    - `Q5.ipynb`

## Environment
- Python version:
- Conda env:
- Key libraries (versions):
    - pandas:
    - numpy:
    - matplotlib:
    - sklearn:
    - nltk:
    - mlxtend:
    - pm4py:
    - sktime:
    - pyspark:

## Data Files
- `groceries_basket.csv`
- `frequent_itemsets.csv`
- `fines_event_log.xes`
- `nlp/` (transcripts)
- `time-series/air_traffic.csv`
- `time-series/flights_train.csv`
- `time-series/flights_test.csv`
- `big-data/event_log.csv` (or `.csv.gz` if provided)
- `docker-compose.yml` (for Spark)

---

# Q1: Market Basket Analysis (23 pts)

## (a) Product overview & frequencies
### a.1 Support ≥ 1% 对应的最小 support count
**Answer:**


- min_count_for_1% = 649

### a.2 Mean support count across all products
**Answer:**
mean_support_count = 21.588217568178393

- mean_support_count = 21.588217568178393

### a.3 Median support count across all products
**Answer:**
- median_support_count = 4.0

## (b) Top-10 most frequent products + max possible support for itemset size 3
**Top-10 table (product_id, abs freq, rel freq):**
- |   product_id |   abs_freq |   rel_freq |
  |-------------:|-----------:|-----------:|
  |        24852 |       9376 |  0.144549  |
  |        13176 |       7701 |  0.118725  |
  |        21137 |       5480 |  0.0844845 |
  |        21903 |       4818 |  0.0742785 |
  |        47626 |       4096 |  0.0631475 |
  |        47766 |       3670 |  0.0565799 |
  |        47209 |       3605 |  0.0555778 |
  |        16797 |       3199 |  0.0493186 |
  |        26209 |       3017 |  0.0465127 |
  |        27966 |       2728 |  0.0420572 |

**Maximum support for 3-item itemset & which product IDs:**
- |   max_support_size3_upper | itemset_product_ids   |
  |--------------------------:|:----------------------|
  |                 0.0844845 | [24852, 13176, 21137] |
- 理论最大值为三个商品中的最小相对支持率。也就是top3:0.0844845

## (c) Orders containing all 3 products + explanation
**Count of orders containing all 3:**
- 3
- |   orders_containing_all_three |   product_id_1 | product_name_1   |   product_id_2 | product_name_2         |   product_id_3 | product_name_3       |
  |------------------------------:|---------------:|:-----------------|---------------:|:-----------------------|---------------:|:---------------------|
  |                             3 |          24852 | Banana           |          13176 | Bag of Organic Bananas |          21137 | Organic Strawberries |

**Explanation (1–2 sentences):**
- 香蕉和一袋有机香蕉是明显重合的商品，而草莓也同属于水果分类。顾客通常只会购买其中一个所以三者同时出现在同一订单的概率低。

## (d) Category plots
1) **Total number of sold products per category (basket items)**  
   **Figure:**
-参考对应代码生成图片

2) **Mean & median number of items per order for each category**  
   **Figure:**
-参考对应代码生成图片

## (e) Beverages: mean vs median interpretation (2–3 sentences)
**Answer:**
- 均值为0,73中位数是0 说明大多数的订单是不单独购买饮料的，但是买的时候会更加集中。整体数据呈现长尾分布。

## (f) Frequent itemsets file analysis
### f.1 Names of products in the largest frequent itemset
**Answer:**
- Bag of Organic Bananas, Organic Strawberries, Organic Raspberries, Organic Hass Avocado

### f.2 # frequent itemsets of size ≥ 2 (abs & rel)
**Answer:**
- 1017个，占比0.5146 具体见图

### f.3 Item appearing in most frequent itemsets + count
**Answer:**
- banana,出现168次

### f.4 # products appearing in only a single frequent itemset
**Answer:**
- 794个

## (g) Category sets for itemsets (size ≥ 2) + frequency plot
**Figure:**
- 具体见图



## (h) Explain 3 observations (each 1–2 sentences) + implication for beverages diversity
### h.1 Why produce-only itemsets dominate
- 果蔬是日常生活中最必要，最常见的产品。通常购买都会买多种也就是size>2，根据类别映射后就会坍缩为{produce}。

### h.2 Why produce appears in nearly all frequent itemsets
- 因为果蔬是生活必需品，购买其他物品也经常会顺带买一些。同时produce类别下产品很多所以很容易在itemsets里面出现。

### h.3 Why four beverage-only frequent itemsets exist & implication
- 结合前面median=0，mean=0.73可以知道，大多数订单要么不买饮料要么就买的比较多。所以在那部分确实买饮料的订单里，顾客很可能会一次买多件、甚至经常以固定搭配一起买（比如两种常一起买的饮料）

## (i) Association rules (min support_count=100, confidence≥0.2)
### i.1 Mean & median lift across all rules
- mean:3.9534   median:2.831

### i.2 # rules with top-10 product(s) in antecedent
- 228

### i.3 # rules with top-10 product(s) in consequent
- 430

## (j) Highest-confidence rule + interpretation + why highest-frequency product as consequent (≤3 sentences)
**Rule:**
- | antecedents                      | consequents        |    support |   confidence |    lift | ante_names                                                              | cons_names                 | ante_contains_top10   | cons_contains_top10   |
  |:---------------------------------|:-------------------|-----------:|-------------:|--------:|:------------------------------------------------------------------------|:---------------------------|:----------------------|:----------------------|
  | frozenset({47209, 21137, 27966}) | frozenset({13176}) | 0.00188086 |     0.622449 | 5.24276 | ['Organic Strawberries', 'Organic Raspberries', 'Organic Hass Avocado'] | ['Bag of Organic Bananas'] | True                  | True                  |

**Interpretation (1–2 sentences):**
-这条规则说明当一个订单同时包含这三种有机莓类/水果（前件）时，有约 62.24% 的概率也会购买 “Bag of Organic Bananas”。并且 lift≈5.24 说明这种同时购买香蕉的概率远高于“随机买到香蕉”的基准水平，属于很强的正相关共购关系 

**Why consequent rather than antecedent (max 3 sentences):**
- 后件香蕉 support = 0.118725（约 11.87% 的订单都买），本身就是非常常见的商品，因此它很容易作为许多规则的 consequent 出现。对一个比较具体的前件组合来说，后件如果是这种“高基数常购品”，规则的 confidence 更容易变高。所以他也容易成为置信度最高的项

## (k) Rules excluding top-10 products: lift observation + explanation (2–3 sentences)
**Observation:**
- 不包含top-10的rule lift相较于包含top10的非常大。

**Explanation:**
- 因为 top10 商品本身出现概率很高：即便没有任何前件，后件也经常会出现，所以很多包含 top10 后件的规则虽然置信度可能高，但相对于基准概率提升有限。lift需要除以后件的支持度，非top10的支持度更低所以除之后lift会变大。

## (l) Bi-directional rules & sequence subsequences support counts
**Table (a→b subsequence count, b→a subsequence count):**
- |    |     a | a_name                                                |     b | b_name                                                |   support_count_<a,b> |   support_count_<b,a> |
  |---:|------:|:------------------------------------------------------|------:|:------------------------------------------------------|----------------------:|----------------------:|
  |  0 | 13176 | Bag of Organic Bananas                                | 21137 | Organic Strawberries                                  |                  1149 |                   404 |
  |  1 | 33754 | Total 2% with Strawberry Lowfat Greek Strained Yogurt | 33787 | Total 2% Lowfat Greek Strained Yogurt with Peach      |                    69 |                    47 |
  |  2 | 28465 | Icelandic Style Skyr Blueberry Non-fat Yogurt         | 36865 | Non Fat Raspberry Yogurt                              |                    59 |                    61 |
  |  3 |  4957 | Total 2% Lowfat Greek Strained Yogurt With Blueberry  | 33754 | Total 2% with Strawberry Lowfat Greek Strained Yogurt |                    53 |                    57 |

**Notes / method (brief):**
- 第一行可以看出加入香蕉之后再加入草莓比较常见，两者的support_count差异明显。具有强倾向性。

## (m) Why rule with support_count≥100, confidence>0.2, lift<1 is impossible (brief)
**Answer:**
- 这是因为要lift<1 也就是confidence(A->B)<support(b). 又因为confidence(A->B)>0.2所以support(B)要大于0.2.但是原题目中最大的support也才0.11。
- 核心就是对根据定义对值的放缩判定。


---

# Q2: Process Mining (17 pts)

## (a) Inductive Miner model analysis (based only on discovered model)
### a.1 Possible start activities
- 

### a.2 Activities & order for traces with appeal to judge
- 

### a.3 Activities that can occur more than once
- 

### a.4 Credit collection without sending via post possible?
- 

## (b) Count cases in log exhibiting behaviors allowed by model
### b.1 Payment made but still sent to credit collection (count)
- 

### b.2 Penalty before sending via post (count)
- 

### b.3 Appeal after a payment (count)
- 

### b.4 Notified about appeal result without any appeal (count)
- 

## (c) Variant analysis
### c.1 Cumulative variant frequency chart
**Figure:**
- 

### c.2 Interpretation (2 sentences)
- 

### c.3 Five most frequent variants & #cases
- 

## (d) Case closure categories pie chart (include relative frequencies)
**Figure:**
- 

**Counts / shares (optional table):**
- 

## (e) Closed cases sublog (57756 cases) → top-5 closed-case variants → Petri net + comment (3–4 sentences)
**Filtered variants (if needed):**
- 

**Discovered Petri net figure:**
- 

**Comment (activities, payment timing, payment vs credit collection):**
- 

## (f) Conformance: token-based replay fitness of full log on filtered-log model
- Perfectly fitting traces (%):
- Log fitness value:
- Explanation why fitness >> perfect-trace-%:
  - 

---

# Q3: Natural Language Processing (20 pts)

## Data loading
- Loaded files:
- Final dataframe shape (expected ~ (1793, 5)):
- Notes:
  - 

## (a) 25 most frequent words (no preprocessing) + 2 problems & fixes
**Histogram figure:**
- 

**Problem 1 + fix:**
- 

**Problem 2 + fix:**
- 

## (b) Preprocess: lowercase + remove punctuation + nltk punkt_tab tokenize + stopword removal
- Function signature / description:
  - 

**Histogram (25 most frequent tokens in 01-introduction, no stopwords):**
- 

## (c) Stacked histogram for tokens by lecture
Tokens: `data, decision, predict, derivative, network, easy, database`  
**Figure:**
-

**Observation (2–3 sentences):**
- 

## (d) n-gram language model generation
- Implementation notes (n-grams, padding <s>, </s>, ConditionalFreqDist, random.choices seed=32133, lexicographic sorting):
  - 

**Generated text for seed “introduction to data” (max 30 generated tokens):**
- n=3:
  - 
- n=4:
  - 
- n=5:
  - 
- n=24:
  - 

**Differences commentary:**
- 

**Expected behavior for n > 5 & why:**
- 

## (e) Hierarchical TF-IDF timestamp retrieval (k=2, m=2)
- Preprocessing pipeline (tokenize + stopwords + stemming):
  - 
- Implementation notes (level-1 lecture docs, level-2 segment docs, cosine similarity via dot):
  - 

### Query 1: “gradient descent approach”
**Top-k lectures + scores + top-m timestamps:**
- Lecture 1:
    - Score:
    - Timestamps:
    - Matching text (brief excerpt):
- Lecture 2:
    - Score:
    - Timestamps:
    - Matching text (brief excerpt):

**Comment (1–2 sentences):**
- 

### Query 2: “beer and diapers”
**Top-k lectures + scores + top-m timestamps:**
- Lecture 1:
    - Score:
    - Timestamps:
    - Matching text (brief excerpt):
- Lecture 2:
    - Score:
    - Timestamps:
    - Matching text (brief excerpt):

**Comment (1–2 sentences):**
- 

---

# Q4: Time Series Analysis (23 pts)

## (a) Exploration
### a.i Load air_traffic.csv
1) Time range covered:
-
2) # months tracked:
-
3) Overall # flights:
-

### a.ii Time series plot: overall flights per month + description
**Figure:**
- 

**Description:**
- 

### a.iii Plot passengers vs flights (time series) + compare trends
**Figure:**
- 

**Comparison:**
- 

## (b) In-depth analysis
### b.i Yearly seasonal plots (flights & passengers) + description (2–3 sentences)
**Figures:**
- 

**Description & consistency notes:**
- 

### b.ii Correlation coefficient passengers vs flights + interpretation
- Correlation:
- Interpretation:
- Relation to a.iii:
  - 

### b.iii Correlograms for differencing {0,1,2}, lags up to 24 + description + most significant lags
**Figures:**
- 

**Description:**
- 

**Most significant lags (excluding 0) across all correlograms:**
- 

### b.iv STL decomposition (period=12) + stationary residual?
**Figure (trend/seasonal/resid):**
- 

**Residual stationary?**
- 

## (c) Forecasting (flights_train/test)
### c.i Suitability for forecasting (2–3 sentences)
- 

### c.ii NaiveForecaster(mean) vs ARIMA (sktime) — find better ARIMA across RMSE/MAE/MAPE
- Naive metrics:
    - RMSE:
    - MAE:
    - MAPE:
- Best ARIMA order (p,d,q):
  - 
- ARIMA metrics:
    - RMSE:
    - MAE:
    - MAPE:
- Notes:
  - 

### c.iii Plot train/test + forecast + comment
**Figure:**
- 

**Is forecast good? Why/why not:**
- 

---

# Q5: Distributed Data Processing (17 pts)

## (a) MapReduce DFG computation: map/reduce signatures + explanation
- Domains used (S, N0, C, A, T):
  - 

**Map function signature(s):**
- 

**Reduce function signature(s):**
- 

**High-level explanation:**
- 

## (b) Spark DFG discovery: RDD transformations used + explanation + DFG figure
**Transformations list + what each does:**
- 

**DFG figure:**
- 

## (c) Remove arcs with frequency < 100
### c.1 Update MapReduce solution (brief)
- 

### c.2 Update Spark implementation: new/changed transformations + explanation
- 

### c.3 DFG after filtering figure
**Figure:**
- 

## (d) Most common resource per activity (Spark)
**Transformations used + explanation:**
- 

**Result table (Activity → most common resource):**
- 

---

## Notes / References
- Any assumptions, edge cases, or implementation details:
  - 
