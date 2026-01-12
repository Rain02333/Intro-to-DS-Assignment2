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
- {'Create Fine'}

### a.2 Activities & order for traces with appeal to judge
- 一定要包含的activities{
  Create Fine, Insert Fine Notification, Send Fine, Payment, Add penalty,
  Appeal to Judge, Send Appeal to Prefecture, Insert Date Appeal to Prefecture,
  Receive Result Appeal from Prefecture, Notify Result Appeal to Offender,
  Send for Credit Collection
  }。
- 一定会在之前发生的事件must_before_blocks = [{'Create Fine'}, {'Insert Fine Notification'}]
- 一定会在这之后发生的事件must_after_blocks = [{'Send for Credit Collection'}]

### a.3 Activities that can occur more than once
- ['Payment']

### a.4 Credit collection without sending via post possible?
- 可能，其中有一条trace是 ('Create Fine',
  'Notify Result Appeal to Offender',
  'Send for Credit Collection'))

## (b) Count cases in log exhibiting behaviors allowed by model
### b.1 Payment made but still sent to credit collection (count)
- 1013 cases

### b.2 Penalty before sending via post (count)
- 0 cases

### b.3 Appeal after a payment (count)
- 10 cases

### b.4 Notified about appeal result without any appeal (count)
- 17 cases

## (c) Variant analysis
### c.1 Cumulative variant frequency chart
**Figure:**
- 见图

### c.2 Interpretation (2 sentences)
- 前 2 个变体就覆盖约 68.44% 的 cases（0.375717 + 0.308688 ≈ 0.684405），说明流程在实际执行中高度集中，主要遵循少数几条主路径。
之后曲线很快趋于平缓；尽管总共有大约 160 个变体，但新增变体对累计覆盖率的贡献很小，呈现明显的长尾分布。

### c.3 Five most frequent variants & #cases
- 见表格

## (d) Case closure categories pie chart (include relative frequencies)
**Figure:**
- 见图

**Counts / shares (optional table):**
- still_open                  13766
  closed_paid_full            28718
  closed_dismissed              966
  closed_credit_collection    28072
- Share
  still_open                  0.1925
  closed_paid_full            0.4015
  closed_dismissed            0.0135
  closed_credit_collection    0.3925

## (e) Closed cases sublog (57756 cases) → top-5 closed-case variants → Petri net + comment (3–4 sentences)


**Comment (activities, payment timing, payment vs credit collection):**
- 与 Q2(a) 用全量日志发现的模型相比，Q2(e) 在 closed sublog 且只保留最常见 5 个变体后得到的模型明显更简洁，只包含主干活动（Create Fine、Send Fine、Insert Fine Notification、Add penalty、Payment、Send for Credit Collection），上诉等长尾活动不再出现。
  从这 5 个变体看，Payment 的时机更“结构化”：要么直接在 Create Fine → Payment（22053 个 case），要么在 … → Add penalty → Payment 之后，甚至存在重复支付 Payment → Payment（1683 个 case）。
  另外，催收在主流 closed 行为中表现为一种与支付互斥的结局：… → Send for Credit Collection 的变体不包含 Payment，而所有包含 Payment 的变体都不包含 Send for Credit Collection。

## (f) Conformance: token-based replay fitness of full log on filtered-log model
- Perfectly fitting traces (%):80.25782276781969
- Log fitness value: 0.9708997543726612
- Explanation why fitness >> perfect-trace-%:
  - 因为 perfectly fitting 是 0/1 的严格判断：只要某条 trace 有一点偏差就不算完全拟合。

---

# Q3: Natural Language Processing (20 pts)

## Data loading
- Final dataframe shape (expected ~ (1793, 5)):1793：4  原本3列加一个lecture就是4列啊 为什么说expected是5啊 没懂。

  

## (a) 25 most frequent words (no preprocessing) + 2 problems & fixes
**Histogram figure:**
- 见图

**Problem 1 + fix:**
- 高频词汇全是停用词信息量低无法反应课程内容。 
- 去停用词（stopwords），必要时再做词形归一（lemma/stem）。

**Problem 2 + fix:**
- 仅使用空格分割导致and  And   say,  这种被分成单独的token导致同义词多token的情况。
- 统一小写 + 去标点/用更规范 tokenizer。

## (b) Preprocess: lowercase + remove punctuation + nltk punkt_tab tokenize + stopword removal

**Histogram (25 most frequent tokens in 01-introduction, no stopwords):**
- 见图

## (c) Stacked histogram for tokens by lecture
Tokens: `data, decision, predict, derivative, network, easy, database`  
**Figure:**
-见图

**Observation (2–3 sentences):**
- data 在几乎所有 lecture 中都大量出现（堆叠柱远高于其他 token），说明它是整门课跨主题的通用高频词。database 几乎只出现在 10-frequent-itemsets，decision 主要集中在 03-decision-trees。

## (d) n-gram language model generation
- Implementation notes (n-grams, padding <s>, </s>, ConditionalFreqDist, random.choices seed=32133, lexicographic sorting):
  - 

**Generated text for seed “introduction to data” (max 30 generated tokens):**
- n=3:
  - introduction to data science machine learning or the other ones so this is perfectly fitting so my error is minimized there so the one that looks like this now the node that is
- n=4:
  - introduction to data science so this lecture will be a group assignment while you are taking this course that you understand there is precisely one that has all kinds of other things that
- n=5:
  - introduction to data
- n=24:
  - introduction to data

**Differences commentary:**
- n=3、n=4 都能生成满 30 个新 token（总长 33），因为上下文只有 2/3 个词，数据里更容易找到匹配的 n-gram，所以模型不容易“卡死”。但也因此约束弱，生成内容会更容易跑题/跳跃。
- n=5 和 n=16 都只输出 seed，说明模型在第一步就停止了。

**Expected behavior for n > 5 & why:**
- 你的结果已经展示了趋势：n 越大越容易提前停止，甚至完全不生成新词。原因是 数据稀疏（sparsity）：n−1 长的上下文组合数量爆炸，但训练语料有限，绝大多数长上下文在 ConditionalFreqDist 里都不存在；
- 一旦遇到没见过的 context，按题目规则就必须停止生成。

## (e) Hierarchical TF-IDF timestamp retrieval (k=2, m=2)
- Preprocessing pipeline (tokenize + stopwords + stemming):
  - 
- Implementation notes (level-1 lecture docs, level-2 segment docs, cosine similarity via dot):
  - 

### Query 1: “gradient descent approach”
**Top-k lectures + scores + top-m timestamps:**
- 匹配结果见输出

**Comment (1–2 sentences):**
- Top-2 lectures 为 04-regression 和 06-neural-networks-1，总体合理：梯度下降既用于回归的损失优化，也用于神经网络训练。
- 06 的片段直接包含 “update the weights / gradient descent”，语义匹配更强；04 的片段更偏泛化的模型/误差表述。

### Query 2: “beer and diapers”
**Top-k lectures + scores + top-m timestamps:**
- 结果见输出

**Comment (1–2 sentences):**
- Top-1 lecture 是 10-frequent-itemsets，并在片段中直接出现 “buy diapers and beer”，与频繁项集/关联规则经典案例完全一致。
- 第二名 01-introduction 也合理，因为导论部分提到 pattern mining 并举了类似购买模式的例子。

---

# Q4: Time Series Analysis (23 pts)

## (a) Exploration
### a.i Load air_traffic.csv
1) Time range covered:
-2003-01-01 to 2023-09-01
2) # months tracked:
-249
3) Overall # flights:
-192100234

### a.ii Time series plot: overall flights per month + description
**Figure:**
- 见图

**Description:**
- 2005到2020逐步下降 2020剧烈下降 后逐步回升

### a.iii Plot passengers vs flights (time series) + compare trends
**Figure:**
- 见图

**Comparison:**
- 两条时间序列整体呈现高度同步：在大多数时期的上升/下降趋势以及每年重复出现的峰谷位置基本一致，说明乘客量与航班量具有明显的正相关关系和共同的季节性。

## (b) In-depth analysis
### b.i Yearly seasonal plots (flights & passengers) + description (2–3 sentences)
**Figures:**
- 见图

**Description & consistency notes:**
- Flights和 Passengers都表现出稳定的 年度季节性，两个序列的季节形状高度一致，说明客流需求的季节变化会同步反映到航班量上。
图中存在一个明显的异常年份2020.

### b.ii Correlation coefficient passengers vs flights + interpretation
- Correlation:0.5698387345242093
- Interpretation:相关系数为 0.570，表明两者存在中等强度的正相关。
- Relation to a.iii:
  - a.iii中pax和fit趋势大体相同，但同时两者并非完全线性绑定：同一时期乘客量还会受到航线等其他因素影响，因此出现“同步但幅度不同”的现象。

### b.iii Correlograms for differencing {0,1,2}, lags up to 24 + description + most significant lags
**Figures:**
- 见图

**Description:**
- 

**Most significant lags (excluding 0) across all correlograms:**
- 12和14 说明存在明显的年度季节性模式。

### b.iv STL decomposition (period=12) + stationary residual?
**Figure (trend/seasonal/resid):**
- 如图

**Residual stationary?**
- yes

## (c) Forecasting (flights_train/test)
### c.i Suitability for forecasting (2–3 sentences)
- 这两个序列都表现出明显的长期趋势和稳定的年度季节性，因此总体上适合用时间序列模型进行预测。
- 但在 2020 年有异常断点,这会降低简单模型在该区间附近的预测可靠性。

### c.ii NaiveForecaster(mean) vs ARIMA (sktime) — find better ARIMA across RMSE/MAE/MAPE
- Naive metrics:
    - RMSE:51192.09540780126
    - MAE:36663.857638888876
    - MAPE:0.049011246090680395
- Best ARIMA order (p,d,q):
  - Best ARIMA order (p,d,q): (12, 0, 9)
- ARIMA metrics:
    - RMSE:13627.400084195984
    - MAE:10423.531177787236
    - MAPE:0.013455019260433196
- Notes:
  - 所选 ARIMA(12,0,9) 在三个评价指标上均显著优于 naive(mean)

### c.iii Plot train/test + forecast + comment
**Figure:**
- 如图

**Is forecast good? Why/why not:**
- Naive(mean) 基本是水平线，无法反映测试期的波动。

ARIMA(12,0,9) 能跟随测试集的起伏，整体更贴近真实值，因此预测效果更好



# Q5: Distributed Data Processing (17 pts)

## (a) MapReduce DFG computation: map/reduce signatures + explanation
- Domains used (S, N0, C, A, T):
  - S: set of strings (text lines, identifiers)
  - N0: set of non-negative integers (line numbers, counts)
  - C ⊆ S: case identifiers (CaseID)
  - A ⊆ S: activity labels (Activity)
  - T ⊆ N0: timestamps (Timestamp)
  - Notation: X* means “a list/multiset of elements from X”.

**Map function signature(s):**
- First Map (parse each row, emit events grouped by case):
  - m1: (N0 × S) → (C × (T × A))*
  - Meaning: input is (lineNumber, lineString), output is a list of pairs (caseID, (timestamp, activity)).
- Second Map (optional identity map between the two jobs):
  - m2: ((A × A) × N0) → ((A × A) × N0)*
  - Meaning: pass through edge-count pairs unchanged.

**Reduce function signature(s):**
- First Reduce (within each case: sort by time, create directly-follows edges):
  - r1: (C × (T × A)*) → (((A × A) × N0))*
  - Meaning: input is (caseID, list of (timestamp, activity)), output is a list of ((a,b), 1) for all adjacent activities after sorting by timestamp.
- Second Reduce (global counting of edges):
  - r2: ((A × A) × N0*) → ((A × A) × N0)*
  - Meaning: input is ((a,b), list of counts), output is ((a,b), sum(counts)).

**High-level explanation:**
- Goal: build a Directly-Follows Graph (DFG) where an edge (a,b) counts how often activity a is immediately followed by b in a case trace.
- Job 1:
  - Map 1 parses each CSV line into (CaseID, (Timestamp, Activity)).
  - Reduce 1 groups all events by CaseID, sorts them by Timestamp, then emits one record ((ai, ai+1), 1) for each adjacent pair in the ordered trace.
- Job 2:
  - Map 2 can be identity (or omitted in Spark-style pipelines).
  - Reduce 2 groups by edge key (a,b) and sums the 1s to obtain the final DFG edge weights.

## (b) Spark DFG discovery: RDD transformations used + explanation + DFG figure
### Transformations
使用 `textFile → map(解析为(case,(ts,act))) → groupByKey(按case聚合) → mapValues(按时间排序得到活动序列) → flatMap(生成相邻活动对边) → reduceByKey(对边计数求和)` 得到 DFG 的每条边 `(a,b)` 及其频次。

### 结果解释（对应你的输出）
- `num edges = 75`：最终 DFG 中共有 75 条不同的 directly-follows 边（不同的 `(a,b)`）。
- `sample`：展示了频次最高的边及其权重，例如 `Create Fine → Send Fine` 出现 99811 次等。

### DFG figure（Top-80）
为保证可读性，只绘制频次最高的 **Top-80** 条边，并在边上标注频次作为权重。


**DFG figure:**
- 如图

## (c) Remove arcs with frequency < 100
### c.1 Update MapReduce solution (brief)
- 不需要新增 MapReduce 作业：保持前两轮 MapReduce 不变（第 1 轮在 case 内排序生成相邻边并输出 `((a,b),1)`；第 2 轮对同一条边求和得到 `((a,b),count)`）。
- 只需在 **第二轮 Reduce 的输出阶段**增加过滤条件：计算出 `count = sum(values)` 后，**仅当 `count >= 100` 才输出该边**；否则丢弃（不写入输出）。


### c.2 Update Spark implementation: new/changed transformations + explanation
- dfg_filtered = dfg_counts.filter(lambda kv: kv[1] >= 100)
- 在 `reduceByKey` 得到 `((a,b),count)` 之后删除所有频次 `< 100` 的边，仅保留 `count >= 100` 的 arcs 用于后续绘图。

### c.3 DFG after filtering figure
**Figure:**
- 如图

## (d) Most common resource per activity (Spark)
**Transformations used + explanation:**
- 使用 `textFile → map(解析出(activity,resource)) → map((activity,resource)→1) → reduceByKey(统计每个(activity,resource)频次) → map(重组为activity维度) → reduceByKey(选最大频次resource，平手按字典序) → sortBy → collect` 得到每个 activity 最常见的 resource。

**Result table (Activity → most common resource):**
- ([('A1', ('1153692000.0', 'Create Fine'), 1),
  ('A100', ('1154469600.0', 'Create Fine'), 1),
  ('A10000', ('1173394800.0', 'Create Fine'), 1),
  ('A10001', ('1174258800.0', 'Create Fine'), 1),
  ('A10004', ('1174345200.0', 'Create Fine'), 1),
  ('A10005', ('1174345200.0', 'Create Fine'), 1),
  ('A10007', ('1174345200.0', 'Create Fine'), 1),
  ('A10008', ('1174345200.0', 'Create Fine'), 1),
  ('A10009', ('1174345200.0', 'Create Fine'), 1),
  ('A1001', ('1154469600.0', 'Create Fine'), 1)],
150370)

---

## Notes / References
- Any assumptions, edge cases, or implementation details:
  - 
