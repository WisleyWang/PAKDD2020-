# 战队名：学不动
> this project is to predict bad disk 


## requirements 
* lightgbm  2.3.0  
* sklearn
* pandas==0.24.2
* pickle
* numpy
* tqdm
* scipy  ==>1.1.0

##data目录下
```text
tcdata|-disk_sample_smart_log_round2 线上测试数据
round1_train|-  所有数据集，包括tag
```

需要按照以下步骤依次执行.py文件

```python
#"generate process data...."
python home/semi/feature/generate.py

# "train model ...."
python home/semi/model/basic_model.py
python home/semi/model/basic_tag_model.py

#"inference ...."
python home/semi/main.py
```
 

特征说明:
```text
主要做了以下特征工程：
在做特征之前，对Nan值进行了处理，采用三次样条差值补全nan。
由于有些数据前后nan值缺失较多，无法插值补齐，故插值之后还有少量nan值，
lgb模型可以自动处理nan值，故插值未对少量nan值做后续处理。
1、**时序的差分特征**
用当前log的feature与之前log的feature做差值，时间间隔可以取 
例如 1 3 7，但由于设备有限，支撑不了这么多特征，
所以选取了部分特征做差值，同时取1 3 7 间隔差值的mean 来将三个
不同时间间隔的特征融合成一个。

2、**初始状态变化特征**
同样用当前log的feature与serial表中，disk的初始状态做差值，同样由于设备内存
限制，只选了几个做，如果全做，效果会好很多。

3、**disk使用时长**
一般来说disk 使用的时间越久，坏的机率越大
serve_time=dt-init_dt

4、**数据丢失率特征**
通过观察数据分布，发现数据缺失越多，越容易发生故障。
miss_data_rate=（disk当前为止所有log数量）/(disk使用时长)

5、**其他特征**
比如加窗的统计特征，由于是时间序列，可以构建加窗的聚合函数
由于设备原因，以及这些特征构建后发现效果不是很明显，故舍去

6、**尝试的特征提取**
对于时序数据，采用LSTM提取时序特征，初步尝试，效果不好，可能是数据
预处理不恰当，后期可以继续尝试。对于连续特征和部分可以当作离散特征的高阶
组合，可以采用DNN，FM 进一步挖掘。

对于数据的预处理还尝试用Log平滑化，以及标准化处理，但是从结果上看，
由于分布差异较大，标准化会改变feature一些特性，所以舍去这部分处理

##  相对于初赛增加了一些特征：
1、curr rate ：当天统计分数，即特征在当天log中的评分
2、model count：发现disk分布中 拥有两种model的disk 坏其中一个的概率较大
3、 gather erro: 选取了几种对于错误描述的smart进行累和。
4、 gct_change： 带窗函数的特征变化率
5、 ewm_calculate: 指数平滑 取mean 和std
6、 ewm_var_diff : var*diff 增大变异程度
7、 scale_smart : raw/normlized 一种规范化的方式
8、 diff_cumsum ： 在原差分的基础上，做了积分，仿照dpi类似的思想，累计误差
9、 data_smoother： 一种加权的线下处理
10、 tag_predict ：给每个disk 打上tag的标签，一定程度上预知坏的类别




```
模型说明:
```text
主要模型还是lgbm，对于数据的效果最佳
初赛时采用分类建模，复赛后转向回归建模，打标签的方式非常重要!!!

tag是个很重要的特征，对于不同类型的损坏应该加以区分，最大化利用tag的信息
lgb多分类模型来预先给disk打上tag标签
```
## 阈值设定
```text
由于赛方要求不能rank求阈值，所以复赛改变了之前的筛选策略，详细可查看
code/evalues.py
制定了多种筛选策略，最后选取了实时分model取top的机制

线上只是简单融合了两个不同参数的lgb模型，
model 放了两个lgb模型，和预测tag的模型，或者运行basic_tag_model.py
和lgb_regressionModel.py训练并生成模型。
线上的predict原始文件也一并保留了
main.py可直接生成predict.csv 和zip 到../prediction_result/

```




