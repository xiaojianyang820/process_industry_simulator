# 过程工业系统控制动作响应曲线稳健估计方法

## 时滞：纯时滞与容量时滞
在过程工业系统中，由于控制指令的作用对象通常是液体或者气体等流体的温度，压力，流量，浓度等指标，这些指标都具有极大的惰性，需要缓慢变化，也就是说控制指令需要一定的时间才能够体现出全部的效果。此外，在过程工业系统中，某一些控制动作是通过管道中的流体传输的，所以从控制动作发出到控制对象察觉到控制动作，会需要一段时间。以上这两种情况都是过程工业系统中的时滞，前一种时滞指的是控制动作克服控制对象惯性，发挥全部影响所需要的时间，称之为**容量时滞**；而后一种时滞指的是控制动作到达控制对象的时长，这一段时间内，控制对象完全无法感知到控制动作，所以这种时滞被称为**纯时滞**。

## 系统响应曲线
在控制系统中，引发被控对象发生变化的因素一般有两个，第一是由控制系统主动发出的动作指令，第二是外界环境变化引发的干扰。这两种因素对于被控对象来说是没有区别的，因为都可以视为一种外来冲击。结合过程工业系统中存在的纯时滞和容量时滞问题，一个外来冲击在被控对象上所带来的影响通常会划分为三个阶段。第一，**无响应阶段**，即控制动作发出，但是由于信号传导的延迟，被控对象还没有感受到相应的变化，所以对控制动作无响应；第二个阶段是**响应上升阶段**，即控制动作所带来的影响开始显现，并逐步上升；第三个阶段是**响应退出阶段**，在这一阶段中本次控制动作对被控对象的影响逐步消退，趋向于无影响。在一个典型的过程工业场景下，通常无响应阶段和响应上升阶段是否存在是未知的，但是响应退出阶段是必须存在的，因为只有具有响应退出阶段的控制过程才是一个负反馈的稳定过程。如果单一控制所带来的影响很长时间都不退出的话，系统所具有的惯性就会过大，导致系统是无法控制的。同时具有这三个阶段的系统响应曲线如下图所示。

![标准的三阶段系统响应图](http://m.qpic.cn/psc?/V12Fwrzs47WSuj/bqQfVz5yrrGYSXMvKr.cqWSk2lT7PBEGnmCK2ezKpmMdpPDYn28MwXX1C66oyuitdWzamTriAx9VOY*3AIbydDeVv77Zw98DzZJyKy6.xaI!/b&bo=VAY4BAAAAAADB0w!&rf=viewer_4)
我们将某一个动作的系统响应曲线记为$R_t(x)$。

一般来说，过程工业系统对于外来冲击的响应曲线有三种最常见的模式，第一，无延时即刻退出模式，第二，短延时对称模式，第三，长延时右偏模式。这三种模式的基本图形对比如下图所示。

![image](http://m.qpic.cn/psc?/V12Fwrzs47WSuj/bqQfVz5yrrGYSXMvKr.cqQAS7B0TS9M.DUWI0nYdHmfWvgE7UeB1.U.5s2sXMSbo9EVemK.Ncovij9qY0td1VPKLhWSEnwhDh9Hf5dxaYP4!/b&bo=VAY4BAAAAAADF1w!&rf=viewer_4)
不同的纯时滞与不同的容量时滞强度加形状可以组合出无数多种系统响应曲线，反映出系统的动态惯性特性。

## 影响权重曲线
系统响应曲线指的是当一个控制动作下发后会在被控系统上引发的响应在时间上推延的结果。反过来说，被控系统在$t$时刻所作出的响应$y_t$，就是$t$时刻及之前多个控制动作所引发的结果。比如管网中某一测温点此刻所呈现的温度`T`，就可能是过往10分钟上10个控制动作所共同决定的。那么每一个动作对于$y_t$的影响权重为多少呢？这一影响权重的具体数值取决于动作的系统响应曲线。

如下表，假设我们需要计算$y_{10}$，认为$y_{10}$是由${x_n}_{n=1}^{10}$加权组合后进行非线性变换而得出，那么相应的加权权重序列$\{w_n\}_{n=1}^{10}$是如何计算出来的呢？显然根据时滞情况进行分析可知，动作$x_{10}$的系统响应曲线目前还处于第0步，动作$x_{9}$的系统响应曲线目前还处于第1步，动作$x_{8}$的系统响应曲线目前还处于第2步，而动作$x_{1}$的系统响应曲线目前处于第9步，即如第三行所示。

| $x_1$ | $x_2$ | $x_3$ | $x_4$ | $x_5$ | $x_6$ | $x_7$ |$x_8$ | $x_9$ | $x_{10}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $w_1$ | $w_2$ | $w_3$ | $w_4$ | $w_5$ | $w_6$ | $w_7$ |$w_8$ | $w_9$ | $w_{10}$ |
| $R_9(x)$ |  $R_8(x)$ |  $R_7(x)$ |  $R_6(x)$ | $R_5(x)$ |  $R_4(x)$ |  $R_3(x)$ | $R_2(x)$ | $R_1(x)$ |  $R_0(x)$ |


显然对于固定的目标状态$y_t$而言，动作序列$\{x_n\}_{n=1}^{10}$对其的影响权重就是固定动作对于未来目标状态$\{y_n\}_{n=1}^{10}$的倒序排列。所以在对某一状态进行仿真建模时，它对于历史冲击的影响权重曲线应该基本符合系统响应曲线的倒序排列才是基本合理的。


## 模型估计方法
通过前面的分析可知，假设过程工业系统中的某个目标量$y_t$受到$M$个外来冲击的影响，这些冲击的系统响应函数可能最长涉及到$N$阶滞后，那么针对该目标量的仿真模型就可以表述为：

$$
y_N = f_2(\sum_{m=1}^M \beta_m f_1(\sum_{n=0}^N w_{m, n} x_{m, n}))=f_2(\sum_{m=1}^M \beta_m f_1(\sum_{n=1}^N R_{N-n}(x_m) x_{m, n}))
$$
这一公式中包括了三个假设，（1）不同冲击因素只在单因素时序求和之后才发生交互作用，或者简单来说，不同冲击因素只在各个因素的累积量上相互影响，而没有时序上的关联关系；（2）同一种冲击因素的系统响应曲线是固定的，与该因素的强度无关；（3）不同冲击因素的系统响应曲线可以不同。

基于这一模型结构就可以构造相应结构的神经网络，基于某一个数据集，对模型中的参数进行估计。我们抽取了3天时间内的供暖管网二次网供回水温度序列数据，作为案例数据。这一场景比较准确地反映了过程工业的一般特性。供水总管处测温点测量到的供水温度可以视为控制动作，而回水总管处测温点测量到的回水温度可以视为被控目标变量。从供水温度开始变动，到回水测温点测量到这一变化，是需要一段时间的，这一段时间就是系统的纯时滞，而首先测量到的温度变化是极为轻微的，因为只有少量的水通过水管网络中的极短路径到达了回水测温点，而更多的热水需要更长的时间逐步到达。随着平均路径长度的热水到达回水测温点，回水测温点温度的变化开始变大，逐步到达顶峰，而后随着最远路径的热水也抵达测温点，本次供水温度波动所带来的全部影响就从回水温度上退出了，之后回水温度所呈现的状态，就是由供水温度上的其他波动所引发的了，跟这一次的调整没有关联了。

### 影响权重序列自由估计
我们基于这一数据集，构建了一个神经网络，对影响权重序列不加任何约束，建立供水温度与回水温度之间仿真模型。所估计得到的影响权重序列$\{R_{N-n}\}_{n=1}^{N}$如下图所示。
![image](http://m.qpic.cn/psc?/V12Fwrzs47WSuj/bqQfVz5yrrGYSXMvKr.cqbvv5mgkDShcm6hXS*frLcyhsUDYYw0abBOJhy4H.AlVQ.jbVTjxXY8j7WVCc5j3YGhKDVMqWzLX*UH9Oy2xz3w!/b&bo=VAY4BAAAAAADB0w!&rf=viewer_4)
这一个图中的横轴相当于时间，2分钟一个单位，共80个时间单位，约160分钟，这个权重序列从右向左来看，就是时间逐步后退，那一刻供水温度对于当前回水温度的影响，比如在刻度为50的位置，就相当于60分钟之前的供水温度对当前（刻度为80的位置）的回水温度所带来的影响权重。尽管该模型的拟合效果较好，但是所估计出来的影响权重序列不可以被合理解释。比如，很难合理说明为什么60分钟之前和40分钟之前的供水温度对此刻的回水温度影响较小，而夹在它们中间的50分钟之前的回水温度影响又非常大。流体系统的信息传导一定是平滑且连续的，如果估计出来的权重序列是这一种高度随机波动的状态，那么就说明这一类仿真算法只不过是在假设空间中无数个能够与数据匹配的假设中随机选择了一个，并没有估计出反映真实物理规律的那一个假设，其可解释性与泛化能力就存疑了。所以我们需要一种能够将数据拟合与机理匹配综合应用的算法系统。

### 影响权重序列形状正则
我们在过程工业系统的系统冲击响应那一部分分析过，由于过程工业系统所具有的流体性质，其控制动作所带来的系统响应曲线往往具有三个特点，（1）头部可能存在无响应区域，尾部一定存在无响应区域，要保证控制动作在长期来看影响会消失；（2）基本上都是单峰的形态，不太可能出现多峰的形态，即该动作对系统的影响已经快消失时又突然开始猛烈影响系统；（3）响应曲线应该足够平滑，这是因为通常被控制的流体状态包括温度，压力，浓度，流量等等，这些都是累积量，也具有较大的惯性，其变化是相对平缓的，不会发生突变。结合这三个特征，我们认为，过程工业系统的系统响应曲线应该基本上满足weibull分布的形态。weibull分布是一个非常宽的分布族，它具有形状参数，尺度参数两个参数，这两个参数的不同取值，weibull分布可以模拟出从即刻衰减到缓慢释放的各种系统响应曲线。上面介绍的三种典型系统响应曲线图示，就是基于weibull分布所生成的。

我们将对权重序列的形状正则加入到上述的神经网络模型中之后，再次进行数据拟合，一共进行350个轮次的拟合训练。前150次训练不对权重序列的形状进行控制，让它自由估计，随后的200次训练中，每进行两次对数据的拟合，就进行一次对权重序列形状的调整。这一过程中，权重序列的变化图示如下：

![image](http://m.qpic.cn/psc?/V12Fwrzs47WSuj/bqQfVz5yrrGYSXMvKr.cqckuJdzlUeKEMohiBFc.utykI1BPOl5c3Y*UXighHFTuVLW4iB1YECtJbcCDVAT02sczrNdukzkIPFSJnpBPZP0!/b&bo=pAEYAQAAAAACJ78!&rf=viewer_4)

从动态演示过程中可以发现，在前期的自由估计阶段，得到了一个不可解释的权重序列，不具有平滑性，单峰型，以及尾部控制影响退出等要求的性质。而在随后进行形状正则之后，算法基于当前权重序列的大致形状猜测出来其潜在服从的平滑形状，并使用这一个平滑形状引导权重序列进行修正，并最终得到如下的权重序列图像：
![image](http://m.qpic.cn/psc?/V12Fwrzs47WSuj/bqQfVz5yrrGYSXMvKr.cqZa5BHNQOW0rui5FDmwX.iHSWG78XyqRNeeC2*30mWiNcO8eSluDA97uW*.cMjcfqeEe.JlUCUB7CFkfg2*PxDg!/b&bo=VAY4BAAAAAADB0w!&rf=viewer_4)
从这个权重序列的图形上我们就可以看出，该二次网系统的纯时滞约为18分钟，即二次网供水温度波动之后，大约18分钟之后，二次网回水温度开始发生变化。大约46分钟后，大部分流体已经完成的本次循环，而全部流体完成循环大约需要100分钟的时间。也就是说，通过形状约束，我们成功地建立了一种可解释的仿真系统。

刚才的介绍只是说明了施加了形状约束之后的流体仿真系统结合了机理特征，具有了更强的可解释性。但是这种施加了形状约束的流体仿真系统与未施加形状约束的流体仿真系统在预测效果上有没有区别呢。可以对比如下两个回归曲线：
![image](http://m.qpic.cn/psc?/V12Fwrzs47WSuj/bqQfVz5yrrGYSXMvKr.cqREZDBCpeWb07Z.gqx3DWqcezaIQyfikribQUhJrWGiXgJPgJ40F8Fnez*b0SOAoRZCj5aDLy0q.diiiyTkfWBY!/b&bo=VAY4BAAAAAADB0w!&rf=viewer_4)
上图为未进行权重序列形状约束的回归曲线，红色实线为预测曲线，黑色虚线为真实曲线。
![image](http://m.qpic.cn/psc?/V12Fwrzs47WSuj/bqQfVz5yrrGYSXMvKr.cqdiL0YD5lRBczGjOkuGLUfX8mZol.dRpV*yvVL..xqDZGhTysEJohao9I9BXF64MGtlsZL2dssCieJ1ExxkHEL8!/b&bo=VAY4BAAAAAADF1w!&rf=viewer_4)
上图为进行了权重序列形状约束的回归曲线，红色实线为预测曲线，黑色虚线为真实曲线。

从这两个图的对比可以很容易发现，自由估计权重序列下，给出的预测曲线有很多不可解释的抖动，这显然是受到了各种噪音的影响。而结合了机理分析的形状约束权重序列下，给出的预测曲线非常平滑，完全能够对抗各种噪音的影响，给出更为准确的预测结果。

## 多冲击因素混合下的模型测试
上述的供回水试验中，只涉及到单一因素的权重序列形状估计问题，比较简单。为了全面的测试本算法解决复杂问题的能力，我们构造了一个虚拟数据集。具体来说，就是基于奥恩斯坦-乌伦贝克随机过程生成三列特征数据，然后将这三列特征数据分别与上面所说的三种典型的系统响应曲线相乘，相乘之后得到三个结果进去求和，构造出目标列数据。当然，为了测试算法的鲁棒性，我们也在特征和目标上都施加了一定强度的噪音。我们进行三个不同的试验。

第一个试验，不对权重序列做任何要求，放开进行自由估计，得到的样本外预测曲线与实际曲线对比为下图：
![image](http://m.qpic.cn/psc?/V12Fwrzs47WSuj/bqQfVz5yrrGYSXMvKr.cqbML7Bk4Nkd2QAF*yhOnjZslvrlhleMteLAOyIbLnmm8hWmASNjBbY9OGIERh8u7CpI10HZ9J0aX.ALGu39pQvk!/b&bo=Ygc4BAAAAAADJ1s!&rf=viewer_4)

第二个试验，要求权重序列相对平滑，但是不对具体形状进行约束，得到的样本外预测曲线与实际曲线对比为下图：
![image](http://m.qpic.cn/psc?/V12Fwrzs47WSuj/bqQfVz5yrrGYSXMvKr.cqWXb3fV1tuUHugyGnUfURbxWSs9a3OuN9cA9y7cWdn9qKSkFEWrngNH5InLpc9JEwL9JjMalrX4RSdtsMeOTDKk!/b&bo=Ygc4BAAAAAADJ1s!&rf=viewer_4)

第三个试验，要求权重序列相对平滑，同时有必须遵循weibull分布族，得到的样本外预测曲线与实际曲线对比为下图：
![image](http://m.qpic.cn/psc?/V12Fwrzs47WSuj/bqQfVz5yrrGYSXMvKr.cqZlwFFdvBxRUt0r3mIc47Ge572gwJLkb39scTw1yn9g1K4u2ucH538dMsuNdWUuqb2GykuMC9LPY7XrrYu6.Kpo!/b&bo=Ygc4BAAAAAADJ1s!&rf=viewer_4)

从三个试验的MAE得分来看，强制平滑估计与强制形状估计都带来了大约5%~10%的准确率提升。强制平滑估计主要是使预测曲线更加的平滑，减少了估计方差，而强制形状估计则有效减少了估计偏差。

接下来我们看一下强制形状约束对于自由估计出来的权重序列所带来的影响。

![image](http://m.qpic.cn/psc?/V12Fwrzs47WSuj/bqQfVz5yrrGYSXMvKr.cqWTJETiqpiGBoSRt88vQEIZaqhWq5AA1cXV5ILDBaiW.Zzfi9QUk*yAwuWggYqSsXVC2z.mi5Zj3JlO8xoee2ss!/b&bo=pAEYAQAAAAACJ78!&rf=viewer_4)
上图为一号特征对应的权重序列拟合与强制形状估计过程，最终得到的权重序列形状和生成虚拟数据所使用的无延时即刻退出模式基本相同。

![image](http://m.qpic.cn/psc?/V12Fwrzs47WSuj/bqQfVz5yrrGYSXMvKr.cqeMLSr3WCW6b5OMEnn2UNIMpkwaQZJuOf4jaJLH9p5rQVcYM*8tmh.mfE0lqSPofrZGlPf1jKsHT9tVKlXquAac!/b&bo=pAEYAQAAAAACJ78!&rf=viewer_4)
上图为二号特征对应的权重序列拟合与强制形状估计过程，最终得到的权重序列形状和生成虚拟数据所使用的短延时对称模式基本相同。

![image](http://m.qpic.cn/psc?/V12Fwrzs47WSuj/bqQfVz5yrrGYSXMvKr.cqVb38*2qVUPXCo0bYCJcG04v0FIt8sZbvWK3jAzJashjJ.LEW8fJqrp2UcQu8oMKxY*fXp9XO6UUK1FJc3XyM90!/b&bo=pAEYAQAAAAACJ78!&rf=viewer_4)
上图为二号特征对应的权重序列拟合与强制形状估计过程，最终得到的权重序列形状和生成虚拟数据所使用的长延时右偏模式基本相同。

从动态的估计过程可以看出，尽管多个特征在数据和权重完成卷积运算之后再次进行了组合运算，本文所提出的方法依然可以准确还原出其使用的权重序列，保证所估计出来的结果优于无约束的自由估计以及对波动率进行限制的平滑估计。由于这里使用的强制形状都是weibull分布，所以仿真模型的参数量就从自由估计的80个参数缩减为三个参数（无响应长度，形状参数，尺度参数），大大压缩了模型规模，降低了训练模型所需要的数据量。

## 强制形状估计的优点

一旦我们在强制权重序列形状的条件下完成了仿真系统构建，那么这会带来极大的好处。包括如下几点：

第一，仿真系统与控制系统的可解释性。由于每一个控制动作对于目标状态的系统响应曲线都被准确估计出来，且足够平滑，有标准的形状，那么每一个控制动作会对目标变量产生多大的影响，什么时候开始产生，什么时候达到顶峰，什么时候影响消退，都可以相当精确的进行描述。由于过程工业系统普遍存在运行安全的要求，如果仿真系统或者控制系统是不可解释的，那么每一个动作所带来的风险就无法评估，这样的仿真系统与控制系统就不可能被过程工业所采用。

第二，泛化性。由于我们强迫权重序列去匹配一个特定的形状，就可以避免权重序列对随机噪音过度学习，从而有效提高了所学习到的模型在未知数据上的泛化能力。

第三，由于准确估计出了权重序列的函数方程，我们就可以将这一神经网络转写成一个标准的动力学方程，这样，经典控制理论的各种安全性检验，稳定性检验就可以基于这一动力学方程进行处理。控制系统能够通过这些经典理论的安全性检验，才有可能真正进入到过程工业控制系统业务当中。

第四，由于这一算法可以相当准确地估计出每一个组件的系统响应曲线，那么所估计出来的系统响应曲线就可以作为设备的健康评估标准。通过对比估计出来的系统响应曲线，与设计原理上的系统响应曲线，以及不同时间段估计出来的多组系统响应曲线，可以对设备的故障与老化作出量化评价，并进行可视化对比，以方便相关维保人员注意设备潜在工作曲线的变化。

## 本算法的设计要点
（1）使用权重序列差分加上平滑矩阵来取代权重序列作为最终的被估计量，确保在不进行形状强制约束的情况下，就可以得到平滑的权重序列
（2）在权重序列与形状分布进行匹配时，与线性相关系数加上分布的熵作为联合判别条件，解决只考虑线性相关系数条件下估计得到的形状分布有偏“胖”趋势的问题
（3）对可使用的形状分布进行了dropout操作，即每一轮形状分布匹配中，只有一部分形状分布可以参加，从而避免在前期有匹配错误时，形成错误螺旋，导致匹配结果无法跳出这个初期错误。

