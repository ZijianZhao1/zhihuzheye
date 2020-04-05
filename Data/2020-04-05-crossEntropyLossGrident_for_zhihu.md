## 目录


  1. 前言
  2. 交叉熵损失函数
  3. 交叉熵损失函数的求导

## 前言
说明：本文只讨论Logistic回归的交叉熵，对Softmax回归的交叉熵类似（Logistic回归和Softmax回归两者本质是一样的，后面我会专门有一篇文章说明两者关系，先在这里挖个坑）。
首先，我们二话不说，先放出交叉熵的公式：

<img src="https://www.zhihu.com/equation?tex=J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)})),
" alt="J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)})),
" class="ee_img tr_noresize" eeimg="1">

以及 <img src="https://www.zhihu.com/equation?tex=J(\theta)" alt="J(\theta)" class="ee_img tr_noresize" eeimg="1"> 对参数 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 的偏导数（用于诸如梯度下降法等优化算法的参数更新），如下：


<img src="https://www.zhihu.com/equation?tex=\frac{\partial}{\partial\theta_{j}}J(\theta) =\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
" alt="\frac{\partial}{\partial\theta_{j}}J(\theta) =\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
" class="ee_img tr_noresize" eeimg="1">

但是在大多论文或数教程中，也就是直接给出了上面两个公式，而未给出推导过程，而且这一过程并不是一两步就可以得到的，这就给初学者造成了一定的困惑，所以我特意在此详细介绍了它的推导过程，跟大家分享。因水平有限，如有错误，欢迎指正。

## 交叉熵损失函数(Logistic Regression代价函数)
我们一共有 <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1"> 组已知样本（ <img src="https://www.zhihu.com/equation?tex=Batch size = m" alt="Batch size = m" class="ee_img tr_noresize" eeimg="1"> ）， <img src="https://www.zhihu.com/equation?tex=(x^{(i)},y^{(i)})" alt="(x^{(i)},y^{(i)})" class="ee_img tr_noresize" eeimg="1"> 表示第  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  组数据及其对应的类别标记。其中 <img src="https://www.zhihu.com/equation?tex=x^{(i)}=(1,x^{(i)}_1,x^{(i)}_2,...,x^{(i)}_p)^T" alt="x^{(i)}=(1,x^{(i)}_1,x^{(i)}_2,...,x^{(i)}_p)^T" class="ee_img tr_noresize" eeimg="1"> 为 <img src="https://www.zhihu.com/equation?tex=p+1" alt="p+1" class="ee_img tr_noresize" eeimg="1"> 维向量（考虑偏置项）， <img src="https://www.zhihu.com/equation?tex=y^{(i)}" alt="y^{(i)}" class="ee_img tr_noresize" eeimg="1"> 则为表示类别的一个数：

- **logistic回归**（是非问题）中， <img src="https://www.zhihu.com/equation?tex=y^{(i)}" alt="y^{(i)}" class="ee_img tr_noresize" eeimg="1"> 取0或者1；
- **softmax回归** （多分类问题）中， <img src="https://www.zhihu.com/equation?tex=y^{(i)}" alt="y^{(i)}" class="ee_img tr_noresize" eeimg="1"> 取1,2...k中的一个表示类别标号的一个数（假设共有k类）。

这里，只讨论logistic回归，输入样本数据 <img src="https://www.zhihu.com/equation?tex=x^{(i)}=(1,x^{(i)}_1,x^{(i)}_2,...,x^{(i)}_p)^T" alt="x^{(i)}=(1,x^{(i)}_1,x^{(i)}_2,...,x^{(i)}_p)^T" class="ee_img tr_noresize" eeimg="1"> ，模型的参数为 <img src="https://www.zhihu.com/equation?tex=\theta=(\theta_0,\theta_1,\theta_2,...,\theta_p)^T" alt="\theta=(\theta_0,\theta_1,\theta_2,...,\theta_p)^T" class="ee_img tr_noresize" eeimg="1"> ,因此有


<img src="https://www.zhihu.com/equation?tex=\theta^T x^{(i)}:=\theta_0+\theta_1 x^{(i)}_1+\dots+\theta_p x^{(i)}_p.
" alt="\theta^T x^{(i)}:=\theta_0+\theta_1 x^{(i)}_1+\dots+\theta_p x^{(i)}_p.
" class="ee_img tr_noresize" eeimg="1">

假设函数（hypothesis function）定义为：


<img src="https://www.zhihu.com/equation?tex=h_\theta(x^{(i)})=\frac{1}{1+e^{-\theta^T x^{(i)}} }.
" alt="h_\theta(x^{(i)})=\frac{1}{1+e^{-\theta^T x^{(i)}} }.
" class="ee_img tr_noresize" eeimg="1">

因为Logistic回归问题就是0/1的二分类问题，可以有


<img src="https://www.zhihu.com/equation?tex=P({\hat{y}}^{(i)}=1|x^{(i)};\theta)=h_\theta(x^{(i)}) \\
P({\hat{y}}^{(i)}=0|x^{(i)};\theta)=1-h_\theta(x^{(i)})
" alt="P({\hat{y}}^{(i)}=1|x^{(i)};\theta)=h_\theta(x^{(i)}) \\
P({\hat{y}}^{(i)}=0|x^{(i)};\theta)=1-h_\theta(x^{(i)})
" class="ee_img tr_noresize" eeimg="1">

现在，我们不考虑“熵”的概念，根据下面的说明，从简单直观角度理解，就可以得到我们想要的损失函数：我们将概率取对数，其单调性不变，有


<img src="https://www.zhihu.com/equation?tex=\log P({\hat{y}}^{(i)}=1|x^{(i)};\theta)=\log h_\theta(x^{(i)})=\log\frac{1}{1+e^{-\theta^T x^{(i)}} } \\
\log P({\hat{y}}^{(i)}=0|x^{(i)};\theta)=\log (1-h_\theta(x^{(i)}))=\log\frac{e^{-\theta^T x^{(i)}}}{1+e^{-\theta^T x^{(i)}} }
" alt="\log P({\hat{y}}^{(i)}=1|x^{(i)};\theta)=\log h_\theta(x^{(i)})=\log\frac{1}{1+e^{-\theta^T x^{(i)}} } \\
\log P({\hat{y}}^{(i)}=0|x^{(i)};\theta)=\log (1-h_\theta(x^{(i)}))=\log\frac{e^{-\theta^T x^{(i)}}}{1+e^{-\theta^T x^{(i)}} }
" class="ee_img tr_noresize" eeimg="1">

那么对于第 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 组样本，假设函数表征正确的组合对数概率为：


<img src="https://www.zhihu.com/equation?tex=I\{y^{(i)}=1\}\log P({\hat{y}}^{(i)}=1|x^{(i)};\theta)+I\{y^{(i)}=0\}\log P({\hat{y}}^{(i)}=0|x^{(i)};\theta)\\
=y^{(i)}\log P({\hat{y}}^{(i)}=1|x^{(i)};\theta)+(1-y^{(i)})\log P({\hat{y}}^{(i)}=0|x^{(i)};\theta)\\
=y^{(i)}\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))
" alt="I\{y^{(i)}=1\}\log P({\hat{y}}^{(i)}=1|x^{(i)};\theta)+I\{y^{(i)}=0\}\log P({\hat{y}}^{(i)}=0|x^{(i)};\theta)\\
=y^{(i)}\log P({\hat{y}}^{(i)}=1|x^{(i)};\theta)+(1-y^{(i)})\log P({\hat{y}}^{(i)}=0|x^{(i)};\theta)\\
=y^{(i)}\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))
" class="ee_img tr_noresize" eeimg="1">

其中， <img src="https://www.zhihu.com/equation?tex=I\{y^{(i)}=1\}" alt="I\{y^{(i)}=1\}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=I\{y^{(i)}=0\}" alt="I\{y^{(i)}=0\}" class="ee_img tr_noresize" eeimg="1"> 为示性函数（indicative function），简单理解为{ }内条件成立时，取1，否则取0，这里不赘言。
那么对于一共 <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1"> 组样本，我们就可以得到模型对于整体训练样本的表现能力：


<img src="https://www.zhihu.com/equation?tex=\sum_{i=1}^{m}y^{(i)}\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))
" alt="\sum_{i=1}^{m}y^{(i)}\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))
" class="ee_img tr_noresize" eeimg="1">

由以上表征正确的概率含义可知，我们希望其值越大，模型对数据的表达能力越好。而我们在参数更新或衡量模型优劣时是需要一个能充分反映模型表现误差的损失函数（Loss function）或者代价函数（Cost function）的，而且我们希望损失函数越小越好。由这两个矛盾，那么我们不妨领代价函数为上述组合对数概率的相反数：


<img src="https://www.zhihu.com/equation?tex=J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))
" alt="J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))
" class="ee_img tr_noresize" eeimg="1">

上式即为大名鼎鼎的交叉熵损失函数。(说明：如果熟悉“[信息熵](http://baike.baidu.com/link?url=1EWQyRQiLUpu50as-PrfzIv-7e_ZP9jk4stpTbK_AKAfz05mKQaH9EQWz_trCW8pJcLXqTklUXLBvHKj2Q0J1K)"的概念 <img src="https://www.zhihu.com/equation?tex=E[-\log p_i]=-\sum_{i=1}^mp_i\log p_i" alt="E[-\log p_i]=-\sum_{i=1}^mp_i\log p_i" class="ee_img tr_noresize" eeimg="1"> ，那么可以有助理解叉熵损失函数，先挖个坑，后面我会专门写一篇讲信息熵的白话文）

## 交叉熵损失函数的求导
这步需要用到一些简单的对数运算公式，这里先以编号形式给出，下面推导过程中使用特意说明时都会在该步骤下脚标标出相应的公式编号，以保证推导的连贯性。

①  <img src="https://www.zhihu.com/equation?tex=\log \frac{a}{b}=\log a-\log b" alt="\log \frac{a}{b}=\log a-\log b" class="ee_img tr_noresize" eeimg="1"> 

②  <img src="https://www.zhihu.com/equation?tex=\log a+\log b=\log (ab)" alt="\log a+\log b=\log (ab)" class="ee_img tr_noresize" eeimg="1"> 

③  <img src="https://www.zhihu.com/equation?tex=a=\log e^a" alt="a=\log e^a" class="ee_img tr_noresize" eeimg="1">    (为了方便这里 <img src="https://www.zhihu.com/equation?tex=\log" alt="\log" class="ee_img tr_noresize" eeimg="1"> 指 <img src="https://www.zhihu.com/equation?tex=\log_e" alt="\log_e" class="ee_img tr_noresize" eeimg="1"> ，即 <img src="https://www.zhihu.com/equation?tex=\ln" alt="\ln" class="ee_img tr_noresize" eeimg="1"> ，其他底数如2,10等，只是前置常数系数不同，对结论毫无影响)

另外，值得一提的是在这里涉及的求导均为矩阵、向量的导数（矩阵微商），这里有一篇[教程](http://download.csdn.net/detail/jasonzzj/9585291)总结得精简又全面，非常棒，推荐给需要的同学。

下面开始推导：

交叉熵损失函数为：


<img src="https://www.zhihu.com/equation?tex=J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))\tag{1}
" alt="J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))\tag{1}
" class="ee_img tr_noresize" eeimg="1">

其中，


<img src="https://www.zhihu.com/equation?tex=\log h_\theta(x^{(i)})=\log\frac{1}{1+e^{-\theta^T x^{(i)}} }=-\log ( 1+e^{-\theta^T x^{(i)}} )\ ,\\ \log(1- h_\theta(x^{(i)}))=\log(1-\frac{1}{1+e^{-\theta^T x^{(i)}} })=\log(\frac{e^{-\theta^T x^{(i)}}}{1+e^{-\theta^T x^{(i)}} })\\=\log (e^{-\theta^T x^{(i)}} )-\log ( 1+e^{-\theta^T x^{(i)}} )=-\theta^T x^{(i)}-\log ( 1+e^{-\theta^T x^{(i)}} ) _{①③}\ .
" alt="\log h_\theta(x^{(i)})=\log\frac{1}{1+e^{-\theta^T x^{(i)}} }=-\log ( 1+e^{-\theta^T x^{(i)}} )\ ,\\ \log(1- h_\theta(x^{(i)}))=\log(1-\frac{1}{1+e^{-\theta^T x^{(i)}} })=\log(\frac{e^{-\theta^T x^{(i)}}}{1+e^{-\theta^T x^{(i)}} })\\=\log (e^{-\theta^T x^{(i)}} )-\log ( 1+e^{-\theta^T x^{(i)}} )=-\theta^T x^{(i)}-\log ( 1+e^{-\theta^T x^{(i)}} ) _{①③}\ .
" class="ee_img tr_noresize" eeimg="1">

由此，得到


<img src="https://www.zhihu.com/equation?tex=J(\theta) =-\frac{1}{m}\sum_{i=1}^m \left[-y^{(i)}(\log ( 1+e^{-\theta^T x^{(i)}})) + (1-y^{(i)})(-\theta^T x^{(i)}-\log ( 1+e^{-\theta^T x^{(i)}} ))\right]\\
=-\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\theta^T x^{(i)}-\theta^T x^{(i)}-\log(1+e^{-\theta^T x^{(i)}})\right]\\
=-\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\theta^T x^{(i)}-\log e^{\theta^T x^{(i)}}-\log(1+e^{-\theta^T x^{(i)}})\right]_{③}\\
=-\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\theta^T x^{(i)}-\left(\log e^{\theta^T x^{(i)}}+\log(1+e^{-\theta^T x^{(i)}})\right)\right] _②\\
=-\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\theta^T x^{(i)}-\log(1+e^{\theta^T x^{(i)}})\right]
" alt="J(\theta) =-\frac{1}{m}\sum_{i=1}^m \left[-y^{(i)}(\log ( 1+e^{-\theta^T x^{(i)}})) + (1-y^{(i)})(-\theta^T x^{(i)}-\log ( 1+e^{-\theta^T x^{(i)}} ))\right]\\
=-\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\theta^T x^{(i)}-\theta^T x^{(i)}-\log(1+e^{-\theta^T x^{(i)}})\right]\\
=-\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\theta^T x^{(i)}-\log e^{\theta^T x^{(i)}}-\log(1+e^{-\theta^T x^{(i)}})\right]_{③}\\
=-\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\theta^T x^{(i)}-\left(\log e^{\theta^T x^{(i)}}+\log(1+e^{-\theta^T x^{(i)}})\right)\right] _②\\
=-\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\theta^T x^{(i)}-\log(1+e^{\theta^T x^{(i)}})\right]
" class="ee_img tr_noresize" eeimg="1">

这次再计算 <img src="https://www.zhihu.com/equation?tex=J(\theta)" alt="J(\theta)" class="ee_img tr_noresize" eeimg="1"> 对第 <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1"> 个参数分量 <img src="https://www.zhihu.com/equation?tex=\theta_j" alt="\theta_j" class="ee_img tr_noresize" eeimg="1"> 求偏导:


<img src="https://www.zhihu.com/equation?tex=\frac{\partial}{\partial\theta_{j}}J(\theta) =\frac{\partial}{\partial\theta_{j}}\left(\frac{1}{m}\sum_{i=1}^m \left[\log(1+e^{\theta^T x^{(i)}})-y^{(i)}\theta^T x^{(i)}\right]\right)\\
=\frac{1}{m}\sum_{i=1}^m \left[\frac{\partial}{\partial\theta_{j}}\log(1+e^{\theta^T x^{(i)}})-\frac{\partial}{\partial\theta_{j}}\left(y^{(i)}\theta^T x^{(i)}\right)\right]\\
=\frac{1}{m}\sum_{i=1}^m \left(\frac{x^{(i)}_je^{\theta^T x^{(i)}}}{1+e^{\theta^T x^{(i)}}}-y^{(i)}x^{(i)}_j\right)\\
=\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
" alt="\frac{\partial}{\partial\theta_{j}}J(\theta) =\frac{\partial}{\partial\theta_{j}}\left(\frac{1}{m}\sum_{i=1}^m \left[\log(1+e^{\theta^T x^{(i)}})-y^{(i)}\theta^T x^{(i)}\right]\right)\\
=\frac{1}{m}\sum_{i=1}^m \left[\frac{\partial}{\partial\theta_{j}}\log(1+e^{\theta^T x^{(i)}})-\frac{\partial}{\partial\theta_{j}}\left(y^{(i)}\theta^T x^{(i)}\right)\right]\\
=\frac{1}{m}\sum_{i=1}^m \left(\frac{x^{(i)}_je^{\theta^T x^{(i)}}}{1+e^{\theta^T x^{(i)}}}-y^{(i)}x^{(i)}_j\right)\\
=\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
" class="ee_img tr_noresize" eeimg="1">

这就是交叉熵对参数的导数：


<img src="https://www.zhihu.com/equation?tex=\frac{\partial}{\partial\theta_{j}}J(\theta) =\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
" alt="\frac{\partial}{\partial\theta_{j}}J(\theta) =\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
" class="ee_img tr_noresize" eeimg="1">

#### 向量形式

前面都是元素表示的形式，只是写法不同，过程基本都是一样的，不过写成向量形式会更清晰，这样就会把 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  和求和符号 <img src="https://www.zhihu.com/equation?tex=\sum" alt="\sum" class="ee_img tr_noresize" eeimg="1"> 省略掉了。我们不妨忽略前面的固定系数项 <img src="https://www.zhihu.com/equation?tex=1/m" alt="1/m" class="ee_img tr_noresize" eeimg="1"> ，交叉墒的损失函数(1)则可以写成下式：

<img src="https://www.zhihu.com/equation?tex=J(\theta) = -\left[ y^T \log h_\theta(x)+(1-y^T)\log(1-h_\theta(x))\right]\tag{2}
" alt="J(\theta) = -\left[ y^T \log h_\theta(x)+(1-y^T)\log(1-h_\theta(x))\right]\tag{2}
" class="ee_img tr_noresize" eeimg="1">
将 <img src="https://www.zhihu.com/equation?tex=h_\theta(x)=\frac{1}{1+e^{-\theta^T x} }" alt="h_\theta(x)=\frac{1}{1+e^{-\theta^T x} }" class="ee_img tr_noresize" eeimg="1"> 带入，得到：

<img src="https://www.zhihu.com/equation?tex=J(\theta) = -\left[ y^T \log \frac{1}{1+e^{-\theta^T x} }+(1-y^T)\log\frac{e^{-\theta^T x}}{1+e^{-\theta^T x} }\right] \\
= -\left[ -y^T \log (1+e^{-\theta^T x}) + (1-y^T) \log e^{-\theta^T x} - (1-y^T)\log (1+e^{-\theta^T x})\right] \\
= -\left[(1-y^T) \log e^{-\theta^T x} - \log (1+e^{-\theta^T x}) \right]\\
= -\left[(1-y^T ) (-\theta^Tx) - \log (1+e^{-\theta^T x}) \right]
" alt="J(\theta) = -\left[ y^T \log \frac{1}{1+e^{-\theta^T x} }+(1-y^T)\log\frac{e^{-\theta^T x}}{1+e^{-\theta^T x} }\right] \\
= -\left[ -y^T \log (1+e^{-\theta^T x}) + (1-y^T) \log e^{-\theta^T x} - (1-y^T)\log (1+e^{-\theta^T x})\right] \\
= -\left[(1-y^T) \log e^{-\theta^T x} - \log (1+e^{-\theta^T x}) \right]\\
= -\left[(1-y^T ) (-\theta^Tx) - \log (1+e^{-\theta^T x}) \right]
" class="ee_img tr_noresize" eeimg="1">
再对 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 求导，前面的负号直接削掉了，

<img src="https://www.zhihu.com/equation?tex=\frac{\partial}{\partial\theta_{j}}J(\theta) = -\frac{\partial}{\partial\theta_{j}}\left[(1-y^T ) (-\theta^Tx) - \log (1+e^{-\theta^T x}) \right] \\
= (1-y^T)x- \frac{e^{-\theta^T }}{1+e^{-\theta^T x} }x \\
= (\frac{1}{1+e^{-\theta^T x} } - y^T)x \\
= \left(h_\theta(x)-y^T \right)x
" alt="\frac{\partial}{\partial\theta_{j}}J(\theta) = -\frac{\partial}{\partial\theta_{j}}\left[(1-y^T ) (-\theta^Tx) - \log (1+e^{-\theta^T x}) \right] \\
= (1-y^T)x- \frac{e^{-\theta^T }}{1+e^{-\theta^T x} }x \\
= (\frac{1}{1+e^{-\theta^T x} } - y^T)x \\
= \left(h_\theta(x)-y^T \right)x
" class="ee_img tr_noresize" eeimg="1">


转载请注明：[赵子健的博客](zijian-zhao.com) » [机器学习系列](https://zijian-zhao.com/tags/#机器学习-ref) » [交叉熵损失函数的求导](zijian-zhao.com/2020/04/crossEntropyLossGrident/) [zijian-zhao.com/2020/04/crossEntropyLossGrident/]

