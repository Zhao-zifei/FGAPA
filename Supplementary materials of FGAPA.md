# Supplementary materials of FGAPA

我们衷心感谢两位审稿人对稿件的细致审阅及其富有建设性的意见。我们非常感激他们对本工作创新性的积极评价及其对所提出的跨域高光谱图像分类方法在整体研究框架和方法动机方面潜力与实际价值的认可。以下是我们对审稿人对FGAPA担忧问题的回复。

## **Response to Reviewer 1**

1.The experiments provide good initial evidence for the method, but need to go further in order to fully establish the value of the approach. The training and transfer hinges on the selection of the source domain, so only one experiment here is very limiting. The source domain needs some properties of potential alignment with the target domains, and must somehow be sufficiently rich to cover the possibilities. So it isn't clear how robust the method is in general, or how to pick the source domain to begin with, or how the approach can be updated as new domains are sampled.

**对于源域的选取，为了方便下游任务的进行，源域的类别数通常要大于或等于目标域类别数，这样训练得到的度量空间才可为目标域任务服务。Chikusei 数据集包含了多样化的景观类别（19类），可为跨域小样本学习提供广泛的知识支持。另外，Chikusei 数据集与目标域存在显著的域间差异，使其成为在具有挑战性的条件下验证我们方法有效性的理想选择。因此，FGAPA使用Chikusei数据集作为源域数据集。**

**为证明我们的方法对于不同的源域都具有很好的鲁棒性和健壮性，我们额外选取了Hanchuan作为源域数据集进行实验。Hanchuan与Chikusei一样拥有丰富的景观类别（16类），并与目标域拥有显著差异。Table 1 展示了以 Hanchuan 作为源域、仅使用五个标记样本时，不同方法在 Indian Pines、Salinas 和 Botswana 数据集（作为目标域）上的性能对比。由于时间问题，我们选取了同为域适应方法的DCFSL、MLPA和利用对比学习捕捉域间差异的CTF-SSCL作为对比方法予以论证。**

<center><p>Table 1:Classification Results (Mean ± Std) with Hanchuan dataset as Source Domain</p></center>

<small>
<table>
    <tr>
        <th rowspan="2">Method</th>
        <th colspan="3" class="dataset-header"><center>Indian Pines</center></th>
        <th colspan="3" class="dataset-header"><center>Salinas</center></th>
        <th colspan="3" class="dataset-header"><center>Botswana</center></th>
    </tr>
    <tr>
        <th><center>OA</center></th>
        <th><center>AA</center></th>
        <th><center>Kappa</center></th>
        <th><center>OA</center></th>
        <th><center>AA</center></th>
        <th><center>Kappa</center></th>
        <th><center>OA</center></th>
        <th><center>AA</center></th>
        <th><center>Kappa</center></th>
    </tr>
    <tr>
        <td class="method-header">DCFSL</td>
        <td>69.53±2.85</td>
        <td>80.72±1.56</td>
        <td>65.77±3.19</td>
        <td>91.06±0.99</td>
        <td>94.95±0.62</td>
        <td>90.07±1.10</td>
        <td>97.15±0.83</td>
        <td>97.06±0.96</td>
        <td>96.91±0.90</td>
    </tr>
    <tr>
        <td class="method-header">CTF-SSCL</td>
        <td>70.59±4.22</td>
        <td>81.69±1.75</td>
        <td>66.96±4.52</td>
        <td>89.79±1.70</td>
        <td>93.82±1.01</td>
        <td>88.66±1.89</td>
        <td>95.34±1.397</td>
        <td>95.60±1.36</td>
        <td>94.95±1.51</td>
    </tr>
    <tr>
        <td class="method-header">MLPA</td>
        <td>68.13±3.21</td>
        <td>79.61±1.84</td>
        <td>64.24±3.51</td>
        <td>90.99±1.07</td>
        <td>94.61±0.85</td>
        <td>89.98±1.18</td>
        <td>96.38±1.13</td>
        <td>96.31±1.18</td>
        <td>96.08±1.23</td>
    </tr>
    <tr>
        <td class="method-header">FGAPA</td>
        <td><strong>76.37±2.56</strong></td>
        <td><strong>85.76±1.18</strong></td>
        <td><strong>73.34±2.78</strong></td>
        <td><strong>92.08±0.87</strong></td>
        <td><strong>95.22±1.03</strong></td>
        <td><strong>91.19±0.97</strong></td>
        <td><strong>97.65±0.97</strong></td>
        <td><strong>97.58±1.09</strong></td>
        <td><strong>97.45±1.05</strong></td>
    </tr>
</table>
</small>

**从Table 1中可以看到，FGAPA在Hanchuan作为源域的情况下，在三个数据集取得的OA、AA、Kappa远优于所有对比方法。以OA为例，与同为域适应方法的DCFSL和MLPA相比，FGAPA的OA在Indian_pines数据集上分别以6.84%，8.5%的优势领先，在Salinas和Botswana上显示出同样趋势。与以泛化性著称的对比学习方法CTF-SSCL相比，我们的方法集成的ACA模块通过对抗的域对齐策略展现出强大性能，同时从方差也可看出我们的方法具有更好的稳定性。综上所述，FGAPA在不同源域下（Chikusei、Hanchuan）均具有很好的稳健性与鲁棒性。**



## **Response to Reviewer 2**

1.Some definitions are missing in the paragraph about FSL in HSIC. What is FSL? What are source and target domains?

**FSL 本质上是一种元学习方法，可以从不同的任务中获取可转移的知识，以快速适应新任务。 通常，它在 N-way K-shot 设置下运行，每类使用 K 个标记样本来训练 N 类分类器 。如图1所示，FSL任务由带标签的样本组成的集合Support set与查询样本组成的集合Query set组成。查询集与支持集构建FSL任务并一同输入特征提取器中进行特征提取。Support features代表支持集经由特征提取器提取得到的特征表示，Query features则是查询集经提取得到的特征表示。FSL通过计算由查询特征到每个类的支持特征之间的距离来判定其类别。通过这样的FSL任务，让模型学会学习，通过给定的支持集判定查询集样本的类别。在HSIC中，也是同样道理。通过选取少量标记的高光谱样本作为支持集，大量无标记的样本作为查询集来进行FSL任务。**

<img src="https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251220151727215.png" alt="替代文本" title="图片标题">

<center><p>Fig1：Few Shot Learning</p></center>

**在跨域少样本学习中，源域是一个数据丰富、标注完善的数据集，模型首先在此进行预训练，学习通用的特征和模式（即让模型学会学习）。而目标域则是我们真正用于下游任务但标记样本极度稀缺的新数据集，其数据分布和类别通常与源域不同。整个过程的核心目标，就是利用从源域获得的先验知识，仅凭目标域中极少的样本（如每类1-5个标记样本），快速适应并解决目标域的新任务。**

2.Figure 1 is not clear. Acronyms are not defined and do not correspond to the text. What are support features? Why don't we see $\mathcal{L}_{fsl}^t$ and $\mathcal{L}_{fsl}^s$? What is the attention score (never mentioned in the text)? Does the prototype bank correspond to $\mathbf p_i$?

![image-20251216203011245](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216203011245.png)

<center><p>Fig2: Overall Framework of FGAPA</p></center>

**Support features代表支持集经由特征提取器提取得到的特征表示，Query features则是查询集经提取得到的特征表示。如图2所示，通过度量Query features与Support features中心形成的原型之间的欧式距离来判定Query features所属类别。**xxxxxx

**FSL任务同时在源域和目标域上进行，其$\mathcal{L}_{fsl}$由$\mathcal{L}_{fsl}^t$和$\mathcal{L}_{fsl}^s$两部分组成，因此我们对其进行省略仅使用$\mathcal{L}_{fsl}$代表。原图2中注意力分数实际对应文中Correlation calculation。同时，$p_i$代表原型库的第i类原型。我们对图2进行更新，以便读者了解FGAPA的整体工作流程。**

3.Equation (5) and (6): what is the dimensionality of the weights and biases in the MLP?

**在Equation (5) and (6)中，MLP层的权重与偏置的维度具体设置如下：在FGAPA中经由特征提取得到的特征最终都被统一为128*1的特征维度，MLP第一层权重分别对应（128，64）其中128代表输入维度，64为输出维度，偏置维度为64。MLP第二层权重则对应（64，128），其中64代表输入维度，128为输出维度，偏置维度为128。**

4.Why do you need to transform the features and prototypes with MLP?

**首先，特征与原型来自于不同的语义层次。通过MLP层将其共同映射到同一个可比较的特征空间。其次MLP层中间的Relu激活函数可以确保模型学习更复杂的非线性关系以提高其表达能力。因此，通过将特征与原型通过MLP层进行转换，能更好的在特征空间中进行相关性计算，并让模型学习更加复杂的表示以提升表达能力。**

5.Why is it a cosine similarity (normalized dot product) in equation (7) and not in equation (10) ?

**事实上公式（7）和公式（10）都使用了余弦相似度。具体来讲，公式（10）与公式（7）的核心思想相同。它们都使用余弦相似度进行相关性计算。使用余弦相似度的原因是其与原型学习天然契合，更适合高维嵌入空间，且优化稳定。同时，余弦相似度配合温度系数可以灵活控制softmax平滑程度，提升训练稳定性。我们会对公式（10）进行对应修改，以便读者能够更清晰地了解方法细节，避免不必要的误解。**

6.What is the total optimization objective of the model in the end? It never appears.

**我们的最终优化目标由少样本学习损失$ L_{\text{fsl}}$,域内对齐损失$L_{\text{in}}$以及跨域对齐损失$L_{\text{cross}}$三部分组成。The final loss is defined as:**
$$
Loss = L_{\text{fsl}} + \lambda_1 L_{\text{in}} + \lambda_2 L_{\text{cross}}
$$
**Here,$\lambda_1$ and $\lambda_2$ represent the weighting hyperparameters for  $L_{\text{in}}$ and $L_{\text{cross}}$. Their values range from 0 to 1, with increments of 0.1. The optimal parameters obtained through tuning are $\lambda_1$ = 0.5 and $\lambda_2$ = 0.3.**

7.You talk about 4 datasets. Where do they come from? Is it public data? Is it published? How do you remove noisy bands?

**本文使用四个数据集进行实验验证，其中Chikusei作为源域，另外三个公开数据集Indian_pines、Salinas和Botswana为目标域数据集。以下为它们的详细信息、数据来源以及噪声去除情况介绍：**

- **Chikusei**

**Chikusei数据集由日本筑西市的 Headwall Hyperspec-VNIR-C 传感器捕获。可在https://naotoyokoya.com/Download.html获取， 它由 2517 × 2335 像素组成，空间分辨率为 2.5 m。 总共提供128个频段，覆盖波长从343到1018 nm，共19个类别。图3展示了Chikusei数据集上的原始图像、Ground Truth以及其19个地物类别标签与每个类别的像素数目。**



<img src="https://gitee.com/abcd123123410513/images/raw/master/imgs/CK_compressed.png" alt="替代文本" title="图片标题" width=500>

<center><p>Figure 3: The land cover types and the number of samples for the Chikusei dataset</p></center>



- **Indian_pines**

**Indian_pines数据集由美国印第安纳州上空的机载可见光/红外成像光谱仪 (AVIRIS) 收集。可在https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes获取， 它包含 145 × 145 像素，空间分辨率约为 20 m。 去除 20 个水吸收带（104-105、150-163 和 220）后，使用 400 至 2500 nm 范围内的 200 个吸收带。 图4展示了Indian_pines数据集上的原始图像、Ground Truth以及其16个地物类别标签与每个类别的像素数目。**



<img src="https://gitee.com/abcd123123410513/images/raw/master/imgs/IP_compressed.png" alt="替代文本" title="图片标题" width=500>

<center><p>Fig4: The land cover types and the number of samples for the Indian_pines dataset</p></center>



- **Salinas**

**Salinas数据集由美国加利福尼亚州萨利纳斯山谷的 AVIRIS 传感器收集。可在https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes获取, 它由 512 × 217 像素组成，空间分辨率约为 3.7 m。 总共使用了覆盖 400-2500 nm 的 204 个波段，分为 16 个类别。图5展示了Salinas数据集上的原始图像、Ground Truth以及其16个地物类别标签与每个类别的像素数目。**



<img src="https://gitee.com/abcd123123410513/images/raw/master/imgs/SA_compressed.png" alt="替代文本" title="图片标题" width=500>

<center><p>Fig5: The land cover types and the number of samples for the Salinas dataset</p></center>



- **Botswana**

**Botswana数据集由 NASA 的 EO1 卫星在 BO 奥卡万戈三角洲上空获取。可在https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes获取。它的尺寸为1476×256像素，空间分辨率约为20m。 在242个光谱波段（400-2500 nm）中，去除噪声波段后使用145个波段（1-9、56-81、98-101、120-133和165-186）。图6展示了Botswana数据集上的原始图像、Ground Truth以及其14个地物类别标签与每个类别的像素数目**。



<img src="https://gitee.com/abcd123123410513/images/raw/master/imgs/BO_compressed.png" alt="替代文本" title="图片标题" width=500>

<center><p>Fig6: The land cover types and the number of samples for the Botswana dataset</p></center>



8.In the experimental protocol, you have to detail the metrics more in detail. What do they measure? How are they calculated? Why are they relevant for the task?

**我们在文中使用了常用于分类评价三个衡量指标，Overall Accuracy(OA)、AA、Kappa。其中，OA用来反映整体分类正确率，AA关注各类别精度的平均表现以避免被样本数占比多的类别主导，Kappa则校正了随机一致性的影响，更可靠地评估分类结果与真实标签的一致程度。它们的定义如下：**

- **Overall Accuracy(OA): **

$$
OA = \frac{\sum_{i=1}^{c} TP_i}{N}
$$
​        其中，$TP_i$代表第i类被正确分类的数目，N为所有样本总数，c代表类别总数。

- **Average Accuracy(AA):** 

$$
AA = \frac{1}{c} \sum_{i=1}^{c} \frac{TP_i}{N_i}
$$

​         其中，$TP_i$代表第i类被正确分类的数目，$N_i$表示第i类样本的总数，c代表类别总数。

-   **Kappa Coefficient(Kappa): **

$$
\kappa = \frac{P_o - P_e}{1 - P_e}
$$
$P_o$ 表示模型与随机分类器做出相同类别判断的概率。
$P_e$ 表示模型与随机分类器基于各自边际分布计算出的理论一致概率。

9.As stated in the global comment, the results are missing statistical significance tests to make sure your method is statistically better on these datasets.

**受限于严格的篇幅要求，论文中部分重要内容未能得到充分展开。为此，我们对FGAPA在Chikusei作源域情况下在Indian Pines、Salinas、Botswana三个目标域上进行相关性能的展示并额外附带标准差，以便论证FGAPA在这些数据集上的性能具有统计意义。其中，表2-4分别表示FGAPA在1-5shot情况下在Indian Pines、Salinas、Botswana上的性能统计。所有实验都运行10次，取平均值。**

<center><p>Table 2: Classification Results (Mean ± Standard Deviation) on Indian_Pines Using 1–5 Labeled Samples per Class</p></center>

<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px;">   <thead>     <tr style="background-color: #f2f2f2;">       <th rowspan="2">Indian Pines</th>       <th colspan="8"><center>Method</center></th>     </tr>     <tr style="background-color: #e0e0e0;">       <th>DFSL+NN</th>       <th>DCFSL</th>       <th>HFSL</th>       <th>DM-MRN</th>       <th>FSCF-SSL</th>       <th>CTF-SSCL</th>       <th>MLPA</th>       <th><strong style="color: black;">FGAPA</strong></th>     </tr>   </thead>   <tbody>     <!-- 1-shot -->     <tr>       <th>1-shot</th>       <td>40.93 ± 4.82</td>       <td>41.32 ± 5.75</td>       <td>42.95 ± 6.74</td>       <td>41.65 ± 5.36</td>       <td>46.44 ± 5.10</td>       <td>43.52 ± 4.84</td>       <td>41.83 ± 5.51</td>       <td><strong style="color: black;">52.16 ± 4.48</strong></td>     </tr>      <!-- 2-shot -->     <tr>       <th>2-shot</th>       <td>50.98 ± 3.09</td>       <td>51.60 ± 4.53</td>       <td>53.78 ± 5.59</td>       <td>55.41 ± 3.88</td>       <td>60.14 ± 4.84</td>       <td>56.82 ± 4.92</td>       <td>53.47 ± 4.82</td>       <td><strong style="color: black;">64.81 ± 4.32</strong></td>     </tr>      <!-- 3-shot -->     <tr>       <th>3-shot</th>       <td>56.47 ± 2.44</td>       <td>56.03 ± 3.45</td>       <td>63.65 ± 4.13</td>       <td>60.85 ± 3.28</td>       <td>67.72 ± 1.62</td>       <td>63.61 ± 3.5</td>       <td>57.65 ± 2.63</td>       <td><strong style="color: black;">71.19 ± 4.18</strong></td>     </tr>      <!-- 4-shot -->     <tr>       <th>4-shot</th>       <td>60.44 ± 2.82</td>       <td>61.96 ± 3.98</td>       <td>69.96 ± 3.93</td>       <td>63.95 ± 3.07</td>       <td>72.92 ± 5.56</td>       <td>67.29 ± 3.04</td>       <td>63.31 ± 2.05</td>       <td><strong style="color: black;">76.12 ± 2.76</strong></td>     </tr>      <!-- 5-shot -->     <tr>       <th>5-shot</th>       <td>63.16 ± 2.92</td>       <td>65.74 ± 2.57</td>       <td>74.03 ± 2.71</td>       <td>69.28 ± 3.30</td>       <td>76.96 ± 2.74</td>       <td>70.86 ± 3.22</td>       <td>66.59 ± 2.81</td>       <td><strong style="color: black;">79.65 ± 2.76</strong></td>     </tr>   </tbody> </table>

<center><p>Table 3: Classification Results (Mean ± Standard Deviation) on Salinas Using 1–5 Labeled Samples per Class</p></center>

<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px;">   <thead>     <tr style="background-color: #f2f2f2;">       <th rowspan="2">Salinas</th>       <th colspan="8"><center>Method</center></th>     </tr>     <tr style="background-color: #e0e0e0;">       <th>DFSL+NN</th>       <th>DCFSL</th>       <th>HFSL</th>       <th>DM-MRN</th>       <th>FSCF-SSL</th>       <th>CTF-SSCL</th>       <th>MLPA</th>       <th><strong style="color: black;">FGAPA</strong></th>     </tr>   </thead>   <tbody>     <!-- 1-shot -->     <tr>       <th>1-shot</th>       <td>75.65 ± 2.46</td>       <td>74.81 ± 4.12</td>       <td>68.41 ± 4.76</td>       <td>76.77 ± 5.72</td>       <td>68.30 ± 4.83</td>       <td>74.19 ± 3.14</td>       <td>73.37 ± 3.08</td>       <td><strong style="color: black;">79.47 ± 3.30</strong></td>     </tr>      <!-- 2-shot -->     <tr>       <th>2-shot</th>       <td>82.16 ± 2.64</td>       <td>82.04 ± 3.26</td>       <td>80.18 ± 2.76</td>       <td>84.89 ± 3.23</td>       <td>81.40 ± 2.99</td>       <td>81.79 ± 3.26</td>       <td>81.63 ± 2.86</td>       <td><strong style="color: black;">87.55 ± 3.13</strong></td>     </tr>      <!-- 3-shot -->     <tr>       <th>3-shot</th>       <td>84.99 ± 1.53</td>       <td>84.88 ± 1.84</td>       <td>85.28 ± 2.65</td>       <td>87.48 ± 3.23</td>       <td>84.33 ± 3.52</td>       <td>85.57 ± 2.17</td>       <td>85.37 ± 2.01</td>       <td><strong style="color: black;">90.70 ± 1.86</strong></td>     </tr>      <!-- 4-shot -->     <tr>       <th>4-shot</th>       <td>88.61 ± 1.53</td>       <td>88.05 ± 1.90</td>       <td>86.98 ± 2.57</td>       <td>90.04 ± 2.03</td>       <td>88.11 ± 3.66</td>       <td>88.58 ± 1.59</td>       <td>89.04 ± 1.50</td>       <td><strong style="color: black;">91.83 ± 1.09</strong></td>     </tr>      <!-- 5-shot -->     <tr>       <th>5-shot</th>       <td>89.74 ± 1.26</td>       <td>90.12 ± 1.22</td>       <td>89.14 ± 2.09</td>       <td>91.08 ± 2.39</td>       <td>90.96 ± 2.06</td>       <td>89.96 ± 1.43</td>       <td>89.94 ± 1.30</td>       <td><strong style="color: black;">93.15 ± 1.00</strong></td>     </tr>   </tbody> </table>

<center><p>Table 4: Classification Results (Mean ± Standard Deviation) on Salinas Using 1–5 Labeled Samples per Class</p></center>

<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px;">   <thead>     <tr style="background-color: #f2f2f2;">       <th rowspan="2">Botswana</th>       <th colspan="8"><center>Method</center></th>     </tr>     <tr style="background-color: #e0e0e0;">       <th>DFSL+NN</th>       <th>DCFSL</th>       <th>HFSL</th>       <th>DM-MRN</th>       <th>FSCF-SSL</th>       <th>CTF-SSCL</th>       <th>MLPA</th>       <th><strong style="color: black;">FGAPA</strong></th>     </tr>   </thead>   <tbody>     <!-- 1-shot -->     <tr>       <th>1-shot</th>       <td>86.88 ± 3.37</td>       <td>85.14 ± 2.96</td>       <td>68.85 ± 5.00</td>       <td>59.91 ± 4.66</td>       <td>68.76 ± 4.89</td>       <td>79.20 ± 3.64</td>       <td>82.72 ± 2.81</td>       <td><strong style="color: black;">90.62 ± 1.71</strong></td>     </tr>      <!-- 2-shot -->     <tr>       <th>2-shot</th>       <td>91.64 ± 2.69</td>       <td>92.79 ± 1.92</td>       <td>82.84 ± 3.75</td>       <td>81.07 ± 3.70</td>       <td>81.55 ± 3.14</td>       <td>89.67 ± 3.57</td>       <td>92.63 ± 1.55</td>       <td><strong style="color: black;">94.95 ± 2.15</strong></td>     </tr>      <!-- 3-shot -->     <tr>       <th>3-shot</th>       <td>94.44 ± 1.94</td>       <td>94.67 ± 1.36</td>       <td>88.83 ± 3.45</td>       <td>86.91 ± 1.93</td>       <td>90.19 ± 3.24</td>       <td>93.31 ± 1.65</td>       <td>94.42 ± 1.98</td>       <td><strong style="color: black;">96.90 ± 1.31</strong></td>     </tr>      <!-- 4-shot -->     <tr>       <th>4-shot</th>       <td>95.54 ± 1.46</td>       <td>95.98 ± 1.23</td>       <td>94.19 ± 2.49</td>       <td>88.82 ± 2.46</td>       <td>93.32 ± 2.36</td>       <td>94.93 ± 0.85</td>       <td>95.30 ± 1.34</td>       <td><strong style="color: black;">97.16 ± 1.35</strong></td>     </tr>      <!-- 5-shot -->     <tr>       <th>5-shot</th>       <td>96.48 ± 0.84</td>       <td>96.90 ± 1.03</td>       <td>94.56 ± 1.61</td>       <td>92.55 ± 1.79</td>       <td>95.81 ± 1.43</td>       <td>96.05 ± 1.32</td>       <td>96.65 ± 1.04</td>       <td><strong style="color: black;">98.22 ± 1.31</strong></td>     </tr>   </tbody> </table>

从表2-4中可以看出，FGAPA在Indian_pines、Salinas、Botswana上，1~5个标记样本的情况下，始终优于其他所有对比方法。例如，5-shot情况下FGAPA通过ACA模块与FFA模块，提取判别特征的同时缩小域间差异，展现出强劲性能，具体来说，在Indian Pines、Salinas、Botswana上分别以2.69%，3.07%，1.32%的优势领先于次优方法。1-shot情况下FFA模块获得的判别性特征能够明确类别分布，提升泛化性。具体来说，在Indian Pines、Salinas、Botswana上分别以5.72%，2.7%，3.74%的优势领先于次优方法。同时FGAPA在所有情况下的标准差普遍低于其他对比方法。这表明FGAPA在有效样本情况下的强大性能，在1shot等极端条件下的高鲁棒性以及突出的稳定性。综上所述，从统计结果来看,FGAPA的性能具有显著优势。

10.There are temperature parameters in equations (7) and (10). How do you tune these? This information is lacking.

Same for hyperparameters in the losses: how are the weights selected? On which data? What impact on the performance of the model?

What is the batch size used? (This is necessary for reproducibility.)

在本实验中，需要设置超参数包含公式（7）和（10）的温度参数τ以及总损失中的λ1和λ2以及模型训练中的batch size,下面我们对这些参数的具体设定进行介绍：

- **For the temperature parameter τ , we set it to a universal value of 0.1 based on experience.**

$$
Loss = L_{\text{fsl}} + \lambda_1 L_{\text{in}} + \lambda_2 L_{\text{cross}}
$$
- **$λ_1$and $λ_2$ represent the weighted hyperparameters for  $L_{\text{in}}$ and $L_{\text{cross}}$. Their values range from 0 to 1 with increments of 0.1. They were tuned using the OA, AA, and Kappa metrics on three datasets, and the optimal parameters obtained were $λ_1$ = 0.5 and $λ_2$ = 0.3.**

- **此外，我们还给出了FGAPA中batch-size的大小设定，其被设置为64。它用来控制模型训练过程中每次参数更新时所使用的样本数。在神经网络的训练过程中，数据分成若干批次，每个批次包含batch size个样本。这样既可以提高模型训练效率，更可以抑制噪声增强模型训练稳定性。**

11.The last paragraph of the introduction does not contain the appropriate acronym (FFE instead of FFA). 

​      Figure 3 is too small. 

我们会在后续的论文中订正将FFA误写成FFE的问题，以提升论文的质量与可读性。另外，我们对图3进行了放大展示以使读者更直观地观察到各方法间的性能差异。

<img src="https://gitee.com/abcd123123410513/images/raw/master/imgs/perfomence.png" alt="替代文本" title="图片标题" width=1200>

<center><p>Figure 7: Revised version of Figure 3 in FGAPA</p></center>

12.I am not an expert in hyperspectral image processing, and it is difficult to understand the task from the text. It would be highly beneficial to add a paragraph that clearly presents the task. For example, what does Figure 2 represent? The bibliography is also somewhat limited.

**我们衷心的感谢审稿人对论文提出的建议，我们也会在FGAPA修订版本对符号定义及核心概念进行更详尽的说明，增强全文可读性。与此同时，我们会补充更多相关参考文献去调研相关工作的研究现状。**

## **The ending**

**我们非常感激审稿人对本工作创新性的积极评价及其对所提出的跨域高光谱图像分类方法在整体研究框架和方法动机方面潜力与实际价值的认可。完全认同审稿人提出的主要关切。我们对审稿人所提出的模型对新源域及目标域的鲁棒性等进一步探讨，FGAPA中基本概念的可读性、实验超参数的设置，方法的细节以及详细实验的优势统计意义给予了全面回复。希望通过本补充材料可以消除审稿人的疑虑。同时再次感谢审稿人提出的宝贵意见，这些建议帮助我们明确了论文中仍需改进的关键方面。我们相信，在充分吸纳这些反馈后，本文的质量与表达将得到显著提升。**

 