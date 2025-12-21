# Supplementary materials of FGAPA

We sincerely thank the two reviewers for their thorough evaluation of our manuscript and for providing constructive and insightful comments. We are also deeply grateful for the reviewers' positive assessment of the novelty of this work, as well as their recognition of the potential and value of the proposed cross-domain hyperspectral image classification method, FGAPA.  Below are our specific responses to the reviewers' concerns regarding FGAPA.

## **Response to Reviewer 1**

1.The experiments provide good initial evidence for the method, but need to go further in order to fully establish the value of the approach. The training and transfer hinges on the selection of the source domain, so only one experiment here is very limiting. The source domain needs some properties of potential alignment with the target domains, and must somehow be sufficiently rich to cover the possibilities. So it isn't clear how robust the method is in general, or how to pick the source domain to begin with, or how the approach can be updated as new domains are sampled.

**To ensure the effectiveness of the proposed method, the source domain is generally required to cover no fewer classes than the target domain, so as to construct a sufficiently representative metric space for downstream tasks. The Chikusei dataset contains 19 diverse land-cover classes, providing rich prior knowledge for cross-domain few-shot learning. In addition, a significant domain shift exists between Chikusei and the target domains, making it well suited for evaluating the generalization capability of few-shot learning methods under challenging conditions. Therefore, Chikusei is selected as the source domain in this study.**

**To further validate the robustness of our method across different source domains, we additionally conduct experiments using the Hanchuan dataset as the source domain. This dataset also contains a rich variety of land-cover classes (16 classes) and exhibits a substantial domain shift from the target domains. Table 1 reports the classification performance of our method on the Indian Pines, Salinas, and Botswana target domains using Hanchuan as the source domain with only five labeled samples per class, in comparison with representative domain adaptation methods, including DCFSL, MLPA, and the contrastive learning-based CTF-SSCL.**

<center><p>Table 1:Classification results (mean ± std) with Hanchuan dataset as source domain</p></center>

<table style="font-size: 0.6em;">
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


**As shown in Table 1, when Hanchuan is used as the source domain, FGAPA achieves superior OA, AA, and Kappa across all three target datasets. In terms of OA, FGAPA outperforms DCFSL and MLPA by 6.84% and 8.5% on Indian Pines, respectively, with similar gains observed on Salinas and Botswana. Compared with the contrastive learning-based CTF-SSCL, our method benefits from the adversarial domain alignment strategy in the ACA module, resulting in stronger performance, while lower variance further confirms its superior stability. Overall, FGAPA demonstrates consistent robustness and reliability across different source domains, including Chikusei and Hanchuan.**



## **Response to Reviewer 2**

1.Some definitions are missing in the paragraph about FSL in HSIC. What is FSL? What are source and target domains?

**Few Shot Learning (FSL) is fundamentally a meta-learning method that can acquire transferable knowledge from different tasks, enabling rapid adaptation to new tasks. It typically follows an N-way K-shot learning setting, where K labeled samples per class are used to train an N-class classifier. As shown in Figure 1, an FSL task consists of a labeled support set and a query set to be predicted. FSL classifies samples by computing the distance between each query feature and the support features of each class, thereby enabling the model to learn how to infer the classes of query samples based on the given support set. In hyperspectral image classification (HSIC), the same framework can be applied: a small number of labeled samples are selected as the support set, while a large number of unlabeled samples serve as the query set, constructing an FSL task for model learning**.



<img src="https://gitee.com/abcd123123410513/images/raw/master/imgs/FSL_compressed.png" alt="替代文本" title="图片标题">

<center><p>Fig1:    Few shot learning</p></center>

**In cross-domain FSL, the source domain refers to a dataset with abundant, fully labeled samples, where the model is initially pre-trained to learn general features and patterns (i.e., to “learn how to learn”). The target domain is the dataset for the downstream task, typically with very few labeled samples and a data distribution and set of classes different from the source domain. The core objective of the entire process is to leverage the prior knowledge acquired from the source domain to quickly adapt to new tasks in the target domain using only a minimal number of labeled samples (e.g., 1–5 per class).**

2.Figure 1 is not clear. Acronyms are not defined and do not correspond to the text. What are support features? Why don't we see $L_{\text{fsl}}^t$ and $L_{\text{fsl}}^s$? What is the attention score (never mentioned in the text)? Does the prototype bank correspond to $\mathbf p_i$?

![image-20251216203011245](https://gitee.com/abcd123123410513/images/raw/master/imgs/overall%20framework%20of%20FGAPA_compressed.png)

<center><p>Fig2: Overall framework of FGAPA</p></center>

**Support features and Query features are the feature representations extracted from the support set and the query set, respectively. As shown in Figure 2, the proposed method determines the category of Query features by computing the Euclidean distance between them and the prototypes formed by the Support features of each class.**

**The FSL task is conducted simultaneously on both the source and target domains. Its loss function $L_{\text{fsl}}$ consists of two components, $L_{\text{fsl}}^t$ and $L_{\text{fsl}}^s$, which are collectively referred to as $\mathcal{L}_{fsl}$ in the text. In the original Figure 2, the term "attention score" corresponds to the "correlation calculation" described in the paper, and $p_i$ denotes the prototype of the $i$-th class in the prototype bank. Figure 2 has been updated accordingly to provide a clearer illustration of the overall workflow of FGAPA.**

3.Equation (5) and (6): what is the dimensionality of the weights and biases in the MLP?

**In FGAPA, the feature extractor outputs features of dimension 128×1. As specified in Equations (5) and (6), the MLP layers are configured as follows: the first layer has weights of size (128, 64) and biases of size 64, and the second layer has weights of size (64, 128) and biases of size 128.**

4.Why do you need to transform the features and prototypes with MLP?

**Since features and prototypes originate from different semantic levels, the MLP layer maps them into a comparable feature space. Meanwhile, the ReLU activation function within MLP further helps  the model to learn more complex nonlinear relationships, thereby enhancing its expressive capability. Therefore, transforming both features and prototypes through the MLP allows for more effective correlation computation in the feature space and facilitates the learning of richer, more expressive representations.**

5.Why is it a cosine similarity (normalized dot product) in equation (7) and not in equation (10) ?

**Both Equation (7) and Equation (10) employ cosine similarity for correlation computation. This approach is naturally aligned with prototype learning, better suited for high-dimensional embedding spaces, and maintains optimization stability. Moreover, the temperature coefficient adjusts the sharpness of the softmax distribution, further enhancing training stability. To clarify the methodological details, Equation (10) will be revised accordingly to avoid potential ambiguity.**

6.What is the total optimization objective of the model in the end? It never appears.

**The overall loss function of our method consists of three components: the few-shot learning loss $L_{\text{fsl}}$, the intra-domain alignment loss $L_{\text{in}}$, and the cross-domain alignment loss $L_{\text{cross}}$. It is defined as follows:**

$$
\text{Loss} = L_{\text{fsl}} + \lambda_1 L_{\text{in}} + \lambda_2 L_{\text{cross}}
$$

**Here, $\lambda_1$ and $\lambda_2$ are weighting hyperparameters for $L_{\text{in}}$ and $L_{\text{cross}}$, respectively. Their values are tuned within the range [0, 1] with a step size of 0.1. After optimization, the best-performing values are set as $\lambda_1 = 0.5$ and $\lambda_2 = 0.3$.**

7.You talk about 4 datasets. Where do they come from? Is it public data? Is it published? How do you remove noisy bands?

**This study conducts experiments on four publicly available hyperspectral datasets. Specifically, Chikusei is used as the source domain, while Indian Pines, Salinas, and Botswana serve as the target domains. The detailed information, data sources, and noise removal procedures for each dataset are as follows:**

- **Chikusei**

**The Chikusei dataset was captured by the Headwall Hyperspec-VNIR-C sensor over Chikusei City, Japan. It is publicly available at https://naotoyokoya.com/Download.html and consists of 2517 × 2335 pixels with a spatial resolution of 2.5 m. The data provides 128 spectral bands covering wavelengths from 343 to 1018 nm and includes 19 land-cover classes. Figure 3 shows the pseudo-color image, the ground truth, and the corresponding class labels along with the number of samples for each class.**



<img src="https://gitee.com/abcd123123410513/images/raw/master/imgs/CK_compressed.png" alt="替代文本" title="图片标题" width=500>

<center><p>Figure 3: The land cover types and the number of samples for the Chikusei dataset</p></center>



- **Indian Pines**

**The Indian Pines dataset was collected by the Airborne Visible/Infrared Imaging Spectrometer (AVIRIS) over Indiana, USA. It is publicly accessible at https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes and comprises 145 × 145 pixels with a spatial resolution of approximately 20 m. After removing 20 water absorption bands (104–105, 150–163, and 220), 200 spectral bands covering the wavelength range from 400 to 2500 nm are used. Figure 4 shows the pseudo-color image, the ground truth, and the corresponding class labels along with the number of samples for each class.**



<img src="https://gitee.com/abcd123123410513/images/raw/master/imgs/IP_compressed.png" alt="替代文本" title="图片标题" width=500>

<center><p>Fig4: The land cover types and the number of samples for the Indian Pines dataset</p></center>



- **Salinas**

**The Salinas dataset was collected by the AVIRIS sensor over the Salinas Valley in California, USA. It is publicly available at https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes and consists of 512 × 217 pixels with a spatial resolution of approximately 3.7 m. A total of 204 spectral bands covering the range of 400–2500 nm are used, divided into 16 land-cover classes. Figure 5 shows the pseudo-color image, the ground truth, and the corresponding class labels along with the number of samples for each class.**

<img src="https://gitee.com/abcd123123410513/images/raw/master/imgs/SA_compressed.png" alt="替代文本" title="图片标题" width=500>

<center><p>Fig5: The land cover types and the number of samples for the Salinas dataset</p></center>



- **Botswana**

**The Botswana dataset was acquired by NASA's EO-1 satellite over the Okavango Delta in Botswana. It is publicly available at https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes. The image size is 1476 × 256 pixels, with a spatial resolution of approximately 20 m. Out of the original 242 spectral bands (400–2500 nm), 145 bands are used after removing noisy ones (specifically bands 1–9, 56–81, 98–101, 120–133, and 165–186). Figure 6 shows the pseudo-color image, the ground truth, and the corresponding class labels along with the number of samples for each class.**



<img src="https://gitee.com/abcd123123410513/images/raw/master/imgs/BO_compressed.png" alt="替代文本" title="图片标题" width=500>

<center><p>Fig6: The land cover types and the number of samples for the Botswana dataset</p></center>



8.In the experimental protocol, you have to detail the metrics more in detail. What do they measure? How are they calculated? Why are they relevant for the task?

**Three commonly used metrics for evaluating classification performance are employed in this study: Overall Accuracy (OA), Average Accuracy (AA), and the Kappa coefficient. OA measures the overall classification correctness, AA calculates the average accuracy across all classes to avoid dominance by classes with a large number of samples, and Kappa provides a more reliable assessment of agreement between the classification results and ground truth by correcting for random consistency. The definitions of these metrics are as follows:**

- **OA:** 

$$
OA = \frac{\sum_{i=1}^{c} TP_i}{N}
$$


​        **where $TP_i$ denotes the number of correctly classified samples for the i-th class, N is the total number of samples, and c represents the total number of classes.**

- **AA:** 

$$
AA = \frac{1}{c} \sum_{i=1}^{c} \frac{TP_i}{N_i}
$$

​         **where $TP_i$ denotes the number of correctly classified samples for the i-th class, $N_i$ represents the total number of samples in the i-th class, and c is the total number of classes.**

-   **Kappa:**

$$
\kappa = \frac{P_o - P_e}{1 - P_e}
$$

​         **where $P_o$ represents the probability that the model and a random classifier assign the same class labels,$P_e$ denotes the theoretical probability of agreement between the model and a random classifier, calculated based on their respective marginal distributions.**

9.As stated in the global comment, the results are missing statistical significance tests to make sure your method is statistically better on these datasets.

**Due to space constraints, some important content in the main text could not be fully elaborated. To address this, we have supplemented the performance results of FGAPA on the three target domains—Indian Pines, Salinas, and Botswana—with Chikusei as the source domain, including standard deviations to demonstrate statistical significance. Tables 2-4 present the performance statistics of FGAPA under 1–5 shot settings on the three target domains, respectively. All experiments were independently run 10 times, and the results are reported as averages.**

<center><p>Table 2: Classification results (mean ± standard deviation) on Indian Pines using 1–5 labeled samples per class</p></center>

<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px;">   <thead>     <tr style="background-color: #f2f2f2;">       <th rowspan="2">Indian Pines</th>       <th colspan="8"><center>Method</center></th>     </tr>     <tr style="background-color: #e0e0e0;">       <th>DFSL+NN</th>       <th>DCFSL</th>       <th>HFSL</th>       <th>DM-MRN</th>       <th>FSCF-SSL</th>       <th>CTF-SSCL</th>       <th>MLPA</th>       <th><strong style="color: black;">FGAPA</strong></th>     </tr>   </thead>   <tbody>     <!-- 1-shot -->     <tr>       <th>1-shot</th>       <td>40.93 ± 4.82</td>       <td>41.32 ± 5.75</td>       <td>42.95 ± 6.74</td>       <td>41.65 ± 5.36</td>       <td>46.44 ± 5.10</td>       <td>43.52 ± 4.84</td>       <td>41.83 ± 5.51</td>       <td><strong style="color: black;">52.16 ± 4.48</strong></td>     </tr>      <!-- 2-shot -->     <tr>       <th>2-shot</th>       <td>50.98 ± 3.09</td>       <td>51.60 ± 4.53</td>       <td>53.78 ± 5.59</td>       <td>55.41 ± 3.88</td>       <td>60.14 ± 4.84</td>       <td>56.82 ± 4.92</td>       <td>53.47 ± 4.82</td>       <td><strong style="color: black;">64.81 ± 4.32</strong></td>     </tr>      <!-- 3-shot -->     <tr>       <th>3-shot</th>       <td>56.47 ± 2.44</td>       <td>56.03 ± 3.45</td>       <td>63.65 ± 4.13</td>       <td>60.85 ± 3.28</td>       <td>67.72 ± 1.62</td>       <td>63.61 ± 3.5</td>       <td>57.65 ± 2.63</td>       <td><strong style="color: black;">71.19 ± 4.18</strong></td>     </tr>      <!-- 4-shot -->     <tr>       <th>4-shot</th>       <td>60.44 ± 2.82</td>       <td>61.96 ± 3.98</td>       <td>69.96 ± 3.93</td>       <td>63.95 ± 3.07</td>       <td>72.92 ± 5.56</td>       <td>67.29 ± 3.04</td>       <td>63.31 ± 2.05</td>       <td><strong style="color: black;">76.12 ± 2.76</strong></td>     </tr>      <!-- 5-shot -->     <tr>       <th>5-shot</th>       <td>63.16 ± 2.92</td>       <td>65.74 ± 2.57</td>       <td>74.03 ± 2.71</td>       <td>69.28 ± 3.30</td>       <td>76.96 ± 2.74</td>       <td>70.86 ± 3.22</td>       <td>66.59 ± 2.81</td>       <td><strong style="color: black;">79.65 ± 2.76</strong></td>     </tr>   </tbody> </table>

<center><p>Table 3: Classification results (mean ± standard deviation) on Salinas using 1–5 labeled samples per class</p></center>

<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px;">   <thead>     <tr style="background-color: #f2f2f2;">       <th rowspan="2">Salinas</th>       <th colspan="8"><center>Method</center></th>     </tr>     <tr style="background-color: #e0e0e0;">       <th>DFSL+NN</th>       <th>DCFSL</th>       <th>HFSL</th>       <th>DM-MRN</th>       <th>FSCF-SSL</th>       <th>CTF-SSCL</th>       <th>MLPA</th>       <th><strong style="color: black;">FGAPA</strong></th>     </tr>   </thead>   <tbody>     <!-- 1-shot -->     <tr>       <th>1-shot</th>       <td>75.65 ± 2.46</td>       <td>74.81 ± 4.12</td>       <td>68.41 ± 4.76</td>       <td>76.77 ± 5.72</td>       <td>68.30 ± 4.83</td>       <td>74.19 ± 3.14</td>       <td>73.37 ± 3.08</td>       <td><strong style="color: black;">79.47 ± 3.30</strong></td>     </tr>      <!-- 2-shot -->     <tr>       <th>2-shot</th>       <td>82.16 ± 2.64</td>       <td>82.04 ± 3.26</td>       <td>80.18 ± 2.76</td>       <td>84.89 ± 3.23</td>       <td>81.40 ± 2.99</td>       <td>81.79 ± 3.26</td>       <td>81.63 ± 2.86</td>       <td><strong style="color: black;">87.55 ± 3.13</strong></td>     </tr>      <!-- 3-shot -->     <tr>       <th>3-shot</th>       <td>84.99 ± 1.53</td>       <td>84.88 ± 1.84</td>       <td>85.28 ± 2.65</td>       <td>87.48 ± 3.23</td>       <td>84.33 ± 3.52</td>       <td>85.57 ± 2.17</td>       <td>85.37 ± 2.01</td>       <td><strong style="color: black;">90.70 ± 1.86</strong></td>     </tr>      <!-- 4-shot -->     <tr>       <th>4-shot</th>       <td>88.61 ± 1.53</td>       <td>88.05 ± 1.90</td>       <td>86.98 ± 2.57</td>       <td>90.04 ± 2.03</td>       <td>88.11 ± 3.66</td>       <td>88.58 ± 1.59</td>       <td>89.04 ± 1.50</td>       <td><strong style="color: black;">91.83 ± 1.09</strong></td>     </tr>      <!-- 5-shot -->     <tr>       <th>5-shot</th>       <td>89.74 ± 1.26</td>       <td>90.12 ± 1.22</td>       <td>89.14 ± 2.09</td>       <td>91.08 ± 2.39</td>       <td>90.96 ± 2.06</td>       <td>89.96 ± 1.43</td>       <td>89.94 ± 1.30</td>       <td><strong style="color: black;">93.15 ± 1.00</strong></td>     </tr>   </tbody> </table>

<center><p>Table 4: Classification results (mean ± standard deviation) on Salinas using 1–5 labeled samples per class</p></center>

<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px;">   <thead>     <tr style="background-color: #f2f2f2;">       <th rowspan="2">Botswana</th>       <th colspan="8"><center>Method</center></th>     </tr>     <tr style="background-color: #e0e0e0;">       <th>DFSL+NN</th>       <th>DCFSL</th>       <th>HFSL</th>       <th>DM-MRN</th>       <th>FSCF-SSL</th>       <th>CTF-SSCL</th>       <th>MLPA</th>       <th><strong style="color: black;">FGAPA</strong></th>     </tr>   </thead>   <tbody>     <!-- 1-shot -->     <tr>       <th>1-shot</th>       <td>86.88 ± 3.37</td>       <td>85.14 ± 2.96</td>       <td>68.85 ± 5.00</td>       <td>59.91 ± 4.66</td>       <td>68.76 ± 4.89</td>       <td>79.20 ± 3.64</td>       <td>82.72 ± 2.81</td>       <td><strong style="color: black;">90.62 ± 1.71</strong></td>     </tr>      <!-- 2-shot -->     <tr>       <th>2-shot</th>       <td>91.64 ± 2.69</td>       <td>92.79 ± 1.92</td>       <td>82.84 ± 3.75</td>       <td>81.07 ± 3.70</td>       <td>81.55 ± 3.14</td>       <td>89.67 ± 3.57</td>       <td>92.63 ± 1.55</td>       <td><strong style="color: black;">94.95 ± 2.15</strong></td>     </tr>      <!-- 3-shot -->     <tr>       <th>3-shot</th>       <td>94.44 ± 1.94</td>       <td>94.67 ± 1.36</td>       <td>88.83 ± 3.45</td>       <td>86.91 ± 1.93</td>       <td>90.19 ± 3.24</td>       <td>93.31 ± 1.65</td>       <td>94.42 ± 1.98</td>       <td><strong style="color: black;">96.90 ± 1.31</strong></td>     </tr>      <!-- 4-shot -->     <tr>       <th>4-shot</th>       <td>95.54 ± 1.46</td>       <td>95.98 ± 1.23</td>       <td>94.19 ± 2.49</td>       <td>88.82 ± 2.46</td>       <td>93.32 ± 2.36</td>       <td>94.93 ± 0.85</td>       <td>95.30 ± 1.34</td>       <td><strong style="color: black;">97.16 ± 1.35</strong></td>     </tr>      <!-- 5-shot -->     <tr>       <th>5-shot</th>       <td>96.48 ± 0.84</td>       <td>96.90 ± 1.03</td>       <td>94.56 ± 1.61</td>       <td>92.55 ± 1.79</td>       <td>95.81 ± 1.43</td>       <td>96.05 ± 1.32</td>       <td>96.65 ± 1.04</td>       <td><strong style="color: black;">98.22 ± 1.31</strong></td>     </tr>   </tbody> </table>

As shown in Tables 2-4, under the 1–5 shot settings, FGAPA consistently outperforms all comparative methods across the three datasets: Indian Pines, Salinas, and Botswana. Specifically, under the 5-shot condition, the collaborative work of its ACA and FFA modules extracts discriminative features while reducing inter-domain differences, leading to outstanding performance—surpassing the second-best method by 2.69%, 3.07%, and 1.32% on the three datasets, respectively. Under the 1-shot condition, the discriminative features captured by the FFA module better characterize class distributions and enhance generalization, achieving advantages of 5.72%, 2.7%, and 3.74%, respectively. Moreover, FGAPA generally exhibits lower standard deviations than other methods across all experimental scenarios, indicating strong stability under varying sample conditions. Overall, these results fully demonstrate the statistically significant superiority of FGAPA.

10.There are temperature parameters in equations (7) and (10). How do you tune these? This information is lacking. Same for hyperparameters in the losses: how are the weights selected? On which data? What impact on the performance of the model? What is the batch size used? (This is necessary for reproducibility.)

**In this experiment, the hyperparameters include the temperature parameter τ in Equations (7) and (10), the weighting coefficients λ₁ and λ₂ in the total loss, and the batch size used during model training. The specific settings for these parameters are described below.**

- **For the temperature parameter τ , we set it to a universal value of 0.1 based on experience.**

- **In the final loss, two hyperparameters controlling the dual-domain adversarial strategy are included, as specifically shown below:**

$$
Loss = L_{\text{fsl}} + \lambda_1 L_{\text{in}} + \lambda_2 L_{\text{cross}}
$$

​         **where $λ_1$ and $λ_2$ represent the weighted hyperparameters for  $L_{\text{in}}$ and $L_{\text{cross}}$. Their values range from 0 to 1 with increments of 0.1. They were tuned using the OA, AA, and Kappa metrics on three datasets, and the optimal parameters obtained were $λ_1$ = 0.5 and $λ_2$ = 0.3.**

- **Furthermore, we set the batch size in FGAPA to 64, which determines the number of samples used in each parameter update during training. Throughout the training process, the data are divided into multiple batches, each containing 64 samples. This configuration not only improves training efficiency but also helps suppress noise and enhances the stability of model training.**

11.The last paragraph of the introduction does not contain the appropriate acronym (FFE instead of FFA). 

​      Figure 3 is too small. 

**Following the suggestions, we have corrected all instances of FFE to FFA in the text. Furthermore, Figure 3 has been enlarged to enable readers to more clearly observe the performance differences among the methods.**

<img src="https://gitee.com/abcd123123410513/images/raw/master/imgs/Revised%20version%20of%20Figure%203%20in%20FGAPA_compressed%20(1).png" alt="替代文本" title="图片标题" width=1200>

<center><p>Figure 7: Revised version of figure 3 in FGAPA</p></center>

12.I am not an expert in hyperspectral image processing, and it is difficult to understand the task from the text. It would be highly beneficial to add a paragraph that clearly presents the task. For example, what does Figure 2 represent? The bibliography is also somewhat limited.

**We sincerely thank the reviewer for the valuable suggestions. In the revised manuscript, we will provide clearer explanations of the notations and core concepts to improve readability. Additionally, we will include relevant references to better reflect the current research landscape.**

## **The ending**

**We sincerely appreciate the reviewers' positive assessment of the novelty of this work and their recognition of the potential and practical value of the proposed cross-domain hyperspectral image classification method within the overall research framework and methodological motivation. We fully acknowledge the main concerns raised by the reviewers and have provided comprehensive responses regarding the robustness of FGAPA to new source and target domains, the readability of its fundamental concepts, the setting of experimental hyperparameters, methodological details, and the statistical significance of the experimental results. We hope that this supplementary material addresses the reviewers' concerns. Once again, we thank the reviewers for their valuable suggestions, which have helped us identify key areas for improvement in the manuscript. We believe that incorporating this feedback will significantly enhance the quality and clarity of the paper.**


 
































