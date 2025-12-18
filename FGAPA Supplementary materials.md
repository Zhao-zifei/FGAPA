# FGAPA Supplementary materials

## **Preface**

**This document aims to help readers who are not focused on hyperspectral image analysis quickly understand the basic concepts and task requirements. While addressing the questions raised by reviewers, it also presents all experiments and supplementary experiments conducted on FGAPA.**

## About FSL

FSL

Few-shot learning (FSL) has attracted attention for its ability to learn and generalize from very limited samples. FSL is essentially a meta-learning approach that can acquire transferable knowledge from different tasks to quickly adapt to new tasks. Typically, it operates under an N-way K-shot setting, where K labeled samples per class are used to train an N-class classifier.

![image-20251216200323548](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216200323548.png) 



Taking natural images as an example, an FSL task consists of a Support set, which is a collection of labeled samples, and a Query set, which is a collection of query samples. The query set and the support set are used to construct the FSL task and are fed into the feature extractor together for feature extraction. **Support features** represent the feature representations obtained from the support set through the feature extractor, while **Query features** are the feature representations obtained from the query set. The loss in FSL is calculated through metric learning, determining the class of a query feature by computing its distance to the support features of each class, bringing features of the same class closer and pushing features of different classes apart in the metric space. Through such FSL tasks, the model learns to classify query samples based on the given support set.

## **About the Source Domain, Target Domain, and Their Selection**

**Source Domain and Target Domain**

In the context of cross-domain few-shot learning, the setting usually consists of source domain data $D_s$ containing $C_s$ classes and target domain data $D_t$ containing $C_t$ classes. The classes in $D_s$ are referred to as base classes, while the classes in $C_t$ are referred to as novel classes, and their actual categories are typically different. To ensure diversity in the training samples, $C_s$ is usually larger than $C_t$. Unlike the abundant labeled samples in $C_s$, $C_t$ contains only a few labeled samples $D_l$ and a large number of unlabeled samples $D_u$. To meet the requirements for constructing meta-learning tasks, labeled samples $D_l$ are augmented using random Gaussian noise. The proposed FGAPA method uses the labeled samples $D_l$ from both the source domain $D_s$ and the target domain $D_t$ to train the feature extractor and ultimately evaluates classification performance on the target domain $D_t$ using the unlabeled samples $D_u$.

**In general, the source domain consists of data with abundant labeled samples, while the target domain contains only a small number of labeled samples for downstream classification tasks. What we aim to do is fully transfer the knowledge acquired by the model from training on the source domain to the target domain for classification tasks.**

**Regarding the selection of the source domain, in order to facilitate downstream tasks and reverse operations, the number of categories in the source domain is usually required to be greater than or equal to that of the target domain. This ensures that the metric space obtained from training can serve the tasks of the target domain. In this study, only Chikusei was used as the source domain, but our method demonstrates good robustness and reliability across different source domains. To this end, we conducted experiments using the Hanchuan source domain. Due to time constraints, only partial validation was performed, with further experiments to be completed later.**

Due to time constraints, we chose **DCFSL** and **MLPA**, **which are both domain adaptation methods**, and compared them with **CTF-SSCL**, based on contrastive learning tasks, and **GCC-FSL**, **which uses graphs to capture the correlation between the two domains**.

**Indian_pines-5shot**

![image-20251218093539862](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251218093539862.png)

**Salinas-5shot**

![image-20251218093740663](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251218093740663.png)

**Botswana-5shot**

![image-20251218094032517](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251218094032517.png)

Obviously, our method not only performs well on the Chikusei source domain, but also shows strong robustness and stability on other source domains such as Hanchuan. **However, the performance has declined compared to before. This is precisely the issue we need to address next, such as adjusting the learning rate and re-optimizing the loss hyperparameters to develop an adversarial domain adaptation strategy for new source domains.**

## **About downstream tasks**

Classification of hyperspectral images is precise down to each hyperspectral pixel, so the reviewer mistakenly thought it was image segmentation. However, image segmentation focuses on dividing regions, where a pixel and its neighboring pixels should be assigned to a representative area of a certain class. From the classification maps of the comparative methods in the paper, it can be seen that some regions have pixels of different classes scattered throughout. Therefore, this corresponds to a classification task rather than semantic segmentation.

![img](https://gitee.com/abcd123123410513/images/raw/master/imgs/DFSL-IP.png)





## **Regarding the issue of text corresponding to the model diagram**![image-20251216203011245](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216203011245.png)



We have revised the issue raised by the reviewer regarding the attention scores not being represented in the text, replacing it with **Correlation calculation**. Meanwhile, $p_i$ represents the prototype of each class in the prototype library.



## **About the method**

- FFA

  Regarding the reviewer's question about why the transformation of features and prototypes is performed through the MLP layer, as well as the issue of the dimensions of the MLP's weights and biases:

  First, features and prototypes come from different semantic levels. They are jointly mapped into the same comparable feature space through an MLP layer. At the same time, the intermediate ReLU in the MLP layer ensures that the model can learn more complex nonlinear relationships to enhance its expressive power.

  The features obtained through feature extraction are ultimately unified into a feature dimension of 128*1. The MLP layer weights correspond to (64, 128), where 64 represents the output dimension, and the bias is 64. The weights of the second layer correspond to (128, 64), with a bias of 128. This design reduces the number of parameters while forcing the model to learn important features.





## **About Hyperparameters and Final Loss**

**For the temperature parameter, we set it to a universal value of 0.1 based on experience.**

**The final loss is defined as:**
$$
Loss = L_{\text{fsl}} + \lambda_1 L_{\text{in}} + \lambda_2 L_{\text{cross}}
$$
$\lambda_1$ and $\lambda_2$ represent the weighting hyperparameters for the in-domain loss and cross-domain loss. Their values range from 0 to 1, with increments of 0.1. The optimal parameters obtained through tuning are $\lambda_1$ = 0.5 and $\lambda_2$ = 0.3.



## **About the innovations of this paper**

1. FFA: By updating through a moving average and incorporating the concept of momentum, we guide the model to improve its ability to extract discriminative features by injecting semantic correlation information of features. We distill between features and prototypes and inject the obtained semantic information to enhance feature representation and improve feature discriminability.
2. CCA: For the first time, we propose a dual-domain alignment function for cross-domain adaptation of hyperspectral images. The idea is that the essence of transfer learning is to find a common metric space where both the source and target domains perform well. In the field of hyperspectral image classification, domain adaptation is usually conducted within a single domain. For the first time, we approach it from a joint perspective, modeling the source and target domains together to achieve dual-domain alignment. Furthermore, in non-hyperspectral fields, dual-domain alignment requires that the categories of the source and target domains correspond. We remove this restriction, allowing the source and target domains to seek a metric space for universal domain adaptation even when their categories differ.



## **About the dataset**

This study uses the publicly available Chikusei dataset as the source domain and validates it on three public datasets: Indian Pines, Salinas, and Botswana.

Chikusei

![image-20251216211502343](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216211502343.png)

This dataset was captured by the Headwall Hyperspec-VNIR-C sensor in Chikusei City, Japan. It consists of 2517 × 2335 pixels with a spatial resolution of 2.5 m. A total of 128 bands are provided, covering wavelengths from 343 to 1018 nm, spanning 19 categories.

**Publish Link**：https://naotoyokoya.com/Download.html

**Publish  paper**：https://www.researchgate.net/profile/Naoto-Yokoya/publication/304013716_Airborne_hyperspectral_data_over_Chikusei/links/5762f36808ae570d6e15c026/Airborne-hyperspectral-data-over-Chikusei.pdf

Indian_pines

![image-20251216211903236](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216211903236.png)

This dataset was collected by the Airborne Visible/Infrared Imaging Spectrometer (AVIRIS) over Indiana, USA. It contains 145 × 145 pixels with a spatial resolution of about 20 m. After removing 20 water absorption bands (104-105, 150-163, and 220), 200 absorption bands in the range of 400 to 2500 nm are used. The dataset includes 16 classes.

**Publish  paper**：https://purr.purdue.edu/publications/1947/1

Salinas

![image-20251216212045332](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216212045332.png)

This dataset was collected by the AVIRIS sensor in the Salinas Valley, California, USA. It consists of 512 × 217 pixels with a spatial resolution of approximately 3.7 m. A total of 204 bands covering 400-2500 nm were used, divided into 16 categories.

**Publish Link**：http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

Botswana

![image-20251216212159994](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216212159994.png)

This dataset was acquired by NASA's EO1 satellite over the Okavango Delta in Botswana. It has a size of 1476×256 pixels and a spatial resolution of approximately 20m. Out of 242 spectral bands (400-2500 nm), 145 bands were used after removing noisy bands (1-9, 56-81, 98-101, 120-133, and 165-186). It contains 14 classes.

**Publish Link**：http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

**During the experiment, we used a publicly available hyperspectral dataset with noise bands already removed.**

## **About the experiment**

**Due to the strict length restrictions of the article, the experiments in the paper were not fully presented, which may have led to misunderstandings by the reviewers. To address this, we are now presenting all the experiments in full detail.**

- **Three Metrics**

  OA: The overall proportion of correctly classified samples by the model, that is, the percentage of all correctly classified samples out of the total number of samples.
  $$
  OA = \frac{\text{Total number of correctly classified samples}}{\text{Total sample size}} = \frac{\sum_{i=1}^{K} TP_i}{N}
  $$
  

​       AA: The average classification accuracy for each category, eliminating the impact of the number of categories.
$$
PA_i = \frac{TP_i}{\text{Number of actual samples in class } i} = \frac{TP_i}{TP_i + FN_i}
$$

$$
AA = \frac{1}{K} \sum_{i=1}^{K} PA_i
$$

​       **Kappa: Consider the consistency difference between classification results and random classification.**
$$
\kappa = \frac{OA - EA}{1 - EA}
$$
​        **OA: Overall Classification Accuracy EA: Expected Agreement**
$$
EA = \frac{\sum_{i=1}^{K} (\text{row sum}_i \times \text{column sum}_i)}{N^2}
$$
$R_i$ is the row sum for class $i$ (actual number of samples in class $i$),  
$C_i$ is the column sum for class $i$ (predicted number of samples in class $i$),  
and $N$ is the total number of samples.



**Comparative Experiment**

**IP-5shot**

![image-20251216214820501](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216214820501.png)

 **IP-4shot**

![image-20251216214856037](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216214856037.png)

**IP-3shot**

![image-20251216214919605](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216214919605.png)

**IP-2shot**

![image-20251216214952750](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216214952750.png)

**IP-1shot**

![image-20251216215025613](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216215025613.png)

**Salinas-5shot**

![image-20251216215216585](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216215216585.png)

**Salinas-4shot**

![image-20251216215230578](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216215230578.png)

**Salinas-3shot**

![image-20251216215308659](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216215308659.png)

**Salinas-2shot**

![image-20251216215356670](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216215356670.png)

**Salinas-1shot**

![image-20251216215421753](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216215421753.png)

**Botswana-5shot**

![image-20251216215757503](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216215757503.png)

**Botswana-4shot**

![image-20251216215815425](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216215815425.png)

**Botswana-3shot**

![image-20251216215840757](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216215840757.png)

**Botswana-2shot**

![image-20251216215858811](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216215858811.png)

**Botswana-1shot**

![image-20251216215927456](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251216215927456.png)





## **About the paper images**

**We apologize for the lack of clarity caused by the inability to display images across columns due to the length constraints of the paper.**

![img](https://gitee.com/abcd123123410513/images/raw/master/imgs/perfomence.png)



## Classification map

**Indian_pines**

![image-20251217172136688](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251217172136688.png)

**Salinas**

![image-20251217173401459](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251217173401459.png)

**Botswana**

![image-20251217174530160](https://gitee.com/abcd123123410513/images/raw/master/imgs/image-20251217174530160.png)

**T-SNE**

**Indian_pines**

CTF-SSCL

![TSNE-CTF-SSCL-IP](https://gitee.com/abcd123123410513/images/raw/master/imgs/TSNE-CTF-SSCL-IP.png)

FSCF-SSL

![TSNE-FSCF-SSL-IP](https://gitee.com/abcd123123410513/images/raw/master/imgs/TSNE-FSCF-SSL-IP.png)

MLPA

![TSNE-MLPA-IP](https://gitee.com/abcd123123410513/images/raw/master/imgs/TSNE-MLPA-IP.png)

FGAPA

![IP_Tsne](D:\课题组\2025.7.9\分类图\IP\IP_Tsne.png)



**Salinas**

CTF-SSCL

![TSNE-CTF-SSCL-SA](https://gitee.com/abcd123123410513/images/raw/master/imgs/TSNE-CTF-SSCL-SA.png)

FSCF-SSL

![TSNE-FSCF-SSL-SA](https://gitee.com/abcd123123410513/images/raw/master/imgs/TSNE-FSCF-SSL-SA.png)

MLPA

![TSNE-MLPA-SA](https://gitee.com/abcd123123410513/images/raw/master/imgs/TSNE-MLPA-SA.png)

FGAPA

![SA_Tsne](D:\课题组\2025.7.9\分类图\SA\SA_Tsne.png)



**Botswana**

CTF-SSCL

![TSNE-CTF-SSCL-BO](https://gitee.com/abcd123123410513/images/raw/master/imgs/TSNE-CTF-SSCL-BO.png)

FSCF-SSL

![TSNE-FSCF-SSL-BO](https://gitee.com/abcd123123410513/images/raw/master/imgs/TSNE-FSCF-SSL-BO.png)

MLPA

![TSNE-MLPA-BO](https://gitee.com/abcd123123410513/images/raw/master/imgs/TSNE-MLPA-BO.png)

FGAPA

![TSNE-PGSMC-BO](https://gitee.com/abcd123123410513/images/raw/master/imgs/TSNE-PGSMC-BO.png)