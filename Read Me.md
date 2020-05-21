1.Introduce

This is the source code for paper "An L1-and-L2-norm-oriented Latent Factor Model for Recommender Systems" that is submitted to IEEE transactions on neural networks and learning systems. The uploaded “L3F_Code.rar” contains 4 files, i.e., L3F-RMSE-MAE-NDCG-Situation1, L3F-Precision-NDCG-Situation2, L3F-Noise, and L3F-Parallelization. Each file can be used to run our algorithm directly with a dataset Jester. “L3F-RMSE-MAE-NDCG-Situation1” is used to test RMSE, MAE, and NDCG of Situation 1. “L3F-Precision-NDCG-Situation2” is used to test Precision and NDCG of Situation 2. “L3F-Noise” is used to test Outlier Data Sensitivity. “L3F-Parallelization” is used to test computational efficiency with a parallel version implemented by Hogwild!. 

2. Run our algorithm

(1)	Download the file “L3F_Code.rar”.
(2)	Unzip the file " L3F_Code.rar", then we have four files.
(3)	Import any one of the four files into a JAVA Integrated Development Environment, like “Eclipse”, refer to https://www.eclipse.org/downloads/.
(4)	Open “Main_Function.java” and run. 

3.Parameters setting. 

Parameters are lised at the end of “Main_Function.java”. Their meaning are explained as follows. 
(1)	Common_Function.initializeRatings("./Jester_train.txt","./Jester_test.txt", "::"): input dataset. A example dataset Jester is provided in each file. 
(2)	int top_k: Top K recommendation. 
(3)	int m: Bias controller, 1 means without Bias, 2 means with Bias. 
(4)	CommonRecomm_NEW.ThreadNUM: core number of parallelization. 
(5)	Common_Function.noise_ratio: control the noise_ratio in the range of [0-1].
(6)	Common_Function.eta: learning rate η .
(7)	Common_Function.lambda: regularization parameter λ.
(8)	Common_Function.featureDimension: Latent factor dimension f. 


