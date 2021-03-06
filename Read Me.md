1.Introduce

This is the source code for paper "An L1-and-L2-norm-oriented Latent Factor Model for Recommender Systems" that is submitted to IEEE transactions on neural networks and learning systems. The uploaded “L3F_Code.rar” contains 4 files, i.e., L3F-RMSE-MAE-NDCG-Situation1, L3F-Precision-NDCG-Situation2, L3F-Noise, and L3F-Parallelization. Each file can be used to run our algorithm directly. “L3F-RMSE-MAE-NDCG-Situation1” is used to test RMSE, MAE, and NDCG of Situation 1. “L3F-Precision-NDCG-Situation2” is used to test Precision and NDCG of Situation 2. “L3F-Noise” is used to test L3F’s Robustness to Outlier Data. “L3F-Parallelization” is used to test computational efficiency with a parallel version implemented by Hogwild!. 

2.Run our algorithm

(1)	Download the file “L3F_Code.rar”.

(2)	Unzip the file " L3F_Code.rar", then we have four files.

(3)	Import any one of the four files into a JAVA Integrated Development Environment, like “Eclipse”, refer to https://www.eclipse.org/downloads/.

(4)	Open “Main_Function.java” and run. 

3.Parameters setting

Parameters are listed  at the end of “Main_Function.java”. Their meaning is explained as follows.

(1)	Common_Function.initializeRatings("./Jester_train.txt","./Jester_test.txt", "::"): input dataset. A example dataset Jester is provided in each file. 

(2)	int top_k: Top K recommendation. 

(3)	int m: Bias controller, 1 means without Bias, 2 means with Bias. 

(4)	CommonRecomm_NEW.ThreadNUM: core number of parallelization. 

(5)	Common_Function.noise_ratio: control the noise_ratio in the range of [0-1].

(6)	Common_Function.eta: learning rate η .

(7)	Common_Function.lambda: regularization parameter λ.

(8)	Common_Function.featureDimension: Latent factor dimension f. 

4.Datasets 

The dataset Jester is provided in the code. The artificial dataset is also provided in the "L3F-Noise" code. The other datasets used in this paper can be downloaded from the following repositories. 

https://www.librec.net/datasets.html

https://grouplens.org/datasets/movielens/

https://grouplens.org/datasets/eachmovie/

https://webscope.sandbox.yahoo.com/catalog.php?datatype=r

http://www.occamslab.com/petricek/data/

5.Code Online

The codes of L3F model for “L3F-RMSE-MAE-NDCG-Situation1” are available to see on Github online. Please refer to the other three files: Common_Function, Main_Function, RTuple. 


