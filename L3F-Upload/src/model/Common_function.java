package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.StringTokenizer;





public abstract class Common_function
{
	

	
	
	
	public int err_type = 1;// 1:RMSE;other:MAE.
	public double minInnerLoopCount = 0;

	public double cacheInnerLoopCount = 0;

	public double minTotalTime = 0;

	public double cacheTotalTime = 0;

	public static boolean flag_B = true;

	public static boolean flag_C = true;

	public static double min_Error_RMSE = 1e10;
	public static double min_Error_MAE = 1e10;

	public double previous_Error = 1e10;

	public double max_Error = 0;

	public int min_Round = 0;

	public static int con_n = 300;

	public static int delayCount = 10;

	// 琛岀壒寰�
	public static int B_Count = 1;

	//鐢ㄤ互鍒濆鍖朆鐨勫熀鏁扮粍锛屼繚璇丅_count涓婂崌鏃讹紝涓嶅悓count瀵瑰簲鐨勫悓涓�涓疄浣撶殑鍜岀浉绛�
	public static double[] B_Base;

	public double[][] B;

	public static double[][] min_B, B_cache, B_delta;

	public double[][] P;

	public static double[][] min_P, P_cache, P_delta;

	// 鍒楃壒寰�
	public static int C_Count = 1;

	//鐢ㄤ互鍒濆鍖朇鐨勫熀鏁扮粍锛屼繚璇丆_count涓婂崌鏃讹紝涓嶅悓count瀵瑰簲鐨勫悓涓�涓疄浣撶殑鍜岀浉绛�
	public static double[]  C_Base;
	
	public double[][] C;

	public static double[][] min_C, C_cache, C_delta;

	public double[][] Q;

	public static double[][] min_Q, Q_cache, Q_delta;

	// 涓�闃舵搴�
	public static double[][] B_gradient, C_gradient, P_gradient, Q_gradient;

	// 鐢ㄦ潵杩涜璁＄畻鍏辫江姊害鐨勭紦瀛樼煩闃�
	public static double[][] B_r, B_p, B_r_prime, C_r, C_p, C_r_prime, P_r,
			P_p, P_r_prime, Q_r, Q_r_prime, Q_p;

	// 鐢ㄦ潵璁板綍Hessian鐭╅樀涓庡悜閲忎箻绉殑缂撳瓨鐭╅樀
	public static double[][] B_hp, C_hp, P_hp, Q_hp;

	// 璁板綍鐢ㄦ埛鍜岄」鐩畆ating鏁伴噺鐨勬暟缁�
	public static double[] user_Rating_count, item_Rating_count;

	// 鐗瑰緛缁存暟
	public static int featureDimension = 20;

	// 璁粌杞暟
	public static int trainingRound = 1000;

	// 鐗瑰緛鍒濆鍊�
	public static double init_Max = 0.004;

	public static double init_Scale = 0.004;

	public static int mapping_Scale = 1000;

	// 鎺у埗鍙傛暟
	public static double eta = 0.0001;

	public static double lambda = 0.01;

	public static double gama = 0.01;

	public static double tau = 0.001;

	public static double epsilon = 0.001;

	public static ArrayList<RTuple> trainData = null;

	public static ArrayList<RTuple> testData = null;

	public static int item_MaxID = 0, user_MaxID = 0;


	
	
	
	
	
	public abstract void train() throws IOException;

	public Common_function() throws NumberFormatException, IOException
	{
		initInstanceFeatures();
	}

	public static void initBiasSettings(boolean ifB, boolean ifC, int B_C,int C_C)
	{
		flag_B = ifB;
		flag_C = ifC;
		B_Count = B_C;
		C_Count = C_C;
		
		B_cache = new double[user_MaxID + 1][B_Count];
		min_B = new double[user_MaxID + 1][B_Count];

		C_cache = new double[item_MaxID + 1][C_Count];
		min_C = new double[item_MaxID + 1][C_Count];
		
		System.gc();

		if(B_Count!=0)
		{
			for (int i = 1; i <= user_MaxID; i++) 
			{
				double tempUB = B_Base[i]/B_Count;
				for (int j = 0; j < B_Count; j++) 
				{
					B_cache[i][j] = tempUB;
					min_B[i][j] = B_cache[i][j];
				}
			}
		}

		if(C_Count!=0)
		{
			for (int i = 1; i <= item_MaxID; i++) 
			{
				double tempIB = C_Base[i]/C_Count;
				for (int j = 0; j < C_Count; j++) 
				{
					C_cache[i][j] = tempIB;
					min_C[i][j] = C_cache[i][j];
				}
			}
		}
	}

	public static void initiStaticArrays() 
	{
		// 鍔�1鏄负浜嗗湪搴忓彿涓婁笌ID淇濇寔涓�鑷�
		user_Rating_count = new double[user_MaxID + 1];
		item_Rating_count = new double[item_MaxID + 1];

		P_cache = new double[user_MaxID + 1][featureDimension];
		Q_cache = new double[item_MaxID + 1][featureDimension];

		//min_P = new double[user_MaxID + 1][featureDimension];
		//min_Q = new double[item_MaxID + 1][featureDimension];

		B_Base = new double[user_MaxID + 1];
		C_Base = new double[item_MaxID + 1];

		// 鍒濆鍖栫壒寰佺煩闃�,閲囩敤闅忔満鍊�,浠庤�屽舰鎴愪竴涓狵闃堕�艰繎
		Random random = new Random(System.currentTimeMillis());
		for (int i = 1; i <= user_MaxID; i++)
		{
			user_Rating_count[i] = 0;
			int tempBB = random.nextInt(mapping_Scale);
			
			
//			鍐冲畾鍔犱笉鍔燽ias鐨勬儏鍐�
//			B_Base[i] = init_Max - init_Scale * tempBB / mapping_Scale;
			B_Base[i] = 0;//鏃燽ias鐨勬儏鍐碉紝鍒欒缃负0
			
						
			for (int j = 0; j < featureDimension; j++) 
			{
				int temp = random.nextInt(mapping_Scale);
				P_cache[i][j] = init_Max - init_Scale * temp / mapping_Scale;
				//min_P[i][j] = P_cache[i][j];
			}
		}
		for (int i = 1; i <= item_MaxID; i++)
		{
			item_Rating_count[i] = 0;
			int tempCB = random.nextInt(mapping_Scale);
			
			
//			鍐冲畾鍔犱笉鍔燽ias鐨勬儏鍐�
//			C_Base[i] = init_Max - init_Scale * tempCB / mapping_Scale;
			C_Base[i] = 0;//鏃燽ias鐨勬儏鍐碉紝鍒欒缃负0
			
			
			
			for (int j = 0; j < featureDimension; j++)
			{
				int temp = random.nextInt(mapping_Scale);
				Q_cache[i][j] = init_Max - init_Scale * temp / mapping_Scale;
				//min_Q[i][j] = Q_cache[i][j];

			}
		}

		for (RTuple tempRating : trainData) 
		{
			user_Rating_count[tempRating.iUserID] += 1;
			item_Rating_count[tempRating.iItemID] += 1;
		}
	}


	public void initInstanceFeatures() {
		B = new double[user_MaxID + 1][B_Count];
		C = new double[item_MaxID + 1][C_Count];
		P = new double[user_MaxID + 1][featureDimension];
		Q = new double[item_MaxID + 1][featureDimension];
		for (int i = 1; i <= user_MaxID; i++) {
			for (int j = 0; j < B_Count; j++) {
				B[i][j] = B_cache[i][j];
			}
			for (int j = 0; j < featureDimension; j++) {
				P[i][j] = P_cache[i][j];
			}
		}
		for (int i = 1; i <= item_MaxID; i++) {
			for (int j = 0; j < C_Count; j++) {
				C[i][j] = C_cache[i][j];
			}
			for (int j = 0; j < featureDimension; j++) {
				Q[i][j] = Q_cache[i][j];
			}
		}
	}

	public void cacheMinFeatures()
	{
		for (int i = 1; i <= user_MaxID; i++) 
		{
			for (int j = 0; j < B_Count; j++) 
			{
				min_B[i][j] = min_B[i][j];
			}
			for (int j = 0; j < featureDimension; j++)
			{
				min_P[i][j] = P[i][j];
			}
		}
		for (int i = 1; i <= item_MaxID; i++) 
		{
			for (int j = 0; j < C_Count; j++)
			{
				min_C[i][j] = C[i][j];
			}
			for (int j = 0; j < featureDimension; j++)
			{
				min_Q[i][j] = Q[i][j];
			}
		}
	}

	public static void initializeRatings(String trainFileName,String testFileName, String separator)
	             throws NumberFormatException, IOException 
	{
		// 鍔犲叆瀵筊atingMap鐨勫垵濮嬪寲
		initTrainData(trainFileName, separator);
		initTestData(testFileName, separator);
	}

	public static void initTrainData(String fileName, String separator)
			throws NumberFormatException, IOException {
		trainData = new ArrayList<RTuple>();
		File personSource = new File(fileName);
		BufferedReader in = new BufferedReader(new FileReader(personSource));

		String tempVoting;
		while (((tempVoting = in.readLine()) != null)) {
			StringTokenizer st = new StringTokenizer(tempVoting, separator);
			String personID = null;
			if (st.hasMoreTokens())
				personID = st.nextToken();
			String movieID = null;
			if (st.hasMoreTokens())
				movieID = st.nextToken();
			String personRating = null;
			if (st.hasMoreTokens())
				personRating = st.nextToken();
			int iUserID = Integer.valueOf(personID);
			int iItemID = Integer.valueOf(movieID);

			// 璁板綍涓嬫渶澶х殑itemid鍜寀serid锛涘洜涓篿temid鍜寀serid鏄繛缁殑锛屾墍浠ユ渶澶х殑itemid鍜寀serid涔熶唬琛ㄤ簡鍚勮嚜鐨勬暟鐩�
			user_MaxID = (user_MaxID > iUserID) ? user_MaxID : iUserID;
			item_MaxID = (item_MaxID > iItemID) ? item_MaxID : iItemID;
			double dRating = Double.valueOf(personRating);

			RTuple temp = new RTuple();
			temp.iUserID = iUserID;
			temp.iItemID = iItemID;
			temp.dRating = dRating;
			trainData.add(temp);
		}
		in.close();
	}

	public static void initTestData(String fileName, String separator)
			throws NumberFormatException, IOException {
		testData = new ArrayList<RTuple>();
		File personSource = new File(fileName);
		BufferedReader in = new BufferedReader(new FileReader(personSource));

		String tempVoting;
		while (((tempVoting = in.readLine()) != null)) {
			StringTokenizer st = new StringTokenizer(tempVoting, separator);
			String personID = null;
			if (st.hasMoreTokens())
				personID = st.nextToken();
			String movieID = null;
			if (st.hasMoreTokens())
				movieID = st.nextToken();
			String personRating = null;
			if (st.hasMoreTokens())
				personRating = st.nextToken();
			int iUserID = Integer.valueOf(personID);
			int iItemID = Integer.valueOf(movieID);

			// 璁板綍涓嬫渶澶х殑itemid鍜寀serid锛涘洜涓篿temid鍜寀serid鏄繛缁殑锛屾墍浠ユ渶澶х殑itemid鍜寀serid涔熶唬琛ㄤ簡鍚勮嚜鐨勬暟鐩�
			user_MaxID = (user_MaxID > iUserID) ? user_MaxID : iUserID;
			item_MaxID = (item_MaxID > iItemID) ? item_MaxID : iItemID;
		  
			
			
			double dRating = Double.valueOf(personRating);

			RTuple temp = new RTuple();
			temp.iUserID = iUserID;
			temp.iItemID = iItemID;
			temp.dRating = dRating;
			testData.add(temp);
		}
//	    System.out.println(item_MaxID);//output the maxID Number of User 
//	      System.out.println(user_MaxID);//output the maxID Number of User 
		in.close();
	}

	// 鍚戦噺鐐逛箻鍑芥暟
	public static double dotMultiply(double[] x, double[] y) {
		double sum = 0;
		for (int i = 0; i < x.length; i++) {
			sum += x[i] * y[i];
		}
		return sum;
	}

	// 鍚戦噺鍔�
	
	public static void VectorDivideVector(double[] first, double[] second,
			double[] result) {
		for (int i = 0; i < first.length; i++) {
			result[i] = first[i]/second[i];
		}
	}
	
	public static void VectorMutiplyVector(double[] first, double[] second,
			double[] result) {
		for (int i = 0; i < first.length; i++) {
			result[i] = first[i]*second[i];
		}
	}
	
	
	
	public static void vectorAdd(double[] first, double[] second,
			double[] result) {
		for (int i = 0; i < first.length; i++) {
			result[i] = first[i] + second[i];
		}
	}

	public static void vectorAdd(double[] first, double[] second) {
		for (int i = 0; i < first.length; i++) {
			first[i] = first[i] + second[i];
		}
	}

	// 鍚戦噺鍑�
	public static void vectorSub(double[] first, double[] second,
			double[] result) {
		for (int i = 0; i < first.length; i++) {
			result[i] = first[i] - second[i];
		}
	}

	public static void vectorSub(double[] first, double[] second) {
		for (int i = 0; i < first.length; i++) {
			first[i] = first[i] - second[i];
		}
	}

	// 鍚戦噺涔�
	public static void vectorMutiply(double[] vector, double time,
			double[] result) {
		for (int i = 0; i < vector.length; i++) {
			result[i] = vector[i] * time;
		}
	}

	public static double[] vectorMutiply(double[] vector, double time) {
		double[] result = new double[vector.length];
		for (int i = 0; i < vector.length; i++) {
			result[i] = vector[i] * time;
		}
		return result;
	}

	public static double[] initZeroVector() {
		double[] result = new double[featureDimension];
		for (int i = 0; i < featureDimension; i++)
			result[i] = 0;
		return result;
	}

	public double getMinPrediction(int userID, int itemID) {
		double ratingHat = 0;
		ratingHat += dotMultiply(min_P[userID], min_Q[itemID]);
		if (flag_B) {
			for (int j = 0; j < B_Count; j++) {
				ratingHat += min_B[userID][j];
			}
		}
		if (flag_C) {
			for (int j = 0; j < C_Count; j++) {
				ratingHat += min_C[itemID][j];
			}
		}
		return ratingHat;
	}

	public double getLocPrediction(int userID, int itemID) {
		double ratingHat = 0;
		ratingHat += dotMultiply(P[userID], Q[itemID]);
//		if (flag_B) {
//			for (int j = 0; j < B_Count; j++) {
//				ratingHat += B[userID][j];
//			}
//		}
//		if (flag_C) {
//			for (int j = 0; j < C_Count; j++) {
//				ratingHat += C[itemID][j];
//			}
//		}
		return ratingHat;
	}
	
	

	public static void getUseBias(double u,double[] b,int[] r,double c,double[] B)
	{ 
		for(int i = 1; i <= item_MaxID; i++)
		{
			 if(r[i] == 0)
			     b[i] = 0;
			 else
			  {
				 B[i] = (b[i] - u * r[i])/(r[i]+c);
			  } 			
		}	
	}
	public static void getItemBias(double u,double[] b,int[] r,double c,double[] sum,double[] B)
	{ 
		for(int j = 1; j <= user_MaxID; j++)
		{			
			 if(r[j] == 0)
			    b[j] = 0;		 
			 else	 
			 {
				 B[j] = (b[j] - u * r[j] - sum[j])/(r[j]+c);
			 }	
			 
		}
	}
	
	
	public int getMaxIndex(double[] array) {
		int temp = -1;
		double max = -1000;
		for (int i = 0; i < array.length - 1; i++) {
			if (max < array[i]) {
				max = array[i];
				temp = i;
			}
		}
		return temp;
	}

	public double testCurrentMAEu(double []Bi,double []Bu,double u) 
	{
		// 璁＄畻鍦ㄦ祴璇曢泦涓婄殑MAE
		double sumMAE = 0, sumCount = 0;
		for (RTuple tempTestRating : testData)
		{
			double actualRating = tempTestRating.dRating-u-Bu[tempTestRating.iUserID]-Bi[tempTestRating.iItemID];
			double ratinghat = this.getLocPrediction(tempTestRating.iUserID,
					tempTestRating.iItemID);

			sumMAE += Math.abs(actualRating - ratinghat);
			sumCount++;
		}
		double MAE = sumMAE / sumCount;
//		 System.out.println(sumCount);
		return MAE;
		 
		
	}

	
	public double testCurrentRMSEu(double []Bi,double []Bu,double u)
	{
		// 璁＄畻鍦ㄦ祴璇曢泦涓婄殑RMSE
		double sumRMSE = 0, sumCount = 0;
		for (RTuple tempTestRating : testData)
		{
			double actualRating = tempTestRating.dRating -u-Bu[tempTestRating.iUserID]-Bi[tempTestRating.iItemID];
			double ratinghat = this.getLocPrediction(tempTestRating.iUserID,
					tempTestRating.iItemID);

			sumRMSE += Math.pow((actualRating - ratinghat), 2);
			sumCount++;
		}
		double RMSE = Math.sqrt(sumRMSE / sumCount);
		return RMSE;
				
	}

	public double testCurrentSumCount(double []Bi,double []Bu,double u)	//娴嬭瘯鍦ㄦ祴璇曢泦涓婅绠楃殑鏁版嵁涓暟
	{
		// 璁＄畻鍦ㄦ祴璇曢泦涓婄殑RMSE
		double  sumCount = 0;
		for (RTuple tempTestRating : testData)
		{
			sumCount++;
		}
		return sumCount;
				
	}
	
	// according the "e:\\UserClassId.txt" to split the original Dataset
    public static int  userclassid[]; //initialize the userclassid[],recording the each cluster's No and their counts
		public static int[] SplitDataset(int TypeOfClustering) throws NumberFormatException, IOException /// return each cluster's  counts
	{
        
			  FileReader reader = new FileReader("e:\\UserClassId.txt");   //è®°å½•Userå±žäºŽçš„ç±»åˆ«æ ‡ç­¾
		      BufferedReader br = new BufferedReader(reader);   
		      String ClassLabel = null;   
		      
		      while((ClassLabel = br.readLine()) != null) 
		      {   

		    	  String newString = ClassLabel.replaceAll("\t", "");
	
		    	  char[] chars = newString.toCharArray();
		  
		    	  //int[] userclassid;
		    	  userclassid=new int[newString.length()];
                    
		      	   for(int j=0;j<chars.length;j++)
		      	   {                       
		      		 userclassid[j] = (int)chars[j]-48;
		      	     //System.out.println(userclassid[j] + "\t"); 
			       } 		   	  
		       }
		                 
            br.close();    
 
			
	        File file1 = new File("E:\\train1.txt");  //å­˜æ”¾æ•°ç»„æ•°æ�®çš„æ–‡ä»¶ 
			File file2 = new File("E:\\train2.txt");  //å­˜æ”¾æ•°ç»„æ•°æ�®çš„æ–‡ä»¶  
			File file3 = new File("E:\\train3.txt"); 
			File file4 = new File("E:\\train4.txt");
			FileWriter out1 = new FileWriter(file1);  //æ–‡ä»¶å†™å…¥æµ�
			FileWriter out2 = new FileWriter(file2);  //æ–‡ä»¶å†™å…¥æµ�
			FileWriter out3 = new FileWriter(file3);  //æ–‡ä»¶å†™å…¥æµ�
	   		FileWriter out4 = new FileWriter(file4);  //æ–‡ä»¶å†™å…¥æµ�
	   		File file5 = new File("E:\\test1.txt");  //å­˜æ”¾æ•°ç»„æ•°æ�®çš„æ–‡ä»¶ 
    		File file6 = new File("E:\\test2.txt");  //å­˜æ”¾æ•°ç»„æ•°æ�®çš„æ–‡ä»¶  
    		File file7 = new File("E:\\test3.txt"); 
    		File file8 = new File("E:\\test4.txt");
    		FileWriter out5 = new FileWriter(file5);  //æ–‡ä»¶å†™å…¥æµ�
    		FileWriter out6 = new FileWriter(file6);  //æ–‡ä»¶å†™å…¥æµ�
    		FileWriter out7 = new FileWriter(file7);  //æ–‡ä»¶å†™å…¥æµ�
    		FileWriter out8 = new FileWriter(file8); 
	          
    
    		
	         for (RTuple tempRating : trainData)// å¦‚æžœæ˜¯æµ‹è¯•é›†å°±æ�¢æˆ� testData
			{
	        	 
	       
			if (TypeOfClustering==1){
				if(userclassid[tempRating.iUserID-1]==1)// å¦‚æžœæ˜¯Itemå°±æ�¢æˆ� iItemID
       	         {
			    
	    		  out1.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
	    		  out1.write("\r\n");
	              }
  	 
  	            else if(userclassid[tempRating.iUserID-1]==2)
  	                  {
    		            out2.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
    		            out2.write("\r\n");
                      }        
  		 		  else if(userclassid[tempRating.iUserID-1]==3)
  	                  {
  		               out3.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
  		               out3.write("\r\n");
   	                  }   
  	          		 else 
  	                        {
	    		            out4.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
	    		           out4.write("\r\n");
                            }    	      			     	 	
	        	 	}
			if (TypeOfClustering!=1){
				if(userclassid[tempRating.iItemID-1]==1)// å¦‚æžœæ˜¯Itemå°±æ�¢æˆ� iItemID
       	         {
			    
	    		  out1.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
	    		  out1.write("\r\n");
	              }
  	 
  	            else if(userclassid[tempRating.iItemID-1]==2)
  	                  {
    		            out2.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
    		            out2.write("\r\n");
                      }        
  		 		  else if(userclassid[tempRating.iItemID-1]==3)
  	                  {
  		               out3.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
  		               out3.write("\r\n");
   	                  }   
  	          		 else 
  	                        {
	    		            out4.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
	    		           out4.write("\r\n");
                            }    	      			     	 	
	        	 	}
				}

	         
	         
	         int cluster1=0, cluster2=0, cluster3=0, cluster4=0;
	         for (RTuple tempRating : testData)// å¦‚æžœæ˜¯æµ‹è¯•é›†å°±æ�¢æˆ� testData
				{
		        	 
		       
				if (TypeOfClustering==1){
					if(userclassid[tempRating.iUserID-1]==1)// å¦‚æžœæ˜¯Itemå°±æ�¢æˆ� iItemID
	       	         {
				    
		    		  out5.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
		    		  out5.write("\r\n");
		    		  cluster1=cluster1+1;
		    		  
		              }
	  	 
	  	            else if(userclassid[tempRating.iUserID-1]==2)
	  	                  {
	    		            out6.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
	    		            out6.write("\r\n");
	    		            cluster2=cluster2+1;
	                      }        
	  		 		  else if(userclassid[tempRating.iUserID-1]==3)
	  	                  {
	  		               out7.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
	  		               out7.write("\r\n");
	  		             cluster3=cluster3+1;
	   	                  }   
	  	          		 else 
	  	                        {
		    		            out8.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
		    		           out8.write("\r\n");
		    		           cluster4=cluster4+1;
	                            }    	      			     	 	
		        	 	}
				if (TypeOfClustering!=1){
					if(userclassid[tempRating.iItemID-1]==1)// å¦‚æžœæ˜¯Itemå°±æ�¢æˆ� iItemID
	       	         {
				    
		    		  out5.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
		    		  out5.write("\r\n");
		    		  cluster1=cluster1+1;
		              }
	  	 
	  	            else if(userclassid[tempRating.iItemID-1]==2)
	  	                  {
	    		            out6.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
	    		            out6.write("\r\n");
	    		            cluster2=cluster2+1;
	                      }        
	  		 		  else if(userclassid[tempRating.iItemID-1]==3)
	  	                  {
	  		               out7.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
	  		               out7.write("\r\n");
	  		              cluster3=cluster3+1;
	   	                  }   
	  	          		 else 
	  	                        {
		    		            out8.write(tempRating.iUserID +"::"+ tempRating.iItemID +"::"+ tempRating.dRating);
		    		           out8.write("\r\n");
		    		           cluster4=cluster4+1;
	                            }    	      			     	 	
		        	 	}
					}
     
 	            out1.close();
 			    out2.close();
 			    out3.close();
 			    out4.close();
                out5.close();
        		out6.close();
        		out7.close();
        		out8.close(); 
        		
        		int [] cluster_result= new int [4];
        	    cluster_result[0]=cluster1;
        	    cluster_result[1]=cluster2;
        	    cluster_result[2]=cluster3;
        	    cluster_result[3]=cluster4;
        	    return cluster_result;    }
		// Over: according the "e:\\UserClassId.txt" to split the original Dataset
	
		public double testCheck(double []Bi,double []Bu,double u) 
		{
			double sumMAE = 0; 
			for (RTuple tempTestRating : trainData)
			{
				double actualRating = tempTestRating.dRating-u-Bu[tempTestRating.iUserID]-Bi[tempTestRating.iItemID];
				double ratinghat = this.getLocPrediction(tempTestRating.iUserID,
						tempTestRating.iItemID);

//				sumMAE += (actualRating - ratinghat)*(actualRating - ratinghat);
				sumMAE += Math.abs(actualRating - ratinghat);
				
			}

			return sumMAE;			
		}
		
		public static double vectorL1(double[] input) {
			double sum = 0;
			for (int i = 0; i < input.length; i++) 
			{			
				sum+= Math.abs(input[i]) ;
			}
			return sum;
		}

		
	    
}

	
	
	
	
	
	
	
