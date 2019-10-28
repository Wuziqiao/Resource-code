package model;

import java.io.IOException;
import java.io.FileWriter;
import java.io.File;


public class Main_Function extends Common_function 
{   
	
	public Main_Function() throws NumberFormatException, IOException
	{
		super();
	}
	

	
	int m=1;

	   double alpha=0.5;//control the L1-norm and L2-norm

	public void train() throws IOException
	{		
         
		double Globalaverage=0;         
		double[] Bi = new double[item_MaxID+1];    
		double[] Bu = new double[user_MaxID+1];        
		
		double[] tempPu = new double[featureDimension];
		double[] tempDeltaPu = new double[featureDimension];
		double[] tempPu1 = new double[featureDimension];
		double[] tempQi = new double[featureDimension];
		double[] tempQi1 = new double[featureDimension];
		double[] tempDeltaQi = new double[featureDimension];
  
			
			double SumLossCumulativeL1=0; double SumLossCumulativeL2=0; 
		for (int round = 1; round <= trainingRound; round++) 
		{
				


				for (RTuple tempRating : trainData)   
			{
				
		        double  ratingHat= tempRating.dRating;
				double rPrediction = this.getLocPrediction(tempRating.iUserID,tempRating.iItemID);
				double err = ratingHat-rPrediction; 
				
							
				
				if(err>0)///////open absolute err>0
				{	
					vectorMutiply(P[tempRating.iUserID], (1 - eta * lambda), tempPu); //eta_u
					vectorMutiply(P[tempRating.iUserID], err * eta*(1-alpha), tempDeltaPu);//eta_i
					
					vectorMutiply(Q[tempRating.iItemID], (1 - eta * lambda), tempQi);//eta_i
					vectorMutiply(Q[tempRating.iItemID], err * eta*(1-alpha), tempDeltaQi);//eta_u

					vectorAdd(tempPu, tempDeltaQi, tempPu1);
					vectorAdd(tempQi, tempDeltaPu, tempQi1);
					
					vectorMutiply(Q[tempRating.iItemID], alpha*eta, tempQi);//eta_u
					vectorMutiply(P[tempRating.iUserID], alpha*eta, tempPu);//eta_i
					
					vectorAdd(tempPu1, tempQi, P[tempRating.iUserID]);
					vectorAdd(tempQi1, tempPu, Q[tempRating.iItemID]);
				}
				
				if(err<0)///////open absolute err>0
				{				
					vectorMutiply(P[tempRating.iUserID], (1 - eta * lambda), tempPu);//eta_u
					vectorMutiply(P[tempRating.iUserID], err * eta*(1-alpha), tempDeltaPu);//eta_i
					
					vectorMutiply(Q[tempRating.iItemID], (1 - eta * lambda), tempQi);//eta_i
					vectorMutiply(Q[tempRating.iItemID], err * eta*(1-alpha), tempDeltaQi);//eta_u

					vectorAdd(tempPu, tempDeltaQi, tempPu1);
					vectorAdd(tempQi, tempDeltaPu, tempQi1);
					
					vectorMutiply(Q[tempRating.iItemID], alpha*eta, tempQi);//eta_u
					vectorMutiply(P[tempRating.iUserID], alpha*eta, tempPu);//eta_i
										
					vectorSub(tempPu1, tempQi, P[tempRating.iUserID]);
					vectorSub(tempQi1, tempPu, Q[tempRating.iItemID]);
					
				}
		
			}//end for traindata on L1

				
			///adaptively updating the alpha1
			double SumLoss1 = 0; //L1 loss sum: update the alpha
			double SumLoss2 = 0; //L2 loss sum: update the alpha

	         for (RTuple tempRating : trainData)   /// 
					{
                    double  ratingHat= tempRating.dRating;
					double rPrediction = this.getLocPrediction(tempRating.iUserID,tempRating.iItemID);
					double Frobenius_err=(ratingHat-rPrediction);
					SumLoss1+=Math.abs(Frobenius_err);
					SumLoss2+=Math.pow(Frobenius_err,2);
					}

	         
//	     /////*****************Alpha=Cumulative Loss, Sum First; 
	             SumLossCumulativeL1=SumLossCumulativeL1+SumLoss1;
		         SumLossCumulativeL2=SumLossCumulativeL2+SumLoss2;
		         double SumLoss1_R=SumLossCumulativeL1/(SumLossCumulativeL1+SumLossCumulativeL2);
		         double SumLoss2_R=SumLossCumulativeL2/(SumLossCumulativeL1+SumLossCumulativeL2);
		         alpha=Math.exp(-SumLoss1_R)/(Math.exp(-SumLoss1_R)+Math.exp(-SumLoss2_R));
//	    /////*****************Alpha=Cumulative Loss, Sum First; 	         

					// ////////testing////////////
					double curErr;double curErr1;
                    curErr = this.testCurrentMAEu(Bi,Bu,Globalaverage);
					
					if (min_Error_MAE > curErr) 
					{
						min_Error_MAE = curErr;
						this.min_Round = round;
					} 
					else if ((round - this.min_Round) >= delayCount)
					{
						break;
					}
	
                    curErr1 = this.testCurrentRMSEu(Bi,Bu,Globalaverage);
					if (min_Error_RMSE > curErr1) 
					{
						min_Error_RMSE = curErr1;
						this.min_Round = round;
					} 
					else if ((round - this.min_Round) >= delayCount)
					{
						break;
					}
					// ////////testing////////////	
		
		
		}
	 		
}

     ////////////////////////main function//////////////////////////////////
	public static void main(String[] argv) throws NumberFormatException,IOException
	{
		    Common_function.initializeRatings("./Jester_train.txt","./Jester_test.txt", "::");
			Common_function.eta =0.001;
			Common_function.lambda = 0.03;
			Common_function.trainingRound = 1000;
			Common_function.delayCount = 20;
			Common_function.featureDimension =20;//
			Common_function.initiStaticArrays();
			      min_Error_RMSE = 1e10;
			      min_Error_MAE= 1e10;
			      Common_function.initBiasSettings(true, true, 1, 1);
			      Main_Function L3F = new Main_Function();   
			      long startTime=System.currentTimeMillis();

			      L3F.train();
		
			      long endTime = System.currentTimeMillis();
			      
			      double seconds = (endTime - startTime) / 1000F;
			      System.out.println("time: "+seconds + "\t");
			      
			      System.out.println("eta"+eta+ "\t" +"lambda"+lambda+ "\t" +"RMSE: "+min_Error_RMSE+"\t" +"MAE: "+min_Error_MAE);


	}
}























