/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.jnp.spark;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 *
 * @author gavalian
 */
public class LogisticRegressionTraining {
    public LogisticRegressionTraining(){
        
    }
    
    public void fit(Dataset<Row> dataFrame){
        LogisticRegression lr = new LogisticRegression()
        .setMaxIter(100)
        .setRegParam(0.00003)
        .setElasticNetParam(0.08);

     // Fit the model
     LogisticRegressionModel lrModel = lr.fit(dataFrame);

     // Print the coefficients and intercept for multinomial logistic regression
     System.out.println("Coefficients: \n"
             + lrModel.coefficientMatrix() + " \nIntercept: " + lrModel.interceptVector());
     LogisticRegressionTrainingSummary trainingSummary = lrModel.summary();
     
     // Obtain the loss per iteration.
     double[] objectiveHistory = trainingSummary.objectiveHistory();
     for (double lossPerIteration : objectiveHistory) {
         System.out.println(lossPerIteration);
     }
    }
}
