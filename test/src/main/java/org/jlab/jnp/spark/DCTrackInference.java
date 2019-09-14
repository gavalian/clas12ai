/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.jnp.spark;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 *
 * @author gavalian
 */
public class DCTrackInference {
    SparkSession spark;
    MultilayerPerceptronClassificationModel model;
    Dataset<Row> dataFrame;
    
    public DCTrackInference(){
        
    }
    
    public void initSession(){

        spark = SparkSession.builder().appName("DCML").master("local[4]").getOrCreate();
        //Logger.getLogger("all").setLevel(Level.OFF);
        //Logger.getLogger("info").setLevel(Level.OFF);
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        spark.sparkContext().setLogLevel("ERROR");
    }
    
    public void loadModel(String filename){
        model = MultilayerPerceptronClassificationModel.load(filename);
        model.toString();
    }
    
    public void loadData(String filename){
        dataFrame = spark.read().format("libsvm").load(filename);
    }
    
    public void evaluate(){
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setMetricName("accuracy");
        Long start_time = System.currentTimeMillis();
        Dataset<Row> result = model.transform(dataFrame);
        for(int i = 0; i < 10; i++){
            if(i%100==0){
                Long currentTime = System.currentTimeMillis() - start_time;
                System.out.printf("%3d : %12d\n",i,currentTime);
            }
            result = model.transform(dataFrame);
        }
        //DataSetResult dataSetResult = SparkUtils.getDataSetResult(result);
        //Long end_time = System.currentTimeMillis();
        //Dataset<Row> predictionAndLabels = result.select("prediction", "label");            
        //double value = evaluator.evaluate(predictionAndLabels);
        
        //Long end_time = System.currentTimeMillis();
        //Long time = end_time - start_time;
        //System.out.println(" RESULT REACHED at " + time + " ms");
        
       /* predictionAndLabels.show();
        Object obj = predictionAndLabels.select("prediction").rdd().collect();
        System.out.println(obj);
        System.out.println( "LABEL = " +  predictionAndLabels.col("label").gt(0.5));
        System.out.println(" EVALUATE = " + value);*/
    }
    
    public static void main(String[] args){
        DCTrackInference infer = new DCTrackInference();
        infer.initSession();
        infer.loadModel("spack_dc_tracking.nnet");
        infer.loadData("evaluator_50.txt");
        infer.evaluate();
    }
}
