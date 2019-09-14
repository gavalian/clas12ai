/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.jnp.spark;

import java.io.IOException;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 *
 * @author gavalian
 */
public class DCTrackFitter {
    
    MultilayerPerceptronClassificationModel model;
    
    public DCTrackFitter(){
        
    }
    public void loadModel(String filename){
        DCSparkSession.getInstance();
        model = MultilayerPerceptronClassificationModel.load(filename);
    }
    
    public void decide(Dataset<Row> dataFrame){
        Dataset<Row> result = model.transform(dataFrame);
        List<Row> resultsList = result.collectAsList();
        result.show(200, false);
        int counter = 0;
        int counterZero = 0;
        int counterNonZero = 0;
        for ( int loop = 0; loop < resultsList.size(); loop++){
            Double value = resultsList.get(loop).getDouble(0);
            Double label = resultsList.get(loop).getDouble(4);
            counter++;
            if(label>0.5){
                counterNonZero++;
            } else {
                counterZero++;
            }
            System.out.println(" prediction = " + value + "  label = " + label);
            
        }
        System.out.println("counter = " + counter + " zero = " + counterZero + " non-zero = " + counterNonZero);
    }
    
    public void loadModel(String filename, Dataset<Row> dataFrame){
        model = MultilayerPerceptronClassificationModel.load(filename);
        Dataset<Row> result = model.transform(dataFrame);
        List<Row> resultsList = result.collectAsList();
        result.show(200, false);
        int counter = 0;
        int counterZero = 0;
        int counterNonZero = 0;
        for ( int loop = 0; loop < resultsList.size(); loop++){
            Double value = resultsList.get(loop).getDouble(0);
            Double label = resultsList.get(loop).getDouble(4);
            counter++;
            if(label>0.5){
                counterNonZero++;
            } else {
                counterZero++;
            }
            System.out.println(" prediction = " + value + "  label = " + label);
            
        }
        System.out.println("counter = " + counter + " zero = " + counterZero + " non-zero = " + counterNonZero);
    }
    
    public void fit(Dataset<Row> dataFrame){
        
        Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.6, 0.4}, 561234L);
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];
        
        int[] layers = new int[] {12, 256, 256, 2};
        MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setBlockSize(128)
                .setSeed(561234L)
                .setStepSize(0.01)
                .setMaxIter(1000);
        
        MultilayerPerceptronClassificationModel model = trainer.fit(train);
        Dataset<Row> result = model.transform(test);
        Dataset<Row> predictionAndLabels = result.select("prediction", "label");
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy");
        System.out.println("\n\n\n***********************************************\n");
        System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));
        System.out.println("***********************************************\n");
        
        try {
            model.save("DCML.nnet");
        } catch (IOException ex) {
            Logger.getLogger(DCTrackFitter.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
    
    public static void main(String[] args){
        DCTrackFitter fitter = new DCTrackFitter();
        DatasetUtils  utils  = new DatasetUtils();
        
        Dataset<Row> dataFrame = utils.createRandom(200);
        
        fitter.fit(dataFrame);
    }
}
