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

import org.apache.spark.SparkConf;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.FeatureHasher;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
//import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
//import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
//import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

/**
 *
 * @author gavalian
 */
public class DCTrackClassifier {
    
    SparkSession spark;
    Dataset<Row> dataFrame;
    
    public DCTrackClassifier(){
        
    }
    
    public void initSpark(String application){
        spark = SparkSession.builder().appName("DCML").master("local[4]").getOrCreate();
        //Logger.getLogger("all").setLevel(Level.OFF);
        //Logger.getLogger("info").setLevel(Level.OFF);
        org.apache.log4j.Logger.getLogger("org").setLevel(org.apache.log4j.Level.OFF);
        org.apache.log4j.Logger.getLogger("akka").setLevel(org.apache.log4j.Level.OFF);
        spark.sparkContext().setLogLevel("ERROR");
        System.out.println("LOG NAME : " + spark.logName());
        org.apache.log4j.Logger.getLogger("org.apache.spark.sql.SparkSession").setLevel(org.apache.log4j.Level.OFF);
        org.apache.log4j.Logger.getLogger("o.s.j.server.handler.ContextHandler").setLevel(org.apache.log4j.Level.ERROR);
        spark.log().info("MAKING AN INFOR STATEMENT");
        System.out.println("DEBUG : " + spark.log().isDebugEnabled());
        System.out.println("ERROR : " + spark.log().isErrorEnabled());
        System.out.println("INFO  : " + spark.log().isInfoEnabled());
        System.out.println("WARN  : " + spark.log().isWarnEnabled());
        System.out.println("TRACE : " + spark.log().isTraceEnabled());
    }
    
    public void initData(String dataFile){
        dataFrame = spark.read().format("libsvm").load(dataFile);
        dataFrame.show();
        
        List<Row> dataList = dataFrame.takeAsList(12);
        for(int i = 0; i < 12; i++){
            Double value = (Double) dataList.get(i).get(0);
            System.out.println("  VALUE = " + value);
        }
    }
    
    public void createFeatures(){
        int[] data = new int[100];
        Row label = RowFactory.create(1,data);
        //Row   row = RowFactory.create(data);
        /*FeatureHasher hasher = new FeatureHasher()
                .setInputCols(new String[]{"real", "bool", "stringNum", "string"})
                .setOutputCol("features");*/
        Integer value = (Integer) label.get(0);
        System.out.println("VALUE = " + value);
        
    }
    
    public void trainDecisionTree(){
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4)
                .fit(dataFrame);
        // Split the data into training and test sets (30% held out for testing).
        Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];
        // Train a DecisionTree model.
        DecisionTreeRegressor dt = new DecisionTreeRegressor()
                .setFeaturesCol("indexedFeatures");
        
        // Chain indexer and tree in a Pipeline.
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{featureIndexer, dt});

        // Train model. This also runs the indexer.
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions.
        Dataset<Row> predictions = model.transform(testData);
        predictions.select("label", "features").show(5);

        // Select (prediction, true label) and compute test error.
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        double rmse = evaluator.evaluate(predictions);
        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);

        DecisionTreeRegressionModel treeModel =
                (DecisionTreeRegressionModel) (model.stages()[1]);
        System.out.println("Learned regression tree model:\n" + treeModel.toDebugString());
    }
    public void trainLinearRegression(){
        LogisticRegression lr = new LogisticRegression()
        .setMaxIter(10)
        .setRegParam(0.3)
        .setElasticNetParam(0.8);

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

         // for multiclass, we can inspect metrics on a per-label basis
         System.out.println("False positive rate by label:");
         int i = 0;
         double[] fprLabel = trainingSummary.falsePositiveRateByLabel();
         for (double fpr : fprLabel) {
             System.out.println("label " + i + ": " + fpr);
             i++;
         }

         System.out.println("True positive rate by label:");
         i = 0;
         double[] tprLabel = trainingSummary.truePositiveRateByLabel();
         for (double tpr : tprLabel) {
             System.out.println("label " + i + ": " + tpr);
             i++;
         }
         
         System.out.println("Precision by label:");
         i = 0;
         double[] precLabel = trainingSummary.precisionByLabel();
         for (double prec : precLabel) {
             System.out.println("label " + i + ": " + prec);
             i++;
         }
         
         System.out.println("Recall by label:");
         i = 0;
         double[] recLabel = trainingSummary.recallByLabel();
         for (double rec : recLabel) {
             System.out.println("label " + i + ": " + rec);
             i++;
         }
         
         System.out.println("F-measure by label:");
         i = 0;
         double[] fLabel = trainingSummary.fMeasureByLabel();
         for (double f : fLabel) {
             System.out.println("label " + i + ": " + f);
             i++;
         }
         
         double accuracy = trainingSummary.accuracy();
         double falsePositiveRate = trainingSummary.weightedFalsePositiveRate();
         double truePositiveRate = trainingSummary.weightedTruePositiveRate();
         double fMeasure = trainingSummary.weightedFMeasure();
         double precision = trainingSummary.weightedPrecision();
         double recall = trainingSummary.weightedRecall();
         System.out.println("Accuracy: " + accuracy);
         System.out.println("FPR: " + falsePositiveRate);
         System.out.println("TPR: " + truePositiveRate);
         System.out.println("F-measure: " + fMeasure);
         System.out.println("Precision: " + precision);
         System.out.println("Recall: " + recall);
    }
    public static void example(){
        SparkSession spark;
        SparkConf conf = new SparkConf();
        conf.setMaster("local[2]");
        spark = SparkSession.builder().appName("DCML").master("local").getOrCreate();
        // Load training data
        
        
        //String path = "dc_training_svm.txt";
        String path = "simple_data.txt";
        Dataset<Row> dataFrame = spark.read().format("libsvm").load(path);
        
// Split the data into train and test
Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.6, 0.4}, 1234L);
Dataset<Row> train = splits[0];
Dataset<Row> test = splits[1];

// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 3 (classes)

int[] layers = new int[] {4, 5, 4, 3};

// create the trainer and set its parameters
MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
        .setLayers(layers)
        .setBlockSize(128)
        .setSeed(1234L)
        .setMaxIter(1000);

//System.out.println();

// train the model
System.out.println("\n\n\n\n\n========= START TRAINING");
MultilayerPerceptronClassificationModel model = trainer.fit(train);
System.out.println("\n\n\n\n\n========= END TRAINING");

// compute accuracy on the test set
Dataset<Row> result = model.transform(test);
Dataset<Row> predictionAndLabels = result.select("prediction", "label");
MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
        .setMetricName("accuracy");

System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));

    }
    
    public static void read(String file){
        MultilayerPerceptronClassificationModel model = MultilayerPerceptronClassificationModel.load(file);
        model.toString();
        
        
        
    }
    
    
    
    
    public static void main(String[] args){
        
        
        DCTrackClassifier dcCL = new DCTrackClassifier();
        dcCL.initSpark("CLASSIFIER");
        dcCL.initData("dc_training_svm_full.txt");
        //dcCL.trainDecisionTree();
        //dcCL.trainLinearRegression();
        dcCL.createFeatures();
        /*
        SparkSession spark;
        SparkConf conf = new SparkConf();
        conf.setMaster("local[2]");
        spark = SparkSession.builder().appName("DCML").master("local").getOrCreate();
        // Load training data
        
        
        String path = "dc_training_svm_full.txt";
        //String path = "simple_data.txt";
        Dataset<Row> dataFrame = spark.read().format("libsvm").load(path);
        
// Split the data into train and test
Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.6, 0.4}, 1234L);
Dataset<Row> train = splits[0];
Dataset<Row> test = splits[1];

// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 3 (classes)

int[] layers = new int[] {4032, 256, 2};

// create the trainer and set its parameters
MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
        .setLayers(layers)
        .setBlockSize(128)
        .setSeed(1234L)
        .setStepSize(0.1)
        .setMaxIter(10);

//System.out.println();

// train the model
System.out.println("\n\n\n\n\n========= START TRAINING");
MultilayerPerceptronClassificationModel model = trainer.fit(train);
System.out.println("\n\n\n\n\n========= END TRAINING");

// compute accuracy on the test set
Dataset<Row> result = model.transform(test);
Dataset<Row> predictionAndLabels = result.select("prediction", "label");
MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
        .setMetricName("accuracy");
System.out.println("\n\n\n***********************************************\n");
System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));
System.out.println("***********************************************\n");
        try {
            model.save("spack_dc_tracking.nnet");
        } catch (IOException ex) {
            Logger.getLogger(DCTrackClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        DCTrackClassifier.read("spack_dc_tracking.nnet");
    }
    
    //DCTrackClassifier.read ("spack_dc_tracking.nnet");
    //DCTrackClassifier.examples();*/
    }
}
