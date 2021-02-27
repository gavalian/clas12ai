package org.crtc_jlab.path.denoise2d.models;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.crtc_jlab.path.denoise2d.dataset.LsvmDataSet;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.writer.impl.misc.SVMLightRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.CollectScoresListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class AbstractCnnDenoisingAutoEncoder {
	protected int width, height, channels;
	protected static long seed = 123L;
	
	protected MultiLayerNetwork model;
	
	public AbstractCnnDenoisingAutoEncoder() {
		width = 112;
		height = 36;
		channels = 1;
	}
	

	public abstract void buildModel();
	
	public void loadModel(String pathToModel) throws IOException {

		model = MultiLayerNetwork.load(new File(pathToModel), true);
	}
	
	public void saveModel(String pathToModel) throws IOException {
		System.out.println("Saving model in " + pathToModel);
		model.save(new File(pathToModel));
	}
	
	public void train(LsvmDataSet trainSet, int nEpochs) {

		System.out.println("Starting training for " + nEpochs + " epochs...");
		model.setListeners(new ScoreIterationListener(1));
//		model.setListeners(new CollectScoresListener(1));
		
		int iteration = 0;
		int numIters = trainSet.getNumBatches();
        for( int epoch=0; epoch<nEpochs; epoch++ ){
        	System.out.println("Starting epoch: " + (epoch + 1) + ". Number of batches: " + numIters);
        	iteration = 0;
        	while(trainSet.hasNext()){
        		System.out.println("Epoch " + (epoch + 1) + "/" + nEpochs + " | Iteration  " + (iteration + 1) + "/" + numIters);
        		INDArray features = trainSet.getNextFeaturesBatch();
        		INDArray labels = trainSet.getNextLabelsBatch();
//        		System.out.println(labels);
        		model.fit(features, labels);
//        		System.out.println(model.output(features));
//        		System.out.println(features.shapeInfoToString());
        		
            	iteration ++;
        	}
            System.out.println("Epoch " + (epoch + 1) + " complete");
            trainSet.resetIterator();
        }
        System.out.println("Done");
	}
	
	public void test(LsvmDataSet testSet) {
	 	RegressionEvaluation eval = new RegressionEvaluation();
		System.out.println("Starting testing");
		int iteration = 0;
		int numIters = testSet.getNumBatches();
    	System.out.println("Number of batches: " + numIters);
    	iteration = 0;
    	while(testSet.hasNext()){
    		System.out.println("Iteration  " + (iteration + 1) + "/" + numIters);
        	INDArray predictions = model.output(testSet.getNextFeaturesBatch());
        	eval.eval(testSet.getNextLabelsBatch(), predictions);
        	iteration ++;
    	}
    	
    	System.out.println(eval.stats());
	}
	
	public void predict(LsvmDataSet predictSet, String resultsPath) throws Exception {
		System.out.println("Starting predictions");
		int iteration = 0;
		int numIters = predictSet.getNumBatches();
    	System.out.println("Number of batches: " + numIters);
    	iteration = 0;
    	
    	Configuration configWriter = new Configuration();
    	configWriter.setInt(SVMLightRecordWriter.FEATURE_FIRST_COLUMN, 0);
    	configWriter.setInt(SVMLightRecordWriter.FEATURE_FIRST_COLUMN, 0);
    	File inputFile = new File(resultsPath+"predict_report.txt");
     	
    	SVMLightRecordWriter writer = new SVMLightRecordWriter();
    	writer.initialize(configWriter, new FileSplit(inputFile), new NumberOfRecordsPartitioner());;
    	while(predictSet.hasNext()){
    		System.out.println("Iteration  " + (iteration + 1) + "/" + numIters);
    		INDArray features = predictSet.getNextFeaturesBatch();
    		INDArray labels = predictSet.getNextLabelsBatch();
        	INDArray predictions = model.output(features);
        	iteration ++;
        	for(int i = 0; i< features.size(0); i++)
        	{
        		List<Writable> record = Arrays.asList(new NDArrayWritable( labels.slice(i).reshape(4032)), new DoubleWritable(2));
        		writer.write(record);
        		record = Arrays.asList(new NDArrayWritable( predictions.slice(i).reshape(4032)), new DoubleWritable(1));
        		writer.write(record);
        		record = Arrays.asList(new NDArrayWritable( features.slice(i).reshape(4032)), new DoubleWritable(0));
        		writer.write(record);
        	}
    	}
    	writer.close();
    	
    	System.out.println("Done");
	}

}
