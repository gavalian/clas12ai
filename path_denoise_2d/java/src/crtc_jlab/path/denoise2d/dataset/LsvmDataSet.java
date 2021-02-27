package org.crtc_jlab.path.denoise2d.dataset;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.impl.misc.SVMLightRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class LsvmDataSet {
	
	private SVMLightRecordReader recordReader;
	private RecordReaderDataSetIterator it;
	private int batchSize, numPossibleLabels;
	private List<INDArray> features;
	public Iterator<INDArray> featuresIterator;
	private List<INDArray> labels;
	public Iterator<INDArray> labelsIterator;

	public LsvmDataSet(String pathToLsvmFile, int numFeatures, int numLabels, boolean zeroBasedIndexing) throws IOException, InterruptedException {
		
		File f = new File(pathToLsvmFile);
		recordReader = new SVMLightRecordReader();
		Configuration conf = new Configuration();
		conf.setInt(SVMLightRecordReader.class.getName()+".numfeatures", numFeatures);
		conf.setInt(SVMLightRecordReader.class.getName()+".numLabels", numLabels);
		conf.setBoolean(SVMLightRecordReader.class.getName()+".zeroBasedIndexing", zeroBasedIndexing);
		recordReader.initialize(conf, new FileSplit(f));
//		
	}
	
	public void readData(int batchSize, int numPossibleLabels, int[] inputLabels, int[] targetLabels, long[] dataShape) {
		this.batchSize = batchSize;
		this.numPossibleLabels = numPossibleLabels;
		it = new RecordReaderDataSetIterator(recordReader, this.batchSize * 2, -1, this.numPossibleLabels);
		features = new ArrayList<>();
		labels = new ArrayList<>();
		
		
		while(it.hasNext()){
			DataSet data = it.next();
			DataSet noisy_input = data.filterBy(inputLabels);
			DataSet ground_truth = data.filterBy(targetLabels);
			dataShape[0] = noisy_input.getFeatures().size(0);
			features.add(noisy_input.getFeatures().reshape(dataShape));
			labels.add(ground_truth.getFeatures().reshape(dataShape));
//			labels.add(ground_truth.getFeatures());
		}
		
		featuresIterator = features.iterator();
		labelsIterator = labels.iterator();
	}
	
	public boolean hasNext() {
		
		return featuresIterator.hasNext() && labelsIterator.hasNext();
	}
	
	public int getNumBatches() {
		
		return features.size();
	}
	
	public INDArray getNextFeaturesBatch() {
		
		return featuresIterator.next();
	}
	
	public INDArray getNextLabelsBatch() {
		
		return labelsIterator.next();
	}
	
	public void resetIterator() {
		featuresIterator = features.iterator();
		labelsIterator = labels.iterator();
	}
	
	public INDArray getAllFeatures() {
		return Nd4j.vstack(features);
	}
	
	public INDArray getAllLabels() {
		
		return Nd4j.vstack(labels);
	}
	

}
