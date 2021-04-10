package org.crtcjlab.denoise2d.dataset;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.impl.misc.SVMLightRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Pad;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

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
        conf.setInt(SVMLightRecordReader.class.getName() + ".numfeatures", numFeatures);
        conf.setInt(SVMLightRecordReader.class.getName() + ".numLabels", numLabels);
        conf.setBoolean(SVMLightRecordReader.class.getName() + ".zeroBasedIndexing", zeroBasedIndexing);
        recordReader.initialize(conf, new FileSplit(f));
    }

    public void readData(int batchSize, int numPossibleLabels, int[] inputLabels, int[] targetLabels, long[] dataShape) {
        this.batchSize = batchSize;
        this.numPossibleLabels = numPossibleLabels;
        it = new RecordReaderDataSetIterator(recordReader, this.batchSize * 2, -1, this.numPossibleLabels);
        features = new ArrayList<>();
        labels = new ArrayList<>();

        while (it.hasNext()) {
            DataSet data = it.next();
            DataSet noisy_input = data.filterBy(inputLabels);
            DataSet ground_truth = data.filterBy(targetLabels);
            dataShape[0] = noisy_input.getFeatures().size(0);
            features.add(noisy_input.getFeatures().reshape(dataShape));
            labels.add(ground_truth.getFeatures().reshape(dataShape));
        }

        featuresIterator = features.iterator();
        labelsIterator = labels.iterator();
    }

    /**
     * Adds padding to the X and Y dimensions of the dataset.
     *
     * @param paddingX X dimension padding to add to the left and right of a data sample
     * @param paddingY Y dimension padding to add to the top and bottom of a data sample
     */
    public void addPadding(final int paddingX, final int paddingY) {
        // Ensure that the padding values are divisible by 2
        if (paddingX % 2 != 0) {
            System.out.println(paddingX);
            System.err.println("Error: X padding must be divisible by 2.");
            System.exit(1);
        }
        if (paddingY % 2 != 0) {
            System.err.println("Error: Y padding must be divisible by 2.");
            System.exit(1);
        }

        final int halfPaddingX = paddingX / 2;
        final int halfPaddingY = paddingY / 2;

        // Pad features and labels
        for (int i = 0; i < features.size(); ++i) {
            INDArray feature = features.get(i);
            INDArray paddedFeature = Nd4j.pad(feature, new int[][]{{0, 0}, {halfPaddingY, halfPaddingY}, {halfPaddingX, halfPaddingX}, {0, 0}}, Pad.Mode.CONSTANT, 0);
            features.set(i, paddedFeature);

            INDArray label = labels.get(i);
            INDArray paddedLabel = Nd4j.pad(label, new int[][]{{0, 0}, {halfPaddingY, halfPaddingY}, {halfPaddingX, halfPaddingX}, {0, 0}}, Pad.Mode.CONSTANT, 0);
            labels.set(i, paddedLabel);
        }
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

    public List<INDArray> getFeatures() {
        return features;
    }

    public List<INDArray> getLabels() {
        return labels;
    }
}
