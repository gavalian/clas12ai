package org.crtcjlab.denoise2d.models;

import org.crtcjlab.denoise2d.dataset.LsvmDataSet;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.writer.impl.misc.SVMLightRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Pad;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.security.sasl.SaslServer;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

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

    /**
     * Load a Keras model. Reads JSON configuration file and a file containing the trained weights.
     *
     * @param pathModelConfig  Path to the JSON file containing the model configuration
     * @param pathModelWeights Path to the HDF5 file containing the model's weights
     * @throws UnsupportedKerasConfigurationException Keras configuration is not supported by DL4J
     * @throws IOException                            Failure to read any of the files
     * @throws InvalidKerasConfigurationException     Keras configuration is invalid
     */
    public void loadKerasModel(final String pathModelConfig, final String pathModelWeights) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        model = KerasModelImport.importKerasSequentialModelAndWeights(pathModelConfig, pathModelWeights);
        System.out.println(model.summary());
    }

    public void saveModel(String pathToModel) throws IOException {
        System.out.println("Saving model in " + pathToModel);
        model.save(new File(pathToModel));
    }

    /**
     * Preprocess a dataset for ingestion by the model. Directly modifies the dataset.
     * Adds padding.
     *
     * @param dataSet Dataset to preprocess
     */
    public void preprocessDataset(LsvmDataSet dataSet, int paddingX, int paddingY) {
        dataSet.addPadding(paddingX, paddingY);
    }

    public void train(LsvmDataSet trainSet, int nEpochs) {
        System.out.println("Starting training for " + nEpochs + " epochs...");
        model.setListeners(new ScoreIterationListener(1));

        int iteration = 0;
        int numIters = trainSet.getNumBatches();
        for (int epoch = 0; epoch < nEpochs; epoch++) {
            System.out.println("Starting epoch: " + (epoch + 1) + ". Number of batches: " + numIters);
            iteration = 0;
            while (trainSet.hasNext()) {
                System.out.println("Epoch " + (epoch + 1) + "/" + nEpochs + " | Iteration  " + (iteration + 1) + "/" + numIters);
                INDArray features = trainSet.getNextFeaturesBatch();
                INDArray labels = trainSet.getNextLabelsBatch();

                model.fit(features, labels);

                iteration++;
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

        while (testSet.hasNext()) {
            System.out.println("Iteration  " + (iteration + 1) + "/" + numIters);
            INDArray predictions = model.output(testSet.getNextFeaturesBatch());
            eval.eval(testSet.getNextLabelsBatch(), predictions);
            iteration++;
        }

        System.out.println(eval.stats());
    }

    public void predict(LsvmDataSet predictSet, String resultsPath, final int paddingX, final int paddingY) {
        System.out.println("Starting predictions");
        int iteration = 0;
        int numIters = predictSet.getNumBatches();
        System.out.println("Number of batches: " + numIters);

        Configuration configWriter = new Configuration();
        configWriter.setInt(SVMLightRecordWriter.FEATURE_FIRST_COLUMN, 0);
        configWriter.setInt(SVMLightRecordWriter.FEATURE_FIRST_COLUMN, 0);
        File inputFile = new File(resultsPath + "predict_report.txt");

        SVMLightRecordWriter writer = new SVMLightRecordWriter();

        try {
            writer.initialize(configWriter, new FileSplit(inputFile), new NumberOfRecordsPartitioner());
        } catch (Exception e) {
            System.err.println("Error: Failed to initialize SVMLightRecordWriter for prediction.");
            e.printStackTrace();
            System.exit(1);
        }

        boolean hasPadding = paddingX > 0 || paddingY > 0;
        final int halfPaddingX = paddingX / 2;
        final int halfPaddingY = paddingY / 2;

        while (predictSet.hasNext()) {
            System.out.println("Iteration  " + (iteration + 1) + "/" + numIters);
            INDArray features = predictSet.getNextFeaturesBatch();
            INDArray labels = predictSet.getNextLabelsBatch();
            INDArray predictions = model.output(features);

            // TODO need to fix prediction output

            // Remove padding from features, labels, and predictions
            if (hasPadding) {
                features = Nd4j.pad(features, new int[][]{{0, 0}, {-halfPaddingY, -halfPaddingY}, {-halfPaddingX, -halfPaddingX}, {0, 0}}, Pad.Mode.CONSTANT, 0);
                labels = Nd4j.pad(labels, new int[][]{{0, 0}, {-halfPaddingY, -halfPaddingY}, {-halfPaddingX, -halfPaddingX}, {0, 0}}, Pad.Mode.CONSTANT, 0);
                predictions = Nd4j.pad(predictions, new int[][]{{0, 0}, {-halfPaddingY, -halfPaddingY}, {-halfPaddingX, -halfPaddingX}, {0, 0}}, Pad.Mode.CONSTANT, 0);
            }

            iteration++;
            for (int i = 0; i < features.size(0); i++) {
                try {
                    List<Writable> record = Arrays.asList(new NDArrayWritable(labels.slice(i).reshape(4032)), new DoubleWritable(2));
                    writer.write(record);

                    record = Arrays.asList(new NDArrayWritable(predictions.slice(i).reshape(4032)), new DoubleWritable(1));
                    writer.write(record);

                    record = Arrays.asList(new NDArrayWritable(features.slice(i).reshape(4032)), new DoubleWritable(0));
                    writer.write(record);
                } catch (IOException e) {
                    System.err.println("Error: Failed to write predictions.");
                    e.printStackTrace();
                    System.exit(1);
                }
            }
            System.exit(1);
        }
        writer.close();

        System.out.println("Done");
    }
}
