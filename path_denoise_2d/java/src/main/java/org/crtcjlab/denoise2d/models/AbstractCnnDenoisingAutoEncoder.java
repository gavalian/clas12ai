package org.crtcjlab.denoise2d.models;

import org.crtcjlab.denoise2d.dataset.LsvmDataSet;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.writer.impl.misc.SVMLightRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Pad;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public abstract class AbstractCnnDenoisingAutoEncoder {
    
    protected int width, height, channels;
    protected static long seed = 123L;

    protected MultiLayerNetwork model;

    protected long  executionTime = 0L;
    protected long executionCount = 0L;
    protected long paddingTime = 0L;
    protected long paddingCount = 0L;
    
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

        int numIters = trainSet.getNumBatches();
        for (int epoch = 0; epoch < nEpochs; epoch++) {
            System.out.println("Starting epoch: " + (epoch + 1) + ". Number of batches: " + numIters);
            int iteration = 0;
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

    private enum PaddingOperation {
        ADD, REMOVE
    }

    /**
     * Add or remove padding for the list of features
     *
     * @param array List to process.
     * @param paddingX Horizontal padding to apply or remove. Half this value is applied or removed left and right.
     * @param paddingY Vertical padding to apply or remove. Half this value is applied or remove top and bottom.
     * @param paddingOp Padding operation to apply. Can either add or remove padding.
     */
    private static void processPadding(final List<INDArray> array, final int paddingX, final int paddingY, PaddingOperation paddingOp) {
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

        final int paddingSign = (paddingOp == PaddingOperation.REMOVE) ? -1 : 1;

        final int halfPaddingX = (paddingX / 2) * paddingSign;
        final int halfPaddingY = (paddingY / 2) * paddingSign;

        // Process padding
        for (int i = 0; i < array.size(); ++i) {
            INDArray element = array.get(i);
            INDArray newElement = Nd4j.pad(element, new int[][]{{0, 0}, {halfPaddingY, halfPaddingY}, {halfPaddingX, halfPaddingX}, {0, 0}}, Pad.Mode.CONSTANT, 0);
            array.set(i, newElement);
        }
    }
    
    public double getExecutionTime(){
        return ( (double) executionTime)/executionCount;
    }
    public double getPaddingTime(){
        return ( (double) paddingTime)/paddingCount;
    }
    /**
     * Predict values with the current model. Applies padding if any padding parameter is passed.
     *
     * @param features List of INDArray features
     * @param paddingX Horizontal padding to apply. Half this value is applied left and right
     * @param paddingY Vertical padding to apply. Half this value is applied top and bottom
     * @param threshold Threshold above which predicted values become 1, and below which they become 0
     * @return List of INDArray predictions
     */
    public List<INDArray> predict(final List<INDArray> features, final int paddingX, final int paddingY, final double threshold) {
        final int totalBatches = features.size();
        final List<INDArray> predictions = new ArrayList<>(totalBatches);

        // Check if we have padding
        final boolean hasPadding = paddingX != 0 || paddingY != 0;

        // Apply padding
        if (hasPadding) {
            long then = System.currentTimeMillis();
            processPadding(features, paddingX, paddingY, PaddingOperation.ADD);
            long now = System.currentTimeMillis();
            paddingTime += (now-then);
            paddingCount++;
        }

        // Predict
        for (int i = 0; i < totalBatches; ++i) {
            //--> Commented by Gagik /System.out.println("Iteration  " + (i + 1) + "/" + totalBatches);
            long then = System.currentTimeMillis();
            INDArray prediction = model.output(features.get(i));
            
            long now = System.currentTimeMillis();
            executionTime += (now - then);
            executionCount++;
            prediction = Transforms.floor(prediction.add(1 - threshold));
            predictions.add(prediction);
            
        }

        // Remove padding from features and predictions
        if (hasPadding) {
            long then = System.currentTimeMillis();
            processPadding(features, paddingX, paddingY, PaddingOperation.REMOVE);
            processPadding(predictions, paddingX, paddingY, PaddingOperation.REMOVE);
            long now = System.currentTimeMillis();
            paddingTime += (now-then);
            paddingCount++;
        }

        return predictions;
    }
    public List<INDArray> predict(final List<INDArray> features, final int paddingX, final int paddingY, final double threshold, final boolean recoverHits) {
        List<INDArray>  predictions = this.predict(features, paddingX, paddingY, threshold);
        if (recoverHits) {
            recoverMissingHits(predictions, features);
        }
        return predictions;
    }
    /**
     * Recovers missing hits from a set of denoised features.
     * All elements within a 1 sensor radius from 1.0 elements in the denoised features that are also 1.0 in the noisy
     * features are set to 1.0. This tries to fill gaps that may have been produced by the denoising process.
     *
     * The following neighbor offset notation is used (row,col):
     * -1,-1  | -1,0  | -1,+1
     *  0,-1  |  0,0  |  0,+1
     * +1,-1  | +1,0  | +1,+1
     *
     * NOTE: Experimental
     *
     * @param denoisedFeatures Denoised features (i.e. output of the ML model). These will be modified by the function.
     * @param noisyFeatures Noisy features (i.e. input to the ML model).
     */
    public void recoverMissingHits(List<INDArray> denoisedFeatures, final List<INDArray> noisyFeatures) {
        final int numBatches = (int) denoisedFeatures.get(0).shape()[0];
        final int numRows = (int) denoisedFeatures.get(0).shape()[1];
        final int numCols = (int) denoisedFeatures.get(0).shape()[2];

        // Iterate samples
        IntStream.range(0, denoisedFeatures.size())
                .parallel()
                .forEach(i -> {
                    INDArray denoisedFeature = denoisedFeatures.get(i);
                    INDArray noisyFeature = noisyFeatures.get(i);

                    // Iterate sample elements
                    for (int batch = 0; batch < numBatches; ++batch) {
                        for (int row = 0; row < numRows; ++row) {
                            for (int col = 0; col < numCols; ++col) {
                                // Get denoised element
                                double denoisedElement = denoisedFeature.getDouble(batch, row, col, 0);

                                // Skip denoised elements that are not hits
                                if (denoisedElement != 1) continue;

                                // Check neighbors
                                boolean hasTopNeighbors = (row - 1) >= 0;
                                boolean hasBottomNeighbors = (row + 1) < numRows;
                                boolean hasLeftNeighbors = (col - 1) >= 0;
                                boolean hasRightNeighbors = (col + 1) < numCols;

                                double neighborNoisyElement;
                                double neighborDenoisedElement;

                                // Neighbor -1, 0
                                if (hasTopNeighbors) {
                                    neighborNoisyElement = noisyFeature.getDouble(batch, row - 1, col, 0);
                                    neighborDenoisedElement = denoisedFeature.getDouble(batch, row - 1, col, 0);

                                    if (neighborNoisyElement == 1 && neighborDenoisedElement == 0) {
                                        denoisedFeature.putScalar(new int[]{batch, row - 1, col, 0}, 1.5);
                                    }

                                    // Neighbor -1, -1
                                    if (hasLeftNeighbors) {
                                        neighborNoisyElement = noisyFeature.getDouble(batch, row - 1, col - 1, 0);
                                        neighborDenoisedElement = denoisedFeature.getDouble(batch, row - 1, col - 1, 0);

                                        if (neighborNoisyElement == 1 && neighborDenoisedElement == 0) {
                                            denoisedFeature.putScalar(new int[]{batch, row - 1, col - 1, 0}, 1.5);
                                        }
                                    }

                                    // Neighbor -1, +1
                                    if (hasRightNeighbors) {
                                        neighborNoisyElement = noisyFeature.getDouble(batch, row - 1, col + 1, 0);
                                        neighborDenoisedElement = denoisedFeature.getDouble(batch, row - 1, col + 1, 0);

                                        if (neighborNoisyElement == 1 && neighborDenoisedElement == 0) {
                                            denoisedFeature.putScalar(new int[]{batch, row - 1, col + 1, 0}, 1.5);
                                        }
                                    }
                                }

                                // Neighbor 0, -1
                                if (hasLeftNeighbors) {
                                    neighborNoisyElement = noisyFeature.getDouble(batch, row, col - 1, 0);
                                    neighborDenoisedElement = denoisedFeature.getDouble(batch, row, col - 1, 0);

                                    if (neighborNoisyElement == 1 && neighborDenoisedElement == 0) {
                                        denoisedFeature.putScalar(new int[]{batch, row, col - 1, 0}, 1.5);
                                    }
                                }
                                // Neighbor 0, +1
                                if (hasRightNeighbors) {
                                    neighborNoisyElement = noisyFeature.getDouble(batch, row, col + 1, 0);
                                    neighborDenoisedElement = denoisedFeature.getDouble(batch, row, col + 1, 0);

                                    if (neighborNoisyElement == 1 && neighborDenoisedElement == 0) {
                                        denoisedFeature.putScalar(new int[]{batch, row, col + 1, 0}, 1.5);
                                    }
                                }

                                // Neighbor +1, 0
                                if (hasBottomNeighbors) {
                                    neighborNoisyElement = noisyFeature.getDouble(batch, row + 1, col, 0);
                                    neighborDenoisedElement = denoisedFeature.getDouble(batch, row + 1, col, 0);

                                    if (neighborNoisyElement == 1 && neighborDenoisedElement == 0) {
                                        denoisedFeature.putScalar(new int[]{batch, row + 1, col, 0}, 1.5);
                                    }

                                    // Neighbor +1, -1
                                    if (hasLeftNeighbors) {
                                        neighborNoisyElement = noisyFeature.getDouble(batch, row + 1, col - 1, 0);
                                        neighborDenoisedElement = denoisedFeature.getDouble(batch, row + 1, col - 1, 0);

                                        if (neighborNoisyElement == 1 && neighborDenoisedElement == 0) {
                                            denoisedFeature.putScalar(new int[]{batch, row + 1, col - 1, 0}, 1.5);
                                        }
                                    }

                                    // Neighbor +1, +1
                                    if (hasRightNeighbors) {
                                        neighborNoisyElement = noisyFeature.getDouble(batch, row + 1, col + 1, 0);
                                        neighborDenoisedElement = denoisedFeature.getDouble(batch, row + 1, col + 1, 0);

                                        if (neighborNoisyElement == 1 && neighborDenoisedElement == 0) {
                                            denoisedFeature.putScalar(new int[]{batch, row + 1, col + 1, 0}, 1.5);
                                        }
                                    }
                                }
                            }
                        }

                        // Convert all 1.5s to 1s
                        denoisedFeature = Transforms.floor(denoisedFeature);
                        denoisedFeatures.set(i, denoisedFeature);
                    }
                });
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
            //-- commented by Gagik / System.out.println("Iteration  " + (iteration + 1) + "/" + numIters);
            INDArray features = predictSet.getNextFeaturesBatch();
            INDArray labels = predictSet.getNextLabelsBatch();
            INDArray predictions = model.output(features);

            // Every value greater than or equal to 0 should become 1, the rest should become 0
            predictions = Transforms.floor(predictions.add(0.5));

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
        }
        writer.close();

        System.out.println("Done");
    }
}
