package org.crtcjlab.denoise2d;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.*;
import org.crtcjlab.denoise2d.dataset.LsvmDataSet;
import org.crtcjlab.denoise2d.models.AbstractCnnDenoisingAutoEncoder;
import org.crtcjlab.denoise2d.models.DenoisingAutoEncoder;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.io.File;
import java.io.IOException;

public class Denoise2d {
    private static Namespace Args;

    public static void main(String[] args) throws Exception {
        parseArgs(args);
        String subCommand = Args.get("subcommand");

        // Run subcommand
        if (subCommand.equals("train")) {
            train();
        } else if (subCommand.equals("test")) {
            test();
        } else if (subCommand.equals("predict")) {
            predict();
        } else {
            System.out.println("Unexpected subcommand given. Quitting.");
            System.exit(1);
        }
    }

    public static void train() {
        final String trainDataPath = Args.getString("training_file");
        final String validationDataPath = Args.getString("validation_file");
        final String resultsDir = Args.getString("results_dir");
        final String network = Args.getString("network");
        final int batchSize = Args.getInt("batch_size");
        final int nEpochs = Args.getInt("epochs");

        // Read LSVM data
        System.out.println("===== Reading  dataset in: " + trainDataPath + "... =====");
        LsvmDataSet trainingLsvmData = null;
        try {
            trainingLsvmData = new LsvmDataSet(trainDataPath, 4032, 1, false);
            trainingLsvmData.readData(batchSize, 3, new int[]{0}, new int[]{1, 2}, new long[]{-1, 36, 112, 1});
        } catch (IOException e) {
            System.err.println("Error: Failed to read training LSVM dataset.");
            e.printStackTrace();
            System.exit(1);
        } catch (InterruptedException e) {
            System.err.println("Error: LSVM training dataset reading was interrupted.");
            e.printStackTrace();
            System.exit(1);
        }
        System.out.println("===== Done =====");

        // Compile model
        AbstractCnnDenoisingAutoEncoder model = DenoisingAutoEncoder.getAutoEncoderByName(network);
        System.out.println("===== Training using model architecture: " + network + " =====");
        model.buildModel();

        // Check if results directory exists, otherwise try to create it
        File file = new File(resultsDir);
        boolean success = true;
        if (!file.exists()) {
            success = file.mkdir();
        }

        // Train and save model
        if (success) {
            model.train(trainingLsvmData, nEpochs);

            try {
                model.saveModel(resultsDir + "cnn_autoenc" + network + ".h5");
            } catch (IOException e) {
                System.err.println("Error: Failed to save Keras model.");
                e.printStackTrace();
                System.exit(1);
            }
        } else {
            System.out.println("Failed to open/create directory to save results!");
            System.exit(1);
        }

        // Validate model
        if (!validationDataPath.isEmpty()) {
            LsvmDataSet validationLsvmData = null;
            try {
                validationLsvmData = new LsvmDataSet(validationDataPath, 4032, 1, false);
                validationLsvmData.readData(batchSize, 3, new int[]{0}, new int[]{1, 2}, new long[]{-1, 36, 112, 1});
            } catch (IOException e) {
                System.err.println("Error: Failed to read validation LSVM dataset.");
                e.printStackTrace();
                System.exit(1);
            } catch (InterruptedException e) {
                System.err.println("Error: LSVM validation dataset reading was interrupted.");
                e.printStackTrace();
                System.exit(1);
            }

            model.test(validationLsvmData);
        }
    }

    public static void test() {
        final String testingDataPath = Args.getString("testing_file");
        final String resultsDir = Args.getString("results_dir");
        final String modelConfigPath = Args.getString("model_config");
        final String modelWeightsPath = Args.getString("model_weights");
        final int batchSize = Args.getInt("batch_size");
        final int paddingX = Args.getInt("px");
        final int paddingY = Args.getInt("py");

        System.out.println("======= Testing =======");
        System.out.println("Model configuration: " + modelConfigPath);
        System.out.println("Model weights: " + modelWeightsPath);

        // Read LSVM data
        LsvmDataSet testingLsvmData = null;
        try {
            testingLsvmData = new LsvmDataSet(testingDataPath, 4032, 1, false);
            testingLsvmData.readData(batchSize, 3, new int[]{0}, new int[]{1, 2}, new long[]{-1, 36, 112, 1});
        } catch (IOException e) {
            System.err.println("Error: Failed to read testing LSVM dataset.");
            e.printStackTrace();
            System.exit(1);
        } catch (InterruptedException e) {
            System.err.println("Error: LSVM testing dataset reading was interrupted.");
            e.printStackTrace();
            System.exit(1);
        }

        // Check if results directory exists, otherwise try to create it
        File file = new File(resultsDir);
        boolean success = true;
        if (!file.exists()) {
            success = file.mkdir();
        }

        // Load and test model
        if (success) {
            DenoisingAutoEncoder model = new DenoisingAutoEncoder();

            try {
                model.loadKerasModel(modelConfigPath, modelWeightsPath);
            } catch (IOException | UnsupportedKerasConfigurationException | InvalidKerasConfigurationException e) {
                System.err.println("Error: Failed to load Keras model for testing.");
                e.printStackTrace();
                System.exit(1);
            }

            System.out.println("Preprocessing dataset");
            model.preprocessDataset(testingLsvmData, paddingX, paddingY);
            System.out.println("Finished preprocessing");

            model.test(testingLsvmData);
        } else {
            System.out.println("Failed to open/create directory to save results!");
            System.exit(1);
        }
    }

    public static void predict() {
        final String predictionDataPath = Args.getString("prediction_file");
        final String resultsDir = Args.getString("results_dir");
        final String modelConfigPath = Args.getString("model_config");
        final String modelWeightsPath = Args.getString("model_weights");
        final int batchSize = Args.getInt("batch_size");
        final int paddingX = Args.getInt("px");
        final int paddingY = Args.getInt("py");

        System.out.println("======= Prediction =======");
        System.out.println("Model configuration: " + modelConfigPath);
        System.out.println("Model weights: " + modelWeightsPath);

        // Read LSVM data
        LsvmDataSet predictionLsvmData = null;
        try {
            predictionLsvmData = new LsvmDataSet(predictionDataPath, 4032, 1, false);
            predictionLsvmData.readData(batchSize, 3, new int[]{0}, new int[]{1, 2}, new long[]{-1, 36, 112, 1});
        } catch (IOException e) {
            System.err.println("Error: Failed to read prediction LSVM dataset.");
            e.printStackTrace();
            System.exit(1);
        } catch (InterruptedException e) {
            System.err.println("Error: LSVM prediction dataset reading was interrupted.");
            e.printStackTrace();
            System.exit(1);
        }

        // Check if results directory exists, otherwise try to create it
        File file = new File(resultsDir);
        boolean success = true;
        if (!file.exists()) {
            success = file.mkdir();
        }

        // Load model and predict
        if (success) {
            DenoisingAutoEncoder model = new DenoisingAutoEncoder();

            try {
                model.loadKerasModel(modelConfigPath, modelWeightsPath);
            } catch (IOException | UnsupportedKerasConfigurationException | InvalidKerasConfigurationException e) {
                System.err.println("Error: Failed to load Keras model for prediction.");
                e.printStackTrace();
                System.exit(1);
            }

            System.out.println("Preprocessing dataset");
            model.preprocessDataset(predictionLsvmData, paddingX, paddingY);
            System.out.println("Finished preprocessing");

            model.predict(predictionLsvmData, resultsDir, paddingX, paddingY);
        } else {
            System.out.println("Failed to open/create directory to save results!");
            System.exit(1);
        }
    }

    public static void parseArgs(String[] args) {
        ArgumentParser parser = ArgumentParsers.newFor("CRTC-JLab Machine Learning 2D Track Denoising CLI").build()
                .defaultHelp(true)
                .description("2D Track events reconstruction using machine learning.");

        Subparsers subparsers = parser.addSubparsers().help("Run the training, testing or predicting sub-command");
        subparsers.dest("subcommand");

        // Training arguments
        Subparser trainingParser = subparsers.addParser("train").help("Train a model, perform testing, and serialize it.");
        trainingParser.addArgument("--training-file", "-t").required(true).type(String.class).help("Path to the SVM file containing the training data.");
        trainingParser.addArgument("--validation-file", "-v").setDefault("").type(String.class).help("Path to the SVM file containing the validation data.");
        trainingParser.addArgument("--results-dir", "-r").required(true).type(String.class).help("Path to the directory to store the model produced by the training and related results.");
        trainingParser.addArgument("--epochs", "-e").setDefault(10).type(Integer.class).help("How many training epochs to go through.");
        trainingParser.addArgument("--batch-size", "-b").setDefault(16).type(Integer.class).help("Size of the training batch.");
        trainingParser.addArgument("--network", "-n").setDefault("0b").type(String.class).help("Neural network architecture to use.");

        // Testing arguments
        Subparser testingParser = subparsers.addParser("test").help("Load a model for testing.");
        testingParser.addArgument("--testing-file", "-t").required(true).type(String.class).help("Path to the SVM file containing the testing data.");
        testingParser.addArgument("--model-config").required(true).type(String.class).help("Path to model JSON configuration file.");
        testingParser.addArgument("--model-weights").required(true).type(String.class).help("Path to model weights HDF5 file.");
        testingParser.addArgument("--results-dir", "-r").required(true).type(String.class).help("Path to the directory to store the testing related results.");
        testingParser.addArgument("--batch-size", "-b").setDefault(16).type(Integer.class).help("Size of the training batch.");
        testingParser.addArgument("--px").setDefault(0).type(Integer.class).help("Padding along the x dimension. Applied in both left and right sides.");
        testingParser.addArgument("--py").setDefault(0).type(Integer.class).help("Padding along the y dimension. Applied in both top and bottom sides.");

        // Prediction arguments
        Subparser predictionParser = subparsers.addParser("predict").help("Load a model and use it for predictions.");
        predictionParser.addArgument("--prediction-file", "-p").required(true).type(String.class).help("Path to the SVM file containing the prediction data.");
        predictionParser.addArgument("--model-config").required(true).type(String.class).help("Path to model JSON configuration file.");
        predictionParser.addArgument("--model-weights").required(true).type(String.class).help("Path to model weights HDF5 file.");
        predictionParser.addArgument("--results-dir", "-r").required(true).type(String.class).help("Path to the directory to store the testing related results.");
        predictionParser.addArgument("--batch-size", "-b").setDefault(16).type(Integer.class).help("Size of the training batch.");
        predictionParser.addArgument("--px").setDefault(0).type(Integer.class).help("Padding along the x dimension. Applied in both left and right sides.");
        predictionParser.addArgument("--py").setDefault(0).type(Integer.class).help("Padding along the y dimension. Applied in both top and bottom sides.");

        try {
            Args = parser.parseArgs(args);
            System.out.println(Args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }
    }
}
