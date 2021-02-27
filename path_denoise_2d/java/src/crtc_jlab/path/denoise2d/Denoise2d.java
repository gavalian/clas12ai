package org.crtc_jlab.path.denoise2d;

import java.io.File;

import org.crtc_jlab.path.denoise2d.dataset.LsvmDataSet;
import org.crtc_jlab.path.denoise2d.models.AbstractCnnDenoisingAutoEncoder;
import org.crtc_jlab.path.denoise2d.models.DenoisingAutoEncoder;
import org.crtc_jlab.path.denoise2d.models.DenoisingAutoEncoder0;
import org.nd4j.linalg.api.ndarray.INDArray;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import net.sourceforge.argparse4j.inf.Subparser;
import net.sourceforge.argparse4j.inf.Subparsers;

public class Denoise2d {
    private static Namespace Args;
    
	public static void main(String[] args) throws Exception {
		
		parseArgs(args);
		String subCommand = Args.get("subcommand");
		if (subCommand.equals("train")) {
			String trainDataPath = Args.getString("training_file");
			String validationDataPath = Args.getString("validation_file");
			String resultsDir = Args.getString("results_dir");
			String network = Args.getString("network"); 
			int batchSize = Args.getInt("batch_size");
			int nEpochs = Args.getInt("epochs");
			
			System.out.println("===== Reading  dataset in: " + trainDataPath + "... =====");
			LsvmDataSet trainingLsvmData = new LsvmDataSet(trainDataPath, 4032, 1, false);
			trainingLsvmData.readData(batchSize, 3, new int[] {0}, new int[] {1, 2}, new long[] {-1, 1,  36, 112} );
			System.out.println("===== Done =====");
//			INDArray features = trainingLsvmData.getNextFeaturesBatch();
//			System.out.println(features.slice(0));
//			System.out.println(features.slice(0).shapeInfoToString());
//			System.out.println(features.slice(0).slice(0).slice(0).getInt(35));
//			System.out.println(features.slice(0).slice(0).slice(0).getInt(36));
//			System.out.println(features.slice(0).slice(0).slice(1).getInt(37));
//			System.out.println(features.slice(0).slice(0).slice(0).getInt(260));
//			System.out.println(features.slice(0).slice(0).slice(0).getInt(261));
//			System.exit(1);
			AbstractCnnDenoisingAutoEncoder denoising_ae = DenoisingAutoEncoder.getAutoEncoderByName(network);
			System.out.println("===== Training using model architecture: " + network + " =====");

	        denoising_ae.buildModel();
	        File file = new File(resultsDir);
	        boolean success = true;
	        if(!file.exists()) {
	        	success = file.mkdir();
	        }
	        if(success) {
		        denoising_ae.train(trainingLsvmData, nEpochs);
		        denoising_ae.saveModel(resultsDir+"cnn_autoenc"+network+".h5");
	        }
	        else {
	        	System.out.println("Failed to open/create directory to save results!");
	        	System.exit(1);
			}

	        
	        if(!validationDataPath.isEmpty()) {
				LsvmDataSet validationLsvmData = new LsvmDataSet(validationDataPath, 4032, 1, false);
				validationLsvmData.readData(batchSize, 3, new int[] {0}, new int[] {1, 2}, new long[] {-1, 1,  36, 112} );
				denoising_ae.test(validationLsvmData);
	        }
		}
		else if(subCommand.equals("test")) {
			String testingDataPath = Args.getString("testing_file");
//			String resultsDir = Args.getString("results_dir");
			String modelPath = Args.getString("model");
			int batchSize = Args.getInt("batch_size");
			
			System.out.println("===== Testing using model at: " + modelPath + " =====");
			
			LsvmDataSet testingLsvmData = new LsvmDataSet(testingDataPath, 4032, 1, false);
			testingLsvmData.readData(batchSize, 3, new int[] {0}, new int[] {1, 2}, new long[] {-1, 1,  36, 112} );
			DenoisingAutoEncoder denoising_ae = new DenoisingAutoEncoder();
	        File file = new File(modelPath);
	        boolean success = true;
	        if(!file.exists()) {
	        	success = file.mkdir();
	        }
	        if(success) {
		        denoising_ae.loadModel(modelPath);
		        denoising_ae.test(testingLsvmData);
	        }
	        else {
	        	System.out.println("Failed to open/create directory to save results!");
	        	System.exit(1);
			}

		}
		else if(subCommand.equals("predict")) {
			String predictionDataPath = Args.getString("prediction_file");
			String resultsDir = Args.getString("results_dir");
			String modelPath = Args.getString("model");
			int batchSize = Args.getInt("batch_size");
			
			System.out.println("===== Predicting using model at: " + modelPath + " =====");
			
			LsvmDataSet predictionLsvmData = new LsvmDataSet(predictionDataPath, 4032, 1, false);
			predictionLsvmData.readData(batchSize, 3, new int[] {0}, new int[] {1, 2}, new long[] {-1, 1,  36, 112} );
			DenoisingAutoEncoder0 denoising_ae = new DenoisingAutoEncoder0();
			denoising_ae.loadModel(modelPath);
	        File file = new File(resultsDir);
	        boolean success = true;
	        if(!file.exists()) {
	        	success = file.mkdir();
	        }
	        if(success) {
	        	denoising_ae.predict(predictionLsvmData, resultsDir);
	        }
	        else {
	        	System.out.println("Failed to open/create directory to save results!");
	        	System.exit(1);
			}
			
		}
		else {
			System.out.println("Unexpected subcommand given. Quitting.");
			System.exit(1);
		}
			
    }
	
	public static void parseArgs(String[] args) {
		ArgumentParser parser = ArgumentParsers.newFor("CRTC-JLab Machine Learning 2D Track Denoising CLI").build()
				.defaultHelp(true)
				.description("2D Track events reconstruction using machine learning.");
	    Subparsers subparsers = parser.addSubparsers().help("Run the training, testing or predicting sub-command");
	    subparsers.dest("subcommand");
	    Subparser parserA = subparsers.addParser("train").help("Train a model, perform testing, and serialize it.");
	    parserA.addArgument("--training-file", "-t").required(true).type(String.class).help("Path to the SVM file containing the training data.");
	    parserA.addArgument("--validation-file", "-v").setDefault("").type(String.class).help("Path to the SVM file containing the validation data.");
	    parserA.addArgument("--results-dir", "-r").required(true).type(String.class).help("Path to the directory to store the model produced by the training and related results.");
	    parserA.addArgument("--epochs", "-e").setDefault(10).type(Integer.class).help("How many training epochs to go through.");
	    parserA.addArgument("--batch-size", "-b").setDefault(16).type(Integer.class).help("Size of the training batch.");
	    parserA.addArgument("--network", "-n").setDefault("0b").type(String.class).help("Neural network architecture to use.");
	    
	    Subparser parserB = subparsers.addParser("test").help("Load a model for testing.");
	    parserB.addArgument("--testing-file", "-t").required(true).type(String.class).help("Path to the SVM file containing the testing data.");
	    parserB.addArgument("--model", "-m").required(true).type(String.class).help("Trained model to load for testing.");
	    parserB.addArgument("--results-dir", "-r").required(true).type(String.class).help("Path to the directory to store the testing related results.");
	    parserB.addArgument("--batch-size", "-b").setDefault(16).type(Integer.class).help("Size of the training batch.");
	    
	    
	    Subparser parserC = subparsers.addParser("predict").help("Load a model and use it for predictions.");
	    parserC.addArgument("--prediction-file", "-p").required(true).type(String.class).help("Path to the SVM file containing the prediction data.");
	    parserC.addArgument("--model", "-m").required(true).type(String.class).help("Trained model to load for prediction.");
	    parserC.addArgument("--results-dir", "-r").required(true).type(String.class).help("Path to the directory to store the testing related results.");
	    parserC.addArgument("--batch-size", "-b").setDefault(16).type(Integer.class).help("Size of the training batch.");
	    
	    
	    try {
	    	Args = parser.parseArgs(args);
	        System.out.println(Args);
	    } catch (ArgumentParserException e) {
	        parser.handleError(e);
	        System.exit(1);
	    }
	}
		
}
