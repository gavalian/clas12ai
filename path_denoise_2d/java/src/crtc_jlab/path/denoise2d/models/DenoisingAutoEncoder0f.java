package org.crtc_jlab.path.denoise2d.models;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class DenoisingAutoEncoder0f extends AbstractCnnDenoisingAutoEncoder{

	@Override
	public void buildModel() {
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	            .seed(seed)
	            .updater(new Nadam.Builder().learningRate(0.001).beta1(0.9).beta2(0.999).epsilon(1e-07).build())
	            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	            .weightInit(WeightInit.XAVIER_UNIFORM)
//	            .biasInit(0)
	            .list()
	            .layer(new ConvolutionLayer.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(4,3).stride(1,1).activation(Activation.RELU)
	                .nIn(1).nOut(28).build())
	            .layer(new ConvolutionLayer.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(4,3).stride(1,1).activation(Activation.RELU)
		                .nOut(28).build())
//	            .layer(new BatchNormalization())
	            .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())

	            .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).activation(Activation.RELU).convolutionMode(ConvolutionMode.Same)
	                .nOut(28).build())
	            .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).activation(Activation.RELU).convolutionMode(ConvolutionMode.Same)
		                .nOut(28).build())
	            
//	            .layer(new BatchNormalization())
	            .layer(new SubsamplingLayer.Builder().kernelSize(3,2).stride(3,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
	            .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).activation(Activation.RELU).convolutionMode(ConvolutionMode.Same)
	                    .nOut(28).build())
	            .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).activation(Activation.RELU).convolutionMode(ConvolutionMode.Same)
	                    .nOut(28).build())
	            
	            .layer(new Upsampling2D.Builder().size(new int[] {3,2}).build())
	            .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).activation(Activation.RELU).convolutionMode(ConvolutionMode.Same)
	                    .nOut(28).build())
	            .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).activation(Activation.RELU).convolutionMode(ConvolutionMode.Same)
	                    .nOut(28).build())
	            
	            .layer(new Upsampling2D.Builder().size(2).build())
//	            .layer(new OutputLayer.Builder().)
	            .layer(new ConvolutionLayer.Builder().kernelSize(4,3).stride(1,1).activation(Activation.RELU).convolutionMode(ConvolutionMode.Same)
	                    .nOut(28).build())
	            .layer(new ConvolutionLayer.Builder().kernelSize(4,3).stride(1,1).activation(Activation.RELU).convolutionMode(ConvolutionMode.Same)
	                    .nOut(28).build())
	            .layer(new ConvolutionLayer.Builder().kernelSize(4,3).stride(1,1).activation(Activation.SIGMOID).convolutionMode(ConvolutionMode.Same)
	                    .nOut(1).build())
	            .layer(new CnnLossLayer.Builder().activation(Activation.IDENTITY).lossFunction(LossFunction.XENT).build())
	            .setInputType(InputType.convolutional(height, width, channels))
	            .build();

	        model = new MultiLayerNetwork(conf);
	        model.init();
	        System.out.println(model.summary(InputType.convolutional(height, width, channels)));        
	    }
}
