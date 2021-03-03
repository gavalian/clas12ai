package org.crtcjlab.denoise2d.models;

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

public class DenoisingAutoEncoder0b extends AbstractCnnDenoisingAutoEncoder {

    @Override
    public void buildModel() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Nadam.Builder().learningRate(0.001).beta1(0.9).beta2(0.999).epsilon(1e-07).build())
//	            .updater(new Adam())
//	            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER_UNIFORM)
                .activation(Activation.RELU)
                .biasInit(0)
                .list()
                .layer(new ConvolutionLayer.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(4, 3).stride(1, 1).activation(Activation.RELU)
                        .nIn(1).nOut(54).build())
                .layer(new ConvolutionLayer.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(4, 3).stride(1, 1).activation(Activation.RELU)
                        .nOut(54).build())
//	            .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2).poolingType(SubsamplingLayer.PoolingType.MAX).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(4, 3).stride(1, 1).activation(Activation.RELU).convolutionMode(ConvolutionMode.Same)
                        .nOut(54).build())
                .layer(new ConvolutionLayer.Builder().kernelSize(4, 3).stride(1, 1).activation(Activation.RELU).convolutionMode(ConvolutionMode.Same)
                        .nOut(54).build())

//	            .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(3, 2).stride(3, 2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
                .layer(new ConvolutionLayer.Builder().kernelSize(4, 3).stride(1, 1).activation(Activation.RELU).convolutionMode(ConvolutionMode.Same)
                        .nOut(54).build())
                .layer(new ConvolutionLayer.Builder().kernelSize(4, 3).stride(1, 1).activation(Activation.RELU).convolutionMode(ConvolutionMode.Same)
                        .nOut(54).build())

                .layer(new Upsampling2D.Builder().size(new int[]{3, 2}).build())
                .layer(new ConvolutionLayer.Builder().kernelSize(4, 3).stride(1, 1).activation(Activation.RELU).convolutionMode(ConvolutionMode.Same)
                        .nOut(54).build())
                .layer(new ConvolutionLayer.Builder().kernelSize(4, 3).stride(1, 1).activation(Activation.RELU).convolutionMode(ConvolutionMode.Same)
                        .nOut(54).build())

                .layer(new Upsampling2D.Builder().size(2).build())
//	            .layer(new OutputLayer.Builder().nOut(4032).activation(Activation.SIGMOID).lossFunction(LossFunction.XENT).build())
                .layer(new ConvolutionLayer.Builder().kernelSize(4, 3).stride(1, 1).activation(Activation.IDENTITY).convolutionMode(ConvolutionMode.Same)
                        .nOut(1).build())
                .layer(new CnnLossLayer.Builder().activation(Activation.SIGMOID).lossFunction(LossFunction.XENT).build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println(model.summary(InputType.convolutional(height, width, channels)));
    }
}
