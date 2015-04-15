/* 
 * Copyright (c) 2015 RobotsByTheC. All rights reserved.
 *
 * Open Source Software - may be modified and shared by FRC teams. The code must
 * be accompanied by the BSD license file in the root directory of the project.
 */
package org.usfirst.frc.team2084.neuralnetwork;

/**
 * @author Ben Wolsieffer
 */
public class Network {

    private double error;
    private double recentAverageError;
    private final Neuron[][] layers;
    private final double[] results;

    /**
     * 
     */
    public Network(final int[] topology, final String transferFunction) {
        // size
        final int numberOfLayers = topology.length;
        layers = new Neuron[numberOfLayers][];

        // create a new layer on each iteration
        for (int layerNumber = 0; layerNumber < numberOfLayers; layerNumber++) {
            final int layerSize = topology[layerNumber];
            final Neuron[] layer = layers[layerNumber] = new Neuron[layerSize];

            // number of outputs to a neuron
            final int numberOutputs = (layerNumber == numberOfLayers - 1) ? 0 : topology[layerNumber + 1];

            // fill layer with neurons and add bias neuron to the layer;
            for (int neuronNumber = 0; neuronNumber <= topology[layerNumber]; neuronNumber++) {
                layer[neuronNumber] = new Neuron(numberOutputs, neuronNumber, transferFunction);
            }
        }

        // Force the bias node's output to 1.0 (it was the last neuron pushed in
        // this layer):
        // TODO: should this be done for each layer?
        Neuron[] outputLayer = layers[layers.length - 1];
        outputLayer[outputLayer.length - 1].setOutputValue(1.0);
        results = new double[outputLayer.length];
    }

    // feedForward - operation to train the network
    public void feedForward(final double[] inputValues) {
        final Neuron[] inputLayer = layers[0];

        if (inputValues.length != inputLayer.length - 1) {
            throw new IllegalArgumentException("inputValues size needs to be the same as the input layer - bias");
        }

        for (int i = 0; i < inputValues.length; i++) {
            inputLayer[i].setOutputValue(inputValues[i]);
        }

        // forward propagation
        // loop each layer and each neuron inside the layer
        Neuron[] prevLayer = inputLayer;
        for (int layerNumber = 1; layerNumber < layers.length; layerNumber++) {
            Neuron[] layer = layers[layerNumber];
            for (Neuron n : layer) {
                n.feedForward(prevLayer);
            }
            prevLayer = layer;
        }
    }

    // backPropagation learning
    public void backPropagation(double[] targetValues, double eta, double alpha) {
        // Calculate overall net error (RMS-root mean square error - of output
        // neuron errors)
        Neuron[] outputLayer = layers[layers.length - 1];

        // overall net error
        error = 0.0;

        for (int n = 0; n < outputLayer.length - 1; n++) {
            final double delta = targetValues[n] - outputLayer[n].getOutputValue();
            error += delta * delta;
        }
        error *= 0.5;

        // error/=(* numPatterns); //get average error squared
        // error/=outputLayer.length - 1; //get average error squared
        // Implement a recent average measurement
        recentAverageError = error;

        // Calculate output layer gradients
        for (int n = 0; n < outputLayer.length - 1; n++) {
            outputLayer[n].calculateOutputGradients(targetValues[n]);
        }

        // Calculate hidden layer gradients
        for (int layerNumber = layers.length - 2; layerNumber > 0; layerNumber--) {
            final Neuron[] hiddenLayer = layers[layerNumber];
            final Neuron[] nextLayer = layers[layerNumber + 1];

            for (Neuron n : hiddenLayer) {
                n.calculateHiddenGradients(nextLayer);
            }
        }

        // For all layers from outputs to first hidden layer,
        // update connection weights
        for (int layerNumber = layers.length - 1; layerNumber > 0; layerNumber--) {
            Neuron[] layer = layers[layerNumber];
            Neuron[] prevLayer = layers[layerNumber - 1];

            for (int n = 0; n < layer.length - 1; n++) {
                layer[n].updateInputWeights(prevLayer, eta, alpha);
            }
        }
    }

    public double[] getResults() {
        Neuron[] outputLayer = layers[layers.length - 1];
        for (int n = 0; n < outputLayer.length - 1; n++) {
            results[n] = outputLayer[n].getOutputValue();
        }
        return results;
    }

    public double getRecentAverageError() {
        return recentAverageError;
    }

    public double[] getLayerValues(int row) {
        final Neuron[] layer = layers[row];
        final double[] values = new double[layer.length];

        for (int i = 0; i < layer.length; i++) {
            values[i] = layer[i].getOutputValue();
        }
        return values;
    }

    public void setLayer(double[] values, int row) {
        final Neuron[] layer = layers[row];

        for (int i = 0; i < layer.length; i++) {
            layer[i].setOutputValue(values[i]);
        }
    }
}
