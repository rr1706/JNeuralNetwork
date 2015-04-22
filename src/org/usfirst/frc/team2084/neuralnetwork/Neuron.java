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
public class Neuron {

    private double sum;
    private double outputValue;
    private double gradient;
    private final TransferFunction transferFunction;
    private final int index;

    private final Connection[] outputConnections;

    public Neuron(final int numberOutputs, final int index, final TransferFunction transferFunction) {
        outputConnections = new Connection[numberOutputs];
        for (int i = 0; i < outputConnections.length; i++) {
            outputConnections[i] = new Connection();
        }
        this.index = index;
        this.transferFunction = transferFunction;
    }

    public void setOutputValue(final double value) {
        outputValue = value;
    }

    public double getOutputValue() {
        return outputValue;
    }

    public Connection[] getOutputConnections() {
        return outputConnections;
    }

    public void feedForward(final Neuron[] prevLayer) {
        double sum = 0.0;

        // Sum the previous layer's outputs (which are our inputs)
        // Include the bias node from the previous layer.
        for (final Neuron n : prevLayer) {
            sum += n.getOutputValue() * n.outputConnections[index].weight;
        }

        // activate function or transfer /sig /gaussian /linear/ step
        outputValue = transferFunction.calculate(sum);
        this.sum = sum;
    }

    public void calculateOutputGradients(final double targetValue) {
        gradient = (targetValue - outputValue) * transferFunction.derivative(sum);
    }

    public double sumDOW(final Neuron[] nextLayer) {
        double sum = 0.0;

        // Sum our contributions of the errors at the nodes we feed.
        for (int n = 0; n < nextLayer.length - 1; n++) {
            sum += outputConnections[n].weight * nextLayer[n].gradient;
        }

        return sum;
    }

    public void calculateHiddenGradients(final Neuron[] nextLayer) {
        gradient = sumDOW(nextLayer) * transferFunction.derivative(sum);
    }

    public void updateInputWeights(final Neuron[] prevLayer, final double eta, final double momentum) {
        // The weights to be updated are in the Connection container
        // in the neurons in the preceding layer
        for (int n = 0; n < prevLayer.length; n++) {
            final Neuron neuron = prevLayer[n];
            final Connection conn = neuron.outputConnections[index];

            final double oldDeltaWeight = conn.deltaWeight;
            // Individual input, magnified by the gradient and train
            // rate:
            final double newDeltaWeight = eta
                    * neuron.getOutputValue()
                    * gradient
                    // Also add momentum = a fraction of the previous
                    // delta weight;
                    + momentum
                    * oldDeltaWeight;

            conn.deltaWeight = newDeltaWeight;
            conn.weight += newDeltaWeight;
        }

    }
}
