/* 
 * Copyright (c) 2015 RobotsByTheC. All rights reserved.
 *
 * Open Source Software - may be modified and shared by FRC teams. The code must
 * be accompanied by the BSD license file in the root directory of the project.
 */
package org.usfirst.frc.team2084.neuralnetwork;

import java.util.Arrays;

/**
 * @author Ben Wolsieffer
 */
public class Neuron {

    private double outputValue;
    private double gradient;
    private String transferFunction;
    private final int index;

    private final Connection[] outputWeights;

    private static double transferFunctionTanH(double x) {
        return Math.tanh(x);
    }

    private static double transferFunctionTanHDerivative(double x) {
        double tanhx = Math.tanh(x);
        return 1.0 - tanhx * tanhx;
    }

    private static double transferFunctionSig(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private static double transferFunctionSigDerivative(double x) {
        double expnegx = Math.exp(-x);
        return expnegx / Math.pow((1 + expnegx), 2);
    }

    private static double transferFunctionStep(double x) {
        return x < 0 ? 0 : x;
    }

    private static double transferFunctionStepDerivative(double x) {
        return 1.0;
    }

    public Neuron(int numberOutputs, int index) {
        outputWeights = new Connection[numberOutputs];
        Arrays.stream(outputWeights).forEach((c) -> c = new Connection());
        this.index = index;
    }

    public Neuron(int numberOutputs, int index, String transferFunction) {
        this(numberOutputs, index);
        this.transferFunction = transferFunction;
    }

    public void setOutputValue(double value) {
        outputValue = value;
    }

    public double getOutputValue() {
        return outputValue;
    }

    public void feedForward(final Neuron[] prevLayer) {
        double sum = 0.0;

        // Sum the previous layer's outputs (which are our inputs)
        // Include the bias node from the previous layer.
        for (final Neuron n : prevLayer) {
            sum += n.getOutputValue() * n.outputWeights[index].weight;
        }

        // activate function or transfer /sig /gaussian /linear/ step
        // TODO: implement more transfer functions
        switch (transferFunction) {
        case "th":
            outputValue = transferFunctionTanH(sum);
        break;
        case "sig":
        default:
            outputValue = transferFunctionSig(sum);
        }
    }

    public void calculateOutputGradients(double targetValue) {
        final double delta = targetValue - outputValue;

        // TODO: implement more transfer functions
        switch (transferFunction) {
        case "th":
            gradient = delta * transferFunctionTanHDerivative(outputValue);
        break;
        case "sig":
        default:
            gradient = delta * transferFunctionSigDerivative(outputValue);
        }
    }

    public double sumDOW(Neuron[] nextLayer) {
        double sum = 0.0;

        // Sum our contributions of the errors at the nodes we feed.
        // TODO: why is it nextLayer.size() - 1?
        for (int n = 0; n < nextLayer.length - 1; n++) {
            sum += outputWeights[n].weight * nextLayer[n].gradient;
        }

        return sum;
    }

    public void calculateHiddenGradients(Neuron[] nextLayer) {
        // TODO: should this select correct transfer function?
        gradient = sumDOW(nextLayer) * transferFunctionTanHDerivative(outputValue);
    }

    public void updateInputWeights(Neuron[] prevLayer, double eta, double alpha) {
        // The weights to be updated are in the Connection container
        // in the neurons in the preceding layer
        for (int n = 0; n < prevLayer.length; n++) {
            final Neuron neuron = prevLayer[n];
            final Connection conn = neuron.outputWeights[index];

            final double oldDeltaWeight = conn.deltaWeight;
            // Individual input, magnified by the gradient and train
            // rate:
            final double newDeltaWeight = eta
                    * neuron.getOutputValue()
                    * gradient
                    // Also add momentum = a fraction of the previous
                    // delta weight;
                    + alpha
                    * oldDeltaWeight;

            conn.deltaWeight = newDeltaWeight;
            conn.weight += newDeltaWeight;
        }

    }
}
