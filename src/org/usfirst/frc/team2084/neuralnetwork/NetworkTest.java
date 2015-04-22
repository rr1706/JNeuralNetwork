/* 
 * Copyright (c) 2015 RobotsByTheC. All rights reserved.
 *
 * Open Source Software - may be modified and shared by FRC teams. The code must
 * be accompanied by the BSD license file in the root directory of the project.
 */
package org.usfirst.frc.team2084.neuralnetwork;

import java.io.File;
import java.util.Arrays;

/**
 * @author Ben Wolsieffer
 */
public class NetworkTest {

    public static final int MAX_EPOCHS = 100000;
    public static final double MAX_ERROR = 0.001;

    public static void main(String[] args) {
        try {
            Data test = new Data(new File("data/out.txt"));
            Network network = test.getNetwork();

            double[][] inputs = test.getInputs();
            double[][] targetOutputs = test.getTargetOutputs();

            System.out.println(Arrays.deepToString(inputs));
            System.out.println(Arrays.deepToString(targetOutputs));

            network.feedForward(new double[] { 1, 0 });
            System.out.println(Arrays.toString(network.getResults()));

            int epochs = 0;
            double averageError = 0;
            while (true) {
                double error = 0;

                epochs++;
                for (int i = 0; i < inputs.length; i++) {
                    double[] input = inputs[i];
                    double[] targetOutput = targetOutputs[i];
                    network.feedForward(input);
                    network.backPropagation(targetOutput);
                    error += network.getRecentAverageError();
                }

                averageError = error / inputs.length;
                if (epochs % 1000 == 0) {
                    System.out.println("Error: " + averageError);
                }

                if (averageError < MAX_ERROR) {
                    System.out.println("==============");
                    System.out.println("Took " + epochs + " epochs to converge.");
                    break;
                } else if (epochs >= MAX_EPOCHS) {
                    System.out.println("==============");
                    System.out.println("Did not converge after " + epochs + " epochs.");
                    break;
                }
            }
            test.save(new File("data/out.txt"));
            // network.feedForward(new double[] { 1, 1 });
            // System.out.println(Arrays.toString(network.getResults()));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
