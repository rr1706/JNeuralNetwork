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
 * A simple test of the neural network code.
 * 
 * @author Ben Wolsieffer
 */
public class NetworkTest {

    public static final int MAX_EPOCHS = 100000;
    public static final double MAX_ERROR = 1;

    public static void main(String[] args) {
        // Uncomment the following lines to choose a test:
        // testFile();
        selfLearning();
    }

    /**
     * A robot "simulation", used to test the unsupervised learning ability.
     */
    public static class Robot {

        public static final double MAX_SPEED = 50;
        public static final double TIME_STEP = 0.05;

        public double heading = 0;

        public void rotate(double speed) {
            System.out.println("speed: " + speed);
            heading += speed * TIME_STEP * MAX_SPEED;
        }
    }

    /**
     * Demonstrates how the network can learn to rotate a "robot" to a desired
     * location. This code is similar to what would be used on an FRC robot.
     */
    public static void selfLearning() {
        try {
            Robot robot = new Robot();
            // Load the robot network characteristics from a file
            Data data = new Data(new File("data/robot.txt"));
            Network network = data.getNetwork();

            while (true) {
                double error = 1;
                // Pick a random desired heading
                double desired = Math.random() * 360;
                System.out.println("DESIRED: " + desired);
                // Wait a second, to make it readable
                Thread.sleep(1000);
                do {
                    // Feed forward the heading error
                    network.feedForward(desired - robot.heading);
                    // Rotate the robot at the speed given by the output of last
                    // hidden layer
                    robot.rotate(network.getLayer(2)[0].getOutputValue());
                    // Apply that rotation speed for a certain amount of time
                    Thread.sleep((int) (Robot.TIME_STEP * 1000));
                    // Set the output neuron to the new heading error. This is
                    // the kind of weird part, because it tricks the
                    // back-propagation algorithm into minimizing the difference
                    // between the real and desired headings.
                    network.setLayerOutputs(3, desired - robot.heading);
                    // Back-propagate to adjust weights to minimize error
                    network.backPropagation(0);
                    error = network.getRecentAverageError();
                    System.out.println("heading: " + robot.heading);
                    System.out.println("error: " + error);
                } while (error > MAX_ERROR);
                System.out.println("+++++++++++++++++++");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Trains the network using a standard input file. This is not that useful
     * for an FRC robot.
     */
    public static void testFile() {
        try {
            Data data = new Data(new File("data/not.txt"));
            Network network = data.getNetwork();

            double[][] inputs = data.getInputs();
            double[][] targetOutputs = data.getTargetOutputs();

            System.out.println(Arrays.deepToString(inputs));
            System.out.println(Arrays.deepToString(targetOutputs));

            network.feedForward(0);
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
            // Save the network to an output file
            data.save(new File("data/out.txt"));
            // network.feedForward(new double[] { 1, 1 });
            // System.out.println(Arrays.toString(network.getResults()));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
