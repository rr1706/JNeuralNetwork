/* 
 * Copyright (c) 2015 RobotsByTheC. All rights reserved.
 *
 * Open Source Software - may be modified and shared by FRC teams. The code must
 * be accompanied by the BSD license file in the root directory of the project.
 */
package org.usfirst.frc.team2084.neuralnetwork;

/**
 * Represents a connection between two {@link Neuron}s, and holds a weight and
 * its delta.
 * 
 * @author Ben Wolsieffer
 */
public class Connection {

    public double weight = Math.random();
    public double deltaWeight = 0;
}
