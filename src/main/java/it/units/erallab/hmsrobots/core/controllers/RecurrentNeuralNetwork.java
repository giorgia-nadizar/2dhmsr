/*
 * Copyright (C) 2021 Eric Medvet <eric.medvet@gmail.com> (as Eric Medvet <eric.medvet@gmail.com>)
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package it.units.erallab.hmsrobots.core.controllers;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron.ActivationFunction;
import it.units.erallab.hmsrobots.core.snapshots.RNNState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;
import it.units.erallab.hmsrobots.util.Parametrized;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * @author eric
 */
public class RecurrentNeuralNetwork implements Serializable, RealFunction, Parametrized, Resettable, Snapshottable {

  private static final double P = 0.01;

  private final int inputNeurons;
  private final int recurrentNeurons;
  private final int outputNeurons;

  private final double[] inputNeuronsValues;
  private final double[] recurrentNeuronsValues;
  private final double[] outputNeuronsValues;

  @JsonProperty
  private final ActivationFunction activationFunction;
  @JsonProperty
  private final double[][] inputWeights;
  @JsonProperty
  private final double[][] recurrentWeights; // r(x,x) = 0
  @JsonProperty
  private final double[][] outputWeights;

  @JsonCreator
  public RecurrentNeuralNetwork(
      @JsonProperty("activationFunction") ActivationFunction activationFunction,
      @JsonProperty("inputWeights") double[][] inputWeights,
      @JsonProperty("recurrentWeights") double[][] recurrentWeights,
      @JsonProperty("outputWeights") double[][] outputWeights
  ) {
    inputNeurons = inputWeights.length;
    recurrentNeurons = recurrentWeights.length;
    outputNeurons = outputWeights[0].length;
    if (inputWeights[0].length != recurrentNeurons) {
      throw new IllegalArgumentException(String.format(
          "Wrong number of input weights: %d expected, %d found",
          recurrentNeurons,
          inputWeights[0].length
      ));
    }
    if (recurrentWeights[0].length != recurrentNeurons) {
      throw new IllegalArgumentException(String.format(
          "Wrong number of recurrent weights: %d expected, %d found",
          recurrentNeurons,
          recurrentWeights[0].length
      ));
    }
    if (outputWeights.length != recurrentNeurons) {
      throw new IllegalArgumentException(String.format(
          "Wrong number of output weights: %d expected, %d found",
          recurrentNeurons,
          outputWeights.length
      ));
    }
    this.activationFunction = activationFunction;
    this.inputWeights = inputWeights;
    this.recurrentWeights = recurrentWeights;
    this.outputWeights = outputWeights;
    inputNeuronsValues = new double[inputNeurons];
    recurrentNeuronsValues = new double[recurrentNeurons];
    outputNeuronsValues = new double[outputNeurons];
    this.reset();
  }

  public RecurrentNeuralNetwork(int inputNeurons, int recurrentNeurons, int outputNeurons, double[] flatWeights) {
    if (flatWeights.length != countWeights(inputNeurons, recurrentNeurons, outputNeurons)) {
      throw new IllegalArgumentException(String.format(
          "Wrong number of weights: %d expected, %d found",
          countWeights(inputNeurons, recurrentNeurons, outputNeurons),
          flatWeights.length
      ));
    }
    double[][][] weights = unflat(flatWeights, inputNeurons, recurrentNeurons, outputNeurons);
    this.inputNeurons = inputNeurons;
    this.recurrentNeurons = recurrentNeurons;
    this.outputNeurons = outputNeurons;
    this.activationFunction = ActivationFunction.TANH;
    this.inputWeights = weights[0];
    this.recurrentWeights = weights[1];
    this.outputWeights = weights[2];
    inputNeuronsValues = new double[inputNeurons];
    recurrentNeuronsValues = new double[recurrentNeurons];
    outputNeuronsValues = new double[outputNeurons];
    this.reset();
  }

  public RecurrentNeuralNetwork(int inputNeurons, int recurrentNeurons, int outputNeurons) {
    this(inputNeurons, recurrentNeurons, outputNeurons, new double[countWeights(inputNeurons, recurrentNeurons, outputNeurons)]);
  }

  @Override
  public double[] apply(double[] inputs) {
    if (inputs.length != inputNeurons) {
      throw new IllegalArgumentException(String.format("Expected input length is %d: found %d", inputNeurons, inputs.length));
    }
    IntStream.range(0, inputNeurons).forEach(i -> inputNeuronsValues[i] = activationFunction.apply(inputs[i]));
    double[] recurrentSynapticInputs = new double[recurrentNeurons];
    for (int i = 0; i < recurrentNeurons; i++) {
      for (int k = 0; k < inputNeurons; k++) {
        recurrentSynapticInputs[i] += inputWeights[k][i] * inputNeuronsValues[k];
      }
      for (int j = 0; j < recurrentNeurons; j++) {
        if (j == i) {
          continue;
        }
        recurrentSynapticInputs[i] += recurrentWeights[j][i] * recurrentNeuronsValues[j];
      }
    }
    for (int i = 0; i < recurrentNeurons; i++) {
      recurrentNeuronsValues[i] = (1 - P) * activationFunction.apply(recurrentSynapticInputs[i]) + P * recurrentNeuronsValues[i];
    }
    double[] outputs = new double[outputNeurons];
    for (int i = 0; i < recurrentNeurons; i++) {
      for (int j = 0; j < outputNeurons; j++) {
        outputs[j] += outputWeights[i][j] * recurrentNeuronsValues[i];
      }
    }
    IntStream.range(0, outputNeurons).forEach(i -> outputNeuronsValues[i] = activationFunction.apply(outputs[i]));
    return outputNeuronsValues;
  }

  @Override
  public int getInputDimension() {
    return inputNeurons;
  }

  @Override
  public int getOutputDimension() {
    return outputNeurons;
  }

  public double[][] getInputWeights() {
    return inputWeights;
  }

  public double[][] getRecurrentWeights() {
    return recurrentWeights;
  }

  public double[][] getOutputWeights() {
    return outputWeights;
  }


  private static double[] flat(double[][] inputWeights, double[][] recurrentWeights, double[][] outputWeights) {
    double[] flatInputWeights = flatWeights(inputWeights);
    double[] flatRecurrentWeights = flatRecurrentWeights(recurrentWeights);
    double[] flatOutputWeights = flatWeights(outputWeights);
    return DoubleStream.concat(DoubleStream.concat(Arrays.stream(flatInputWeights), Arrays.stream(flatRecurrentWeights)), Arrays.stream(flatOutputWeights)).toArray();
  }

  private static double[][][] unflat(double[] flatWeights, int inputNeurons, int recurrentNeurons, int outputNeurons) {
    int[] sizes = {inputNeurons * recurrentNeurons, (recurrentNeurons - 1) * recurrentNeurons, outputNeurons * recurrentNeurons};
    double[][][] unflatWeights = new double[3][][];
    unflatWeights[0] = unflatWeights(
        Arrays.copyOfRange(flatWeights, 0, sizes[0]),
        inputNeurons,
        recurrentNeurons
    );
    unflatWeights[1] = unflatRecurrentWeights(
        Arrays.copyOfRange(flatWeights, sizes[0], sizes[0] + sizes[1]),
        recurrentNeurons
    );
    unflatWeights[2] = unflatWeights(
        Arrays.copyOfRange(flatWeights, sizes[0] + sizes[1], sizes[0] + sizes[1] + sizes[2]),
        recurrentNeurons,
        outputNeurons
    );
    return unflatWeights;
  }

  private static double[] flatWeights(double[][] weights) {
    return Arrays.stream(weights).flatMapToDouble(Arrays::stream).toArray();
  }

  private static double[][] unflatWeights(double[] flatWeights, int startNeurons, int destNeurons) {
    return IntStream.range(0, startNeurons)
        .mapToObj(n -> Arrays.copyOfRange(flatWeights, n * destNeurons, destNeurons * (n + 1)))
        .toArray(double[][]::new);
  }

  private static double[] flatRecurrentWeights(double[][] recurrentWeights) {
    int recurrentNeurons = recurrentWeights.length;
    double[] flatRecurrentWeights = new double[(recurrentNeurons - 1) * recurrentNeurons];
    int flatWeightsIndex = 0;
    for (int i = 0; i < recurrentNeurons; i++) {
      for (int j = 0; j < recurrentNeurons; j++) {
        if (i == j) {
          continue;
        }
        flatRecurrentWeights[flatWeightsIndex] = recurrentWeights[i][j];
        flatWeightsIndex++;
      }
    }
    return flatRecurrentWeights;
  }

  private static double[][] unflatRecurrentWeights(double[] flatRecurrentWeights, int recurrentNeurons) {
    double[][] recurrentWeights = new double[recurrentNeurons][recurrentNeurons];
    int flatWeightsIndex = 0;
    for (int i = 0; i < recurrentNeurons; i++) {
      for (int j = 0; j < recurrentNeurons; j++) {
        if (i == j) {
          continue;
        }
        recurrentWeights[i][j] = flatRecurrentWeights[flatWeightsIndex];
        flatWeightsIndex++;
      }
    }
    return recurrentWeights;
  }

  @Override
  public double[] getParams() {
    return flat(inputWeights, recurrentWeights, outputWeights);
  }

  @Override
  public void setParams(double[] params) {
    double[][][] unflatParams = unflat(params, inputNeurons, recurrentNeurons, outputNeurons);
    for (int i = 0; i < unflatParams[0].length; i++) {
      System.arraycopy(unflatParams[0][i], 0, inputWeights[i], 0, unflatParams[0][i].length);
    }
    for (int i = 0; i < unflatParams[1].length; i++) {
      System.arraycopy(unflatParams[1][i], 0, recurrentWeights[i], 0, unflatParams[1][i].length);
    }
    for (int i = 0; i < unflatParams[2].length; i++) {
      System.arraycopy(unflatParams[2][i], 0, outputWeights[i], 0, unflatParams[2][i].length);
    }
  }

  public static int countWeights(int inputNeurons, int recurrentNeurons, int outputNeurons) {
    return inputNeurons * recurrentNeurons + recurrentNeurons * (recurrentNeurons - 1) + recurrentNeurons * outputNeurons;
  }

  @Override
  public Snapshot getSnapshot() {
    return new Snapshot(
        new RNNState(inputNeuronsValues, recurrentNeuronsValues, outputNeuronsValues, inputWeights, recurrentWeights, outputWeights, activationFunction.getDomain()),
        getClass()
    );
  }

  @Override
  public void reset() {
    IntStream.range(0, inputNeurons).forEach(i -> inputNeuronsValues[i] = 0);
    IntStream.range(0, recurrentNeurons).forEach(i -> recurrentNeuronsValues[i] = 0);
    IntStream.range(0, outputNeurons).forEach(i -> outputNeuronsValues[i] = 0);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    RecurrentNeuralNetwork that = (RecurrentNeuralNetwork) o;
    return inputNeurons == that.inputNeurons && recurrentNeurons == that.recurrentNeurons && outputNeurons == that.outputNeurons && activationFunction == that.activationFunction && Arrays.equals(inputWeights, that.inputWeights) && Arrays.equals(recurrentWeights, that.recurrentWeights) && Arrays.equals(outputWeights, that.outputWeights);
  }

  @Override
  public int hashCode() {
    int result = Objects.hash(inputNeurons, recurrentNeurons, outputNeurons, activationFunction);
    result = 31 * result + Arrays.hashCode(inputWeights);
    result = 31 * result + Arrays.hashCode(recurrentWeights);
    result = 31 * result + Arrays.hashCode(outputWeights);
    return result;
  }

  @Override
  public String toString() {
    return "RNN." + activationFunction.toString().toLowerCase() + "[" +
        inputNeurons + "," + recurrentNeurons + "," + outputNeurons + "]";
  }
}
