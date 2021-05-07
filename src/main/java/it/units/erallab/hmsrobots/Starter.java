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
package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.*;
import it.units.erallab.hmsrobots.core.controllers.snn.*;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.AverageFrequencySpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.MovingAverageSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.SpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.UniformWithMemoryValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.ValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.objects.ControllableVoxel;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.*;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.FramesImageBuilder;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.GridOnlineViewer;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.dyn4j.dynamics.Settings;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

/**
 * @author Eric Medvet <eric.medvet@gmail.com>
 */
public class Starter {

  private static void sampleExecution() throws IOException {
    final Locomotion locomotion = new Locomotion(
        20,
        Locomotion.createTerrain("flat"),
        new Settings()
    );
    final ControllableVoxel hardMaterialVoxel = new ControllableVoxel(
        Voxel.SIDE_LENGTH,
        Voxel.MASS_SIDE_LENGTH_RATIO,
        50d,
        Voxel.SPRING_D,
        Voxel.MASS_LINEAR_DAMPING,
        Voxel.MASS_ANGULAR_DAMPING,
        Voxel.FRICTION,
        Voxel.RESTITUTION,
        Voxel.MASS,
        Voxel.LIMIT_CONTRACTION_FLAG,
        Voxel.MASS_COLLISION_FLAG,
        Voxel.AREA_RATIO_MAX_DELTA,
        Voxel.SPRING_SCAFFOLDINGS,
        ControllableVoxel.MAX_FORCE,
        ControllableVoxel.ForceMethod.DISTANCE
    );
    final ControllableVoxel softMaterialVoxel = new ControllableVoxel(
        Voxel.SIDE_LENGTH,
        Voxel.MASS_SIDE_LENGTH_RATIO,
        5d,
        Voxel.SPRING_D,
        Voxel.MASS_LINEAR_DAMPING,
        Voxel.MASS_ANGULAR_DAMPING,
        Voxel.FRICTION,
        Voxel.RESTITUTION,
        Voxel.MASS,
        Voxel.LIMIT_CONTRACTION_FLAG,
        Voxel.MASS_COLLISION_FLAG,
        Voxel.AREA_RATIO_MAX_DELTA,
        EnumSet.of(Voxel.SpringScaffolding.SIDE_EXTERNAL, Voxel.SpringScaffolding.CENTRAL_CROSS),
        ControllableVoxel.MAX_FORCE,
        ControllableVoxel.ForceMethod.DISTANCE
    );
    int w = 20;
    int h = 5;
    Robot<ControllableVoxel> robot = new Robot<>(
        new TimeFunctions(Grid.create(
            w, h,
            (x, y) -> (Double t) -> Math.sin(-2 * Math.PI * t + Math.PI * ((double) x / (double) w))
        )),
        Grid.create(
            w, h,
            (x, y) -> (y == 0) ? SerializationUtils.clone(hardMaterialVoxel) : SerializationUtils.clone(softMaterialVoxel)
        )
    );
    FramesImageBuilder framesImageBuilder = new FramesImageBuilder(
        5, 5.5, 0.1, 300, 200, FramesImageBuilder.Direction.HORIZONTAL
    );
    Outcome result = locomotion.apply(robot, framesImageBuilder);
    BufferedImage image = framesImageBuilder.getImage();
    System.out.println("Outcome: " + result);
  }

  public static void main(String[] args) {
    //comb();
    //lShaped();
    //snn();
    bipeds();
    //rollingOne();
    //jumpingTwo();
    //rollingBall();
    //breakingWorm();
    //plainWorm();
    //cShaped();
    //multiped();
  }

  private static void lShaped() {
    //one voxel robot
    Grid<SensingVoxel> body = Grid.create(5, 5, (x, y) -> (x == 0 || y == 4) ? new SensingVoxel(List.of(
        new Velocity(true, 8d, Velocity.Axis.X, Velocity.Axis.Y),
        new Derivative(new Velocity(true, 4d, Velocity.Axis.X, Velocity.Axis.Y)),
        new Angle(),
        new Lidar(10, Map.of(Lidar.Side.E, 4))
    )) : null);
    Robot<?> robot = new Robot<>(
        new TimeFunctions(Grid.create(5, 5, t -> t > 2 ? 1d : 0d)),
        body
    );
    //episode
    Locomotion locomotion = new Locomotion(
        5,
        Locomotion.createTerrain("flat"),
        new Settings()
    );
    Robot<?> r00 = robot;
    Robot<?> r01 = SerializationUtils.clone(robot, SerializationUtils.Mode.JAVA);
    Robot<?> r02 = SerializationUtils.clone(robot, SerializationUtils.Mode.JSON);
    System.out.println(locomotion.apply(r00));
    System.out.println(locomotion.apply(r01));
    System.out.println(locomotion.apply(r02));
    Robot<?> r10 = robot;
    Robot<?> r11 = SerializationUtils.clone(robot, SerializationUtils.Mode.JAVA);
    Robot<?> r12 = SerializationUtils.clone(robot, SerializationUtils.Mode.JSON);
    GridOnlineViewer.run(locomotion, List.of(r01, r11, r12));
    System.out.println(locomotion.apply(r10));
    System.out.println(locomotion.apply(r11));
    System.out.println(locomotion.apply(r12));
    //Robot<?> r21 = SerializationUtils.clone(robot, SerializationUtils.Mode.JAVA);
    //r21.reset();
  }

  private static void comb() {
    //one voxel robot
    Grid<? extends SensingVoxel> body = RobotUtils.buildSensorizingFunction("uniform-t+a+vxy-0.01").apply(RobotUtils.buildShape("comb-9x4"));
    Robot<?> robot = new Robot<>(
        new TimeFunctions(Grid.create(5, 5, t -> t > 2 ? 1d : 0d)),
        body
    );
    //episode
    Locomotion locomotion = new Locomotion(
        5,
        Locomotion.createTerrain("flat"),
        new Settings()
    );
    Robot<?> r00 = robot;
    Robot<?> r01 = SerializationUtils.clone(robot, SerializationUtils.Mode.JAVA);
    Robot<?> r02 = SerializationUtils.clone(robot, SerializationUtils.Mode.JSON);
    System.out.println(locomotion.apply(r00));
    System.out.println(locomotion.apply(r01));
    System.out.println(locomotion.apply(r02));
    Robot<?> r10 = robot;
    Robot<?> r11 = SerializationUtils.clone(robot, SerializationUtils.Mode.JAVA);
    Robot<?> r12 = SerializationUtils.clone(robot, SerializationUtils.Mode.JSON);
    GridOnlineViewer.run(locomotion, List.of(r01, r11, r12));
    System.out.println(locomotion.apply(r10));
    System.out.println(locomotion.apply(r11));
    System.out.println(locomotion.apply(r12));
    //Robot<?> r21 = SerializationUtils.clone(robot, SerializationUtils.Mode.JAVA);
    //r21.reset();
  }

  private static void snn() {
    Random random = new Random();
    Grid<? extends SensingVoxel> body = RobotUtils.buildSensorizingFunction("spinedTouch-t-f-0").apply(RobotUtils.buildShape("biped-4x3"));
    //centralized sensing
    CentralizedSensing centralizedSensing = new CentralizedSensing(body);
    MultilayerSpikingNetworkWithConverters msn = new MultilayerSpikingNetworkWithConverters(
        centralizedSensing.nOfInputs(),
        new int[]{2},
        centralizedSensing.nOfOutputs(),
        (BiFunction<Integer, Integer, SpikingFunction>) (l, i) -> new LIFNeuron()
    );
    double[] ws = msn.getParams();
    IntStream.range(0, ws.length).forEach(i -> ws[i] = random.nextDouble() * 2d - 1d);
    msn.setParams(ws);
    centralizedSensing.setFunction(msn);
    Robot<SensingVoxel> robot = new Robot<>(
        centralizedSensing,
        SerializationUtils.clone(body)
    );
    Locomotion locomotion = new Locomotion(
        10,
        Locomotion.createTerrain("hilly-0.3-1-0"),
        new Settings()
    );
    //GridOnlineViewer.run(locomotion, robot);
    try {
      System.out.println("doing 1st video");
      GridFileWriter.save(locomotion, robot, 300, 200, 0, 25, VideoUtils.EncoderFacility.FFMPEG_SMALL, new File("/home/eric/video.1.mp4"));
      robot = SerializationUtils.clone(robot);
      System.out.println("doing 2nd video");
      GridFileWriter.save(locomotion, robot, 300, 200, 0, 25, VideoUtils.EncoderFacility.FFMPEG_SMALL, new File("/home/eric/video.2.mp4"));
    } catch (IOException e) {
      e.printStackTrace();
    }
    /*FramesImageBuilder framesImageBuilder = new FramesImageBuilder(5, 7, .75, 600, 300, FramesImageBuilder.Direction.VERTICAL);
    locomotion.apply(robot, framesImageBuilder);
    try {
      ImageIO.write(framesImageBuilder.getImage(), "png", new File("/home/eric/frames-multiped.png"));
    } catch (IOException e) {
      e.printStackTrace();
    }*/
  }

  private static void bipeds() {
    Grid<? extends SensingVoxel> body = RobotUtils.buildSensorizingFunction("spinedTouchFootSighted-f-f-0.01").apply(RobotUtils.buildShape("biped-4x3"));
    //simple
    double f = 1d;
    Robot<ControllableVoxel> phasesRobot = new Robot<>(
        new TimeFunctions(Grid.create(
            body.getW(),
            body.getH(),
            (final Integer x, final Integer y) -> (Double t) -> Math.sin(-2 * Math.PI * f * t + Math.PI * ((double) x / (double) body.getW()))
        )),
        SerializationUtils.clone(body)
    );
    //distribute sensing
    Random random = new Random();
    /*DistributedSensing distributedSensing = new DistributedSensing(body, 1);
    for (Grid.Entry<? extends SensingVoxel> entry : body) {
      MultiLayerPerceptron mlp = new MultiLayerPerceptron(
          MultiLayerPerceptron.ActivationFunction.TANH,
          distributedSensing.nOfInputs(entry.getX(), entry.getY()),
          new int[]{2},
          distributedSensing.nOfOutputs(entry.getX(), entry.getY())
      );
      double[] ws = mlp.getParams();
      IntStream.range(0, ws.length).forEach(i -> ws[i] = random.nextDouble() * 2d - 1d);
      mlp.setParams(ws);
      distributedSensing.getFunctions().set(entry.getX(), entry.getY(), mlp);
    }
    Robot<SensingVoxel> distHetero = new Robot<>(
        distributedSensing,
        SerializationUtils.clone(body)
    );*/
    DistributedSpikingSensing distributedSpikingSensing = new DistributedSpikingSensing(body, 1, new LIFNeuronWithHomeostasis(), new UniformWithMemoryValueToSpikeTrainConverter(), new AverageFrequencySpikeTrainToValueConverter());
    for (Grid.Entry<? extends SensingVoxel> entry : body) {
      MultilayerSpikingNetwork multilayerSpikingNetwork = new MultilayerSpikingNetwork(
          distributedSpikingSensing.nOfInputs(entry.getX(), entry.getY()),
          new int[]{2},
          distributedSpikingSensing.nOfOutputs(entry.getX(), entry.getY()),
          new LIFNeuronWithHomeostasis()
      );
      double[] ws = multilayerSpikingNetwork.getParams();
      IntStream.range(0, ws.length).forEach(i -> ws[i] = random.nextDouble() * 8d - 3d);
      multilayerSpikingNetwork.setParams(ws);
      distributedSpikingSensing.getFunctions().set(entry.getX(), entry.getY(), multilayerSpikingNetwork);
    }
    Robot<SensingVoxel> distHetero = new Robot<>(
        distributedSpikingSensing,
        SerializationUtils.clone(body)
    );

    //centralized sensing
    CentralizedSensing centralizedSensing = new CentralizedSensing(body, true);
    MultiLayerPerceptron mlp = new MultiLayerPerceptron(
        MultiLayerPerceptron.ActivationFunction.TANH,
        centralizedSensing.nOfInputs(),
        new int[]{2},
        centralizedSensing.nOfOutputs()
    );
    double[] ws = mlp.getParams();
    IntStream.range(0, ws.length).forEach(i -> ws[i] = random.nextDouble() * 2d - 1d);
    mlp.setParams(ws);
    centralizedSensing.setFunction(mlp);
    MultilayerSpikingNetworkWithConverters msn = new MultilayerSpikingNetworkWithConverters(
        centralizedSensing.nOfInputs(), new int[]{2}, centralizedSensing.nOfOutputs(), new LIFNeuron()
    );
    double[] msnWeights = msn.getParams();
    IntStream.range(0, msnWeights.length).forEach(i -> msnWeights[i] = random.nextDouble() * 2d - 1d);
    msn.setParams(msnWeights);
    centralizedSensing.setFunction(msn);
    Robot<SensingVoxel> centralized = new Robot<>(
        centralizedSensing,
        SerializationUtils.clone(body)
    );
    //episode
    Locomotion locomotion = new Locomotion(
        10,
        Locomotion.createTerrain("flat"),
        new Settings()
    );
    Grid<Pair<String, Robot<?>>> namedSolutionGrid = Grid.create(1, 3);
    namedSolutionGrid.set(0, 0, Pair.of("dist-hetero", distHetero));
    namedSolutionGrid.set(0, 1, Pair.of("centralized", centralized));
    namedSolutionGrid.set(0, 2, Pair.of("phasesRobot", phasesRobot));
    GridOnlineViewer.run(locomotion, namedSolutionGrid);
  }

  private static void multiped() {
    double f = 1d;
    Grid<Boolean> body = Grid.create(7, 2, (x, y) -> y == 1 || (x % 2 == 0));
    Robot<?> robot = new Robot<>(
        new TimeFunctions(Grid.create(
            body.getW(),
            body.getH(),
            (final Integer x, final Integer y) -> (Double t) -> Math.sin(
                -2 * Math.PI * f * t + 2 * Math.PI * ((double) x / (double) body.getW()) + Math.PI * ((double) y / (double) body.getH())
            )
        )),
        RobotUtils.buildSensorizingFunction("uniform-a-0.01").apply(body)
    );
    Locomotion locomotion = new Locomotion(
        10,
        Locomotion.createTerrain("hilly-0.3-1-0"),
        new Settings()
    );
    GridOnlineViewer.run(locomotion, robot);
    /*FramesImageBuilder framesImageBuilder = new FramesImageBuilder(5, 7, .75, 600, 300, FramesImageBuilder.Direction.VERTICAL);
    locomotion.apply(robot, framesImageBuilder);
    try {
      ImageIO.write(framesImageBuilder.getImage(), "png", new File("/home/eric/frames-multiped.png"));
    } catch (IOException e) {
      e.printStackTrace();
    }*/
  }

  private static void breakingWorm() {
    Grid<? extends SensingVoxel> body = RobotUtils.buildSensorizingFunction("spinedTouch-f-t-0").apply(RobotUtils.buildShape("worm-4x3"));
    double f = 1d;
    Robot<ControllableVoxel> unbreakableRobot = new Robot<>(
        new TimeFunctions(Grid.create(
            body.getW(),
            body.getH(),
            (final Integer x, final Integer y) -> (Double t) -> Math.sin(-2 * Math.PI * f * t + Math.PI * ((double) x / (double) body.getW()))
        )),
        body
    );
    Robot<?> breakableRobot = RobotUtils.buildRobotTransformation(
        "breakable-area-1000/500-3/0.5-0",
        new Random(0)
    ).apply(SerializationUtils.clone(unbreakableRobot));
    //episode
    Locomotion locomotion = new Locomotion(
        60,
        Locomotion.createTerrain("hilly-0.5-5-0"),
        new Settings()
    );
    Grid<Pair<String, Robot<?>>> namedSolutionGrid = Grid.create(1, 2);
    namedSolutionGrid.set(0, 0, Pair.of("unbreakable", unbreakableRobot));
    namedSolutionGrid.set(0, 1, Pair.of("breakable", breakableRobot));
    GridOnlineViewer.run(locomotion, namedSolutionGrid);
  }

  private static void plainWorm() {
    Grid<? extends SensingVoxel> body = RobotUtils
        .buildSensorizingFunction("uniform-l1-0.01")
        //.buildSensorizingFunction("uniform-l5-0")
        //.buildSensorizingFunction("uniformAll-0")
        .apply(RobotUtils.buildShape("worm-5x2"));
    double f = 1d;
    Robot<?> robot = new Robot<>(
        new TimeFunctions(Grid.create(
            body.getW(),
            body.getH(),
            (final Integer x, final Integer y) -> (Double t) -> Math.sin(
                -2 * Math.PI * f * t + 2 * Math.PI * ((double) x / (double) body.getW()) + Math.PI * ((double) y / (double) body.getH())
            )
            //(x, y) -> t -> Math.signum(Math.sin(-2 * Math.PI * (f + (x > body.getW() / 2d ? 1 : 0)) * t))
        )),
        SerializationUtils.clone(body)
    );
    robot = RobotUtils.buildRobotTransformation("broken-0.0-0", new Random(0)).apply(robot);
    //episode
    Locomotion locomotion = new Locomotion(
        10,
        Locomotion.createTerrain("flatWithStart-2"),
        new Settings()
    );
    GridOnlineViewer.run(locomotion, robot);
  }

  private static void cShaped() {
    Grid<Boolean> shape = Grid.create(8, 8, (x, y) -> x < 2 || y < 2 || y >= 6);
    Grid<? extends SensingVoxel> body = RobotUtils.buildSensorizingFunction("uniform-t-0").apply(shape);
    Robot<?> robot = new Robot<>(
        new CentralizedSensing(body),
        SerializationUtils.clone(body)
    );
    //episode
    Locomotion locomotion = new Locomotion(
        10,
        Locomotion.createTerrain("flatWithStart-2"),
        new Settings()
    );
    //GridOnlineViewer.run(locomotion, robot);
    try {
      GridFileWriter.save(locomotion, robot, 300, 200, 0, 25, VideoUtils.EncoderFacility.FFMPEG_SMALL, new File("/home/eric/cshaped.mp4"));
    } catch (IOException e) {
      e.printStackTrace();
    }
    GridOnlineViewer.run(locomotion, robot);
  }

  private static void rollingOne() {
    //one voxel robot
    Grid<SensingVoxel> oneBody = Grid.create(1, 1, new SensingVoxel(List.of(
        new Velocity(true, 8d, Velocity.Axis.X, Velocity.Axis.Y),
        new Derivative(new Velocity(true, 4d, Velocity.Axis.X, Velocity.Axis.Y)),
        new Angle(),
        new Lidar(10, Map.of(Lidar.Side.E, 4))
    )));
    Robot<SensingVoxel> robot = new Robot<>(
        new CentralizedSensing(oneBody, RealFunction.build(in -> new double[]{0d}, 1, 1)),
        oneBody
    );
    //episode
    Locomotion locomotion = new Locomotion(
        60,
        new double[][]{new double[]{0, 10, 30, 31, 100, 1000, 1010}, new double[]{100, 100, 80, 10, 10, 0, 100}},
        new Settings()
    );
    GridOnlineViewer.run(locomotion, robot);
  }

  public static SpikingNeuron[][] jumpingTwo() {
    //two voxels robot
    Grid<SensingVoxel> oneBody = Grid.create(1, 2, (x, y) -> new SensingVoxel(List.of(new Touch())));
    SpikingNeuron[][] spikingNeurons = new SpikingNeuron[2][2];
    for (int i = 0; i < spikingNeurons.length; i++)
      for (int j = 0; j < spikingNeurons[i].length; j++)
        spikingNeurons[i][j] = new IzhikevicNeuron(true);
    double[][][] weights = new double[1][2][2];
    weights[0][0][0] = 10;
    weights[0][0][1] = 0;
    weights[0][1][0] = 0;
    weights[0][1][1] = 0;
    MultilayerSpikingNetworkWithConverters m = new MultilayerSpikingNetworkWithConverters(spikingNeurons, weights,
        new ValueToSpikeTrainConverter[]{
            new UniformWithMemoryValueToSpikeTrainConverter(50),
            new UniformWithMemoryValueToSpikeTrainConverter(50)},
        new SpikeTrainToValueConverter[]{
            new MovingAverageSpikeTrainToValueConverter(50, 5),
            new MovingAverageSpikeTrainToValueConverter(50, 5),
        });
    //MultiLayerPerceptron m = new MultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.TANH,2,new int[]{}, 2, new double[]{0,-100,0,0,0,0});
    Robot<SensingVoxel> robot = new Robot<>(
        new CentralizedSensing(oneBody, m),
        oneBody
    );
    //episode
    Locomotion locomotion = new Locomotion(
        5,
        //new double[][]{new double[]{0, 10, 30, 31, 100, 1000, 1010}, new double[]{100, 100, 80, 10, 10, 0, 100}},
        Locomotion.createTerrain("flat"),
        new Settings()
    );
    GridOnlineViewer.run(locomotion, robot);
    return spikingNeurons;
  }

  private static void rollingBall() {
    Random random = new Random();
    Grid<Boolean> shape = RobotUtils.buildShape("ball-7");
    Grid<? extends SensingVoxel> body = RobotUtils.buildSensorizingFunction("uniform-ax+t+l5-0").apply(shape);
    //centralized sensing
    CentralizedSensing centralizedSensing = new CentralizedSensing(body);
    MultiLayerPerceptron mlp = new MultiLayerPerceptron(
        MultiLayerPerceptron.ActivationFunction.TANH,
        centralizedSensing.nOfInputs(),
        new int[0],
        centralizedSensing.nOfOutputs()
    );
    double[] ws = mlp.getParams();
    IntStream.range(0, ws.length).forEach(i -> ws[i] = random.nextGaussian());
    mlp.setParams(ws);
    centralizedSensing.setFunction(mlp);
    Robot<SensingVoxel> robot = new Robot<>(
        centralizedSensing,
        SerializationUtils.clone(body)
    );
    //episode
    Locomotion locomotion = new Locomotion(
        60,
        Locomotion.createTerrain("downhill-30"),
        new Settings()
    );
    GridOnlineViewer.run(locomotion, robot);
  }

}
