package it.units.erallab.hmsrobots.tasks.periodiclocomotion;

import it.units.erallab.hmsrobots.behavior.BehaviorUtils;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;

import java.util.Objects;
import java.util.stream.Collectors;

public class PeriodicOutcome extends Outcome {

  private final double length;

  public PeriodicOutcome(Outcome outcome, double length) {
    super(outcome.getObservations());
    this.length = length;
  }

  // TODO reason on goals positioning: maybe the "border" of the terrain should be the target
  public double getCoverage() {
    boolean forward = true;
    double[] robotPositions = getObservations().values().stream().mapToDouble(PeriodicOutcome::getRobotCenterPosition).toArray();
    double leftGoal = robotPositions[0];
    double rightGoal = length - leftGoal;
    double coverage = 0;
    for (double position : robotPositions) {
      if (forward && position >= rightGoal) {
        coverage += 1;
        forward = false;
      }
      if (!forward && position <= leftGoal) {
        coverage += 1;
        forward = true;
      }
    }
    double additionalCoverage;
    if (forward) {
      additionalCoverage = robotPositions[robotPositions.length - 1] - leftGoal;
    } else {
      additionalCoverage = rightGoal - robotPositions[robotPositions.length - 1];
    }
    return coverage + Math.max(0, additionalCoverage / (rightGoal - leftGoal));
  }

  private static double getRobotCenterPosition(Observation observation) {
    return BehaviorUtils.center(observation.getVoxelPolies().values().stream().filter(Objects::nonNull).collect(Collectors.toList())).x;
  }

}
