package it.units.erallab.hmsrobots.tasks.periodiclocomotion;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.snapshots.SnapshotListener;
import it.units.erallab.hmsrobots.tasks.AbstractTask;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import org.dyn4j.dynamics.Settings;

import static it.units.erallab.hmsrobots.tasks.locomotion.Locomotion.TERRAIN_BORDER_HEIGHT;
import static it.units.erallab.hmsrobots.tasks.locomotion.Locomotion.TERRAIN_BORDER_WIDTH;

public class PeriodicLocomotion extends AbstractTask<Robot<?>, PeriodicOutcome> {

  private final double finalT;
  private final int length;
  private final double[][] groundProfile;

  public PeriodicLocomotion(double finalT, int length, Settings settings) {
    super(settings);
    this.finalT = finalT;
    this.length = length;
    groundProfile = createTerrain(length);
  }

  private static double[][] createTerrain(int length) {
    return new double[][]{
        new double[]{0, TERRAIN_BORDER_WIDTH, length - TERRAIN_BORDER_WIDTH, length},
        new double[]{TERRAIN_BORDER_HEIGHT, 5, 5, TERRAIN_BORDER_HEIGHT}
    };
  }

  @Override
  public PeriodicOutcome apply(Robot<?> solution, SnapshotListener listener) {
    Locomotion locomotion = new Locomotion(finalT, groundProfile, settings);
    Outcome outcome = locomotion.apply(solution, listener);
    return new PeriodicOutcome(outcome, length);
  }
}
