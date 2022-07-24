from openmm import State
from openmm.app import Simulation
from openmm.unit import *
from . import NumpyAppendFile


class PotentialEnergyReporter:
    def __init__(self, filename: str, report_interval: int):
        '''
        Initialize a ForceReporter object.

        Parameters
        ----------
        filename : str
            The filename of the file to write to.
        report_interval : int
            The interval (in time steps) at which to write frames.
        '''
        self.filename = filename
        self.report_interval = report_interval

    def __del__(self):
        '''
        Close the file to delete the ForceReporter object.
        '''
        self.fd.close()

    def report(self, simulation: Simulation, state: State):
        '''
        Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The simulation to generate a report for.
        state : State
            The current state of the simulation.
        '''
        potential_energy = state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
        with NumpyAppendFile(self.filename) as file:
            # potential_energies is a scalar - wrap it in a 1-element array
            file.append([potential_energy])

    def describeNextReport(self, simulation: Simulation):
        '''
        Get information about the next report that will be generated.

        Parameters
        ----------
        simulation : Simulation
            The simulation to generate a report for.
        '''
        steps = self.report_interval - simulation.currentStep % self.report_interval
        return (steps, False, False, False, True, None)
