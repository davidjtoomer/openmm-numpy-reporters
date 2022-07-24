# Adaptation of the ForceReporter class from
# http://docs.openmm.org/latest/userguide/application/04_advanced_sim_examples.html#extracting-and-reporting-forces-and-other-data
# using appendable Numpy files.

from openmm import State
from openmm.app import Simulation
from openmm.unit import *
from . import NumpyAppendFile


class ForceReporter:
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
        forces = state.getForces(
            asNumpy=True).value_in_unit(
            kilocalories_per_mole /
            angstrom)
        with NumpyAppendFile(self.filename) as file:
            file.append(forces)

    def describeNextReport(self, simulation: Simulation):
        '''
        Get information about the next report that will be generated.

        Parameters
        ----------
        simulation : Simulation
            The simulation to generate a report for.
        '''
        steps = self.report_interval - simulation.currentStep % self.report_interval
        return (steps, False, False, True, False, None)
