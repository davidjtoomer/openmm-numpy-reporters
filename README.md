# openmm-numpy-reporters
This repository provides reporters for molecular dynamics (MD) simulations in OpenMM saved in the NumPy file format (.npy). These reporters are ideal for storing quantities that are neither automatically stored by [OpenMM reporters nor by MDTraj reporters](https://mdtraj.org/development/api/reporters.html#:~:text=MDTraj%20currently%20provides%20three%20reporters,featured%20trajectory%20file%20format%20available.) (e.g. forces). The resulting files are easily loaded into Python programs using <code>numpy.load(...)</code>.

Previous approaches to storing data not covered by OpenMM or MDTraj include

1. Storing all of the data in a growing NumPy array during the simulation and saving it to a NumPy file at the end

2. Appending data to a text file during the simulation, and then loading and converting that data to a NumPy array to save after the simulation

In either case, all of the data has to be loaded into an array at one or more timepoints during the program before it can be saved as a NumPy file. For long-time MD simulations of potentially large systems, this is extremely memory-intensive.

These reporters are memory-efficient because they avoid loading all of the data at any given point in time. Instead, these reporters append to an existing NumPy file. However, <code>numpy.save</code> does not have an append mode. To work around this, these reporters manipulate the NumPy file format to append the data. Doing so reduces the space complexity down to the number of reports in one simulation step rather than in the total number of simulation steps. 
