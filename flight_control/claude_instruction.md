# Background
We have the newest communication from the competition stored in `background_info/email_1.md` but we are only in the virutal qualifier round 1

Right now we do not have the DCL software setup because we don't have the access keys so we'll have to run simulations to smoke test our code

There is a rough draft of a simulator in ./simulation to view the 3d model of the gates. We may need to pip install a python library to get a high fidelity simulation


# Tasks
## Flight control logic
Existing code in `/flight_control` is just a rough draft and needs to be optimized and upgraded
* Ensure that the code will work with the simulation & accepts the outputs from the gate detection algorithm
* For now, we should fly towards the nearest gate that is highlighted

## Simulation
* The simulated environment should create a random sequence of gates within a confined 3D space and highlight the gate that the drone passes through. After the drone passes through a gate that's highlighted, it must highlight the next gate and un-highlight the passed gate

* After all the gates have been passed the simulation should terminate

* When testing our algorithms in the simulation testbench, we should be able to view a 3rd person POV as well as a 1st person POV of the simulation. In the 1st person POV it should display the gate detection boundary box that's being drawn up with the confidence score

## Tuning gate detection logic for phase 1:
- Given that the gates for phase 1 is highlighted we can write a phase_1 logic, separate from what we have right now, that takes advantage of the fact that the gates will be highlighted 



