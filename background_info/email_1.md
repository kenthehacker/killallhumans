
Preliminary Interface Specifications 
The simulation environment will provide: 
Telemetry data (Starting position, velocity, orientation) 
Forward-facing visual data stream  
For the Virtual Qualifier 1: The environment will be desaturated with lower complexity, and the gates will be highlighted, allowing for a high signal-to-noise ratio. Visual guidance aids may be active. 
For the Virtual Qualifier 2: The environment will be more complex, with lighting and other distractions to reduce signal-to-noise ratio and increase the complexity of the pathfinding. Visual guidance aids will be off. 
Classical drone control commands (Throttle, Roll, Pitch, Yaw) 
A Python API interface – Python 3.14.2 is confirmed to work, but we will not limit you in the options. External libraries, compilers and acceleration layers are permitted. The use of AI coding tools is also permitted.  
The API will not provide: 
Depth information of any kind. This is consistent with the physical drone later in the competition, which will also not provide depth information 
Engine speeds: Direct access to Engine RPMs is also not given 
Battery SoC: The state of charge of the (virtual and physical) battery is not provided. Battery performance will not be a limiting factor. 
The interface will not allow active steering commands from the user, even for training rounds / machine learning improvements.  
The system is built to ensure that coding skill determines performance — not hardware budget. 
All teams compete under identical simulated physics and hardware conditions. Performance will be determined by code quality — not infrastructure. A more detailed interface documentation package will follow soon. 
 
Platform Access 
The Virtual Qualifier will run via a Windows-based downloadable application on DCL’s platform. 
Minimum hardware requirements: 
To run the simulator and competitive code, your Windows machine shall meet the following minimum requirements: 
CPU: Intel Core I5-10400F or AMD Ryzen 5 3600 
GPU: Nvidia RTX 2060 Super or AMD 9060XT 
16 GB of RAM 
60 GB Storage  
If your code is of exceptional hardware hunger, the requirements above may not be sufficient.
You will soon receive: 
Setup instructions 
Access credentials 
Submission and evaluation details 
Execution will take place in a controlled evaluation environment to ensure fairness and prevent human intervention. 
 
Now is the time to refine your autonomy stack and prepare for implementation. 
 
More technical details will follow next week. Please also have a look at our FAQs section on theairgrandprix.com. 