# Background
You are a cracked software engineer that is competing in the AI Grand Prix Drone Racing competition hosted by Anduril
Scan this website for more background information: https://theaigrandprix.com/

We will be developing software that will enable a drone to autonomously fly through gates in the correct sequencing. 
The first phase is to create the MVP

This is the most recent email sent to us by the competition:

```
Hi, 
 
As promised, here are more details about the 1st Virtual Qualifier (Round 1) - and what you should prepare for. 
 
What Round 1 Looks Like: 
The first qualifier will take place inside a virtual environment designed to focus purely on autonomy performance. No visual gimmics. 
 
You can expect: 
A structured 3D racecourse with a defined number of standardized gates. 
Clear gate sequencing — all gates must be passed in the correct order. 
Realistic drone physics and flight dynamics. 
A time-based scoring system — fastest valid run wins but the main goal is to pass gates. 
The emphasis is on precision, stability, and speed under realistic physical constraints. 
Technical Framework & Code Submission: 
Round 1 will be executed through a Windows-based downloadable application built on DCL’s competition platform. Exact hardware requirements will be communicated in the upcoming newsletters, but a recent mid-tier PC with a dedicated graphics card should generally do. We will make sure that the team with the best coding skills is fastest, not the one with the most hardware budget. 
 
Your task: 
Prepare your Python-based autonomy stack, which will be integrated into the simulator and executed in a controlled evaluation environment. In the coming weeks, we will provide more information on interface specifications, input/output definitions, detailed submission instructions, exact opening & deadline dates, etc. 
 
Your Code needs to handle the following: 
Gate recognition: Detect and locate gates in the virtual environment using available sensor data and a visual data feed. Gates will mostly be standardized. 
Drone control: Command the drone's flight dynamics (speed, orientation, thrust) with precision. There will be a balance to be found between speed and accuracy. 
Path planning & navigation: Plot and follow an efficient route through all gates in the correct order, under realistic physical limitations of the drone’s capabilities.  

Code Ownership — Important Clarification: 
We’ve received several questions regarding intellectual property. You retain full ownership of your algorithm, source code, and documentation. By submitting your entry, you grant the AI Grand Prix permission to use your code strictly for operating, monitoring, and judging the competition and only for the duration of the competition period. We will use this access to monitor the code to avoid cheating, exploitation and human interference, but there is no transfer of IP to Anduril or any founding partner. 
 
Entry Fee: 
We would like to emphasize again that there are no entry fees for teams. However, each participating team is responsible for covering its own expenses related to the AI Grand Prix, including travel, accommodation, and any additional costs incurred. 
 
What’s Next: 
Next week, we will share preliminary technical interface details and additional specifications and updated FAQs to help you prepare.  
 
Now is the time to test, iterate, and stress-test your autonomy stack. 

```
# Roadmap

This is a high level overview of a potential architecture:
```
Camera → [Gate Detection (CV/ML)] → Gate Position
                                         ↓
                              [Path Planner (optimization)]
                                         ↓
                              [MPC Controller (deterministic)]
                                         ↓
                              [Low-level PID (deterministic)]
                                         ↓
                                   Motor Commands
```

We do not have to follow this exact architecture; its one of many possible set ups. For now, we are inteested in gate-detection & gate position detection due to the lack of information at the current moment.

## MVP Roadmap Plan:

Because the API and technical documents have not yet been released, we should focus on the aspects that are agnostic to whatever new information will come out.

### Gate Detection:
We need to write software such that given a live camera feed we will detect the presence of all the gates in the camera's field of view with very low latency






# Misc:



