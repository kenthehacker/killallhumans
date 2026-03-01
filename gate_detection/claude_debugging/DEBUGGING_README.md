# Debugging
There is something that went wrong, or things need to be updated. Read the following:

## Context:
This was ran:
```
python3 tests/video_detection_demo.py --dataset ../external_data/drone-racing-dataset --flight flight-01a-ellipse --preset orange --delay 33
```

**Ghost Detection**
the `gate_detection/claude_debugging/ceiling.png` screenshot shows that there was a false positive on a gate being detected

**No Detection At All**
the `gate_detection/claude_debugging/full_frame_no_detection.png` screenshot shows a scenario where there's clearly a gate but the code did not pick up there was a gate


## Investigation Conducted:
* read the `gate_detection/claude_debugging/gate_detection_analysis.md` investigation
* claude code was able to generate `gate_detection/claude_debugging/gate_detector_v2.py` and we can visualize the effects with `gate_detection/claude_debugging/comparison_frame999.png`
* claude code tried to make a preset agnostic code as shown in `gate_detection/claude_debugging/gate_detector_v3.py` but there's some issues with the code as it doesn't get the true borders of the gate correct. You should only use this as a reference but not adopt the logic unless it is truly sound. This is just a prototype suggestion for ideas

The issue is that the current implementation, which requires us to pass in a preset, will force us to pick up on certain colors, hues, etc


## Action Items
Given the issues raised we need to rectify the issues. Look at the TODO list in `gate_detection/claude_instructions.md` 
Feel free to add any other action items other than those listed in the claude_instructions.md if they will be beneficial.


