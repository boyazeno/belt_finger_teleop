# Panda Control

## Description:
This package provides functions to start the gripper (customized) and robot service node based on deoxys_control.

This also provides a pick environment that coordinates cameras and tactile sensors to formulate a gym style env for policy rollout.

## Installation:
```bash
pip install -e .
```
The torch version need to be adapted according to the machine you run.

## How to run:
* To start the belt gripper service node:
```bash
cd scripts
python3 run_belt_gripper.py
```

* To start the parallel gripper service node:
```bash
cd scripts
python3 run_gripper.py
```

* `run_actor.py` is for inferencing a trained RL policy (not used for belt_gripper specifically).