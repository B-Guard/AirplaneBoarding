# Boarding Simulation (SimPy)

This project simulates airplane boarding as a discrete-event system with a single centre aisle, no overtakes, and per-row aisle capacity. 
You can run Task 2 experiments via `run.py`, which writes CSV outputs into `./results/`. 
Install requirements with `pip install simpy`.

## File walkthrough

### `model.py`
Implements the SimPy boarding simulation engine and all shared model state. 
Passengers move row-by-row through an aisle segmented by row, with each segment having a fixed capacity (default 2) to prevent overtakes. 
Seating time is computed from a base time plus a penalty for already-occupied “blocker” seats between aisle and target seat, plus an optional luggage add-time. 
The “infinite queue outside” is represented by a simple entry gate that only lets passengers attempt entry in manifest order.
The `simulate(...)` function is the main API: it runs one replication and returns the finish time and an optional trace for face validation.

### `policies.py`
Contains boarding strategies that generate passenger seat assignments and a boarding sequence. 
It builds a list of all seats in the aircraft and then shuffles/sorts it to represent different strategies.
Implemented policy is `random` (Task 2). 
Each policy returns a manifest (ordered list) of `Passenger` objects with assigned seats and luggage flags. 

### `experiments.py`
Provides replication runners and task-level experiment functions that write results to CSV. 
The `run_replications(...)` function is the core loop: it generates a manifest via a policy, runs the simulation, and computes summary statistics. 
Task 2 helpers include `task2_face_validation()` and `task2_sweep_seat_times()`. 

### `run.py`
A small command-line entry point for running experiments. It parses a task name argument and calls the corresponding function in `experiments.py`. 
Results are written into `./results/` by default, and the script prints the output file paths when finished.

## Usage

```bash
pip install simpy
python run.py task2_face_validation
python run.py task2_sweep
python run.py task3_many_boarding_strategies

## To run with 2 entry doors
python run.py task2_face_validation --two_entries
python run.py task2_sweep --two_entries
python run.py task3_many_boarding_strategies --two_entries
```