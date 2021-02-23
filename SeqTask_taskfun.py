# SeqTask_taskfun
# This module contains various functions for creating input-output values of
# a Sequential task under different conditions.
# Programmers: Eva + Jonathan + Mehrdad
# Date: Aug 1st, 2020

# Importing Libraries
import numpy as np
import torch
import random


# Simple Gaussian force
# Task Structure: Pre->S1->S2...Sn->Mem->RT->E1->E2...En
# Force profile: crude gaussian force profile
def rnn_IO_gaussian(simparams):
    # Update simparams:
    # Instruction time: (Cue-on + Cue-off) for all targets + Memory period
    simparams.update({"instTime": (simparams["cueOn"] + simparams["cueOff"]) * simparams["numTargetTrial"] +
                                  +simparams["cueOn"] + simparams["memPeriod"]})
    # Movement time: Force IPI for all targets + Force width
    simparams.update({"moveTime": (simparams["forceIPI"] * simparams["numTargets"]) + simparams["forceWidth"]})
    # Total Trial Time: Instruction + Reaction + Movement
    simparams.update({"trialTime": simparams["instTime"] + simparams["RT"] + simparams["moveTime"]})

    trial_n = simparams["numEpisodes"]
    in_data = np.zeros([simparams["trialTime"], simparams["numEpisodes"], simparams["numTargets"]+1])
    out_data = np.zeros([simparams["trialTime"], trial_n, simparams["numTargets"]])
    GoTrial = random.choices([0, 1], weights=[1-simparams["GoTrial"], simparams["GoTrial"]], k=simparams["numEpisodes"])
    y = gaussian()
    # Zero matrices for task info
    # inputs_history: A zero matrix for recording task's input timing
    # NumEpisodes, NumTargets + 1 GoSignal, 3(label, start-time, end_time)
    inputs_history = np.zeros((trial_n, simparams["numTargetTrial"]+1, 3))
    inputs_history[:, -1, 0] = GoTrial
    targets_history = np.zeros((trial_n, simparams["numTargetTrial"], 3))
    for i in range(trial_n):
        seq_data = np.random.randint(simparams["minTarget"], high=simparams["maxTarget"], size=[1, simparams["numTargetTrial"]])
        inputs_history[i, :-1, 0] = seq_data  # Saving labels for the current finger sequence for the current trial
        targets_history[i, :, 0] = seq_data
        t = simparams["preTime"]
        for j in range(simparams["numTargetTrial"]): # define targets
            inputs_history[i, j, 1:] = [t, t+simparams["cueOn"]]   # Saving start and end time for each instruction
            t_inp = range(t, t+simparams["cueOn"])
            in_data[t_inp, i, int(seq_data[0, j])] = 1
            t = t + simparams["cueOn"]+simparams["cueOff"]
        # whether go or no-go trial
        if GoTrial[i] == 1:  # go trial
            in_data[range(t+simparams["memPeriod"], t+simparams["memPeriod"] +
                      simparams["cueOn"]), i, simparams["numTargets"]] = 1  # go signal
            # Saving GoCue signal interval
            inputs_history[i, -1, 1:] = [t+simparams["memPeriod"], t+simparams["memPeriod"]+simparams["cueOn"]]
            # expected output
            t = simparams["instTime"]+simparams["RT"]
            for j in range(simparams["numTargetTrial"]):
                t_out = range(t, t+simparams["forceWidth"])
                targets_history[i, j, 1:] = [t, t+simparams["forceWidth"]]
                previous = out_data[t_out, i, int(seq_data[0, j])]
                target = y
                out_data[t_out, i, int(seq_data[0, j])] = np.maximum(previous, target)
                t = t + simparams["forceIPI"]
    inputs = torch.from_numpy(in_data)
    target_outputs = torch.from_numpy(out_data)
    return inputs.float().to(simparams["device"]), target_outputs.float().to(simparams["device"]), inputs_history, targets_history


def rnn_IO_gaussian_continuous(simparams):
    # Time for one cue even to happen
    simparams.update({"CueTime": (simparams["cueOn"] + simparams["cueOff"] + simparams["PostCue"])})
    # Total time for input and output signal
    trialTime = simparams['CueTime']*simparams["MemorySpan"] + simparams['CueTime']*2*simparams["numTargetTrial"] + simparams["numTargetTrial"]*(simparams["forceWidth"] + simparams['postPress'])+simparams["preTime"]
    simparams.update({"trialTime": trialTime})
    # Number of trials is augmented since first ones are memorized (action comes with delay)
    n_Trial = simparams["numTargetTrial"] + simparams["MemorySpan"]
    # Zero tensors
    in_data = np.zeros((simparams["trialTime"], simparams["numEpisodes"], simparams["numTargets"]+1))
    out_data = np.zeros((simparams["trialTime"], simparams["numEpisodes"], simparams["numTargets"]))
    # Zero tensors for history
    inputs_history = np.zeros((simparams["numEpisodes"], n_Trial, 5))
    targets_history = np.zeros((simparams["numEpisodes"], n_Trial, 3))

    seq_data = np.random.randint(simparams["minTarget"], high=simparams["maxTarget"], size=[simparams["numEpisodes"], n_Trial])
    # Save finger numbers in history
    inputs_history[:, :, 0] = seq_data
    targets_history[:, :, 0] = seq_data
    for trial in range(simparams["numEpisodes"]):
        t = simparams["preTime"]
        for target in range(n_Trial):

            # For first targets, just memory no action
            if target < simparams["MemorySpan"]:

                t_inp = range(t, t + simparams["cueOn"])
                # Save the cues in history [instruction start time,instruction end time]
                inputs_history[trial, target, 1:3] = [t, t + simparams["cueOn"]]
                in_data[t_inp, trial, seq_data[trial, target]] = 1
                t = t + simparams["cueOn"] + simparams["cueOff"] + simparams["PostCue"]

            else:
                # turn on a go cue
                t_inp = range(t, t + simparams["cueOn"])
                # Save the go cues in history
                inputs_history[trial, target-simparams["MemorySpan"], 3:5] = [t, t + simparams["cueOn"]]
                in_data[t_inp, trial, simparams["numTargets"]] = 1
                t_t = t + simparams["cueOn"] + simparams["cueOff"] + simparams["RT"]
                t = t + simparams["cueOn"] + simparams["cueOff"] + simparams["PostCue"] + simparams["forceWidth"] + simparams['postPress']

                # Produce an output
                y = gaussian()
                t_out = range(t_t, t_t + simparams["forceWidth"])
                # Save execution time
                targets_history[trial, target-simparams["MemorySpan"], 1:3] = [t_t, t_t + simparams["forceWidth"]]
                previous = out_data[t_out, trial, seq_data[trial, target]]
                out_data[t_out, trial, seq_data[trial, target-simparams["MemorySpan"]]] = np.maximum(previous, y)

                # turn on a instruction cue
                # Go for next target, except for last Memspan trials (No input is required for last ones)
                t_inp = range(t, t + simparams["cueOn"])
                if target < n_Trial - simparams["MemorySpan"]:
                    in_data[t_inp, trial, seq_data[trial, target]] = 1
                    # Save instruction cue time
                    inputs_history[trial, target, 1:3] = [t, t + simparams["cueOn"]]
                else:
                    in_data[t_inp, trial, seq_data[trial, target]] = 0
                t = t + simparams["cueOn"] + simparams["cueOff"] + simparams["PostCue"]

    inputs = torch.from_numpy(in_data)
    target_outputs = torch.from_numpy(out_data)
    return inputs.float().to(simparams["device"]), target_outputs.float().to(simparams["device"]), inputs_history, targets_history





# convolve expected output force profile with a Gaussian window - for now hard-coded


def gaussian():
    x = np.arange(-12.5, 12.5, 1)
    s = 3
    y = 1./np.sqrt(2.*np.pi*s**2) * np.exp(-x**2/(2.*s**2))
    y = y/np.max(y)
    return y


def taskplot(rnn_input, rnn_target):
    import matplotlib.pyplot as plt
    inputs = rnn_input.cpu().numpy()
    targets = rnn_target.cpu().numpy()
    rand_trial_inx = np.random.randint(0, inputs.shape[1], 3)

    fig, axs = plt.subplots(2, 3)
    for i, inx in enumerate(rand_trial_inx, 0):
        axs[0, i].plot(inputs[:, inx, :] * np.array([1, 2, 3, 4, 5, 6]))
        axs[0, i].set_title(f'Trial #{inx+1}\nInput')
        axs[0, i].legend(['Finger 1', 'Finger 2', 'Finger 3', 'Finger 4', 'Finger 5', 'Go'])
        axs[0, i].set_ylim([0, 6.2])
        axs[1, i].plot(targets[:, inx, :])
    plt.show()



