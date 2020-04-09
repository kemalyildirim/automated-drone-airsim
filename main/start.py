#!/usr/bin/env python
# coding: utf-8
from pynput.keyboard import Key, Listener
import numpy as np
import keras
import airsim
import time
import droneEnvironment
from dqn_net_keras import DQNet
from dqn_net_keras import Dataset


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


np.set_printoptions(precision=3, suppress=True)
#test_model = keras.models.load_model("14july-final.h5")
test_model = keras.models.load_model("14v1.h5")
#test_model = None

settings = {}

settings["learning_rate"] = 0.003
settings["reward_decay"] = 0.9
settings["memory_length"] = 81920
settings["batch_size"] = 32
settings["epochs"] = 1
settings["replace_target_iter"] = 50
settings["model"] = test_model
settings["n_actions"] = 7
settings["n_features"] = 6

#aim = airsim.Vector3r(76, 49, 10)
#aim = airsim.Vector3r(205, 40, -26)
#aim = airsim.Vector3r(32, 38, -4)
#start = airsim.Vector3r(0, 0, -5)

start = [0, 0, -5]
aim = [32, 38, -4]

agent = DQNet(settings)
env = droneEnvironment.GridWorld(aim)
agent.model.summary()

current = 0
count = 0
learns = 0
epochs = 0
succ = 0
learn = 0
rc = 0  #Reward counter

#Initial state
s = env.reset()

while True:

    action = agent.choose_action(s)
    s_, r, done, info = env.step(action)

    env.render(extra1 = str(count), extra2 = "reward: " + str(int(r))  + " Epochs: " + str(epochs))
    count += 1
    current += 1
    if info == "success":
        succ += 1
    out = False
    if current > 100:
        out = True
        info = "out of steps"

    rc += r
    counter = agent.add_data(s, action, r, s_)

    if out:
        agent.model.save("test.h5")
        print("Reset due to ", info, "Epoch reward: ", rc)
        print("--------------------------------reset---------------------------")
        rc = 0
        current = 0
        out = False
        env.reset()

    if done:
        current = 0
        epochs += 1
        agent.model.save("test.h5")
        if info == "success":
            print("Info: ", info, "Epoch reward:", rc)
            break
        print("Reset due to ", info, "Epoch reward: ", rc)
        print("--------------------------------reset---------------------------")
        rc = 0
        env.reset()

    if count > 1:
        agent.learn(times = 2)
        learn += 1

    s = s_


##BREAK ON SUCCESS
agent.model.save("test.h5")

##Landing

while True:
    env.client.landAsync()
    landed = env.client.getMultirotorState().landed_state
    print(landed)
    
    #a = input("q for exit")
    #if a == "q":
    #    break
