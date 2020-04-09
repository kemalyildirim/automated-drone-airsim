import numpy as np
import keras

class Dataset:
    def __init__(self, memory_length, memory_size=4):
        self.memory_length = memory_length
        self.memory_size = memory_size
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_length, self.memory_size))

    def add(self, data):
        if data.ndim == 1:
            data = np.array([data])

        for i in range(data.shape[0]):
            self.memory[self.memory_counter % self.memory_length, :] = data[i]
            self.memory_counter += 1

        return self.memory_counter

    def next_batch(self, batch_size):
        index_size = min(self.memory_counter, self.memory_length)
        index = np.random.choice(index_size, batch_size)
        return self.memory[index, :]

    def memory_init(self):
        self.memory_counter = 0
        return


init_settings = {}

class DQNet:

    def __init__(self, settings):
        self.settings = init_settings.copy()
        self.settings.update(settings)

        self.n_actions = self.settings["n_actions"]
        self.n_features = self.settings["n_features"]
        self.lr = self.settings["learning_rate"]
        self.gamma = self.settings["reward_decay"]
        self.epsilon = 1.0
        #self.epsilon = 0.4
        self.e_min = 0.2
        self.e_decay = 0.995
        self.memory_length = self.settings["memory_length"]
        self.batch_size = self.settings["batch_size"]
        self.epochs = self.settings["epochs"]
        self.replace_target_iter = self.settings["replace_target_iter"]

        self.training_counter = 0
        self.dataset = Dataset(memory_length = self.memory_length, memory_size = self.n_features * 2 + 2)
        
        #build network
        if self.settings["model"] == None:
            model = keras.models.Sequential()
            model.add(keras.layers.Dense(100, activation='relu', input_dim=self.n_features))
            model.add(keras.layers.Dense(100, activation='relu'))
            model.add(keras.layers.Dense(100, activation='relu'))
            model.add(keras.layers.Dense(100, activation='relu'))
            model.add(keras.layers.Dense(self.n_actions))
            model.compile(keras.optimizers.Adam(lr=self.lr), 'mse')
            self.model = model
        else:
            self.model = self.settings["model"]

        self.target_model = keras.models.clone_model(self.model)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        #if np.random.uniform() < self.epsilon:
        if np.random.uniform() > self.epsilon:
            actions_value = self.model.predict(observation)
            action = np.argmax(actions_value)
            print(self.epsilon, 'learned')
        else:
            action = np.random.randint(0, self.n_actions)
            print("random action:", action)
        return action

    #remember
    def add_data(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        counter = self.dataset.add(transition)
        return counter

    def learn(self, times = 1):
        res = ''

        if self.training_counter % (self.replace_target_iter) == 0:
            self.target_model.set_weights(self.model.get_weights())
            res += 'Net updated'
            print(res)

        self.training_counter += 1

        for i in range(times):
            tdata = self.dataset.next_batch(self.batch_size * times)
            s = tdata[:, 0:self.n_features]
            a = tdata[:, self.n_features]
            r = tdata[:, self.n_features+1]
            s_ = tdata[:, -self.n_features:]
            q_next = self.target_model.predict(s_)
            q_value = self.target_model.predict(s)

            q_target = q_value

            batch_index = range(self.batch_size * times)
            action_index = a.astype(int)
            q_target[batch_index, action_index] = r + self.gamma * np.max(q_next, axis = 1)

            train_X = s
            train_Y = q_target
            if self.epsilon > self.e_min:
                self.epsilon *= self.e_decay

            self.model.fit(train_X, train_Y, epochs = self.epochs, verbose = 0)
        return res
