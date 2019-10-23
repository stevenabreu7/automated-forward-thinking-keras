from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
import keras
import random
import time


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class AutoForwardThinking:
    def __init__(self, possible_widths, pool_size, max_layers, data):
        """
            self.in_layer
            self.standby_layers
            self.frozen_hidden
            self.training_hidden
            self.out_layer
        """

        # parameters
        self.optimizer = RMSprop()
        self.loss = 'categorical_crossentropy'

        # load the data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.input_dim = self.x_train.shape[1]
        self.output_dim = self.y_train.shape[1]

        # standby layers (will be added into network)
        self.possible_widths = possible_widths
        self.max_layers = max_layers
        self.pool_size = pool_size
        self._layers = []

        # initialize model
        self.model = Sequential()

        # add first trainable layer (will be deleted again)
        dim = 100
        trainl = Dense(dim,
                       activation='tanh',
                       input_dim=self.input_dim)
        trainl.trainable = True
        trainl.name = 'hidden_1_dense_' + str(dim)
        self.model.add(trainl)

        # add output layer
        outl = Dense(self.output_dim,
                     activation='softmax')
        outl.trainable = True
        outl.name = 'output_layer'
        self.model.add(outl)

        # compile model
        self.compile_model()

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=['accuracy'])

    def train(self, final_epochs=-1, cand_epochs=2,
              batch_size=128, stopping_comp=0.0):
        hist, times, cand_scores = [], [], []

        # callbacks (time and early stopping)
        callbacks = [TimeHistory()]
        if final_epochs < 0:
            patience = -1 * final_epochs
            final_epochs = 1000
            callbacks.append(
                keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience=patience)
            )

        opt_width = None

        for layer_i in range(self.max_layers):

            if opt_width:
                candidate_pool = [e for e in self.possible_widths
                                  if e <= opt_width]
            else:
                candidate_pool = list(self.possible_widths)
            random.shuffle(candidate_pool)
            candidate_pool = candidate_pool[:self.pool_size]

            candidate_scores = {}

            for candidate_i in range(len(candidate_pool)):
                width = candidate_pool[candidate_i]

                print('\nLayer {}. Candidate {}/{} [{} units]\n'.format(
                    layer_i + 1, candidate_i + 1, len(candidate_pool), width
                ))

                # remove output and last hidden layer
                if layer_i == 0:
                    # first layer -> replace everything
                    self.model = Sequential()
                elif candidate_i == 0:
                    # first candidate -> remove only output layer
                    # leave the last hidden layer
                    self.model.pop()
                else:
                    # remove the output layer and the last hidden
                    # layer (because it's the previous candidate)
                    self.model.pop()
                    self.model.pop()

                # add hidden layer
                if layer_i == 0:
                    trainl = Dense(width, activation='tanh',
                                   input_dim=self.input_dim)
                else:
                    trainl = Dense(width, activation='tanh')
                trainl.trainable = True
                trainl.name = 'hidden_{}_dense_{}'.format(
                    layer_i + 1, width
                )
                self.model.add(trainl)

                # add output layer
                outl = Dense(self.output_dim, activation='softmax')
                outl.trainable = True
                outl.name = 'output_layer'
                self.model.add(outl)

                # recompile model
                self.compile_model()

                # run training for set number of epochs
                h = self.model.fit(self.x_train, self.y_train,
                                   epochs=cand_epochs,
                                   batch_size=batch_size,
                                   validation_data=(self.x_test, self.y_test),
                                   callbacks=callbacks[:1])
                t = callbacks[0].times

                candidate_scores[width] = float('{:.4f}'.format(
                    h.history['val_accuracy'][-1]
                ))

            # choose best candidate
            opt_width = max(candidate_scores, key=candidate_scores.get)
            self._layers.append(opt_width)

            print('\nLayer {}. Chosen candidate with {} units.'.format(
                layer_i + 1, opt_width
            ))

            # remove output and last hidden layer
            self.model.pop()
            self.model.pop()

            # add hidden layer
            if layer_i == 0:
                trainl = Dense(opt_width, activation='tanh',
                               input_dim=self.input_dim)
            else:
                trainl = Dense(opt_width, activation='tanh')
            trainl.trainable = True
            trainl.name = 'hidden_{}_dense_{}'.format(
                layer_i + 1, opt_width
            )
            self.model.add(trainl)

            # add output layer
            outl = Dense(self.output_dim, activation='softmax')
            outl.trainable = True
            outl.name = 'output_layer'
            self.model.add(outl)

            # recompile model
            self.compile_model()

            # print the model
            self.print_model()

            # train this layer
            h = self.model.fit(self.x_train, self.y_train,
                               epochs=final_epochs,
                               batch_size=batch_size,
                               validation_data=(self.x_test, self.y_test),
                               callbacks=callbacks)
            t = callbacks[0].times

            # freeze layers
            for i in range(len(self.model.layers)):
                self.model.layers[i].trainable = False
            self.compile_model()

            hist.append(h.history)
            times.append(t)
            cand_scores.append(candidate_scores)

            # # check if we continue training
            # if layer_i > 0:
            #     prev = hist[-2]['val_acc'][-1]
            #     curr = mean(hist[-1]['val_acc'])
            #     curr = hist[-1]['val_acc'][-1]
            #     print('{:.3f}\n{:.3f}'.format(prev, curr))
            #     if prev >= curr:
            #         print('Training converged.')
            #         break

        # save all stats in one dictionary
        stats = {
            'stage': [], 'epoch_stage': [],
            # 'epoch': [],
            'time': [], 'val_loss': [], 'val_accuracy': [],
            'train_loss': [], 'train_accuracy': [],
            'comments': [], 'new_layer': [],
            'candidate_scores': []
        }
        for i in range(len(times)):
            for j in range(len(times[i])):
                # assuming tanh, softmax, cat cross entropy, rms prop
                comments = '[{}]'.format(', '.join(map(str, self._layers)))
                comments += ', tanh, softmax, cat crossentropy, rms prop'
                candidates = str(cand_scores[i])
                stats['stage'] += [i + 1]
                stats['epoch_stage'] += [j + 1]
                # stats['epoch'] += [i * epochs_per_layer + j + 1]
                stats['time'] += ['{:.3f}'.format(times[i][j])]
                stats['val_loss'] += ['{:.4f}'.format(hist[i]['val_loss'][j])]
                stats['val_accuracy'] += ['{:.4f}'.format(hist[i]['val_accuracy'][j])]
                stats['train_loss'] += ['{:.4f}'.format(hist[i]['loss'][j])]
                stats['train_accuracy'] += ['{:.4f}'.format(hist[i]['accuracy'][j])]
                stats['comments'] += [comments]
                stats['new_layer'] += [j == 0]
                stats['candidate_scores'] += [candidates] if j == 0 else ['']

        # return statistics dictionary
        self.stats = stats.copy()
        return stats

    def print_model(self):
        print('\nNetwork:')
        for layer in self.model.layers:
            print(layer.name, layer.trainable)
        print()
