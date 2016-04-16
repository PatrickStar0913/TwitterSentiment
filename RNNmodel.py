import numpy as np
import theano as theano
import theano.tensor as T
import operator


class RNNTheano:

    def __init__(self, word_dim, hidden_dim=500, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        w = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), word_dim)
        Wx = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        Wh = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        # Theano: Created shared variables
        self.w = theano.shared(name='w', value=w.astype(theano.config.floatX))
        self.Wx = theano.shared(name='Wx', value=Wx.astype(theano.config.floatX))
        self.Wh = theano.shared(name='Wh', value=Wh.astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        w, Wx, Wh = self.w, self.Wx, self.Wh
        x = T.ivector('x')
        y = T.ivector('y')
        h0 = np.zeros(self.hidden_dim)
        h0 = theano.shared(name='h0', value=h0.astype(theano.config.floatX))
        b = theano.shared(0.5, "b")

        def forward_prop_step(x_t, s_t_prev, w, Wx, Wh):
            s_t = T.tanh(Wx[:, x_t] + Wh.dot(s_t_prev))
            o_t = T.nnet.sigmoid(w.dot(s_t))
            return [o_t, s_t]

        [o,s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, h0],
            non_sequences=[w, Wx, Wh],)

        o_error = ((y[0]/4 - o[-1]) ** 2)

        prediction = T.switch(T.ge(o[-1], b), 1, 0)
        acc = T.switch(T.eq(prediction, y[0]/4), 1, 0)

        # Gradients
        dw = T.grad(o_error, w)
        dWx = T.grad(o_error, Wx)
        dWh = T.grad(o_error, Wh)

        y_i = y[0]/4

        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction,
                mode=theano.Mode(linker='vm'))
        self.ce_error = theano.function([x, y], o_error)
        self.acc = theano.function([x, y], acc,
                mode=theano.Mode(linker='vm'))
        self.bptt = theano.function([x, y], [dw, dWx, dWh])
        self.get_y = theano.function([y], y_i)

        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], o_error,
                      updates=[(self.w, self.w - learning_rate * dw),
                              (self.Wx, self.Wx - learning_rate * dWx),
                              (self.Wh, self.Wh - learning_rate * dWh)])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x, y in zip(X, Y)])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        # num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X, Y)

    def calculate_total_acc(self, X, Y):
        return np.sum([self.acc(x, y) for x, y in zip(X, Y)])

    def calculate_acc(self, X, Y):
        return self.calculate_total_acc(X, Y)

    def print_pre(self, x,y):
        print x
        print self.predict(x)
        print self.get_y(y)
        print self.acc(x,y)
        print '\n'


def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['w', 'Wx', 'Wh']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)

