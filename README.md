# NAME

AI::FANN

# SYNOPSIS

    #                                   Hidden
    #                             Input    |   Output
    #                                  \   |  /
    my $ann = AI::FANN.new: layers => [ 3, 2, 1 ];

    $ann.set-activation-function: FANN_SIGMOID_SYMMETRIC;

# DESCRIPTION

# METHODS

The methods described below follow roughly the same structure as that used
in the documentation of [`libfann`](http://libfann.github.io/fann/docs).

Whenever possible, the underlying method that is being called will be
indicated next to the method signatures.

Please refer to the `libfann` documentation for additional details.

## Creation and Execution

### new

    # fann_create_standard
    multi method new (
               :@layers,
        Num()  :$connection-rate,
        Bool() :$shortcut,
    ) returns AI::FANN

    # fann_create_from_file
    multi method new (
        IO()   :$path,
    ) returns AI::FANN

Creates a new AI::FANN neural network. The constructor can be called in one
of two ways.

If the `path` parameter is set, it will be coerced to a [IO::Path] and the
network will be created based on the contents of that file (see
[save](#save) for how this file can be created).

Alternatively, a list of integers can be passed as the `layers` parameter to
specify the number of neurons in each layer, with the input layer being the
first in the list, the output layer being the last in the list, and any
remaining ones describing hidden layers.

By default, this will create a fully connected backpropagation neural network.
There will be a bias neuron in each layer (except the output layer), and this
bias neuron will be connected to all neurons in the next layer. When running
the network, the bias nodes always emits 1.

To create a neural network that is not fully connected, a `connection-rate`
parameter can be set to a number between 0 and 1, where 0 is a network with
no connections, and 1 is a fully connected network.

If the `shortcut` flag is set, the resulting network will be fully connected,
and it will have connections between neurons in non-contiguous layers. A fully
connected network with shortcut connections is a network where all neurons are
connected to all neurons in later layers, including direct connections from
the input layer to the output layer.

The `connection-rate` and `shortcut` parameters are not compatible, and using
both is an error.

### connection-rate

    # fann_get_connection_rate
    method connection-rate returns Num

Get the connection rate used when the network was created.

### num-input

    # fann_get_num_input
    method num-input returns Int

Get the number of input neurons.

### num-layers

    # fann_get_num_layers
    method num-layers returns Int

Get the number of layers in the network.

### num-output

    # fann_get_num_output
    method num-output returns Int

Get the number of output neurons.

### total-connections

    # fann_get_total_connection
    method total-connections returns Int

Get the total number of connections in the entire network.

### total-neurons

    # fann_get_total_neurons
    method total-neurons returns Int

Get the total number of neurons in the entire network. This number includes
the bias neurons, so a 2-4-2 network has 2+4+2 neurons, plus 2 bias neurons
(one for each layer except the output one) for a total of 10.

### network-type

    # fann_get_network_type
    method network-type returns AI::FANN::NetType

Get the type of neural network it was created as.

### layer-array

    # fann_get_layer_array
    method layer-array returns List

Get the number of neurons in each layer in the network.

Bias is not included so the layers match the ones used in the constructor.

### bias-array

    # fann_get_bias_array
    method bias-array returns List

Get the number of bias in each layer in the network.

### connection-array

    # fann_get_connection_array
    method connection-array returns [List] of AI::FANN::Connection

Get the connections in the network.

### print-connections

    # fann_print_connections
    method print-connections returns Nil

Will print the connections of the network in a compact matrix, for easy
viewing of its internals.

As an example, this is the output from a small (2 2 1) network trained on the
xor problem:

    Layer / Neuron 012345
    L   1 / N    3 BBa...
    L   1 / N    4 BBA...
    L   1 / N    5 ......
    L   2 / N    6 ...BBA
    L   2 / N    7 ......

This network has five real neurons and two bias neurons. This gives a total of
seven neurons named from 0 to 6. The connections between these neurons can be
seen in the matrix.

A period (".") indicates there is no connection, while a character tells how
strong the connection is on a scale from a-z. The two real neurons in the
hidden layer (neuron 3 and 4 in layer 1) have connections from the three
neurons in the previous layer as is visible in the first two lines. The output
neuron (6) has connections from the three neurons in the hidden layer 3 - 5,
as shown in the fourth line.

To simplify the matrix output, neurons are not visible as neurons that
connections can come from, and input and bias neurons are not visible as
neurons that connections can go to.

### print-parameters

    # fann_print_parameters
    method print-parameters returns Nil

Prints all of the parameters and options of the network.

### run

    # fann_run
    multi method run (
        CArray[num32] :$input
    ) returns CArray[num32]

    multi method run (
        :@input
    ) returns List

Run the input through the neural network, returning an array of outputs. The
output array will have one value per neuron in the output layer.

The type of the return value depends on the type of the input.

If the input is provided as a [CArray[num32]][CArray] object, it will be used
as-is and the return value will be of the same type. This is the fastest way
to call this method.

If the input is passed as a [List] or [Array], it will be internally converted
to its C representation, and the return value will be a [List] object.

## File Input / Output

### save

    # fann_save
    method save ( IO() :$path ) returns Bool

Save the entire network to a configuration file.

The configuration file contains all information about the neural network and
can be passed as the `path` parameter to the constructor to create an exact
copy of the network and all of the associated parameters.

The only parameters that are not saved are the callback, error log, and user
data, since they cannot safely be ported to a different location. Note that
temporary parameters generated during training, like the mean square error,
are also not saved.

## Training

### activation-function

    # fann_get_activation_function
    multi method activation-function (
        Int :$layer!,
        Int :$neuron!,
    ) returns AI::FANN::ActivationFunc

    # fann_set_activation_function
    # fann_set_activation_function_layer
    multi method activation-function (
        AI::FANN::ActivationFunc $function!,
        Int                     :$layer!,
        Int                     :$neuron,
    ) returns AI::FANN:;ActivationFunc

    # fann_set_activation_function_hidden
    # fann_set_activation_function_output
    multi method activation-function (
        AI::FANN::ActivationFunc $function!,
        Bool()                  :$hidden,
        Bool()                  :$output,
    ) returns AI::FANN:;ActivationFunc

If called with no positional arguments, this method returns the activation
function for neuron number and layer specified in the `neuron` and `layer`
parameters respectively, counting the input layer as layer 0. It is not
possible to get activation functions for the neurons in the input layer:
doing so is an error.

If called with a member of the AI::FANN::ActivationFunc enum as the first
positional argument, then this function will instead _set_ this as the
activation function for the specified layer and neuron, and return the
value that has been set.

When used as a setter, specifying the layer is always required. This can
be done with the `layer` parameter, as described above, or with the `hidden`
or `output` flags. The `hidden` flag will set the activation function for
all neurons in _all_ hidden layers, while the `output` flag will do so only
for those in the output layer.

When setting the activation function using the `layer` parameter, the `neuron`
parameter is optional. If none is set, all neurons in the specified layer
will be modified.

### train

    # fann_train
    multi method train (
        CArray[num32] :$input!,
        CArray[num32] :$output!,
    ) returns Nil

    multi method train (
        :@input!,
        :@output!,
    ) returns Nil

    # fann_train_on_data
    multi method train (
        AI::FANN::TrainData :$data!,
                            :$max-epochs!,
                            :$epochs-between-reports!,
        Num()               :$desired-error!,
    ) returns Nil

    # fann_train_on_file
    multi method train (
        IO()  :$path!,
              :$max-epochs!,
              :$epochs-between-reports!,
        Num() :$desired-error!,
    ) returns Nil

This method is used to train the neural network.

The first two candidates train a single iteration using the specified set of
inputs and desired outputs in the `input` and `output` parameters. Inputs
and outputs can be passed as [CArray[num32]][CArray] objects, or as arrays
of numeric values, which will be converted internally to their C
representation.

Since only one pattern is presented, training done this way is always
incremental training (`FANN_TRAIN_INCREMENTAL` in the AI::FANN::Train enum).

The last two candidates train instead on an entire dataset, for a period of
time. The first one takes an AI::FANN::TrainData object in the `data`
parameter, while the second generates one internally from the file specified
in the `path` parameter.

In both cases, the training uses the algorithm set with
[training-algorithm](#training-algorithm) (NYI), and the parameters set for
these training algorithms.

### test

    # fann_test
    multi method test (
        CArray[num32] :$input!,
        CArray[num32] :$output!,
    ) returns CArray[num32]

    multi method test (
        :@input!,
        :@output!,
    ) returns List

    multi method test (
        AI::FANN::TrainData :$data!
    ) returns Num

    multi method train (
        IO() :$path!,
    ) returns Num

Test the network with a set of inputs and desired outputs. This operation
updates the mean square error, but does not change the network in any way.

Inputs and outputs can be passed as CArray[num32] objects, or as arrays of
numeric values, which will be converted internally to their C representation.

These candidates return the same as the equivalent invokations of [run](#run).

Two more calling patterns are offered as shortcuts.

A AI::FANN::TrainData object can be passed as the `data` parameter, in which
case the network will be tested with all the input and output data it
contains.

Alternatively, the `path` parameter can be set to a value that can be coerced
to a [IO::Path] object. In this case, an AI::FANN::TrainData will be
internally read from the contents of this file and used as above.

These candidates return the updated mean square error for the network.

### reset-error

    # fann_reset_MSE
    method reset-error returns Nil

Resets the mean square error from the network, and the number of bits that
fail.

### mean-square-error

    # fann_get_MSE
    method mean-square-error returns Num

Reads the mean square error from the network. This value is calculated during
training or testing (see [train](#train) and [test](#test) above), and can
therefore sometimes be a bit off if the weights have been changed since the
last calculation of the value.

## COPYRIGHT AND LICENSE

Copyright 2021 José Joaquín Atria

This library is free software; you can redistribute it and/or modify it under
the Artistic License 2.0.

[List]: https://docs.raku.org/type/List
[Array]: https://docs.raku.org/type/Array
[IO::Path]: https://docs.raku.org/type/IO::Path
[CArray]: https://docs.raku.org/language/nativecall#Arrays
