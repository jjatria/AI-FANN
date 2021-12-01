# NAME

AI::FANN

# SYNOPSIS

    # See below for details on export tags
    use AI::FANN :enum;

    #                               Hidden
    #                         Input    |    Output
    #                              \   |   /
    given AI::FANN.new: layers => [ 2, 3, 1 ] {

        # A sample data set for solving the XOR problem
        my $data = AI::FANN::TrainData.new: pairs => [
            [ -1, -1 ] => [ -1 ],
            [ -1,  1 ] => [  1 ],
            [  1, -1 ] => [  1 ],
            [  1,  1 ] => [ -1 ],
        ];

        .activation-function: FANN_SIGMOID_SYMMETRIC;

        .train: $data,
            desired-error          => 0.001,
            max-epochs             => 500_000,
            epochs-between-reports => 0;       # Do not print reports

        say .run: [ 1, -1 ];
    }
    # OUTPUT:
    # (0.9508717060089111)

# DESCRIPTION

This distribution provides native bindings for the Fast Artificial Neural
Network library (FANN). The aim of the library is to be easy to use, which
makes it a good entry point and suitable for working on machine learning
prototypes.

Creating networks, training them, and running them on input data can be done
without much knowledge of the internals of ANNs, although the ANNs created
will still be powerful and effective. Users with more experience and desiring
more control will also find methods to parameterize most of the aspects of the
ANNs, allowing for the creation of specialized and highly optimal ANNs.

## Installation

The bindings for Raku make use of the system version of FANN. Please refer to
your platform's instructions on how to install the library, or follow the
instructions for [compiling from source](https://github.com/libfann/fann#to-install).

# METHODS

The methods described below include readers, mutators, and methods that
operate on the internal state of the network in more complex ways.

Some methods, like [num-input](#num-input) are only for reading the
internal state of the network, and will always return the value that was
requested.

Other methods, like [activation-function](#activation-function) will act as
both readers and mutators depending on the arguments that are passed.

When acting as readers, named parameters may be used to specify the scope
of the reading. Some of these may be mandatory.

When acting as mutators, the new value should be passed as one or more
positional arguments, with any named parameters specifying the possible scope
of the mutation. All mutators always return the calling object, to allow
for chaining. These will be marked in the signatures as `returns self`.

Most other methods, like [reset-error](#reset-error) or [train](#train), will
also return the calling object, and may take named parameters. Some methods
have different return values, like [test](#test) or [save](#save) that reflect
the result of the operation. In all cases, the signature should specify the
return value.

The sections below follow roughly the same structure as that used
in the documentation of [libfann](http://libfann.github.io/fann/docs).

Whenever possible, the underlying method that is being called will be
indicated next to the method signatures.

Please refer to the libfann documentation for additional details.

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

### run

    # fann_run
    multi method run (
        CArray[num32] $input
    ) returns CArray[num32]

    multi method run (
        *@input
    ) returns List

Run the input through the neural network, returning an array of outputs. The
output array will have one value per neuron in the output layer.

The type of the return value depends on the type of the input.

If the input is provided as a [CArray[num32]][CArray] object, it will be used
as-is and the return value will be of the same type. This is the fastest way
to call this method.

If the input is passed as a [List] or [Array], it will be internally converted
to its C representation, and the return value will be a [List] object.

### bit-fail

    # fann_get_bit_fail
    method bit-fail returns Int

Returns the number of fail bits, or the number of output neurons which
differ more than the bit fail limit (see [bit-fail-limit](#bit-fail-limit)).
The bits are counted in all of the training data, so this number can be
higher than the number of training data.

This value is reset by [reset-error](#reset-error) and updated by all the
same functions which also update the mean square error (eg. [test](#test)).

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

### randomize-weights

    # fann_randomize_weights
    method randomize-weights (
        Range:D $range,
    ) returns self

Give each connection a random weight between the endpoints of the specified
[Range] object.

From the beginning the weights are random between -0.1 and 0.1.

This method is an alias for [randomise-weights](#randomise-weights).

### randomise-weights

    # fann_randomize_weights
    method randomise-weights (
        Range:D $range,
    ) returns self

Give each connection a random weight between the endpoints of the specified
[Range] object.

From the beginning the weights are random between -0.1 and 0.1.

This method is an alias for [randomize-weights](#randomize-weights).

### print-connections

    # fann_print_connections
    method print-connections returns self

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
    method print-parameters returns self

Prints all of the parameters and options of the network.

### clone

    # fann_copy
    method clone returns AI::FANN

Returns an exact copy of the calling AI::FANN object.

### destroy

    method destroy returns Nil

Destroy the internal representation of this dataset. This is called
automatically by the garbage collector, but can be called manually.

## File Input / Output

### save

    # fann_save
    method save ( IO() $path ) returns Bool

Save the entire network to a configuration file.

The configuration file contains all information about the neural network and
can be passed as the `path` parameter to the constructor to create an exact
copy of the network and all of the associated parameters.

The only parameters that are not saved are the callback, error log, and user
data, since they cannot safely be ported to a different location. Note that
temporary parameters generated during training, like the mean square error,
are also not saved.

## Training

The methods in this section support fixed topology training.

When using this method of training, the size and topology of the ANN is
determined in advance and the training alters the weights in order to minimize
the difference between the desired output values and the actual output values.

For evolving topology training, see the [Cascade Training](#cascade-training)
section below.

### train

    multi method train (
        @input,
        @output,
    ) returns self

    # fann_train
    multi method train (
        CArray[num32] $input,
        CArray[num32] $output,
    ) returns self

    # fann_train_on_data
    multi method train (
        AI::FANN::TrainData:D $data,
        Int() :$max-epochs!,
        Int() :$epochs-between-reports!,
        Num() :$desired-error!,
    ) returns self

    # fann_train_on_file
    multi method train (
        IO() $path,
        Int() :$max-epochs!,
        Int() :$epochs-between-reports!,
        Num() :$desired-error!,
    ) returns self

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
[training-algorithm](#training-algorithm), and the parameters set for
these training algorithms.

### test

    multi method test (
        @input,
        @output,
    ) returns List

    # fann_test
    multi method test (
        CArray[num32] $input,
        CArray[num32] $output,
    ) returns CArray[num32]

    multi method test (
        AI::FANN::TrainData $data,
    ) returns Num

    multi method train (
        IO() $path,
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

### activation-function

    # fann_get_activation_function
    multi method activation-function (
        Int    :$layer!,
        Int    :$neuron!,
    ) returns AI::FANN::ActivationFunc

    # fann_set_activation_function
    # fann_set_activation_function_layer
    multi method activation-function (
        AI::FANN::ActivationFunc $function,
        Int    :$layer!,
        Int    :$neuron,
    ) returns self

    # fann_set_activation_function_hidden
    # fann_set_activation_function_output
    multi method activation-function (
        AI::FANN::ActivationFunc $function,
        Bool() :$hidden,
        Bool() :$output,
    ) returns self

If called with no positional arguments, this method returns the activation
function for the neuron number and layer specified in the `:neuron` and
`:layer` parameters respectively, counting the input layer as layer 0. It is
not possible to get activation functions for the neurons in the input layer:
trying to do so is an error.

If called with a member of the AI::FANN::ActivationFunc enum as the first
positional argument, then this function will instead _set_ this as the
activation function for the specified layer and neuron, and return the
calling AI::FANN object.

When used as a setter, specifying the layer is always required. This can
be done with the `:layer` parameter, as described above, or with the `:hidden`
or `:output` flags. The `:hidden` flag will set the activation function for
all neurons in _all_ hidden layers, while the `:output` flag will do so only
for those in the output layer.

When setting the activation function using the `:layer` parameter, the
`:neuron` parameter is optional. If none is set, all neurons in the specified
layer will be modified.

### activation-steepness

    # fann_get_activation_steepness
    multi method activation-steepness (
        Int    :$layer!,
        Int    :$neuron!,
    ) returns Num

    # fann_set_activation_steepness
    # fann_set_activation_steepness_layer
    multi method activation-steepness (
        Num()   $steepness,
        Int    :$layer!,
        Int    :$neuron,
    ) returns self

    # fann_set_activation_steepness_hidden
    # fann_set_activation_steepness_output
    multi method activation-steepness (
        Num()   $steepness,
        Bool() :$hidden,
        Bool() :$output,
    ) returns self

If called with no positional arguments, this method returns the activation
steepness for the neuron number and layer specified in the `:neuron` and
`:layer` parameters respectively, counting the input layer as layer 0. It is
not possible to get activation functions for the neurons in the input layer:
trying to do so is an error.

If called with a positional argument, it will be coerced to a [Num] and this
function will instead _set_ this as the activation steepenss for the specified
layer and neuron and return the calling AI::FANN object.

When used as a setter, specifying the layer is always required. This can
be done with the `:layer` parameter, as described above, or with the `:hidden`
or `output` flags. The `:hidden` flag will set the activation function for
all neurons in _all_ hidden layers, while the `output` flag will do so only
for those in the output layer.

When setting the activation steepness using the `:layer` parameter, the
`:neuron` parameter is optional. If none is set, all neurons in the specified
layer will be modified.

### training-algorithm

    # fann_get_training_algorithm
    multi method training-algorithm returns AI::FANN::Train

    # fann_set_training_algorithm
    multi method training-algorithm (
        AI::FANN::Train $algorithm,
    ) returns self

If called with no positional arguments, this method returns the training
algorithm as per the AI::FANN::Train enum. The training algorithm is used
eg. when running [train](#train) or [cascade-train](#cascade-train) with
a AI::FANN::TrainData object.

If a member of that enum is passed as the first positional argument, this
method instead sets that as the new training algorithm and returns it.

Note that only `FANN_TRAIN_RPROP` and `FANN_TRAIN_QUICKPROP` are allowed
during cascade training.

The default training algorithm is `FANN_TRAIN_RPROP`.

### train-error-function

    # fann_get_train_error_function
    multi method train-error-function returns AI::FANN::ErrorFunc

    # fann_set_train_error_function
    multi method train-error-function (
        AI::FANN::ErrorFunc $function,
    ) returns self

If called with no positional arguments, this method returns the error function
used during training as per the AI::FANN::ErrorFunc enum.

If a member of that enum is passed as the first positional argument, this
method instead sets that as the new training error function and returns it.

The default training error function if `FANN_ERRORFUNC_TANH`.

### train-stop-function

    # fann_get_train_stop_function
    multi method train-stop-function returns AI::FANN::StopFunc

    # fann_set_train_stop_function
    multi method train-stop-function (
        AI::FANN::StopFunc $function,
    ) returns self

If called with no positional arguments, this method returns the stop function
used during training as per the AI::FANN::StopFunc enum.

If a member of that enum is passed as the first positional argument, this
method instead sets that as the new training stop function and returns it.

The default training stop function if `FANN_STOPFUNC_MSE`.

### bit-fail-limit

    # fann_get_bit_fail_limit
    multi method bit-fail-limit returns Num

    # fann_set_bit_fail_limit
    multi method bit-fail-limit (
        Num() $limit,
    ) returns self

If called with no positional arguments, this method returns the bit fail limit
used during training. If called with a positional argument, it will be coerced
to a [Num] and set as the new limit. In that case, this method returns the
value that has been set.

The bit fail limit is used during training when the stop function is set to
`FANN_STOPFUNC_BIT` (see [train-stop-function](#train-stop-function)).

The limit is the maximum accepted difference between the desired output and
the actual output during training. Each output that diverges more than this
limit is counted as an error bit. This difference is divided by two when
dealing with symmetric activation functions, so that symmetric and asymmetric
activation functions can use the same limit.

The default bit fail limit is 0.35.

### learning-rate

    multi method learning-rate returns Num

    multi method learning-rate (
        Num() $rate,
    ) returns self

If called with no positional arguments, this method returns the learning rate
used during training. If called with a positional argument, it will be coerced
to a [Num] and set as the new learning rate.

The learning rate is used to determine how aggressive training should be for
some of the training algorithms (`FANN_TRAIN_INCREMENTAL`, `FANN_TRAIN_BATCH`,
`FANN_TRAIN_QUICKPROP`). Do however note that it is not used in
`FANN_TRAIN_RPROP`.

The default learning rate is 0.7.

### learning-momentum

    multi method learning-momentum returns Num

    multi method learning-momentum (
        Num() $momentum,
    ) returns self

If called with no positional arguments, this method returns the learning
momentum used during training. If called with a positional argument, it will
be coerced to a [Num] and set as the new learning momentum.

The learning momentum can be used to speed up `FANN_TRAIN_INCREMENTAL`
training. Too high a momentum will however not benefit training. Setting the
momentum to 0 will be the same as not using the momentum parameter. The
recommended value of this parameter is between 0 and 1.

The default momentum is 0.

### reset-error

    # fann_reset_MSE
    method reset-error returns self

Resets the mean square error from the network, and the number of bits that
fail.

### mean-square-error

    # fann_get_MSE
    method mean-square-error returns Num

Reads the mean square error from the network. This value is calculated during
training or testing (see [train](#train) and [test](#test) above), and can
therefore sometimes be a bit off if the weights have been changed since the
last calculation of the value.

## Cascade Training

Cascade training differs from ordinary training in that it starts with an
empty neural network and then adds neurons one by one, while it trains the
neural network. The main benefit of this approach is that you do not have to
guess the number of hidden layers and neurons prior to training, but cascade
training has also proved better at solving some problems.

The basic idea of cascade training is that a number of candidate neurons are
trained separate from the real network, then the most promising of these
candidate neurons is inserted into the neural network. Then the output
connections are trained and new candidate neurons are prepared. The candidate
neurons are created as shortcut connected neurons in a new hidden layer, which
means that the final neural network will consist of a number of hidden layers
with one shortcut connected neuron in each.

For methods supporting ordinary, or fixed topology training, see the
[Training](#training) section above.

### cascade-train

    # fann_cascadetrain_on_data
    multi method cascade-train (
        AI::FANN::TrainData:D $data,
        Int() :$max-neurons!,
        Int() :$neurons-between-reports!,
        Num() :$desired-error!,
    ) returns self

    # fann_cascadetrain_on_file
    multi method cascade-train (
        IO() $path,
        Int() :$max-neurons!,
        Int() :$neurons-between-reports!,
        Num() :$desired-error!,
    ) returns self

Trains the network on an entire dataset for a period of time using the
Cascade2 training algorithm. The dataset can be passed as an
AI::FANN::TrainData object in the `data` parameter. Alternatively, if
the `path` is set, it will be coerced to an [IO::Path] object and the
training data will be read from there instead.

This algorithm adds neurons to the neural network while training, which means
that it needs to start with an ANN without any hidden layers. The neural
network should also use shortcut connections, so the `shortcut` flag should
be used when invoking [new](#new), like this

    my $ann = AI::FANN.new: :shortcut,
        layers => [ $data.num-input, $data.num-output ];

### cascade-num-candidates

    # fann_get_cascade_num_candidates
    multi method cascade-num-candidates returns Int

    # fann_set_cascade_num_candidates
    multi method cascade-num-candidates ( Int $groups ) returns self

If called with no positional arguments, this method returns the number of
candidates used during training. If called with an Int as a positional
argument, it will be set as the new value. In that case, this method returns
the value that has been set.

The number of candidates is calculated by multiplying the value returned by
[cascade-activation-functions-count](#cascade-activation-functions-count),
[cascade-activation-steepnesses-count](#cascade-activation-steepnesses-count),
and [cascade-num-candidate-groups](#cascade-num-candidate-groups).

The actual candidates is defined by the
[cascade-activation-functions](#cascade-activation-functions) and
[cascade-activation-steepnesses](#cascade-activation-steepnesses) arrays.
These arrays define the activation functions and activation steepnesses used
for the candidate neurons. If there are 2 activation functions in the
activation function array and 3 steepnesses in the steepness array, then there
will be 2x3=6 different candidates which will be trained. These 6 different
candidates can be copied into several candidate groups, where the only
difference between these groups is the initial weights. If the number of
groups is set to 2, then the number of candidate neurons will be 2x3x2=12.
The number of candidate groups can be set with
[cascade-num-candidate-groups](#cascade-num-candidate-groups).

The default number of candidates is 6x4x2 = 48

### cascade-num-candidate-groups

    # fann_get_cascade_num_candidate_groups
    multi method cascade-num-candidate-groups returns Int

    # fann_set_cascade_num_candidate_groups
    multi method cascade-num-candidate-groups ( Int $groups ) returns self

If called with no positional arguments, this method returns the number of
candidate groups used during training. If called with an Int as a positional
argument, it will be set as the new value. In that case, this method returns
the value that has been set.

The number of candidate groups is the number of groups of identical candidates
which will be used during training.

This number can be used to have more candidates without having to define new
parameters for the candidates.

See [cascade-num-candidates](#cascade-num-candidates) for a description of
which candidate neurons will be generated by this parameter.

The default number of candidate groups is 2

### cascade-activation-steepnesses

    # fann_get_cascade_activation_steepnesses
    multi method cascade-activation-steepnesses returns List

    # fann_set_cascade_activation_steepnesses
    multi method cascade-activation-steepnesses (
        CArray[num32] $steepnesses,
    ) returns self

    multi method cascade-activation-steepnesses (
        *@steepnesses,
    ) returns self

If called with no positional arguments, this method returns the array of
activation steepnesses used by the candidates. See
[cascade-num-candidates](#cascade-num-candidates) for a description of which
candidate neurons will be generated by this array.

If called with a [CArray[num32]][CArray] object as the first positional
argument, this method will instead use that as the new value. Alternatively,
the values that would be in that array can be passed as positional arguments
and they'll be internally converted to a C representation to use instead.

In either case, the new array must be just as long as defined by the count
(see [cascade-activation-steepnesses-count](#cascade-activation-steepnesses-count)).

The default activation steepnesses are [ 0.25, 0.50, 0.75, 1.00 ].

### cascade-activation-functions

    # fann_get_cascade_activation_functions
    multi method cascade-activation-functions returns List

    # fann_set_cascade_activation_functions
    multi method cascade-activation-functions (
        CArray[num32] $functions,
    ) returns self

    multi method cascade-activation-functions (
        *@functions,
    ) returns self

If called with no positional arguments, this method returns the array of
activation functions used by the candidates. See
[cascade-num-candidates](#cascade-num-candidates) for a description of which
candidate neurons will be generated by this array.

If called with a [CArray[num32]][CArray] object as the first positional
argument, this method will instead use that as the new value. Alternatively,
the values that would be in that array can be passed as positional arguments
and they'll be internally converted to a C representation to use instead.

In either case, the new array must be just as long as defined by the count
(see [cascade-activation-functions-count](#cascade-activation-functions-count)).

The default activation functions are [ `FANN_SIGMOID`,
`FANN_SIGMOID_SYMMETRIC`, `FANN_GAUSSIAN`, `FANN_GAUSSIAN_SYMMETRIC`,
`FANN_ELLIOT`, `FANN_ELLIOT_SYMMETRIC`, `FANN_SIN_SYMMETRIC`,
`FANN_COS_SYMMETRIC`, `FANN_SIN`, `FANN_COS` ].

# EXPORT TAGS

AI::FANN exports nothing by default. However, the following enums are
available and can be exported using the `:enum` tag to export *all* enums, or
the `:error` tag to export only the AI::FANN::Error enum.

## AI::FANN::NetType

  * FANN_NETTYPE_LAYER

  * FANN_NETTYPE_SHORTCUT

## AI::FANN::ActivationFunc

The activation functions used for the neurons during training. The activation
functions can either be defined for a group of neurons by calling
[activation-function](#activation-function) with the `:hidden` or `:output`
parameters or it can be defined for a single neuron or layer with the `:layer`
and `:neuron` parameters.

The steepness of an activation function is defined in the same way by calling
[activation-steepness](#activation-steepness).

See the documentation for those functions for details.

The functions are described with functions where

  * `x` is the input to the activation function

  * `y` is the output

  * `s` is the steepness

  * `d` is the derivation.

<!-- -->

  * FANN_LINEAR

    Linear activation function.

        -∞ < y < ∞
        y = x⋅s
        d = s

  * FANN_THRESHOLD

    Threshold activation function. Cannot be used during training.

        y = 0 if x < 0
        y = 1 if x ≥ 0

  * FANN_THRESHOLD_SYMMETRIC

    Symmetric threshold activation function. Cannot be used during training.

        y = -1 if x < 0
        y =  1 if x ≥ 0

  * FANN_SIGMOID

    Sigmoid activation function. This function is very commonly used.

        0 < y < 1
        y = 1 / ( 1 + exp( -2⋅s⋅x ) ) - 1
        d = 2⋅s⋅y⋅( 1 - y² )

  * FANN_SIGMOID_STEPWISE

    Stepwise linear approximation to sigmoid. Faster than sigmoid, but a
    little less precise.

  * FANN_SIGMOID_SYMMETRIC

    Symmetric sigmoid activation function, also known as "tanh". This function
    is very commonly used.

        -1 < y < 1
        y = tanh(s⋅x) = 2 / ( 1 + exp( -2⋅s⋅x ) ) - 1
        d = s⋅( 1 - y² )

  * FANN_SIGMOID_SYMMETRIC_STEPWISE

    Stepwise linear approximation to symmetric sigmoid. Faster than symmetric
    sigmoid, but a little less precise.

  * FANN_GAUSSIAN

    Gaussian activation function.

        0 < y < 1
        y = 0 when x = -∞
        y = 1 when x = 0
        y = 0 when x = ∞
        y = exp( -x⋅s⋅x⋅s )
        d = -2⋅x⋅s⋅y⋅s

  * FANN_GAUSSIAN_SYMMETRIC

    Symmetric Gaussian activation function.

        -1 < y < 1
        y = -1 when x = -∞
        y =  1 when x = 0
        y = -1 when x = ∞
        y = exp( -x⋅s⋅x⋅s )⋅2 - 1
        d = -2⋅x⋅s⋅y⋅s

  * FANN_GAUSSIAN_STEPWISE

    Not yet implemented.

  * FANN_ELLIOT

    Fast (sigmoid like) activation function defined by David Elliott.

        0 < y < 1
        y = x⋅s / 2 / ( 1 + |x⋅s| ) + 0.5
        d = s / ( 2 ⋅ ( 1 + |x⋅s| )² )

  * FANN_ELLIOT_SYMMETRIC

    Fast (symmetric sigmoid like) activation function defined by David Elliott.

        -1 < y < 1
        y = x⋅s  / ( 1 + |x⋅s| )
        d = s / ( 1 + |x⋅s| )²

  * FANN_LINEAR_PIECE

    Bounded linear activation function.

        0 ≤ y ≤ 1
        y = x⋅s
        d = s

  * FANN_LINEAR_PIECE_SYMMETRIC

    Bounded linear activation function.

        -1 ≤ y ≤ 1
        y = x⋅s
        d = s

  * FANN_SIN_SYMMETRIC

    Periodical sinus activation function.

        -1 ≤ y ≤ 1
        y = sin( x⋅s )
        d = s⋅cos( x⋅s )

  * FANN_COS_SYMMETRIC

    Periodical cosinus activation function.

        -1 ≤ y ≤ 1
        y = cos( x⋅s )
        d = s⋅-sin( x⋅s )

  * FANN_SIN

    Periodical sinus activation function.

        0 ≤ y ≤ 1
        y = sin( x⋅s ) / 2 + 0.5
        d = s⋅cos( x⋅s ) / 2

  * FANN_COS

    Periodical cosinus activation function.

        0 ≤ y ≤ 1
        y = cos( x⋅s ) / 2 + 0.5
        d = s⋅-sin( x⋅s ) / 2

## AI::FANN::Train

The training algorithms used when training on AI::FANN::TrainData with
functions like [train](#train) with the `:path` or `:data` arguments.  The
incremental training alters the weights after each time it is presented an
input pattern, while batch only alters the weights once after it has been
presented to all the patterns.

  * FANN_TRAIN_INCREMENTAL

    Standard backpropagation algorithm, where the weights are updated after
    each training pattern. This means that the weights are updated many
    times during a single epoch. For this reason some problems will train very
    fast with this algorithm, while other more advanced problems will not
    train very well.

  * FANN_TRAIN_BATCH

    Standard backpropagation algorithm, where the weights are updated after
    calculating the mean square error for the whole training set. This means
    that the weights are only updated once during an epoch. For this reason
    some problems will train slower with this algorithm. But since the mean
    square error is calculated more correctly than in incremental training,
    some problems will reach better solutions with this algorithm.

* FANN_TRAIN_RPROP

    A more advanced batch training algorithm which achieves good results for
    many problems. The RPROP training algorithm is adaptive, and does
    therefore not use the value set with [learning-rate](#learning-rate).
    Some other parameters can however be set to change the way the RPROP
    algorithm works, but it is only recommended for users with insight in how
    the RPROP training algorithm works. The RPROP training algorithm is
    described by [Riedmiller and Braun, 1993], but the actual learning
    algorithm used here is the iRPROP- training algorithm which is described
    by [Igel and Husken, 2000] which is a variant of the standard RPROP
    training algorithm.

* FANN_TRAIN_QUICKPROP

    A more advanced batch training algorithm which achieves good results
    for many problems. The quickprop training algorithm uses the
    [learning-rate](#learning-rate) parameter along with other more
    advanced parameters, but it is only recommended to change these advanced
    parameters, for users with insight in how the quickprop training
    algorithm works. The quickprop training algorithm is described by
    [Fahlman, 1988].

* FANN_TRAIN_SARPROP

    This is the same algorithm described in
    ["The SARPROP algorithm: a simulated annealing enhancement to resilient back propagation"][SARPROP]

## AI::FANN::ErrorFunc

Error function used during training.

  * FANN_ERRORFUNC_LINEAR

    Standard linear error function.

  * FANN_ERRORFUNC_TANH

    Tanh error function, usually better but can require a lower learning
    rate. This error function aggressively targets outputs that differ much
    from the desired, while not targeting outputs that only differ a little
    that much. This activation function is not recommended for cascade
    training and incremental training.

## AI::FANN::StopFunc

Stop criteria used during training.

  * FANN_STOPFUNC_MSE

    Stop criterion is Mean Square Error (MSE) value.

  * FANN_STOPFUNC_BIT

    Stop criterion is number of bits that fail. The number of bits means the
    number of output neurons which differ more than the bit fail limit (see
    [bit-fail-limit](#bit-fail-limit)). The bits are counted in all of the
    training data, so this number can be higher than the number of training
    data.

## AI::FANN::Error

Used to define error events on AI::FANN and AI::FANN::TrainData objects.

  * FANN_E_NO_ERROR

    No error.

  * FANN_E_CANT_OPEN_CONFIG_R

    Unable to open configuration file for reading.

  * FANN_E_CANT_OPEN_CONFIG_W

    Unable to open configuration file for writing.

  * FANN_E_WRONG_CONFIG_VERSION

    Wrong version of configuration file.

  * FANN_E_CANT_READ_CONFIG

    Error reading info from configuration file.

  * FANN_E_CANT_READ_NEURON

    Error reading neuron info from configuration file.

  * FANN_E_CANT_READ_CONNECTIONS

    Error reading connections from configuration file.

  * FANN_E_WRONG_NUM_CONNECTIONS

    Number of connections not equal to the number expected.

  * FANN_E_CANT_OPEN_TD_W

    Unable to open train data file for writing.

  * FANN_E_CANT_OPEN_TD_R

    Unable to open train data file for reading.

  * FANN_E_CANT_READ_TD

    Error reading training data from file.

  * FANN_E_CANT_ALLOCATE_MEM

    Unable to allocate memory.

  * FANN_E_CANT_TRAIN_ACTIVATION

    Unable to train with the selected activation function.

  * FANN_E_CANT_USE_ACTIVATION

    Unable to use the selected activation function.

  * FANN_E_TRAIN_DATA_MISMATCH

    Irreconcilable differences between two AI::FANN::TrainData objects.

  * FANN_E_CANT_USE_TRAIN_ALG

    Unable to use the selected training algorithm.

  * FANN_E_TRAIN_DATA_SUBSET

    Trying to take subset which is not within the training set.

  * FANN_E_INDEX_OUT_OF_BOUND

    Index is out of bound.

  * FANN_E_SCALE_NOT_PRESENT

    Scaling parameters not present.

  * FANN_E_INPUT_NO_MATCH

    The number of input neurons in the ANN and data don’t match.

  * FANN_E_OUTPUT_NO_MATCH

    The number of output neurons in the ANN and data don’t match.

# COPYRIGHT AND LICENSE

Copyright 2021 José Joaquín Atria

This library is free software; you can redistribute it and/or modify it under
the Artistic License 2.0.

[Array]: https://docs.raku.org/type/Array
[CArray]: https://docs.raku.org/language/nativecall#Arrays
[IO::Path]: https://docs.raku.org/type/IO::Path
[List]: https://docs.raku.org/type/List
[Range]: https://docs.raku.org/type/Range
[SARPROP]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.8197&rep=rep1&type=pdf
