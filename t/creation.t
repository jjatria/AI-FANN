#!/usr/bin/env raku

use Test;
use AI::FANN;
use AI::FANN::Constants;

#      Input     Hidden    Output
#           \    / | \    /
my @layers = 2, 3, 5, 3, 1;

subtest 'Standard' => {
    ok my $nn = AI::FANN.new( :@layers ), 'new';
    LEAVE $nn.destroy;

    is $nn.bit-fail, 0, 'bit-fail';
    is $nn.connection-rate, 1, 'connection-rate';
    is $nn.network-type, FANN_NETTYPE_LAYER, 'network-type';
    is $nn.num-input,  @layers.head, 'num-input';
    is $nn.num-layers, +@layers, 'num-layers';
    is $nn.num-output, @layers.tail, 'num-output';
    is $nn.total-connections, 51, 'total-connections';
    is $nn.total-neurons, @layers.sum + @layers - 1, 'total-neurons';
    is $nn.activation-function( layer => 1, neuron => 0 ),
        FANN_SIGMOID_STEPWISE, 'activation-function';

    is-deeply $nn.layer-array, @layers.List, 'layer-array';
    is-deeply $nn.bias-array, @layers.List, 'bias-array';

    is-deeply $nn.connection-array.map(*.^name),
        [ 'AI::FANN::Connection' xx $nn.total-connections ].List,
        'connection-array';
}

subtest 'Sparse' => {
    ok my $nn = AI::FANN.new( :@layers, connection-rate => 0.75 ), 'new';
    LEAVE $nn.destroy;

    is $nn.bit-fail, 0, 'bit-fail';
    is $nn.connection-rate, 0.75, 'connection-rate';
    is $nn.network-type, FANN_NETTYPE_LAYER, 'network-type';
    is $nn.num-input,  @layers.head, 'num-input';
    is $nn.num-layers, +@layers, 'num-layers';
    is $nn.num-output, @layers.tail, 'num-output';
    is $nn.total-connections, 42, 'total-connections';
    is $nn.total-neurons, @layers.sum + @layers - 1, 'total-neurons';
    is $nn.activation-function( layer => 1, neuron => 0 ),
        FANN_SIGMOID_STEPWISE, 'activation-function';

    is-deeply $nn.layer-array, @layers.List, 'layer-array';
    is-deeply $nn.bias-array, @layers.List, 'bias-array';

    is-deeply $nn.connection-array.map(*.^name),
        [ 'AI::FANN::Connection' xx $nn.total-connections ].List,
        'connection-array';
}

subtest 'Shortcut' => {
    ok my $nn = AI::FANN.new( :@layers, :shortcut ), 'new';
    LEAVE $nn.destroy;

    is $nn.bit-fail, 0, 'bit-fail';
    is $nn.connection-rate, 1, 'connection-rate';
    is $nn.network-type, FANN_NETTYPE_SHORTCUT, 'network-type';
    is $nn.num-input,  @layers.head, 'num-input';
    is $nn.num-layers, +@layers, 'num-layers';
    is $nn.num-output, @layers.tail, 'num-output';
    is $nn.total-connections, 86, 'total-connections';
    is $nn.total-neurons, 15, 'total-neurons';
    is $nn.activation-function( layer => 1, neuron => 0 ),
        FANN_SIGMOID_STEPWISE, 'activation-function';

    is-deeply $nn.layer-array, @layers.List, 'layer-array';
    is-deeply $nn.bias-array, @layers.List, 'bias-array';

    is-deeply $nn.connection-array.map(*.^name),
        [ 'AI::FANN::Connection' xx $nn.total-connections ].List,
        'connection-array';
}

done-testing;
