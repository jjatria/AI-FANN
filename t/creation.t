#!/usr/bin/env raku

use Test;
use AI::FANN;
use AI::FANN::Constants;

#      Input     Hidden    Output
#           \    / | \    /
my @layers = 2, 3, 5, 3, 1;

sub test ( $o, $checks ) {
    for $checks.kv -> $name, $want {
        my $have = $o."$name"();

        if $have ~~ Positional {
            is-deeply $have, $want, $name;
        }
        else {
            is $have, $want, $name;
        }
    }
}

subtest 'Standard' => {
    ok my $nn = AI::FANN.new( :@layers ), 'new';
    LEAVE $nn.destroy;

    $nn.&test: {
        bias-array        => @layers.List,
        bit-fail          => 0,
        connection-rate   => 1,
        layer-array       => @layers.List,
        network-type      => FANN_NETTYPE_LAYER,
        num-input         => @layers.head,
        num-layers        => @layers.elems,
        num-output        => @layers.tail,
        total-connections => 51,
        total-neurons     => @layers.sum + @layers - 1,
    };

    is-deeply $nn.connection-array.map(*.^name),
        [ 'AI::FANN::Connection' xx $nn.total-connections ].List,
        'connection-array';

    subtest 'activation-function' => {
        my &get-hidden = { .activation-function: neuron => 0, layer => 1 };
        my &get-output = { .activation-function: neuron => 0, layer => @layers.elems - 1 };

        is $nn.&get-hidden, FANN_SIGMOID_STEPWISE, 'getter';

        is $nn.activation-function( FANN_SIGMOID,   :hidden ).&get-hidden, FANN_SIGMOID,  'hidden setter';
        is $nn.activation-function( FANN_GAUSSIAN,  :output ).&get-output, FANN_GAUSSIAN, 'output setter';
        is $nn.activation-function( FANN_ELLIOT, layer => 1 ).&get-hidden, FANN_ELLIOT,   'layer setter';

        ok $nn.activation-function( FANN_THRESHOLD, layer => 1, neuron => 0 );
        is $nn.&get-hidden, FANN_THRESHOLD, 'setter';

        is $nn.activation-function( neuron => 2, layer => 1 ),
            FANN_ELLIOT, 'other neurons remain untouched';
    }

    subtest 'training-algorithm' => {
        is $nn.training-algorithm, FANN_TRAIN_RPROP, 'getter';
        is $nn.training-algorithm(FANN_TRAIN_INCREMENTAL)
            .training-algorithm, FANN_TRAIN_INCREMENTAL, 'setter';
        is $nn.training-algorithm(1)
            .training-algorithm, FANN_TRAIN_BATCH, 'setter with Int';
    }

    subtest 'train-error-function' => {
        is $nn.train-error-function, FANN_ERRORFUNC_TANH, 'getter';
        is $nn.train-error-function(FANN_ERRORFUNC_LINEAR)
            .train-error-function, FANN_ERRORFUNC_LINEAR, 'setter';
        is $nn.train-error-function(0)
            .train-error-function, FANN_ERRORFUNC_LINEAR, 'setter';
    }

    subtest 'train-stop-function' => {
        is $nn.train-stop-function, FANN_STOPFUNC_MSE, 'getter';
        is $nn.train-stop-function(FANN_STOPFUNC_BIT)
            .train-stop-function, FANN_STOPFUNC_BIT, 'setter';
        is $nn.train-stop-function(0)
            .train-stop-function, FANN_STOPFUNC_MSE, 'setter';
    }

    subtest 'bit-fail-limit' => {
        is $nn.bit-fail-limit.round(0.01), 0.35, 'getter';
        is $nn.bit-fail-limit(1.5).bit-fail-limit, 1.5, 'setter';
    }

    subtest 'learning-rate' => {
        is $nn.learning-rate.round(0.1), 0.7, 'getter';
        is $nn.learning-rate(1).learning-rate, 1, 'setter';
    }

    subtest 'learning-momentum' => {
        is $nn.learning-momentum, 0, 'getter';
        is $nn.learning-momentum(1).learning-momentum, 1, 'setter';
    }

    given $nn.connection-array {
        is $_».weight.min.round(0.1), -0.1, 'Minimum weight ~-0.1';
        is $_».weight.max.round(0.1),  0.1, 'Maximum weight ~+0.1';
    }

    is $nn.randomise-weights(1..2), $nn, 'randomise-weights returns self';
    is $nn.randomize-weights(1..2), $nn, 'randomize-weights returns self';

    given $nn.connection-array {
        is $_».weight.min.round(1), 1, 'Scaled minimum weight';
        is $_».weight.max.round(1), 2, 'Scaled maximum weight';
    }

    with $nn.clone {
        is .num-input, $nn.num-input, 'Can clone network';
        .?destroy;
    }
}

subtest 'Sparse' => {
    my $connection-rate = 0.75;

    ok my $nn = AI::FANN.new( :@layers, :$connection-rate ), 'new';
    LEAVE $nn.destroy;

    $nn.&test: {
        bias-array        => @layers.List,
        bit-fail          => 0,
        connection-rate   => $connection-rate,
        layer-array       => @layers.List,
        network-type      => FANN_NETTYPE_LAYER,
        num-input         => @layers.head,
        num-layers        => @layers.elems,
        num-output        => @layers.tail,
        total-connections => 42,
        total-neurons     => @layers.sum + @layers - 1,
    };

    is-deeply $nn.connection-array.map(*.^name),
        [ 'AI::FANN::Connection' xx $nn.total-connections ].List,
        'connection-array';
}

subtest 'Shortcut' => {
    ok my $nn = AI::FANN.new( :@layers, :shortcut ), 'new';
    LEAVE $nn.destroy;

    $nn.&test: {
        bias-array        => @layers.List,
        bit-fail          => 0,
        connection-rate   => 1,
        layer-array       => @layers.List,
        network-type      => FANN_NETTYPE_SHORTCUT,
        num-input         => @layers.head,
        num-layers        => @layers.elems,
        num-output        => @layers.tail,
        total-connections => 86,
        total-neurons     => 15,
    };

    is-deeply $nn.connection-array.map(*.^name),
        [ 'AI::FANN::Connection' xx $nn.total-connections ].List,
        'connection-array';
}

done-testing;
