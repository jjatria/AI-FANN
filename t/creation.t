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
