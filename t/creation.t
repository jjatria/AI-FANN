#!/usr/bin/env raku

use Test;
use AI::FANN :enum;

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

todo 'Weight modification is flaky'; # FIXME
subtest 'Weights with connection' => {
    my $nn = AI::FANN.new: layers => [ 1, 1 ];
    LEAVE $nn.?destroy;

    my @connections = $nn.connection-array;
    for @connections.kv -> $i, $c {
        $c.weight = $i.Num;
    }

    is $nn.weights(@connections), $nn, 'weights with list returns self';
    is $nn.weights, ( 0e0, 1e0 ), 'Can set weights with connections';

    @connections[0].weight = 42e0;

    is $nn.weights(@connections[0]), $nn, 'weights with connection returns self';
    is $nn.weights, ( 42e0, 1e0 ), 'Can set weights with connections';

    is $nn.weights(
        32,
        from => @connections[0].from-neuron,
        to   => @connections[0].to-neuron,
    ).weights, ( 32e0, 1e0 ), 'Can set weights with values'
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

    subtest 'activation-steepness' => {
        my &get-hidden = { .activation-steepness( :0neuron, :1layer ).round: 0.1 };
        my &get-output = { .activation-steepness( :0neuron, layer => @layers.elems - 1 ).round: 0.1 };

        is $nn.&get-hidden, 0.5, 'getter';

        is $nn.activation-steepness( 0.3, :hidden ).&get-hidden, 0.3, 'hidden setter';
        is $nn.activation-steepness( 0.7, :output ).&get-output, 0.7, 'output setter';
        is $nn.activation-steepness( 0.2, :1layer ).&get-hidden, 0.2, 'layer setter';

        ok $nn.activation-steepness( 0.1, layer => 1, neuron => 0 );
        is $nn.&get-hidden, 0.1, 'setter';

        is $nn.activation-steepness( :2neuron, :1layer ).round(0.1),
            0.2, 'other neurons remain untouched';
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

    given $nn.weights {
        is .min.round(0.1), -0.1, 'Minimum weight ~-0.1';
        is .max.round(0.1),  0.1, 'Maximum weight ~+0.1';
    }

    is $nn.randomise-weights(1..2), $nn, 'randomise-weights returns self';
    is $nn.randomize-weights(1..2), $nn, 'randomize-weights returns self';

    given $nn.weights {
        is .min.round(1), 1, 'Scaled minimum weight';
        is .max.round(1), 2, 'Scaled maximum weight';
    }

    given AI::FANN::TrainData.new: pairs => [ 0 => 1 ] {
        is $nn.init-weights($_), $nn, 'init-weights returns self';
        .?destroy;
    }

    with $nn.clone {
        is .num-input, $nn.num-input, 'Can clone network';
        .?destroy;
    }
}

subtest 'Sparse' => {
    my $connection-rate = 0.75;

    ok $_ = AI::FANN.new( :@layers, :$connection-rate ), 'new';
    LEAVE .?destroy;

    .&test: {
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

    is-deeply .connection-array.map(*.^name),
        [ 'AI::FANN::Connection' xx .total-connections ].List,
        'connection-array';
}

subtest 'Shortcut' => {
    ok $_ = AI::FANN.new( :@layers, :shortcut ), 'new';
    LEAVE .?destroy;

    .&test: {
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

    is-deeply .connection-array.map(*.^name),
        [ 'AI::FANN::Connection' xx .total-connections ].List,
        'connection-array';
}

subtest 'File I/O' => {
    use File::Temp;

    ok my $src = AI::FANN.new: layers => [ 2, 3, 1 ];
    LEAVE $src.?destroy;

    my ( $path, $handle ) = tempfile;
    is $src.save($path),    $src, 'Can save with Str';
    is $src.save($path.IO), $src, 'Can save with IO::Path';

    given AI::FANN.new: :$path {
        ok .defined, 'Can read from Str';
        is .num-input, $src.num-input, 'Read data is equivalent';

        LEAVE .?destroy;
    }

    given AI::FANN.new: path => $path.IO {
        ok .defined, 'Can read from IO::Path';
        LEAVE .?destroy;
    }

    subtest 'File must be writable to save' => {
        throws-like { $src.save: $*HOME.parent.child('forbidden') },
            X::AI::FANN, code => FANN_E_CANT_OPEN_CONFIG_W;
    }

    subtest 'File must be readable to read' => {
        my ($dir) = tempdir;
        throws-like { AI::FANN::TrainData.new: path => $dir.IO.child('missing.net') },
            X::AdHoc, message => /'Cannot read from file: '/;
    }
}

done-testing;
