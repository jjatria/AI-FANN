#!/usr/bin/env raku

use Test;
use AI::FANN :error;

my $ann = AI::FANN.new: layers => [ 2, 3, 1 ];
END $ann.destroy;

subtest 'new' => sub {
    throws-like { AI::FANN.new: path => 'missing' },
        X::AdHoc, message => "Cannot read from file: 'missing'";
}

subtest 'save' => sub {
    throws-like {
        AI::FANN.new( layers => [ 1, 1 ] ).save:
            $*HOME.parent.child('forbidden') # Hopefully...
    }, X::AI::FANN, code => FANN_E_CANT_OPEN_CONFIG_W;
}

subtest 'activation-function' => sub {
    throws-like { $ann.activation-function: layer => -1, neuron => 1 },
        X::OutOfRange, what => 'Layer index', range => '1 2';

    throws-like { $ann.activation-function: layer => 0, neuron => 1 },
        X::OutOfRange, what => 'Layer index', range => '1 2';

    throws-like { $ann.activation-function: layer => 4, neuron => 1 },
        X::OutOfRange, what => 'Layer index', range => '1 2';

    throws-like { $ann.activation-function: layer => 1, neuron => 4 },
        X::OutOfRange, what => 'Neuron index', range => '0 1 2 3';

    throws-like { $ann.activation-function: layer => 1, neuron => -1 },
        X::OutOfRange, what => 'Neuron index', range => '0 1 2 3';

    throws-like { $ann.activation-function: 42, :hidden },
        X::AdHoc, message => 'Invalid activation function: must be a value in AI::FANN::ActivationFunc';

    lives-ok { $ann.activation-function: 5, :hidden },
        'Accepts plain numeric values';
}

subtest 'activation-steepness' => sub {
    throws-like { $ann.activation-steepness: layer => -1, neuron => 1 },
        X::OutOfRange, what => 'Layer index', range => '1 2';

    throws-like { $ann.activation-steepness: layer => 0, neuron => 1 },
        X::OutOfRange, what => 'Layer index', range => '1 2';

    throws-like { $ann.activation-steepness: layer => 4, neuron => 1 },
        X::OutOfRange, what => 'Layer index', range => '1 2';

    throws-like { $ann.activation-steepness: layer => 1, neuron => 4 },
        X::OutOfRange, what => 'Neuron index', range => '0 1 2 3';

    throws-like { $ann.activation-steepness: layer => 1, neuron => -1 },
        X::OutOfRange, what => 'Neuron index', range => '0 1 2 3';

    lives-ok { $ann.activation-steepness: 5, :hidden },
        'Accepts plain numeric values';
}

subtest 'training-algorithm' => {
    throws-like { $ann.training-algorithm: 42 },
        X::AdHoc, message => 'Invalid training algorithm: must be a value in AI::FANN::Train';
}

subtest 'train-error-function' => {
    throws-like { $ann.train-error-function: 42 },
        X::AdHoc, message => 'Invalid error function: must be a value in AI::FANN::ErrorFunc';
}

subtest 'train-stop-function' => {
    throws-like { $ann.train-stop-function: 42 },
        X::AdHoc, message => 'Invalid stop function: must be a value in AI::FANN::StopFunc';
}

subtest 'train' => {
    throws-like {
        $ann.train: 'missing',
            max-epochs             => 1,
            epochs-between-reports => 1,
            desired-error          => 1;

    }, X::AdHoc, message => "Cannot read from file: 'missing'";
}

subtest 'cascade-train' => {
    throws-like {
        $ann.cascade-train: 'missing',
            max-neurons             => 1,
            neurons-between-reports => 1,
            desired-error           => 1;

    }, X::AdHoc, message => "Cannot read from file: 'missing'";
}

subtest 'test' => {
    throws-like { $ann.test: 'missing' },
        X::AdHoc, message => "Cannot read from file: 'missing'";
}

done-testing;
