#!/usr/bin/env raku

use AI::FANN;
use AI::FANN::Constants;

say 'Creating network.';

my $dir = $*PROGRAM.parent;

my $data = AI::FANN::TrainData.new:
    path => $dir.child('data/mushroom.train');

my $ann = AI::FANN.new:
    layers => [ $data.num-input, 32, $data.num-output ];

say 'Training network.';

$ann.activation-function( :hidden, FANN_SIGMOID_SYMMETRIC )
    .activation-function( :output, FANN_SIGMOID )
    .train: :$data,
        max-epochs             => 300,
        epochs-between-reports => 10,
        desired-error          => 0.0001;

say 'Testing network.';

$ann.reset-error.test: path => $dir.child('data/mushroom.test');

say 'MSE error on test data: %f'.sprintf: $ann.mean-square-error;

say 'Saving network.';

$ann.save: path => $dir.child('output/mushroom-float.net');

say 'Cleaning up.';

$data.destroy;
$ann.destroy;
