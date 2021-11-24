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

$ann.set-activation-function: :hidden, FANN_SIGMOID_SYMMETRIC;
$ann.set-activation-function: :output, FANN_SIGMOID;

$ann.train: :$data,
    max-epochs             => 300,
    epochs-between-reports => 10,
    desired-error          => 0.0001;

say 'Testing network.';

my $test = AI::FANN::TrainData.new:
    path => $dir.child('data/mushroom.test');


$ann.reset-MSE;
for ^$test.length {
    $ann.test: input => $test.input[$_], output => $test.output[$_];
}

say 'MSE error on test data: %f'.sprintf: $ann.get-MSE;

say 'Saving network.';

$ann.save: path => $dir.child('mushroom_float.net');

say 'Cleaning up.';

$data.destroy;
$test.destroy;
$ann.destroy;
