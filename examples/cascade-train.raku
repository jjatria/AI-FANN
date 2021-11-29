#!/usr/bin/env raku
#
use AI::FANN;
use AI::FANN::Constants;

unit sub MAIN (
           :$training-algorithm = FANN_TRAIN_RPROP,
    Bool() :$multi,
);

my $dir = $*PROGRAM.parent;

say 'Reading data.';

my $train = AI::FANN::TrainData.new:
    path => $dir.child('data/parity8.train');

END $train.destroy;

my $test = AI::FANN::TrainData.new:
    path => $dir.child('data/parity8.test');

END $test.destroy;

$train.scale: -1 .. 1;
$test.scale:  -1 .. 1;

say 'Creating network.';

my $ann = AI::FANN.new: :shortcut,
    layers => [ $train.num-input, $train.num-output ];

END $ann.destroy;

$ann.training-algorithm:   $training-algorithm;
$ann.activation-function:  FANN_SIGMOID_SYMMETRIC, :hidden;
$ann.activation-function:  FANN_LINEAR,            :output;
$ann.train-error-function: FANN_ERRORFUNC_LINEAR;

unless $multi {
    $ann.cascade-activation-steepnesses: 1;
    $ann.cascade-activation-functions:   FANN_SIGMOID_SYMMETRIC;
    $ann.cascade-num-candidate-groups:   8;
}

if $training-algorithm == FANN_TRAIN_QUICKPROP {
    $ann.learning-rate: 0.35;
    $ann.randomize-weights: -2 .. 2;
}

$ann.bit-fail-limit: 0.9;
$ann.train-stop-function: FANN_STOPFUNC_BIT;
$ann.print-parameters;

$ann.save: $dir.child('output/cascade-train2.net');

say 'Training network.';

$ann.cascade-train:
    data                    => $train,
    max-neurons             => 30,
    neurons-between-reports => 1,
    desired-error           => 0;

$ann.print-connections;

my      $mse-train = $ann.test: data => $train;
my $bit-fail-train = $ann.bit-fail;

my      $mse-test  = $ann.test: data => $test;
my $bit-fail-test  = $ann.bit-fail;

say Q:b'\nTrain error: %f, Train bit-fail: %d, Test error: %f, Test bit-fail: %d\n'.sprintf:
    $mse-train, $bit-fail-train, $mse-test, $bit-fail-test;

say 'Saving network.';

$ann.save: $dir.child('output/cascade-train.net');

say 'Cleaning up.';
