#!/usr/bin/env raku

use AI::FANN;
use AI::FANN::Constants;

my $dir = $*PROGRAM.parent;

my $ann = AI::FANN.new: layers => [ 2, 3, 1 ];

END $ann.destroy;

$ann.activation-function: FANN_SIGMOID_SYMMETRIC, :hidden, :output;

$ann.train:
    path                   => $dir.child('/data/xor.data'),
    desired-error          => 0.001,
    max-epochs             => 500_000,
    epochs-between-reports => 1_000;

$ann.save: path => $dir.child('/output/xor_float.net')
    or note 'Could not save network data';
