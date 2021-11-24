#!/usr/bin/env raku

use Test;
use AI::FANN;

my $ann = AI::FANN.new: layers => [ 2, 3, 1 ];
END $ann.destroy;

throws-like { $ann.activation-function: layer => 0, neuron => 1 },
    X::AdHoc, message => /'Cannot access the activation function of the input layer'/;

done-testing;
