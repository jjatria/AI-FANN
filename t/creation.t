#!/usr/bin/env raku

use Test;
use AI::FANN;
use AI::FANN::Constants;

#      Input     Hidden    Output
#           \    / | \    /
my @layers = 2, 3, 5, 3, 1;

{
    ok my $nn = AI::FANN.new( :@layers ), 'new';
    LEAVE $nn.destroy;

    is $nn.num-layers, +@layers, 'num-layers';
    is $nn.num-input,  @layers.head, 'num-input';
    is $nn.num-output, @layers.tail, 'num-output';
    is $nn.total-neurons, @layers.sum + @layers - 1, 'total-neurons';
    is $nn.total-connections, 51, 'total-connections';
    is $nn.network-type, FANN_NETTYPE_LAYER, 'network-type';
    is $nn.connection-rate, 1, 'connection-rate';

    is-deeply $nn.layer-array, @layers.List, 'layer-array';
    is-deeply $nn.bias-array, @layers.List, 'bias-array';

    is-deeply $nn.connection-array.map(*.^name),
        [ 'AI::FANN::Raw::Base::fann_connection' xx $nn.total-connections ].List,
        'connection-array';
}

done-testing;
