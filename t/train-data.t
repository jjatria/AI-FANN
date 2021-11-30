#!/usr/bin/env raku

use Test;
use AI::FANN;

subtest 'Create ' => {
    ok $_ = AI::FANN::TrainData.new(
        pairs => [
            [   0,   0 ] => [   0 ],
            [   0, 255 ] => [ 255 ],
            [ 255,   0 ] => [ 255 ],
            [ 255, 255 ] => [   0 ],
        ],
    ), 'Create from pairs';

    LEAVE .?destroy;

    is .num-input,  2, 'Num input';
    is .num-output, 1, 'Num output';
    is .num-data,   4, 'Num data';

    is .input[2][0,1],    [ 255, 0 ], 'Can set input data';
    is .output[0,1]».[0], [ 0, 255 ], 'Can set output data';

    is .scale( -1..1, :input ), $_, 'Scale returns self';

    is .input[2][0,1]».round,    [   1,  -1 ], 'Can scale input data';
    is .output[0,1]».[0]».round, [   0, 255 ], 'Output is unchanged';

    is .scale( -10..10, :output ), $_, 'Scale returns self';

    is .input[2][0,1]».round,    [   1,  -1 ], 'Input is unchanged';
    is .output[0,1]».[0]».round, [ -10,  10 ], 'Can scale output data';

    is .scale( ^256 ), $_, 'Scale returns self';

    is .input[2][0,1]».round,    [ 255, 0 ], 'Can scale input data';
    is .output[0,1]».[0]».round, [ 0, 255 ], 'Can scale output data';

    is .shuffle, $_, 'Shuffle returns self';
}

subtest 'Create from pairs' => {
    ok $_ = AI::FANN::TrainData.new(
        pairs => [
            [ -1, -1 ] => [ -1 ],
            [ -1,  1 ] => [  1 ],
            [  1, -1 ] => [  1 ],
            [  1,  1 ] => [ -1 ],
        ],
    ), 'Create from pairs';

    LEAVE .?destroy;

    is .num-input,  2, 'Num input';
    is .num-output, 1, 'Num output';
    is .num-data,   4, 'Num data';

    subtest ':pairs must be Pairs' => {
        throws-like
            { AI::FANN::TrainData.new: pairs => [ 0, 1 ] },
            X::AdHoc, message => 'Values in :pairs must be Pair objects';
    }

    subtest 'Inputs must be consistent' => {
        throws-like
            { AI::FANN::TrainData.new: pairs => [ [ 0, 1 ] => 1, 1 => 2 ] },
            X::AdHoc, message => 'Number of inputs must be consistent';
    }

    subtest 'Outputs must be consistent' => {
        throws-like
            { AI::FANN::TrainData.new: pairs => [ 1 => [1], 1 => [1, 2] ] },
            X::AdHoc, message => 'Number of outputs must be consistent';
    }

    subtest 'Inputs must not be empty' => {
        throws-like
            { AI::FANN::TrainData.new: pairs => [ [] => [1] ] },
            X::AdHoc, message => 'Data must have at least one input';
    }

    subtest 'Outputs must not be empty' => {
        throws-like
            { AI::FANN::TrainData.new: pairs => [ [1] => [] ] },
            X::AdHoc, message => 'Data must have at least one output';
    }
}

subtest 'Subset and merge' => {
    my $a = AI::FANN::TrainData.new: pairs => [
        [ -1, -1 ] => [ -1 ],
        [ -1,  1 ] => [  1 ],
        [  1, -1 ] => [  1 ],
        [  1,  1 ] => [ -1 ],
    ];

    LEAVE $a.?destroy;

    is $a.num-data, 4, 'Original data length';

    my $b = $a.subset: 0, 1;
    is $b.num-data, 1, 'Can subset from start';

    LEAVE $b.?destroy;

    my $c = $a.subset: 1, 3;
    is $c.num-data, 3, 'Can subset from middle';

    LEAVE $c.?destroy;

    given AI::FANN::TrainData.new: data => [ $b, $c ] {
        is .num-data, 4, 'Can combine subsets';
        .?destroy;
    }

    given $a.clone {
        is .num-data, 4, 'Can clone data set';
        .?destroy;
    }

    subtest 'Subset position must be positive' => {
        throws-like { $a.subset: -1, 2 },
            X::OutOfRange, what => 'Subset position', range => '0 1 2 3';
    }

    subtest 'Subset position must be within data' => {
        throws-like { $a.subset: 4, 2 },
            X::OutOfRange, what => 'Subset position', range => '0 1 2 3';
    }

    subtest 'Subset length depends on pos' => {
        throws-like { $a.subset: 0, 5 },
            X::OutOfRange, what => 'Subset length', range => '1 2 3 4';

        throws-like { $a.subset: 2, 3 },
            X::OutOfRange, what => 'Subset length', range => '1 2';
    }
}

done-testing;
