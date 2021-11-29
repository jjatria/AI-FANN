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

    is .num-input,  2, 'Num input';
    is .num-output, 1, 'Num output';
    is .num-data,   4, 'Num data';

    is .input[2][0,1],    [ 255, 0 ], 'Can set input data';
    is .output[0,1]».[0], [ 0, 255 ], 'Can set output data';

    is .scale( -1..1 ), $_, 'Scale returns self';

    is .input[2][0,1],    [  1, -1 ], 'Can scale input data';
    is .output[0,1]».[0], [ -1,  1 ], 'Can scale output data';

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

done-testing;
