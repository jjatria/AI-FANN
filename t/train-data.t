#!/usr/bin/env raku

use Test;
use AI::FANN;

subtest 'Create ' => {
    ok $_ = AI::FANN::TrainData.new(
        pairs => [
            [ -1,  1 ] => [ 0 ],
            [ -1, -1 ] => [ 0 ],
            [  1,  1 ] => [ 1 ],
            [ -1,  1 ] => [ 1 ],
        ],
    ), 'Create from pairs';

    is .num-input,  2, 'Num input';
    is .num-output, 1, 'Num output';
    is .num-data,   4, 'Num data';
}

subtest 'Create from pairs' => {
    ok $_ = AI::FANN::TrainData.new(
        pairs => [
            [ -1,  1 ] => [ 0 ],
            [ -1, -1 ] => [ 0 ],
            [  1,  1 ] => [ 1 ],
            [ -1,  1 ] => [ 1 ],
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
