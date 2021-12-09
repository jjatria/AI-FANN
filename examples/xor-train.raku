#!/usr/bin/env raku

use AI::FANN :enum;

my $dir = $*PROGRAM.parent;

say 'Creating network.';

my $ann  = AI::FANN.new: layers => [ 2, 3, 1 ];
END $ann.?destroy;

my $data = AI::FANN::TrainData.new: path => $dir.child('/data/xor.data');
END $data.?destroy;

given $ann {
    .activation-steepness: 1;
    .activation-function: FANN_SIGMOID_SYMMETRIC, :hidden, :output;
    .train-stop-function: FANN_STOPFUNC_BIT;
    .bit-fail-limit: 0.01;
    .training-algorithm: FANN_TRAIN_RPROP;
    .init-weights: $data;
}

say 'Training network.';

$ann.train: $data,
    desired-error          => 0,
    max-epochs             => 1_000,
    epochs-between-reports => 10;

printf "Testing network. %f\n", $ann.test: $data;

for ^$data.num-data {
    my $input  = $data.input[$_];
    my $output = $data.output[$_];
    my $result = $ann.run: $input;

    printf "XOR test (% d,% d) -> % f, should be % d, difference = %f\n",
        $input[0,1],
        $result[0],
        $output[0],
        abs( $result[0] - $output[0] );
}

say 'Saving network.';

$ann.save: $dir.child('output/xor-float.net')
    or note 'Could not save network data';
