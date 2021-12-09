#!/usr/bin/env raku

use AI::FANN :enum;

unit sub MAIN (
    Int   :$epochs-between-reports = 1_000,
    Int   :$max-epochs             = 500_000,
    Num() :$desired-error          = 0.001,
    Num() :$steepness-start        = 1,
    Num() :$steepness-end          = 20,
    Num() :$steepness-step         = 0.1,
);

my $dir = $*PROGRAM.parent;

my $ann = AI::FANN.new: layers => [ 2, 3, 1 ];
LEAVE $ann.?destroy;

my $data = AI::FANN::TrainData.new: path => $dir.child('data/xor.data');
LEAVE $data.?destroy;

$ann.activation-function: FANN_SIGMOID_SYMMETRIC;
$ann.training-algorithm: FANN_TRAIN_QUICKPROP;

say 'Training network on steepness file.';

my $steepness = $steepness-start;
$ann.activation-steepness: $steepness-start;

for ^$max-epochs {
    my $error = $ann.train($data).mean-square-error;

    FIRST printf "Max epochs %8d. Desired error: %.10f\n", $max-epochs, $desired-error
        if $epochs-between-reports;

    if $epochs-between-reports {
        printf "Epochs     %8d. Current error: %.10f\n", $_, $error
            if $_ %% $epochs-between-reports
            || $_ == $max-epochs
            || $_ == 0
            || $error < $desired-error;
    }

    if $error < $desired-error {
        $steepness += $steepness-step;

        last if $steepness > $steepness-end;

        printf "Steepness: %f\n", $steepness;
        $ann.activation-steepness: $steepness;
    }
}

$ann.activation-function: FANN_THRESHOLD_SYMMETRIC;

for ^$data.num-data {
    my $input  := $data.input[$_];
    my $output := $data.output[$_];

    my $result = $ann.run: $input;

    printf "XOR test (% f, % f) -> % f, should be % f, difference = %f\n",
        $input[0],
        $input[1],
        $result[0],
        $output[0],
        abs( $result[0] - $output[0] );
}

$ann.save: $dir.child('output/xor-float.net');
