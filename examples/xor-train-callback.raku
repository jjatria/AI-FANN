#!/usr/bin/env raku

use AI::FANN;
use AI::FANN::Constants;

my $dir = $*PROGRAM.parent;

my $ann = AI::FANN.new: layers => [ 2, 3, 1 ];

END $ann.destroy;

$ann.activation-function: FANN_SIGMOID_SYMMETRIC, :hidden, :output;

my @output = ( -1, 1, 1, -1 );
my @input = (
    [ -1, -1 ],
    [ -1,  1 ],
    [  1, -1 ],
    [  1,  1 ],
);

use NativeCall;
my $data = AI::FANN::TrainData.new:
    num-input  => 2,
    num-output => 1,
    num-data   => 4,
    callback   => sub (
        uint32 $num,
        uint32 $num-input,
        uint32 $num-output,
        CArray[num32] $input,
        CArray[num32] $output,
    ) {
        $input[ ^$num-input ] = @input[$num]».Num;
        $output[0]            = @output[$num].Num;
    }

$ann.train: :$data,
    desired-error          => 0.001,
    max-epochs             => 500_000,
    epochs-between-reports => 1_000;

$ann.save: path => $dir.child('output/xor-float.net')
    or note 'Could not save network data';
