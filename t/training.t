#!/usr/bin/env raku

use Test;
use AI::FANN :enum;

my $nn = AI::FANN.new: layers => [ 2, 3, 1 ];
LEAVE $nn.?destroy;

my $data = AI::FANN::TrainData.new: pairs => [
    [ -1, -1 ] => [ -1 ],
    [ -1,  1 ] => [  1 ],
    [  1, -1 ] => [  1 ],
    [  1,  1 ] => [ -1 ],
];

LEAVE $data.?destroy;

my @epochs;
my $callback = sub (
        $fann,
        $data,
    Int $max-epochs,
    Int $epochs-between-reports,
    Num $desired-error,
    Int $epoch,
    --> Int
) {
    @epochs.push: $epoch;
    return $epoch >= 8 ?? -1 !! 1; # Stop at epoch 8
}

is $nn.callback($callback), $nn, 'callback returns self';

my $train = $nn.train: $data,
    desired-error          => 0.001,
    max-epochs             => 500_000,
    epochs-between-reports => 2;

is $train, $nn, 'train returns self';
is @epochs, [ 1, 2, 4, 6, 8 ],
    'Callback ran every 2 epochs and could be stopped';


is $nn.callback(:delete), $nn, 'Clearing a callback returns self';

throws-like { $nn.callback }, X::AdHoc, message => /'The callback method'/;

subtest 'Scaling' => {
    my $bad = AI::FANN::TrainData.new: pairs => [ 1 => 0 ];
    LEAVE $bad.?destroy;

    my $input  =  0..1;
    my $output = -2..-1;

    throws-like { $nn.scale: $data },
        X::AI::FANN, code => FANN_E_SCALE_NOT_PRESENT,
        'Scale needs scaling factor';

    throws-like { $nn.descale: $data },
        X::AI::FANN, code => FANN_E_SCALE_NOT_PRESENT,
        'Descale needs scaling factor';

    is $nn.scaling( $data, :$input, :$output ), $nn, 'Scaling returns self';
    is $nn.scaling( $data, :$input           ), $nn, 'Scaling with only input returns self';
    is $nn.scaling( $data,          :$output ), $nn, 'Scaling with only output returns self';

    is $nn.scale($data), $nn, 'Scale returns self';

    is $data.input[1][^$data.num-input], (  0,  1 ), 'Scaled input data';
    is $data.output[0,1]».[0],           ( -2, -1 ), 'Scaled output data';

    is $nn.descale($data), $nn, 'Descale returns self';

    is $data.input[1][^$data.num-input], ( -1, 1 ), 'Descaled input data';
    is $data.output[0,1]».[0],           ( -1, 1 ), 'Descaled output data';

    use NativeCall;

    given CArray[num32].new: ( -1, 1 )».Num -> $input {
        is $nn.scale(:$input), $nn, 'Scaling CArray input returns self';
        is $input[0, 1], ( 0, 1 )».Num, 'Scaled CArray input';

        is $nn.descale(:$input), $nn, 'Descaling CArray input returns self';
        is $input[0, 1], ( -1, 1 )».Num, 'Descaled CArray input';
    }

    given CArray[num32].new: -1.Num -> $output {
        is $nn.scale(:$output), $nn, 'Scaling CArray output returns self';
        is $output[0], -2.Num, 'Scaled CArray output';

        is $nn.descale(:$output), $nn, 'Descaling CArray output returns self';
        is $output[0], -1.Num, 'Descaled CArray output';
    }

    given [ -1, 1 ] -> @input {
        is $nn.scale(:@input), $nn, 'Scaling array input returns self';
        is @input, ( 0, 1 ), 'Scaled array input';

        is $nn.descale(:@input), $nn, 'Descaling array input returns self';
        is @input, ( -1, 1 ), 'Descaled array input';
    }

    given [ -1 ] -> @output {
        is $nn.scale(:@output), $nn, 'Scaling array output returns self';
        is @output, ( -2 ), 'Scaled array output';

        is $nn.descale(:@output), $nn, 'Descaling array output returns self';
        is @output, ( -1 ), 'Descaled array output';
    }

    is $nn.scaling(:delete), $nn, 'Clearing scaling returns self';

    throws-like { $nn.scaling( $bad, :$input, :$output ) },
        X::AI::FANN, code => FANN_E_TRAIN_DATA_MISMATCH,
        'Scaling needs matching dataset';
}

done-testing;
