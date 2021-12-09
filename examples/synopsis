# See below for details on export tags
use AI::FANN :enum;

#                               Hidden
#                         Input    |    Output
#                              \   |   /
given AI::FANN.new: layers => [ 2, 3, 1 ] {

    # A sample data set for solving the XOR problem
    my $data = AI::FANN::TrainData.new: pairs => [
        [ -1, -1 ] => [ -1 ],
        [ -1,  1 ] => [  1 ],
        [  1, -1 ] => [  1 ],
        [  1,  1 ] => [ -1 ],
    ];

    .activation-function: FANN_SIGMOID_SYMMETRIC;

    .train: $data,
        desired-error          => 0.001,
        max-epochs             => 500_000,
        epochs-between-reports => 0;

    say .run: [ 1, -1 ];
}
