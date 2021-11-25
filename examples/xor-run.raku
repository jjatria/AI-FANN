#!/usr/bin/env raku

use AI::FANN;

my $ann = AI::FANN.new:
    path => $*PROGRAM.parent.child('output/xor-float.net');

END $ann.destroy;

for (
    [ -1,  1 ],
    [ -1, -1 ],
    [  1,  1 ],
    [  1, -1 ],
) -> @input {
    my $output = $ann.run: :@input;
    say '(% d, % d) -> % f'.sprintf: @input[0], @input[1], $output[0];
}
