unit module AI::FANN::Raw::Error;

use NativeCall;
use AI::FANN::Raw::Base;

my constant LIB = 'fann';

sub fann_set_error_log(fann_error, Pointer) is export is native(LIB) {*}
sub fann_get_errno(fann_error) returns fann_errno_enum is export is native(LIB) {*}
sub fann_reset_errno(fann_error) is export is native(LIB) {*}
sub fann_reset_errstr(fann_error) is export is native(LIB) {*}
sub fann_get_errstr(fann_error) returns Str is export is native(LIB) {*}
sub fann_print_error(fann_error) is export is native(LIB) {*}
