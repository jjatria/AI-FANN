unit module AI::FANN::Raw::Base;

use NativeCall;

constant float                  is export = num32;
constant fann_type              is export = num32;
constant fann_activationfunc    is export = int32;
constant fann_nettype           is export = int32;
constant fann_errno             is export = int32;
constant fann_train             is export = int32;

class fann            is repr('CPointer') is export {*}
class fann_error      is repr('CPointer') is export {*}
class fann_connection is repr('CPointer') is export {*}

class fann_train_data is repr('CStruct') is export {
    has int32   $!errno;
    has Pointer $!error-log;
    has Str     $!errstr;
    has uint32  $.num-data;
    has uint32  $.num-input;
    has uint32  $.num-output;
    has CArray[CArray[fann_type]] $.input;
    has CArray[CArray[fann_type]] $.output;
}
