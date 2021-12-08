unit module AI::FANN::Raw::Base;

use NativeCall;

constant float                    is export = num32;
constant fann_type                is export = num32;
constant fann_activationfunc_enum is export = int32;
constant fann_errorfunc_enum      is export = int32;
constant fann_stopfunc_enum       is export = int32;
constant fann_nettype_enum        is export = int32;
constant fann_errno_enum          is export = int32;
constant fann_train_enum          is export = int32;

class fann_connection is repr('CPointer') is export {*}

class fann_error is repr('CStruct') is export {
    has int32   $.errno;
    has Pointer $!error-log;
    has Str     $.errstr;
}

class fann is repr('CStruct') is export {
    has int32   $.errno;
    has Pointer $!error-log;
    has Str     $.errstr;

    method error { nativecast( fann_error, self ) }
}

class fann_train_data is repr('CStruct') is export {
    has int32   $.errno;
    has Pointer $!error-log;
    has Str     $.errstr;
    has uint32  $.num-data;
    has uint32  $.num-input;
    has uint32  $.num-output;
    has CArray[CArray[fann_type]] $.input;
    has CArray[CArray[fann_type]] $.output;

    method error { nativecast( fann_error, self ) }
}
