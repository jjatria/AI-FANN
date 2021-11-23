unit module AI::FANN::Raw::Base;

use NativeCall;

constant float                  is export = num32;
constant fann_type              is export = num32;
constant fann_activationfunc    is export = int32;
constant fann_nettype           is export = int32;
constant fann_errno             is export = int32;
constant fann_train             is export = int32;

class fann_connection is repr('CStruct') is export {
    has uint32 $from-neuron;
    has uint32 $to-neuron;
    has fann_type $weight;
}

class fann              is repr('CPointer') is export {*}
class fann_train_data   is repr('CPointer') is export {*}
class fann_error        is repr('CPointer') is export {*}
