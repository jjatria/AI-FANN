use NativeCall;

unit class AI::FANN::Connection is repr('CStruct');

has uint32 $.from-neuron;
has uint32 $.to-neuron;
has num32  $.weight; # Should be same as fann_type
