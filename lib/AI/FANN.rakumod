use NativeCall;
use AI::FANN::Raw;
use AI::FANN::Constants ();

# See https://stackoverflow.com/a/43554058/807650
my role StructArray[Mu:U \T where .REPR eq 'CStruct'] does Positional[T] {
    has $.bytes;
    has $.elems;

    method new(UInt \n) {
        self.bless(bytes => buf8.allocate(n * nativesizeof T), elems => n);
    }

    method AT-POS(UInt \i where ^$!elems) {
        nativecast(T, Pointer.new(nativecast(Pointer, $!bytes) + i * nativesizeof T));
    }

    method pointer {
        nativecast(Pointer[T], $!bytes);
    }
}

class AI::FANN {
    has fann $!fann is built;

    class Connection is repr('CStruct') {
        has uint32    $.from-neuron;
        has uint32    $.to-neuron;
        has fann_type $.weight;
    }

    class TrainData {
        trusts AI::FANN;

        has fann_train_data $!data
            handles < num-data num-input num-output input output >;

        method !data { $!data }

        multi method BUILD (
            Int :$num-data!,
            Int :$num-input!,
            Int :$num-output!,
            --> TrainData
        ) {
            $!data = fann_create_train( $num-data, $num-input, $num-output );
        }

        multi method BUILD ( IO() :$path --> TrainData ) {
            $!data = fann_read_train_from_file( "$path" );
        }

        method length { fann_length_train_data($!data) }

        method destroy { $.DESTROY }

        submethod DESTROY { fann_destroy_train($!data) if $!data; $!data = Nil }
    }

    multi method BUILD ( IO() :$path! ) {
        $!fann = fann_create_from_file("$path");
    }

    multi method BUILD (
        :@layers!,
        Bool() :$shortcut,
        Num()  :connection-rate($rate),
    ) {
        my $layers = CArray[uint32].new: |@layers;
        my $n      = @layers.elems;

        $!fann = $shortcut     ?? fann_create_shortcut_array(      $n, $layers )
              !! $rate.defined ?? fann_create_sparse_array( $rate, $n, $layers )
              !!                  fann_create_standard_array(      $n, $layers );
    }

    method connection-rate   ( --> Num ) { fann_get_connection_rate($!fann) }
    method num-input         ( --> Int ) { fann_get_num_input($!fann) }
    method num-layers        ( --> Int ) { fann_get_num_layers($!fann) }
    method num-output        ( --> Int ) { fann_get_num_output($!fann) }
    method total-connections ( --> Int ) { fann_get_total_connections($!fann) }
    method total-neurons     ( --> Int ) { fann_get_total_neurons($!fann) }
  # method decimal-point     ( --> Int ) { fann_get_decimal_point($!fann) }
  # method multiplier        ( --> Int ) { fann_get_multiplier($!fann) }

    method network-type      ( --> AI::FANN::NetType ) {
        AI::FANN::NetType.^enum_from_value: fann_get_network_type($!fann);
    }

    method print-connections ( --> Nil ) { fann_print_connections($!fann) }
    method print-parameters  ( --> Nil ) { fann_print_parameters($!fann) }

    method layer-array {
        my $out = CArray[uint32].allocate($.num-layers);
        fann_get_layer_array($!fann, $out);
        $out.list;
    }

    method bias-array {
        my $out = CArray[uint32].allocate($.num-layers);
        fann_get_layer_array($!fann, $out);
        $out.list;
    }

    method connection-array {
        my $out = StructArray[AI::FANN::Connection].new($.total-connections);
        fann_get_connection_array($!fann, $out.pointer);
        [ $out[ ^$out.elems ] ];
    }

    method activation-function (
        :$layer!,
        :$neuron!,
        --> AI::FANN::ActivationFunc
    ) {
        AI::FANN::ActivationFunc.^enum_from_value:
            fann_get_activation_function( $!fann, $layer, $neuron );
    }

    method run ( :@input ) {
        fann_run( $!fann, CArray[float].new( |@input.map(*.Num) ) );
    }

    method save ( IO() :$path --> Bool() ) {
        !fann_save($!fann, "$path")
    }

    method reset-MSE ( --> Nil ) { fann_reset_MSE($!fann) }

    method get-MSE ( --> Num ) { fann_get_MSE($!fann) }

    method set-activation-function (
        $function,
        Bool() :$hidden is copy,
        Bool() :$output is copy,
        Int    :$layer,
        Int    :$neuron,
    ) {
        if $layer.defined {
            return fann_set_activation_function( $!fann, $function, $layer, $neuron )
                if $neuron.defined;

            return fann_set_activation_function_layer( $!fann, $function, $layer );
        }

        $hidden = $output = True unless $hidden || $output;

        fann_set_activation_function_hidden( $!fann, $function ) if $hidden;
        fann_set_activation_function_output( $!fann, $function ) if $output;
    }

    multi method train ( :$input!, :$output! ) {
        fann_train( $!fann, $input, $output )
    }

    multi method train (
        TrainData :$data,
        IO() :$path,
        :$max-epochs!,
        :$epochs-between-reports!,
        Num() :$desired-error!,
    ) {
        return fann_train_on_data(
            $!fann,
            $data!AI::FANN::TrainData::data,
            $max-epochs,
            $epochs-between-reports,
            $desired-error,
        ) if $data;

        fann_train_on_file(
            $!fann,
            "$path",
            $max-epochs,
            $epochs-between-reports,
            $desired-error,
        );
    }

    method test ( :$input!, :$output! ) { fann_test( $!fann, $input, $output ) }

    method destroy { $.DESTROY }

    submethod DESTROY { fann_destroy($!fann) if $!fann; $!fann = Nil }
}
