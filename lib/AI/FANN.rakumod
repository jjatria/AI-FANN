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

    multi method new ( IO() :$path! --> AI::FANN ) {
        self.bless: fann => fann_create_from_file("$path");
    }

    multi method new (
        :@layers!,
        Bool() :$shortcut,
        Num()  :connection-rate($rate),
        --> AI::FANN
    ) {
        my $layers = CArray[uint32].new: |@layers;
        my $n      = @layers.elems;

        my $fann = $shortcut     ?? fann_create_shortcut_array(      $n, $layers )
                !! $rate.defined ?? fann_create_sparse_array( $rate, $n, $layers )
                !!                  fann_create_standard_array(      $n, $layers );

        self.bless: :$fann;
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
        my $out = StructArray[fann_connection].new($.total-connections);
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
        :$data,
        IO() :$path,
        :$max-epochs!,
        :$epochs-between-reports!,
        Num() :$desired-error!,
    ) {
        return fann_train_on_data(
            $!fann,
            $data,
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

    method destroy { $.DESTROY }

    submethod DESTROY { fann_destroy($!fann) if $!fann; $!fann = Nil }
}
