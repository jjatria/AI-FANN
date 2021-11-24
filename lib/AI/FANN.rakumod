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
        ) {
            $!data = fann_create_train( $num-data, $num-input, $num-output );
        }

        multi method BUILD ( IO() :$path ) {
            $!data = fann_read_train_from_file( "$path" );
        }

        method length ( --> Int ) { fann_length_train_data($!data) }

        method scale ( Range:D $range  --> Nil ) {
            die 'Cannot use an infinite range to set scale' if $range.infinite;
            fann_scale_train_data( $!data, |$range.minmax».Num )
        }

        method destroy ( --> Nil ) { $.DESTROY }

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
        die 'The :shortcut and :connection-rate parameters are not compatible'
            if $rate.defined && $shortcut;

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
    method bit-fail          ( --> Int ) { fann_get_bit_fail($!fann) }

    method network-type      ( --> AI::FANN::NetType ) {
        AI::FANN::NetType.^enum_from_value: fann_get_network_type($!fann);
    }

    method layer-array ( --> List ) {
        my $out = CArray[uint32].allocate($.num-layers);
        fann_get_layer_array($!fann, $out);
        $out.list;
    }

    method bias-array ( --> List ) {
        my $out = CArray[uint32].allocate($.num-layers);
        fann_get_layer_array($!fann, $out);
        $out.list;
    }

    method connection-array ( --> List ) {
        my $out = StructArray[AI::FANN::Connection].new($.total-connections);
        fann_get_connection_array($!fann, $out.pointer);
        [ $out[ ^$out.elems ] ];
    }

    method print-connections ( --> Nil ) { fann_print_connections($!fann) }
    method print-parameters  ( --> Nil ) { fann_print_parameters($!fann) }

    method randomise-weights (|c) { $.randomize-weights: |c } # We love our British users
    method randomize-weights ( Range:D $range  --> Nil ) {
        die 'Cannot use an infinite range to randomize weights' if $range.infinite;
        fann_randomize_weights($!fann, |$range.minmax».Num)
    }

    multi method run ( CArray[fann_type] :$input --> CArray[fann_type] ) {
        fann_run( $!fann, $input )
    }

    multi method run ( :@input --> List ) {
        fann_run( $!fann, CArray[fann_type].new( |@input».Num ) ).list
    }

    method save ( IO() :$path --> Bool() ) {
        !fann_save($!fann, "$path")
    }

    method reset-error ( --> Nil ) { fann_reset_MSE($!fann) }

    method mean-square-error ( --> Num ) { fann_get_MSE($!fann) }

    proto method activation-function ( :$layer, :$neuron, | ) {
        die "Invalid layer index: $layer. Cannot access the activation function of the input layer"
            if $layer.defined && $layer <= 0;
        {*}
    }

    multi method activation-function (
        Int :$layer!,
        Int :$neuron!,
        --> AI::FANN::ActivationFunc
    ) {
        AI::FANN::ActivationFunc.^enum_from_value:
            fann_get_activation_function( $!fann, $layer, $neuron );
    }

    multi method activation-function (
        AI::FANN::ActivationFunc $function!,
        Int:D                   :$layer!,
        Int                     :$neuron,
        --> AI::FANN::ActivationFunc
    ) {
        return fann_set_activation_function( $!fann, $function, $layer, $neuron )
            if $neuron.defined;

        fann_set_activation_function_layer( $!fann, $function, $layer );

        $function;
    }

    multi method activation-function (
        AI::FANN::ActivationFunc $function!,
        Bool()                  :$hidden is copy,
        Bool()                  :$output is copy,
        --> AI::FANN::ActivationFunc
    ) {
        $hidden = $output = True unless $hidden || $output;

        fann_set_activation_function_hidden( $!fann, $function ) if $hidden;
        fann_set_activation_function_output( $!fann, $function ) if $output;

        $function;
    }

    multi method training-algorithm ( --> AI::FANN::Train ) {
        AI::FANN::Train.^enum_from_value: fann_get_training_algorithm($!fann);
    }

    multi method training-algorithm (
        AI::FANN::Train $algorithm,
        --> AI::FANN::Train
    ) {
        fann_set_training_algorithm( $!fann, $algorithm );
        $algorithm;
    }

    multi method train-error-function ( --> AI::FANN::ErrorFunc ) {
        AI::FANN::ErrorFunc.^enum_from_value: fann_get_train_error_function($!fann);
    }

    multi method train-error-function (
        AI::FANN::ErrorFunc $function,
        --> AI::FANN::ErrorFunc
    ) {
        fann_set_train_error_function( $!fann, $function );
        $function;
    }

    multi method train-stop-function ( --> AI::FANN::StopFunc ) {
        AI::FANN::StopFunc.^enum_from_value: fann_get_train_stop_function($!fann);
    }

    multi method train-stop-function (
        AI::FANN::StopFunc $function,
        --> AI::FANN::StopFunc
    ) {
        fann_set_train_stop_function( $!fann, $function );
        $function;
    }

    multi method bit-fail-limit ( --> Num ) {
        fann_get_bit_fail_limit($!fann);
    }

    multi method bit-fail-limit ( Num() $limit --> Num ) {
        fann_set_bit_fail_limit( $!fann, $limit );
        $limit;
    }

    multi method cascade-num-candidate-groups ( --> Int ) {
        fann_get_cascade_num_candidate_groups($!fann);
    }

    multi method cascade-num-candidate-groups (
        Int:D $groups,
        --> Int
    ) {
        fann_set_cascade_num_candidate_groups( $!fann, $groups );
        $groups;
    }

    multi method cascade-activation-steepnesses ( --> List ) {
        fann_get_cascade_activation_steepnesses($!fann).list;
    }

    multi method cascade-activation-steepnesses (
        CArray[fann_type] $steepnesses,
        --> Nil
    ) {
        fann_set_cascade_activation_steepnesses( $!fann, $steepnesses, $steepnesses.elems )
    }

    multi method cascade-activation-steepnesses (
        *@steepnesses,
        --> Nil
    ) {
        fann_set_cascade_activation_steepnesses(
            $!fann,
            CArray[fann_type].new(|@steepnesses».Num),
            @steepnesses.elems
        );
    }

    multi method cascade-activation-functions ( --> List ) {
        fann_get_cascade_activation_functions($!fann).list;
    }

    multi method cascade-activation-functions (
        CArray[fann_activationfunc] $functions,
        --> Nil
    ) {
        fann_set_cascade_activation_functions( $!fann, $functions, $functions.elems )
    }

    multi method cascade-activation-functions (
        *@functions,
        --> Nil
    ) {
        fann_set_cascade_activation_functions(
            $!fann,
            CArray[fann_activationfunc].new(|@functions».Int),
            @functions.elems
        );
    }

    multi method train ( :@input!, :@output! --> Nil ) {
        fann_train( $!fann,
            CArray[fann_type].new(|@input».Num),
            CArray[fann_type].new(|@output».Num),
        );
    }

    multi method train (
        CArray[fann_type] :$input!,
        CArray[fann_type] :$output!,
        --> Nil
    ) {
        fann_train( $!fann, $input, $output )
    }

    multi method train (
        TrainData :$data,
                  :$max-epochs,
                  :$epochs-between-reports,
        Num()     :$desired-error,
        --> Nil
    ) {
        fann_train_on_data(
            $!fann,
            $data!AI::FANN::TrainData::data,
            $max-epochs,
            $epochs-between-reports,
            $desired-error,
        );
    }

    multi method train (
        IO()  :$path,
              :$max-epochs,
              :$epochs-between-reports,
        Num() :$desired-error,
        --> Nil
    ) {
        fann_train_on_file(
            $!fann,
            "$path",
            $max-epochs,
            $epochs-between-reports,
            $desired-error,
        );
    }

    multi method cascade-train (
        TrainData :$data,
                  :$max-neurons,
                  :$neurons-between-reports,
        Num()     :$desired-error,
        --> Nil
    ) {
        fann_cascadetrain_on_data(
            $!fann,
            $data!AI::FANN::TrainData::data,
            $max-neurons,
            $neurons-between-reports,
            $desired-error,
        );
    }

    multi method cascade-train (
        IO()  :$path,
              :$max-neurons,
              :$neurons-between-reports,
        Num() :$desired-error,
        --> Nil
    ) {
        fann_cascadetrain_on_file(
            $!fann,
            "$path",
            $max-neurons,
            $neurons-between-reports,
            $desired-error,
        );
    }

    multi method test ( :@input!, :@output! --> List ) {
        fann_test( $!fann,
            CArray[fann_type].new(|@input».Num),
            CArray[fann_type].new(|@output».Num),
        ).list;
    }

    multi method test (
        CArray[fann_type] :$input!,
        CArray[fann_type] :$output!,
        --> CArray[fann_type]
    ) {
        fann_test( $!fann, $input, $output )
    }

    multi method test ( TrainData :$data --> Num ) {
        fann_test_data( $!fann, $data!AI::FANN::TrainData::data )
    }

    multi method test ( IO() :$path --> Num ) {
        my $data = AI::FANN::TrainData.new: :$path;
        $.test: :$data;
        $data.destroy;
    }

    method destroy ( --> Nil ) { $.DESTROY }

    submethod DESTROY { fann_destroy($!fann) if $!fann; $!fann = Nil }
}
