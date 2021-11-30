unit class AI::FANN;

use NativeCall;
use AI::FANN::Raw;

our enum NetType is export(:enum) «
    FANN_NETTYPE_LAYER
    FANN_NETTYPE_SHORTCUT
»;

our enum ActivationFunc is export(:enum) «
    FANN_LINEAR
    FANN_THRESHOLD
    FANN_THRESHOLD_SYMMETRIC
    FANN_SIGMOID
    FANN_SIGMOID_STEPWISE
    FANN_SIGMOID_SYMMETRIC
    FANN_SIGMOID_SYMMETRIC_STEPWISE
    FANN_GAUSSIAN
    FANN_GAUSSIAN_SYMMETRIC
    FANN_GAUSSIAN_STEPWISE
    FANN_ELLIOT
    FANN_ELLIOT_SYMMETRIC
    FANN_LINEAR_PIECE
    FANN_LINEAR_PIECE_SYMMETRIC
    FANN_SIN_SYMMETRIC
    FANN_COS_SYMMETRIC
    FANN_SIN FANN_COS
»;

our enum Train is export(:enum) «
    FANN_TRAIN_INCREMENTAL
    FANN_TRAIN_BATCH
    FANN_TRAIN_RPROP
    FANN_TRAIN_QUICKPROP
    FANN_TRAIN_SARPROP
»;

our enum ErrorFunc is export(:enum) «
    FANN_ERRORFUNC_LINEAR
    FANN_ERRORFUNC_TANH
»;

our enum StopFunc is export(:enum) «
    FANN_STOPFUNC_MSE
    FANN_STOPFUNC_BIT
»;

our enum Error is export(:enum, :error) «
    FANN_E_NO_ERROR
    FANN_E_CANT_OPEN_CONFIG_R
    FANN_E_CANT_OPEN_CONFIG_W
    FANN_E_WRONG_CONFIG_VERSION
    FANN_E_CANT_READ_CONFIG
    FANN_E_CANT_READ_NEURON
    FANN_E_CANT_READ_CONNECTIONS
    FANN_E_WRONG_NUM_CONNECTIONS
    FANN_E_CANT_OPEN_TD_W
    FANN_E_CANT_OPEN_TD_R
    FANN_E_CANT_READ_TD
    FANN_E_CANT_ALLOCATE_MEM
    FANN_E_CANT_TRAIN_ACTIVATION
    FANN_E_CANT_USE_ACTIVATION
    FANN_E_TRAIN_DATA_MISMATCH
    FANN_E_CANT_USE_TRAIN_ALG
    FANN_E_TRAIN_DATA_SUBSET
    FANN_E_INDEX_OUT_OF_BOUND
    FANN_E_SCALE_NOT_PRESENT
    FANN_E_INPUT_NO_MATCH
    FANN_E_OUTPUT_NO_MATCH
»;

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

has fann $!fann;
has $!layers; # Used for error reporting

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
            :&callback
    ) {
        $!data = &callback
            ?? fann_create_train_from_callback( $num-data, $num-input, $num-output, &callback )
            !! fann_create_train( $num-data, $num-input, $num-output );
    }

    multi method BUILD ( IO() :$path! ) {
        die "Cannot read from file: '$path'" unless $path.r;
        $!data = fann_read_train_from_file( "$path" );
    }

    multi method BUILD ( :@pairs! ) {
        my ( $inputs, $outputs );

        for @pairs.kv -> $i, $_ {
            die 'Values in :pairs must be Pair objects' unless $_ ~~ Pair;

            my @have = .key.List;
            my @want = .value.List;

            die 'Data must have at least one input'  unless @have.elems;
            die 'Data must have at least one output' unless @want.elems;

            if $i == 0 {
                $inputs  = @have.elems;
                $outputs = @want.elems;

                $!data = fann_create_train(
                    @pairs.elems,
                    $inputs,
                    $outputs,
                );
            }
            else {
                die 'Number of inputs must be consistent'  if $inputs  != @have.elems;
                die 'Number of outputs must be consistent' if $outputs != @want.elems;
            }

            $!data.input[ $i][ ^$inputs  ] = |@have».Num;
            $!data.output[$i][ ^$outputs ] = |@want».Num;
        }
    }

    multi method BUILD ( :@data! where { .all ~~ TrainData:D } ) {
        $!data = @data.reduce: {
            fann_merge_train_data(
                $^a!AI::FANN::TrainData::data,
                $^b!AI::FANN::TrainData::data,
            );
        }
    }

    multi method BUILD ( fann_train_data :$data! ) {
        $!data = $data;
    }

    method subset ( Int $pos, Int $length --> TrainData ) {
        X::OutOfRange.new(
           what  => 'Subset position',
           got   => $pos,
           range => ^$.num-data,
        ).throw if $pos !~~ ^$.num-data;

        X::OutOfRange.new(
           what  => 'Subset length',
           got   => $length,
           range => 1..( $.num-data - $pos ),
        ).throw if $length !~~ 1..( $.num-data - $pos );

        self.new: data => fann_subset_train_data( $!data, $pos, $length );
    }

    method clone ( --> TrainData ) {
        self.new: data => fann_duplicate_train_data($!data);
    }

    method scale (
        Range:D $range,
        Bool() :$input,
        Bool() :$output,
    ) {
        die 'Cannot use an infinite range to set scale' if $range.infinite;

        if $input || $output {
            fann_scale_input_train_data(  $!data, |$range.minmax».Num ) if $input;
            fann_scale_output_train_data( $!data, |$range.minmax».Num ) if $output;
        }
        else {
            fann_scale_train_data( $!data, |$range.minmax».Num );
        }

        self;
    }

    method save ( IO() $path --> Bool() ) {
        die "Cannot write to file: '$path'" unless $path.w;
        !fann_save_train("$!data");
    }

    method shuffle ( --> ::?CLASS:D ) {
        fann_shuffle_train_data($!data);
        self;
    }

    method destroy ( --> Nil ) { $.DESTROY }

    submethod DESTROY { fann_destroy_train($!data) if $!data; $!data = Nil }
}

multi method BUILD ( IO() :$path! ) {
    die "Cannot read from file: '$path'" unless $path.r;
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
    $!layers   = @layers;

    $!fann = $shortcut     ?? fann_create_shortcut_array(      $n, $layers )
          !! $rate.defined ?? fann_create_sparse_array( $rate, $n, $layers )
          !!                  fann_create_standard_array(      $n, $layers );
}

multi method BUILD ( fann :$fann! ) {
    $!fann = $fann;
}

method connection-rate   ( --> Num ) { fann_get_connection_rate($!fann) }
method num-input         ( --> Int ) { fann_get_num_input($!fann) }
method num-layers        ( --> Int ) { fann_get_num_layers($!fann) }
method num-output        ( --> Int ) { fann_get_num_output($!fann) }
method total-connections ( --> Int ) { fann_get_total_connections($!fann) }
method total-neurons     ( --> Int ) { fann_get_total_neurons($!fann) }
method bit-fail          ( --> Int ) { fann_get_bit_fail($!fann) }

method network-type ( --> AI::FANN::NetType ) {
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

method print-connections ( --> ::?CLASS:D ) {
    fann_print_connections($!fann);
    self;
}

method print-parameters ( --> ::?CLASS:D ) {
    fann_print_parameters($!fann);
    self;
}

method randomise-weights ( |c --> AI::FANN ) { $.randomize-weights: |c } # We love our British users
method randomize-weights ( Range:D $range  --> AI::FANN ) {
    die 'Cannot use an infinite range to randomize weights' if $range.infinite;
    fann_randomize_weights($!fann, |$range.minmax».Num);
    self;
}

multi method run ( CArray[fann_type] $input --> CArray[fann_type] ) {
    fann_run( $!fann, $input )
}

multi method run ( *@input --> List() ) {
    .[ ^$.num-output ]
        with fann_run( $!fann, CArray[fann_type].new: |@input».Num )
}

method clone ( --> AI::FANN ) {
    self.new: fann => fann_copy($!fann);
}

method save ( IO() $path --> Bool() ) {
    die "Cannot write to file: '$path'" unless $path.w;
    !fann_save($!fann, "$path")
}

method reset-error ( --> ::?CLASS:D ) { fann_reset_MSE($!fann); self }

method mean-square-error ( --> Num ) { fann_get_MSE($!fann) }

proto method activation-function ( :$layer, :$neuron, | ) {
    $!layers //= $.layer-array;

    X::OutOfRange.new(
       what  => 'Layer index',
       got   => $layer,
       range => 1..^$!layers.elems,
    ).throw if $layer.defined && $layer !~~ 1..^$!layers.elems;

    if $neuron.defined {
        die "Cannot set :neuron without setting :layer" unless $layer.defined;

        X::OutOfRange.new(
           what  => 'Neuron index',
           got   => $neuron,
           range => 0..$!layers[$layer],
        ).throw unless $neuron ~~ 0..$!layers[$layer];
    }

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
    --> ::?CLASS:D
) {
    $neuron.defined
        ?? fann_set_activation_function( $!fann, $function, $layer, $neuron )
        !! fann_set_activation_function_layer( $!fann, $function, $layer );

    self;
}

multi method activation-function (
    AI::FANN::ActivationFunc $function!,
    Bool()                  :$hidden is copy,
    Bool()                  :$output is copy,
    --> ::?CLASS:D
) {
    $hidden = $output = True unless $hidden || $output;

    fann_set_activation_function_hidden( $!fann, $function ) if $hidden;
    fann_set_activation_function_output( $!fann, $function ) if $output;

    self;
}

multi method activation-function ( $other, |c ) {
    my $value = AI::FANN::ActivationFunc.^enum_from_value($other)
        // die "Invalid activation function: must be a value in AI::FANN::ActivationFunc";
    nextwith( $value, |c );
}

multi method training-algorithm ( --> AI::FANN::Train ) {
    AI::FANN::Train.^enum_from_value: fann_get_training_algorithm($!fann);
}

multi method training-algorithm (
    AI::FANN::Train $algorithm,
    --> ::?CLASS:D
) {
    fann_set_training_algorithm( $!fann, $algorithm );
    self;
}

multi method training-algorithm ( $other, |c --> AI::FANN ) {
    my $value = AI::FANN::Train.^enum_from_value($other)
        // die "Invalid training algorithm: must be a value in AI::FANN::Train";
    $.training-algorithm: $value, |c;
}

multi method train-error-function ( --> AI::FANN::ErrorFunc ) {
    AI::FANN::ErrorFunc.^enum_from_value: fann_get_train_error_function($!fann);
}

multi method train-error-function (
    AI::FANN::ErrorFunc $function,
    --> ::?CLASS:D
) {
    fann_set_train_error_function( $!fann, $function );
    self;
}

multi method train-error-function ( $other, |c --> AI::FANN ) {
    my $value = AI::FANN::ErrorFunc.^enum_from_value($other)
        // die "Invalid error function: must be a value in AI::FANN::ErrorFunc";
    $.train-error-function: $value, |c;
}

multi method train-stop-function ( --> AI::FANN::StopFunc ) {
    AI::FANN::StopFunc.^enum_from_value: fann_get_train_stop_function($!fann);
}

multi method train-stop-function (
    AI::FANN::StopFunc $function,
    --> ::?CLASS:D
) {
    fann_set_train_stop_function( $!fann, $function );
    self;
}

multi method train-stop-function ( $other, |c --> AI::FANN ) {
    my $value = AI::FANN::StopFunc.^enum_from_value($other)
        // die "Invalid stop function: must be a value in AI::FANN::StopFunc";
    $.train-stop-function: $value, |c;
}

multi method bit-fail-limit ( --> Num ) {
    fann_get_bit_fail_limit($!fann);
}

multi method bit-fail-limit (
    Num() $limit,
    --> ::?CLASS:D
) {
    fann_set_bit_fail_limit( $!fann, $limit );
    self;
}

multi method learning-rate ( --> Num ) {
    fann_get_learning_rate($!fann);
}

multi method learning-rate (
    Num() $rate,
    --> ::?CLASS:D
) {
    fann_set_learning_rate( $!fann, $rate );
    self;
}

multi method learning-momentum ( --> Num ) {
    fann_get_learning_momentum($!fann);
}

multi method learning-momentum (
    Num() $momentum,
    --> ::?CLASS:D
) {
    fann_set_learning_momentum( $!fann, $momentum );
    self;
}

multi method cascade-num-candidate-groups ( --> Int ) {
    fann_get_cascade_num_candidate_groups($!fann);
}

multi method cascade-num-candidate-groups (
    Int:D $groups,
    --> ::?CLASS:D
) {
    fann_set_cascade_num_candidate_groups( $!fann, $groups );
    self;
}

multi method cascade-activation-steepnesses-count ( --> Int ) {
    fann_get_cascade_activation_steepnesses_count($!fann);
}

multi method cascade-activation-steepnesses ( --> List() ) {
    .[ ^$.cascade-activation-steepnesses-count ]
        with fann_get_cascade_activation_steepnesses($!fann)
}

multi method cascade-activation-steepnesses (
    CArray[fann_type] $steepnesses,
    --> ::?CLASS:D
) {
    fann_set_cascade_activation_steepnesses(
        $!fann, $steepnesses, $steepnesses.elems );
    self;
}

multi method cascade-activation-steepnesses (
    *@steepnesses,
    --> ::?CLASS:D
) {
    fann_set_cascade_activation_steepnesses(
        $!fann,
        CArray[fann_type].new(|@steepnesses».Num),
        @steepnesses.elems
    );
    self;
}

multi method cascade-activation-functions ( --> List() ) {
    .[ ^$.cascade-activation-functions-count ]
        with fann_get_cascade_activation_functions($!fann)
}

multi method cascade-activation-functions (
    CArray[fann_activationfunc_enum] $functions,
    --> ::?CLASS:D
) {
    fann_set_cascade_activation_functions(
        $!fann, $functions, $functions.elems );
    self;
}

multi method cascade-activation-functions (
    *@functions,
    --> ::?CLASS:D
) {
    fann_set_cascade_activation_functions(
        $!fann,
        CArray[fann_activationfunc_enum].new(|@functions».Int),
        @functions.elems
    );
    self;
}

multi method train ( @input, @output --> ::?CLASS:D ) {
    fann_train( $!fann,
        CArray[fann_type].new(|@input».Num),
        CArray[fann_type].new(|@output».Num),
    );
    self;
}

multi method train (
    CArray[fann_type] $input,
    CArray[fann_type] $output,
    --> ::?CLASS:D
) {
    fann_train( $!fann, $input, $output );
    self;
}

multi method train (
    TrainData:D $data,
    Int() :$max-epochs!,
    Int() :$epochs-between-reports!,
    Num() :$desired-error!,
    --> ::?CLASS:D
) {
    fann_train_on_data(
        $!fann,
        $data!AI::FANN::TrainData::data,
        $max-epochs,
        $epochs-between-reports,
        $desired-error,
    );
    self;
}

multi method train (
    IO()   $path,
    Int() :$max-epochs!,
    Int() :$epochs-between-reports!,
    Num() :$desired-error!,
    --> ::?CLASS:D
) {
    die "Cannot read from file: '$path'" unless $path.r;
    fann_train_on_file(
        $!fann,
        "$path",
        $max-epochs,
        $epochs-between-reports,
        $desired-error,
    );
    self;
}

multi method cascade-train (
    TrainData:D $data,
    Int() :$max-neurons!,
    Int() :$neurons-between-reports!,
    Num() :$desired-error!,
    --> ::?CLASS:D
) {
    fann_cascadetrain_on_data(
        $!fann,
        $data!AI::FANN::TrainData::data,
        $max-neurons,
        $neurons-between-reports,
        $desired-error,
    );
    self;
}

multi method cascade-train (
    IO()   $path,
    Int() :$max-neurons!,
    Int() :$neurons-between-reports!,
    Num() :$desired-error!,
    --> ::?CLASS:D
) {
    die "Cannot read from file: '$path'" unless $path.r;
    fann_cascadetrain_on_file(
        $!fann,
        "$path",
        $max-neurons,
        $neurons-between-reports,
        $desired-error,
    );
    self;
}

multi method test ( @input, @output --> List() ) {
    fann_test( $!fann,
        CArray[fann_type].new(|@input».Num),
        CArray[fann_type].new(|@output».Num),
    ).[ ^$.num-output ]
}

multi method test (
    CArray[fann_type] $input,
    CArray[fann_type] $output,
    --> CArray[fann_type]
) {
    fann_test( $!fann, $input, $output );
}

multi method test ( TrainData:D $data --> Num ) {
    fann_test_data( $!fann, $data!AI::FANN::TrainData::data );
}

multi method test ( IO() $path --> Num ) {
    my $data = AI::FANN::TrainData.new: :$path;
    LEAVE $data.?destroy;
    $.test: $data;
}

method destroy ( --> Nil ) { $.DESTROY }

submethod DESTROY { fann_destroy($!fann) if $!fann; $!fann = Nil }
