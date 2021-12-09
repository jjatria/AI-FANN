use AI::FANN::Raw;

package AI::FANN {
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
}

class X::AI::FANN is Exception {
    has AI::FANN::Error $.code;
    has Str $.message;
}

my subset MaybeError where * ~~ Nil | X::AI::FANN;

my sub error ( fann_error $e ) {
    LEAVE {
        fann_reset_errstr($e);
        fann_reset_errno($e);
    }

    $e.errno == 0 ?? Nil !! X::AI::FANN.new(
        code    => AI::FANN::Error.^enum_from_value($e.errno),
        message => $e.errstr.chomp.subst( /\.$/, '', :nth(*) ),
    );
}

class AI::FANN {
    use NativeCall;

    # Disable printing error directly to STDERR
    fann_set_error_log( fann_error, Pointer );

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
        has fann_type $.weight is rw;
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
            $!data = fann_read_train_from_file( "$path" )
                or die 'Unable to create train data';
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
                    ) or die 'Unable to create train data';
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
            $!data = @data.reduce: -> $a, $b {
                LEAVE $a!error.throw;

                fann_merge_train_data(
                    $a!AI::FANN::TrainData::data,
                    $b!AI::FANN::TrainData::data,
                );
            }
        }

        multi method BUILD ( fann_train_data :$data! ) {
            $!data = $data;
        }

        method !error ( --> MaybeError ) { $!data.error.&error }

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
            my $data = fann_duplicate_train_data($!data)
                // die 'Unable to create train data';
            self.new: :$data;
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

        method save ( IO() $path --> ::?CLASS:D ) {
            LEAVE self!error.throw;
            fann_save_train($!data, "$path");
            self;
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
        $!fann = fann_create_from_file("$path")
            // die 'Unable to create network';
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

        die 'Unable to create network' unless $!fann;
    }

    multi method BUILD ( fann:D :$fann! ) {
        $!fann = $fann;
    }

    method !error ( --> MaybeError ) { $!fann.error.&error }

    # no error
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

    multi method weights ( --> List() ) { $.connection-array».weight }

    multi method weights (
        Num()  $weight,
        Int() :$from! where * >= 0,
        Int() :$to!   where * >= 0,
        --> ::?CLASS:D
    ) {
        fann_set_weight( $!fann, $from, $to, $weight );
        self;
    }

    multi method weights (
        *@connections where { .all ~~ AI::FANN::Connection },
        --> ::?CLASS:D
    ) {
        for @connections -> $c {
            fann_set_weight( $!fann, $c.from-neuron, $c.to-neuron, $c.weight )
        }
        self;
    }

    method randomise-weights ( |c --> ::?CLASS:D ) { $.randomize-weights: |c } # We love our British users
    method randomize-weights ( Range:D $range  --> ::?CLASS:D ) { # no error
        die 'Cannot use an infinite range to randomize weights' if $range.infinite;
        fann_randomize_weights($!fann, |$range.minmax».Num);
        self;
    }

    method init-weights ( AI::FANN::TrainData:D $data --> ::?CLASS:D ) {
        fann_init_weights( $!fann, $data!AI::FANN::TrainData::data );
        self;
    }

    method layer-array ( --> List ) { # no error
        my $out = CArray[uint32].allocate($.num-layers);
        fann_get_layer_array($!fann, $out);
        $out.list;
    }

    method bias-array ( --> List ) { # no error
        my $out = CArray[uint32].allocate($.num-layers);
        fann_get_layer_array($!fann, $out);
        $out.list;
    }

    method connection-array ( --> List ) { # no error
        my $out = StructArray[AI::FANN::Connection].new($.total-connections);
        fann_get_connection_array($!fann, $out.pointer);
        [ $out[ ^$out.elems ] ];
    }

    method print-connections ( --> ::?CLASS:D ) {
        LEAVE self!error.throw; # FANN_E_CANT_ALLOCATE_MEM ...
        fann_print_connections($!fann);
        self;
    }

    method print-parameters ( --> ::?CLASS:D ) { # no error
        fann_print_parameters($!fann);
        self;
    }

    multi method callback () {
        die 'The callback method can only be used to set the callback, or clear it with :delete';
    }

    multi method callback ( :$delete! where :so --> ::?CLASS:D ) { # no error
        fann_set_callback( $!fann, Code );
        self;
    }

    multi method callback ( &cb --> AI::FANN ) { # no error
        die 'Unsupported callback: it must be able to accept '
            ~ :( AI::FANN, AI::FANN::TrainData, uint32, uint32, num32, uint32 ).raku
            unless &cb.cando: \( AI::FANN, AI::FANN::TrainData, uint32, uint32, num32, uint32 );

        fann_set_callback( $!fann, sub ( $fann, $data, |c ) {
            given cb( AI::FANN.new(:$fann), AI::FANN::TrainData.new(:$data), |c ) {
                return .so ?? 0 !! -1 when Bool;
                return .Int;
            }
        });

        self;
    }

    multi method run ( CArray[fann_type] $input --> CArray[fann_type] ) {
        LEAVE self!error.throw;
        fann_run( $!fann, $input )
    }

    multi method run ( *@input --> List() ) {
        LEAVE self!error.throw;
        .[ ^$.num-output ]
            with fann_run( $!fann, CArray[fann_type].new: |@input».Num )
    }

    method clone ( --> ::?CLASS:D ) {
        LEAVE self!error.throw; # FANN_E_CANT_ALLOCATE_MEM ...
        self.new: fann => fann_copy($!fann);
    }

    multi method scale ( TrainData:D $data --> ::?CLASS:D ) {
        LEAVE self!error.throw;
        fann_scale_train( $!fann, $data!AI::FANN::TrainData::data );
        self;
    }

    multi method descale ( TrainData:D $data --> ::?CLASS:D ) {
        LEAVE self!error.throw;
        fann_descale_train( $!fann, $data!AI::FANN::TrainData::data );
        self;
    }

    multi method scale (
        CArray[fann_type] :$input,
        CArray[fann_type] :$output,
        --> ::?CLASS:D
    ) {
        LEAVE self!error.throw;
        die 'Must specify input or output data to scale' unless $input || $output;

        fann_scale_input(  $!fann, $input  ) if $input;
        fann_scale_output( $!fann, $output ) if $output;

        self;
    }

    multi method descale (
        CArray[fann_type] :$input,
        CArray[fann_type] :$output,
        --> ::?CLASS:D
    ) {
        LEAVE self!error.throw;
        die 'Must specify input or output data to descale' unless $input || $output;

        fann_descale_input(  $!fann, $input  ) if $input;
        fann_descale_output( $!fann, $output ) if $output;

        self;
    }

    multi method scale (
        :@input,
        :@output,
        --> ::?CLASS:D
    ) {
        LEAVE self!error.throw;
        die 'Must specify input or output data to scale' unless @input || @output;

        if @input {
            my $data = CArray[fann_type].new: |@input».Num;
            fann_scale_input( $!fann, $data );
            @input[*] = |$data.list;
        }

        if @output {
            my $data = CArray[fann_type].new: |@output».Num;
            fann_scale_output( $!fann, $data );
            @output[*] = $data.list;
        }

        self;
    }

    multi method descale (
        :@input,
        :@output,
        --> ::?CLASS:D
    ) {
        LEAVE self!error.throw;
        die 'Must specify input or output data to descale' unless @input || @output;

        if @input {
            my $data = CArray[fann_type].new: |@input».Num;
            fann_descale_input( $!fann, $data );
            @input[*] = |$data.list;
        }

        if @output {
            my $data = CArray[fann_type].new: |@output».Num;
            fann_descale_output( $!fann, $data );
            @output[*] = $data.list;
        }

        self;
    }

    multi method scaling (
        TrainData:D $train,
        Range      :$output,
        Range      :$input,
        --> ::?CLASS:D
    ) {
        LEAVE self!error.throw;
        die 'Must specify input or output scaling parameters'
            unless $input || $output;

        die 'Cannot use an infinite range to set input scaling parameters'
            if $input && $input.infinite;

        die 'Cannot use an infinite range to set output scaling parameters'
            if $output && $output.infinite;

        my $ret = 0;

        $ret = fann_set_input_scaling_params(  $!fann, $train!AI::FANN::TrainData::data, |$input.minmax».Num ) if $input && $ret == 0;
        $ret = fann_set_output_scaling_params( $!fann, $train!AI::FANN::TrainData::data, |$output.minmax».Num ) if $output && $ret == 0;

        self;
    }

    multi method scaling ( :$delete! where :so --> ::?CLASS:D ) {
        LEAVE self!error.throw;
        fann_clear_scaling_params( $!fann );
        self;
    }

    method save ( IO() $path --> ::?CLASS:D ) {
        LEAVE self!error.throw;
        fann_save($!fann, "$path");
        self;
    }

    method reset-error ( --> ::?CLASS:D ) { fann_reset_MSE($!fann); self } # no error

    method mean-square-error ( --> Num ) { fann_get_MSE($!fann) } # no error

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

    multi method activation-function ( # no error
        Int :$layer!,
        Int :$neuron!,
        --> AI::FANN::ActivationFunc
    ) {
        AI::FANN::ActivationFunc.^enum_from_value:
            fann_get_activation_function( $!fann, $layer, $neuron );
    }

    multi method activation-function ( # no error
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

    multi method activation-function ( # no error
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

    multi method activation-function ( $other, |c ) { # no error
        my $value = AI::FANN::ActivationFunc.^enum_from_value($other)
            // die "Invalid activation function: must be a value in AI::FANN::ActivationFunc";
        nextwith( $value, |c );
    }

    proto method activation-steepness ( :$layer, :$neuron, | ) {
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

    multi method activation-steepness ( # no error
        Int :$layer!,
        Int :$neuron!,
        --> Num
    ) {
        fann_get_activation_steepness( $!fann, $layer, $neuron );
    }

    multi method activation-steepness ( # no error
        Num()  $value!,
        Int:D :$layer!,
        Int   :$neuron,
        --> ::?CLASS:D
    ) {
        $neuron.defined
            ?? fann_set_activation_steepness( $!fann, $value, $layer, $neuron )
            !! fann_set_activation_steepness_layer( $!fann, $value, $layer );

        self;
    }

    multi method activation-steepness ( # no error
        Num()   $value!,
        Bool() :$hidden is copy,
        Bool() :$output is copy,
        --> ::?CLASS:D
    ) {
        $hidden = $output = True unless $hidden || $output;

        fann_set_activation_steepness_hidden( $!fann, $value ) if $hidden;
        fann_set_activation_steepness_output( $!fann, $value ) if $output;

        self;
    }

    multi method training-algorithm ( --> AI::FANN::Train ) { # no error
        AI::FANN::Train.^enum_from_value: fann_get_training_algorithm($!fann);
    }

    multi method training-algorithm ( # no error
        AI::FANN::Train $algorithm,
        --> ::?CLASS:D
    ) {
        fann_set_training_algorithm( $!fann, $algorithm );
        self;
    }

    multi method training-algorithm ( $other, |c --> ::?CLASS:D ) { # no error
        my $value = AI::FANN::Train.^enum_from_value($other)
            // die "Invalid training algorithm: must be a value in AI::FANN::Train";
        $.training-algorithm: $value, |c;
    }

    multi method train-error-function ( --> AI::FANN::ErrorFunc ) { # no error
        AI::FANN::ErrorFunc.^enum_from_value: fann_get_train_error_function($!fann);
    }

    multi method train-error-function ( # no error
        AI::FANN::ErrorFunc $function,
        --> ::?CLASS:D
    ) {
        fann_set_train_error_function( $!fann, $function );
        self;
    }

    multi method train-error-function ( $other, |c --> ::?CLASS:D ) { # no error
        my $value = AI::FANN::ErrorFunc.^enum_from_value($other)
            // die "Invalid error function: must be a value in AI::FANN::ErrorFunc";
        $.train-error-function: $value, |c;
    }

    multi method train-stop-function ( --> AI::FANN::StopFunc ) { # no error
        AI::FANN::StopFunc.^enum_from_value: fann_get_train_stop_function($!fann);
    }

    multi method train-stop-function ( # no error
        AI::FANN::StopFunc $function,
        --> ::?CLASS:D
    ) {
        fann_set_train_stop_function( $!fann, $function );
        self;
    }

    multi method train-stop-function ( $other, |c --> ::?CLASS:D ) { # no error
        my $value = AI::FANN::StopFunc.^enum_from_value($other)
            // die "Invalid stop function: must be a value in AI::FANN::StopFunc";
        $.train-stop-function: $value, |c;
    }

    # no error
    multi method bit-fail-limit ( --> Num ) { fann_get_bit_fail_limit($!fann) }
    multi method bit-fail-limit ( Num() $value --> ::?CLASS:D ) {
        fann_set_bit_fail_limit( $!fann, $value );
        self;
    }

    # no error
    multi method learning-rate ( --> Num ) { fann_get_learning_rate($!fann) }
    multi method learning-rate ( Num() $value --> ::?CLASS:D ) {
        fann_set_learning_rate( $!fann, $value );
        self;
    }

    # no error
    multi method learning-momentum ( --> Num ) { fann_get_learning_momentum($!fann) }
    multi method learning-momentum ( Num() $value --> ::?CLASS:D ) {
        fann_set_learning_momentum( $!fann, $value );
        self;
    }

    # no error
    multi method quickprop-decay ( --> Num ) { fann_get_quickprop_decay($!fann) }
    multi method quickprop-decay ( Num() $value --> ::?CLASS:D ) {
        die "The decay value must be less than or equal to 0; got instead $value" if $value > 0;
        fann_set_quickprop_decay( $!fann, $value );
        self;
    }

    # no error
    multi method quickprop-mu ( --> Num ) { fann_get_quickprop_mu($!fann) }
    multi method quickprop-mu ( Num() $value --> ::?CLASS:D ) {
        fann_set_quickprop_mu( $!fann, $value );
        self;
    }

    # no error
    multi method quickprop-mu ( --> Num ) { fann_get_quickprop_mu($!fann) }
    multi method quickprop-mu ( Num() $value --> ::?CLASS:D ) {
        fann_set_quickprop_mu( $!fann, $value );
        self;
    }

    # no error
    multi method rprop-increase ( --> Num ) { fann_get_rprop_increase_factor($!fann) }
    multi method rprop-increase ( Num() $value --> ::?CLASS:D ) {
        die "The RPROP increase value must be greater than 1; got instead $value" if $value <= 1;
        fann_set_rprop_increase_factor( $!fann, $value );
        self;
    }

    # no error
    multi method rprop-decrease ( --> Num ) { fann_get_rprop_decrease_factor($!fann) }
    multi method rprop-decrease ( Num() $value --> ::?CLASS:D ) {
        die "The RPROP increase value must be smaller than 1; got instead $value" if $value >= 1;
        fann_set_rprop_decrease_factor( $!fann, $value );
        self;
    }

    # no error
    multi method rprop-delta-range ( --> Range ) {
        fann_get_rprop_delta_min($!fann) .. fann_get_rprop_delta_max($!fann)
    }

    multi method rprop-delta-range ( Range:D $value --> ::?CLASS:D ) {
        die 'Cannot use an infinite range to set scale' if $value.infinite;
        fann_set_rprop_delta_min( $!fann, $value.min );
        self;
    }

    # no error
    multi method rprop-delta-zero ( --> Range ) { fann_get_rprop_delta_zero($!fann) }
    multi method rprop-delta-zero ( Num() $value --> ::?CLASS:D ) {
        die "The delta zero must be greater than 0; got instead $value" if $value <= 0;
        fann_set_rprop_delta_zero( $!fann, $value );
        self;
    }

    # no error
    multi method sarprop-weight-decay-shift ( --> Range ) { fann_get_sarprop_weight_decay_shift($!fann) }
    multi method sarprop-weight-decay-shift ( Num() $value --> ::?CLASS:D ) {
        fann_set_sarprop_weight_decay_shift( $!fann, $value );
        self;
    }

    # no error
    multi method sarprop-step-error-threshold ( --> Range ) { fann_get_sarprop_step_error_threshold_factor($!fann) }
    multi method sarprop-step-error-threshold ( Num() $value --> ::?CLASS:D ) {
        fann_set_sarprop_step_error_threshold_factor( $!fann, $value );
        self;
    }

    # no error
    multi method sarprop-step-error-shift ( --> Range ) { fann_get_sarprop_step_error_shift($!fann) }
    multi method sarprop-step-error-shift ( Num() $value --> ::?CLASS:D ) {
        fann_set_sarprop_step_error_shift( $!fann, $value );
        self;
    }

    # no error
    multi method sarprop-temperature ( --> Range ) { fann_get_sarprop_temperature($!fann) }
    multi method sarprop-temperature ( Num() $value --> ::?CLASS:D ) {
        fann_set_sarprop_temperature( $!fann, $value );
        self;
    }

    multi method train ( @input, @output --> ::?CLASS:D ) {
        LEAVE self!error.throw; # FANN_E_CANT_ALLOCATE_MEM FANN_E_CANT_TRAIN_ACTIVATION ...
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
        LEAVE self!error.throw; # FANN_E_CANT_ALLOCATE_MEM FANN_E_CANT_TRAIN_ACTIVATION ...
        fann_train( $!fann, $input, $output );
        self;
    }

    multi method train ( TrainData:D $data --> ::?CLASS:D ) {
        LEAVE self!error.throw; # FANN_E_CANT_ALLOCATE_MEM FANN_E_CANT_TRAIN_ACTIVATION ...
        fann_train_epoch( $!fann, $data!AI::FANN::TrainData::data );
        self;
    }

    multi method train ( IO() $path --> ::?CLASS:D ) {
        my $data = AI::FANN::TrainData.new: :$path;
        LEAVE $data.?destroy;
        $.train: $data;
    }

    multi method train (
        TrainData:D $data,
        Int() :$max-epochs             where *.defined,
        Int() :$epochs-between-reports where *.defined,
        Num() :$desired-error          where *.defined,
        --> ::?CLASS:D
    ) {
        LEAVE self!error.throw; # FANN_E_CANT_ALLOCATE_MEM FANN_E_CANT_TRAIN_ACTIVATION ...
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
        Int() :$max-epochs             where *.defined,
        Int() :$epochs-between-reports where *.defined,
        Num() :$desired-error          where *.defined,
        --> ::?CLASS:D
    ) {
        LEAVE self!error.throw; # FANN_E_CANT_ALLOCATE_MEM FANN_E_CANT_TRAIN_ACTIVATION ...
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

    multi method train (
        Int() :$max-epochs,
        Int() :$epochs-between-reports,
        Num() :$desired-error,
        |c
    ) {
        die 'You must specify none or all three of :max-epochs, :epochs-between-reports, and :desired-error'
            if ( $max-epochs, $epochs-between-reports, $desired-error ).grep(*.defined) != 3;
        nextsame;
    }

    multi method test ( @input, @output --> List() ) {
        LEAVE self!error.throw; # FANN_E_CANT_USE_ACTIVATION ...
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
        LEAVE self!error.throw; # FANN_E_CANT_USE_ACTIVATION ...
        fann_test( $!fann, $input, $output );
    }

    multi method test ( TrainData:D $data --> Num ) {
        LEAVE self!error.throw; # FANN_E_CANT_USE_ACTIVATION ...
        fann_test_data( $!fann, $data!AI::FANN::TrainData::data );
    }

    multi method test ( IO() $path --> Num ) {
        my $data = AI::FANN::TrainData.new: :$path;
        LEAVE $data.?destroy;
        $.test: $data;
    }

    # Cascade methods

    multi method cascade-train (
        TrainData:D $data,
        Int() :$max-neurons!,
        Int() :$neurons-between-reports!,
        Num() :$desired-error!,
        --> ::?CLASS:D
    ) {
        LEAVE self!error.throw; # FANN_E_CANT_ALLOCATE_MEM FANN_E_CANT_TRAIN_ACTIVATION ...
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
        LEAVE self!error.throw; # FANN_E_CANT_ALLOCATE_MEM FANN_E_CANT_TRAIN_ACTIVATION ...
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

    # no error
    method cascade-num-candidates ( --> Int ) { fann_get_cascade_num_candidates($!fann) }

    # no error
    multi method cascade-candidate-limit ( --> Num ) { fann_get_cascade_candidate_limit($!fann) }
    multi method cascade-candidate-limit ( Num() $value --> ::?CLASS:D ) {
        fann_set_cascade_candidate_limit( $!fann, $value );
        self;
    }

    # no error
    multi method cascade-weight-multiplier ( --> Num ) { fann_get_cascade_weight_multiplier($!fann) }
    multi method cascade-weight-multiplier ( Num() $value --> ::?CLASS:D ) {
        fann_set_cascade_weight_multiplier( $!fann, $value );
        self;
    }

    # no error
    multi method cascade-output-change-fraction ( --> Num ) { fann_get_cascade_output_change_fraction($!fann) }
    multi method cascade-output-change-fraction ( Num() $value --> ::?CLASS:D ) {
        fann_set_cascade_output_change_fraction( $!fann, $value );
        self;
    }

    # no error
    multi method cascade-candidate-change-fraction ( --> Num ) { fann_get_cascade_candidate_change_fraction($!fann) }
    multi method cascade-candidate-change-fraction ( Num() $value --> ::?CLASS:D ) {
        fann_set_cascade_candidate_change_fraction( $!fann, $value );
        self;
    }

    # no error
    multi method cascade-num-candidate-groups ( --> Int ) { fann_get_cascade_num_candidate_groups($!fann) }
    multi method cascade-num-candidate-groups ( Int:D $value --> ::?CLASS:D ) {
        fann_set_cascade_num_candidate_groups( $!fann, $value );
        self;
    }

    # no error
    multi method cascade-candidate-stagnation-epochs ( --> Int ) { fann_get_cascade_candidate_stagnation_epochs($!fann) }
    multi method cascade-candidate-stagnation-epochs ( Int:D $value --> ::?CLASS:D ) {
        fann_set_cascade_candidate_stagnation_epochs( $!fann, $value );
        self;
    }

    # no error
    multi method cascade-output-stagnation-epochs ( --> Int ) { fann_get_cascade_output_stagnation_epochs($!fann) }
    multi method cascade-output-stagnation-epochs ( Int:D $value --> ::?CLASS:D ) {
        fann_set_cascade_output_stagnation_epochs( $!fann, $value );
        self;
    }

    # no error
    multi method cascade-activation-steepnesses-count ( --> Int ) { fann_get_cascade_activation_steepnesses_count($!fann) }

    multi method cascade-candidate-epochs ( --> Range ) {
        fann_get_cascade_max_cand_epochs($!fann) .. fann_get_cascade_min_cand_epochs($!fann);
    }

    multi method cascade-candidate-epochs (
        Range $value,
        --> ::?CLASS:D
    ) {
        die 'Cannot use an infinite range to set cascade candidate epochs' if $value.infinite;
        fann_set_cascade_max_cand_epochs($!fann, $value.min.Int);
        fann_set_cascade_min_cand_epochs($!fann, $value.max.Int);
        self;
    }

    multi method cascade-candidate-epochs (
        Int :$min,
        Int :$max,
        --> ::?CLASS:D
    ) {
        fann_set_cascade_min_cand_epochs($!fann, $min) if $min.defined;
        fann_set_cascade_max_cand_epochs($!fann, $max) if $max.defined;
        self;
    }

    multi method cascade-output-epochs ( --> Range ) {
        fann_get_cascade_max_out_epochs($!fann) .. fann_get_cascade_min_out_epochs($!fann);
    }

    multi method cascade-output-epochs (
        Range $value,
        --> ::?CLASS:D
    ) {
        die 'Cannot use an infinite range to set cascade output epochs' if $value.infinite;
        fann_set_cascade_max_out_epochs($!fann, $value.min.Int);
        fann_set_cascade_min_out_epochs($!fann, $value.max.Int);
        self;
    }

    multi method cascade-output-epochs (
        Int :$min,
        Int :$max,
        --> ::?CLASS:D
    ) {
        fann_set_cascade_min_out_epochs($!fann, $min) if $min.defined;
        fann_set_cascade_max_out_epochs($!fann, $max) if $max.defined;
        self;
    }

    multi method cascade-activation-steepnesses ( --> List() ) {
        .[ ^$.cascade-activation-steepnesses-count ]
            with fann_get_cascade_activation_steepnesses($!fann)
    }

    multi method cascade-activation-steepnesses (
        CArray[fann_type] $steepnesses,
        --> ::?CLASS:D
    ) {
        LEAVE self!error.throw; # FANN_E_CANT_ALLOCATE_MEM
        fann_set_cascade_activation_steepnesses(
            $!fann, $steepnesses, $steepnesses.elems );
        self;
    }

    multi method cascade-activation-steepnesses (
        *@steepnesses,
        --> ::?CLASS:D
    ) {
        LEAVE self!error.throw; # FANN_E_CANT_ALLOCATE_MEM
        fann_set_cascade_activation_steepnesses(
            $!fann,
            CArray[fann_type].new(|@steepnesses».Num),
            @steepnesses.elems
        );
        self;
    }

    multi method cascade-activation-functions ( --> List() ) { # no error
        .[ ^$.cascade-activation-functions-count ]
            with fann_get_cascade_activation_functions($!fann)
    }

    multi method cascade-activation-functions (
        CArray[fann_activationfunc_enum] $functions,
        --> ::?CLASS:D
    ) {
        LEAVE self!error.throw; # FANN_E_CANT_ALLOCATE_MEM
        fann_set_cascade_activation_functions(
            $!fann, $functions, $functions.elems );
        self;
    }

    multi method cascade-activation-functions (
        *@functions,
        --> ::?CLASS:D
    ) {
        LEAVE self!error.throw; # FANN_E_CANT_ALLOCATE_MEM
        fann_set_cascade_activation_functions(
            $!fann,
            CArray[fann_activationfunc_enum].new(|@functions».Int),
            @functions.elems
        );
        self;
    }

    method destroy ( --> Nil ) { $.DESTROY }

    submethod DESTROY { fann_destroy($!fann) if $!fann; $!fann = Nil }
}
