#!/usr/bin/env raku

use Test;

is-deeply do {
    my @before = MY::.keys;
    my @after  = do { use AI::FANN; MY::.keys };

    @after (-) @before;
}, < AI >.Set, 'Plain use only modifies AI namespace';

is-deeply do {
    my @before = MY::.keys;
    my @after  = do { use AI::FANN::Raw; MY::.keys };

    @after (-) @before;
}, «
    &fann_cascadetrain_on_data
    &fann_cascadetrain_on_file
    &fann_clear_scaling_params
    &fann_copy
    &fann_create_from_file
    &fann_create_shortcut
    &fann_create_shortcut_array
    &fann_create_sparse
    &fann_create_sparse_array
    &fann_create_standard
    &fann_create_standard_array
    &fann_create_train
    &fann_create_train_from_callback
    &fann_descale_input
    &fann_descale_output
    &fann_descale_train
    &fann_destroy
    &fann_destroy_train
    &fann_duplicate_train_data
    &fann_get_MSE
    &fann_get_activation_function
    &fann_get_activation_steepness
    &fann_get_bias_array
    &fann_get_bit_fail
    &fann_get_bit_fail_limit
    &fann_get_cascade_activation_functions
    &fann_get_cascade_activation_functions_count
    &fann_get_cascade_activation_steepnesses
    &fann_get_cascade_activation_steepnesses_count
    &fann_get_cascade_candidate_change_function
    &fann_get_cascade_candidate_limit
    &fann_get_cascade_candidate_stagnation_epochs
    &fann_get_cascade_max_cand_epochs
    &fann_get_cascade_max_out_epochs
    &fann_get_cascade_min_cand_epochs
    &fann_get_cascade_min_out_epochs
    &fann_get_cascade_num_candidate_groups
    &fann_get_cascade_num_candidates
    &fann_get_cascade_output_change_function
    &fann_get_cascade_output_stagnation_epochs
    &fann_get_cascade_weight_multiplier
    &fann_get_connection_array
    &fann_get_connection_rate
    &fann_get_errno
    &fann_get_errstr
    &fann_get_layer_array
    &fann_get_learning_momentum
    &fann_get_learning_rate
    &fann_get_network_type
    &fann_get_num_input
    &fann_get_num_layers
    &fann_get_num_output
    &fann_get_quickprop_decay
    &fann_get_quickprop_mu
    &fann_get_rprop_decrease_factor
    &fann_get_rprop_delta_max
    &fann_get_rprop_delta_min
    &fann_get_rprop_delta_zero
    &fann_get_rprop_increase_factor
    &fann_get_sarprop_step_error_shift
    &fann_get_sarprop_step_error_threshold_factor
    &fann_get_sarprop_temperature
    &fann_get_sarprop_weight_decay_shift
    &fann_get_total_connections
    &fann_get_total_neurons
    &fann_get_train_error_function
    &fann_get_train_stop_function
    &fann_get_training_algorithm
    &fann_get_user_data
    &fann_init_weights
    &fann_length_train_data
    &fann_merge_train_data
    &fann_num_input_train_data
    &fann_num_output_train_data
    &fann_print_connections
    &fann_print_error
    &fann_print_parameters
    &fann_randomize_weights
    &fann_read_train_from_file
    &fann_reset_MSE
    &fann_reset_errno
    &fann_reset_errstr
    &fann_run
    &fann_save
    &fann_save_train
    &fann_save_train_to_fixed
    &fann_scale_input
    &fann_scale_input_train_data
    &fann_scale_output
    &fann_scale_output_train_data
    &fann_scale_train
    &fann_scale_train_data
    &fann_set_activation_function
    &fann_set_activation_function_hidden
    &fann_set_activation_function_layer
    &fann_set_activation_function_output
    &fann_set_activation_steepness
    &fann_set_activation_steepness_hidden
    &fann_set_activation_steepness_layer
    &fann_set_activation_steepness_output
    &fann_set_bit_fail_limit
    &fann_set_callback
    &fann_set_cascade_activation_functions
    &fann_set_cascade_activation_steepnesses
    &fann_set_cascade_candidate_change_function
    &fann_set_cascade_candidate_limit
    &fann_set_cascade_candidate_stagnation_epochs
    &fann_set_cascade_max_cand_epochs
    &fann_set_cascade_max_out_epochs
    &fann_set_cascade_min_cand_epochs
    &fann_set_cascade_min_out_epochs
    &fann_set_cascade_num_candidate_groups
    &fann_set_cascade_output_change_function
    &fann_set_cascade_output_stagnation_epochs
    &fann_set_cascade_weight_multiplier
    &fann_set_error_log
    &fann_set_input_scaling_params
    &fann_set_learning_momentum
    &fann_set_learning_rate
    &fann_set_output_scaling_params
    &fann_set_quickprop_decay
    &fann_set_quickprop_mu
    &fann_set_rprop_decrease_factor
    &fann_set_rprop_delta_max
    &fann_set_rprop_delta_min
    &fann_set_rprop_delta_zero
    &fann_set_rprop_increase_factor
    &fann_set_sarprop_step_error_shift
    &fann_set_sarprop_step_error_threshold_factor
    &fann_set_sarprop_temperature
    &fann_set_sarprop_weight_decay_shift
    &fann_set_scaling_params
    &fann_set_train_error_function
    &fann_set_train_stop_function
    &fann_set_training_algorithm
    &fann_set_user_data
    &fann_set_weight
    &fann_set_weight_array
    &fann_shuffle_train_data
    &fann_subset_train_data
    &fann_test
    &fann_test_data
    &fann_train
    &fann_train_epoch
    &fann_train_on_data
    &fann_train_on_file
    fann
    fann_activationfunc_enum
    fann_connection
    fann_errno_enum
    fann_error
    fann_errorfunc_enum
    fann_nettype_enum
    fann_stopfunc_enum
    fann_train_data
    fann_train_enum
    fann_type
    float
».Set, 'Raw exports';

my %imports = (
    enum => <
        FANN_COS_SYMMETRIC
        FANN_ELLIOT
        FANN_ELLIOT_SYMMETRIC
        FANN_ERRORFUNC_LINEAR
        FANN_ERRORFUNC_TANH
        FANN_E_CANT_ALLOCATE_MEM
        FANN_E_CANT_OPEN_CONFIG_R
        FANN_E_CANT_OPEN_CONFIG_W
        FANN_E_CANT_OPEN_TD_R
        FANN_E_CANT_OPEN_TD_W
        FANN_E_CANT_READ_CONFIG
        FANN_E_CANT_READ_CONNECTIONS
        FANN_E_CANT_READ_NEURON
        FANN_E_CANT_READ_TD
        FANN_E_CANT_TRAIN_ACTIVATION
        FANN_E_CANT_USE_ACTIVATION
        FANN_E_CANT_USE_TRAIN_ALG
        FANN_E_INDEX_OUT_OF_BOUND
        FANN_E_INPUT_NO_MATCH
        FANN_E_NO_ERROR
        FANN_E_OUTPUT_NO_MATCH
        FANN_E_SCALE_NOT_PRESENT
        FANN_E_TRAIN_DATA_MISMATCH
        FANN_E_TRAIN_DATA_SUBSET
        FANN_E_WRONG_CONFIG_VERSION
        FANN_E_WRONG_NUM_CONNECTIONS
        FANN_GAUSSIAN
        FANN_GAUSSIAN_STEPWISE
        FANN_GAUSSIAN_SYMMETRIC
        FANN_LINEAR
        FANN_LINEAR_PIECE
        FANN_LINEAR_PIECE_SYMMETRIC
        FANN_NETTYPE_LAYER
        FANN_NETTYPE_SHORTCUT
        FANN_SIGMOID
        FANN_SIGMOID_STEPWISE
        FANN_SIGMOID_SYMMETRIC
        FANN_SIGMOID_SYMMETRIC_STEPWISE
        FANN_SIN FANN_COS
        FANN_SIN_SYMMETRIC
        FANN_STOPFUNC_BIT
        FANN_STOPFUNC_MSE
        FANN_THRESHOLD
        FANN_THRESHOLD_SYMMETRIC
        FANN_TRAIN_BATCH
        FANN_TRAIN_INCREMENTAL
        FANN_TRAIN_QUICKPROP
        FANN_TRAIN_RPROP
        FANN_TRAIN_SARPROP
    >,
    error => <
        FANN_E_CANT_ALLOCATE_MEM
        FANN_E_CANT_OPEN_CONFIG_R
        FANN_E_CANT_OPEN_CONFIG_W
        FANN_E_CANT_OPEN_TD_R
        FANN_E_CANT_OPEN_TD_W
        FANN_E_CANT_READ_CONFIG
        FANN_E_CANT_READ_CONNECTIONS
        FANN_E_CANT_READ_NEURON
        FANN_E_CANT_READ_TD
        FANN_E_CANT_TRAIN_ACTIVATION
        FANN_E_CANT_USE_ACTIVATION
        FANN_E_CANT_USE_TRAIN_ALG
        FANN_E_INDEX_OUT_OF_BOUND
        FANN_E_INPUT_NO_MATCH
        FANN_E_NO_ERROR
        FANN_E_OUTPUT_NO_MATCH
        FANN_E_SCALE_NOT_PRESENT
        FANN_E_TRAIN_DATA_MISMATCH
        FANN_E_TRAIN_DATA_SUBSET
        FANN_E_WRONG_CONFIG_VERSION
        FANN_E_WRONG_NUM_CONNECTIONS
    >,
);

for %imports.kv -> $tag, $expected {
    is-deeply do { EVAL "use AI::FANN :{ $tag }; MY::.keys.grep(/ 'FANN_' /).Set" },
        $expected.Set, "Importing :{ $tag } imports correct names";
}

done-testing;
