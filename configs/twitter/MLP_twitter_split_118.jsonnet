local base_env = import '../base_env.jsonnet';

local seed = 3;
local train_epoch = 100;
local train_batch_size = 128;
local valid_batch_size = 128;
local test_batch_size = 128;

local dropout = 0.5;
local save_interval = 1;

local override = {

  platform_type: 'pytorch',
  ignore_pretrained_weights: [],
  experiment_name: 'mlp_twitter_split_118',
  seed: seed,
  model_config: {
    base_model: 'MLP',
    ModelClass: 'BaseModel',
    EncoderModelClass: 'MLP',
    EncoderModelConfig: {
      num_layers: 2,
      input_dim: 768,
      hidden_dim: 256,
      output_dim: 256,
      dropout: dropout,
    },
    DecoderModelClass: 'MLP',
    DecoderModelConfig: {
      num_layers: 2,
      input_dim: 256,
      hidden_dim: 256,
      output_dim: 4,
      dropout: dropout,
    },
  },
  data_loader: {
    type: 'DataLoaderForGraph',
    dummy_data_loader: 0,
    additional: {},
    dataset_modules: {

      module_list: ['LoadTwitterData', 'LoadSplits', 'LoadDataLoader'],
      module_dict:
        {
          LoadTwitterData: {
            type: 'LoadTwitterData',
            option: 'default',
            config: {
              preprocess: [''],
              name: 'twitter',
              path: 'TwitterData/',
            },
          },
          LoadSplits: {
            type: 'LoadSplits',
            option: 'default',
            path: 'TwitterData/processed/',
            use_column: 'twitter',
            split_ratio: {
              train: 0.1,
              valid: 0.1,
              test: 0.8,
            },
          },
          LoadDataLoader: {
            type: 'LoadDataLoader',
            option: 'default',
            use_column: 'twitter',
            config: {
              train: [
                {
                  dataset_type: 'MLP_twitter',
                  split: 'train',

                },
              ],
              valid: [
                {
                  dataset_type: 'MLP_twitter',
                  split: 'valid',
                },
              ],
              test: [
                {
                  dataset_type: 'MLP_twitter',
                  split: 'valid',
                },
                {
                  dataset_type: 'MLP_twitter',
                  split: 'test',
                },
              ],
            },
          },
        },
    },
  },
  train: {
    type: 'MLPExecutor',
    epochs: train_epoch,
    batch_size: train_batch_size,
    lr: 0.005,
    wd: 0.0,
    scheduler: 'none',
    load_epoch: -1,
    load_model_path: '',
    load_best_model: 1,
    save_interval: save_interval,
    additional: {
      save_top_k: 1,
      save_top_k_metric: 'valid/MLP_twitter.valid/f1_weighted',
      save_top_k_mode: 'max',
      target_node_type: 'user',
      early_stop_patience: 5,
    },
  },
  valid: {
    batch_size: valid_batch_size,
  },
  test: {
    batch_size: test_batch_size,
  },
  metrics: [{ name: 'compute_classification' }],
};

std.mergePatch(base_env, override)
