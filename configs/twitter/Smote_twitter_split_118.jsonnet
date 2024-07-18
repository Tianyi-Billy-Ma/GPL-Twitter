local base_env = import '../base_env.jsonnet';


local seed = 3;
local train_epoch = 500;
local train_batch_size = 128;
local valid_batch_size = 128;
local test_batch_size = 128;

local dropout = 0.3;
local save_interval = 1;

local override = {

  platform_type: 'pytorch',
  ignore_pretrained_weights: [],
  experiment_name: 'smote_twitter_split_118',
  seed: seed,
  model_config: {
    base_model: 'ReWeight',
    SmoteClass: 'SMOTE',
    SmoteClassConfig: {
      hidden_dim: 768,
      k: 3,
    },
    ModelClass: 'MLP',
    ModelClassConfig: {
      input_dim: 768,
      hidden_dim: 256,
      output_dim: 4,
      num_layers: 3,
      dropout: 0.5,
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
            option: 'reload',
            config: {
              preprocess: ['build_baseline', 'build_metapath'],
              name: 'twitter',
              path: 'TwitterData/',
              save_or_load_name: 'twitter_baseline',
              metapaths: [
                [
                  ['user', 'post-->', 'tweet'],
                  ['tweet', '<--engage', 'user'],
                ],
                [
                  ['user', 'post-->', 'tweet'],
                  ['tweet', 'include-->', 'keyword'],
                  ['keyword', '<--tag', 'tweet'],
                  ['tweet', '<--post', 'user'],
                ],
                [
                  ['user', 'profile-->', 'keyword'],
                  ['keyword', '<--profile', 'user'],
                ],
              ],
            },
          },
          LoadSplits: {
            type: 'LoadSplits',
            option: 'reload',
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
                  dataset_type: 'Smote_twitter',
                  split: 'train',

                },
              ],
              valid: [
                {
                  dataset_type: 'Smote_twitter',
                  split: 'valid',
                },
              ],
              test: [
                {
                  dataset_type: 'Smote_twitter',
                  split: 'valid',
                },
                {
                  dataset_type: 'Smote_twitter',
                  split: 'test',
                },
              ],
            },
          },
        },
    },
  },
  train: {
    type: 'SmoteExecutor',
    epochs: train_epoch,
    batch_size: train_batch_size,
    lr: 0.005,
    wd: 0.0,
    scheduler: 'none',
    load_epoch: -1,
    load_model_path: '',
    load_best_model: true,
    save_interval: save_interval,
    additional: {
      save_top_k_metric: 'valid/Smote_twitter.valid/f1_macro',
      save_top_k_mode: 'max',
      target_node_type: 'user',
      early_stop_patience: 10,
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
