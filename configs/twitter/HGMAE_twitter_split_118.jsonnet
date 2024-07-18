local base_env = import '../base_env.jsonnet';


local drop_edge_rate = 0.5;

local seed = 3;
local train_epoch = 500;
local train_batch_size = 128;
local valid_batch_size = 128;
local test_batch_size = 128;

local dropout = 0.8;
local save_interval = 1;

local override = {

  platform_type: 'pytorch',
  ignore_pretrained_weights: [],
  experiment_name: 'hgmae_twitter_split_118',
  seed: seed,
  model_config: {
    base_model: 'HGMAE',
    ModelClass: 'HGMAE',
    EncoderModelClass: 'HAN',
    EncoderModelConfig: {
      drop_edge_rate: drop_edge_rate,
      num_layers: 2,
      input_dim: 768,
      hidden_dim: 256,
      num_heads: 4,
      activation: 'prelu',
      dropout: dropout,
      norm: 'batchnorm',
    },
    DecoderModelClass: 'HAN',
    DecoderModelConfig: {
      num_layers: 2,
      input_dim: 256,
      hidden_dim: 256,
      output_dim: 768,
      num_heads: 1,
      activation: 'prelu',
      dropout: dropout,
      norm: 'batchnorm',
    },
    MPModelConfig: {
      edge_mask_rate: 0.5,
      feat_mask_rate: 0.5,
      edge_alpha_l: 3,
      feature_alpha_l: 2,
      hidden_dim: 64,
      dropout: 0.2,
    },
    ClassifierModelClass: 'LogReg',
    ClassifierModelConfig: {
      input_dim: 256,
      num_classes: 4,
    },
    additional: {
      loss_fn: 'sce',
      alpha_l: 3,
      replace_rate: 0.3,
      leave_unchanged: 0.2,
    },
    loss_weights: {
      tar_loss_weight: 0.3,
      pfp_loss_weight: 0.3,
      mer_loss_weight: 0.4,
    },

  },
  data_loader: {
    type: 'DataLoaderForGraph',
    dummy_data_loader: 0,
    additional: {},
    dataset_modules: {

      module_list: ['LoadTwitterData', 'LoadBinaryData', 'LoadPositionEmb', 'LoadSplits', 'LoadDataLoader'],
      module_dict:
        {
          LoadTwitterData: {
            type: 'LoadTwitterData',
            option: 'default',
            config: {
              preprocess: ['build_metapath_from_config'],
              name: 'twitter',
              path: 'TwitterData/',
              save_or_load_name: 'twitter',
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
          LoadBinaryData: {
            use_column: 'twitter',
          },
          LoadPositionEmb: {
            type: 'LoadPositionEmb',
            option: 'default',
            use_column: 'twitter',
            path: 'TwitterData/processed/',
            config: {
              node_type: 'user',
              file_name: 'position_emb.pt',
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
                  dataset_type: 'HGMAE_twitter',
                  split: 'train',

                },
              ],
              valid: [
                {
                  dataset_type: 'HGMAE_twitter',
                  split: 'valid',
                },
              ],
              test: [
                {
                  dataset_type: 'HGMAE_twitter',
                  split: 'valid',
                },
                {
                  dataset_type: 'HGMAE_twitter',
                  split: 'test',
                },
              ],
            },
          },
        },
    },
  },
  train: {
    type: 'HGMAEExecutor',
    epochs: train_epoch,
    batch_size: train_batch_size,
    lr: 0.005,
    wd: 0.0,
    scheduler: 'none',
    load_epoch: -1,
    load_model_path: '',
    load_best_model: 0,
    save_interval: save_interval,
    additional: {
      // save_top_k_metric: 'valid/HGMAE_twitter.valid/f1_macro',
      save_top_k_metric: 'train/total_loss',
      save_top_k_mode: 'max',
      target_node_type: 'user',
      early_stop_patience: 50,
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
