local base_env = import '../base_env.jsonnet';


local drop_edge_rate = 0.5;

local seed = 3;
local train_epoch = 500;
local train_batch_size = 128;
local valid_batch_size = 128;
local test_batch_size = 128;

local dropout = 0.4;
local save_interval = 1;

local override = {

  platform_type: 'pytorch',
  ignore_pretrained_weights: [],
  experiment_name: 'ihgt_twitter_split_118',
  seed: seed,
  model_config: {
    base_model: 'iHGT',
    PretrainModelName: 'model',
    // PretrainModelClass: 'HGMAE',
    // PretrainModelCkptPath: 'Experiments/hgmae_twitter_split_118_run4/train/saved_model/last.ckpt',
    // PretrainModelConfig: {
    //   ModelClass: 'HGMAE',
    //   EncoderModelClass: 'HAN',
    //   EncoderModelConfig: {
    //     drop_edge_rate: drop_edge_rate,
    //     num_layers: 2,
    //     input_dim: 768,
    //     hidden_dim: 256,
    //     num_heads: 4,
    //     activation: 'prelu',
    //     dropout: dropout,
    //     norm: 'batchnorm',
    //   },
    //   DecoderModelClass: 'HAN',
    //   DecoderModelConfig: {
    //     num_layers: 2,
    //     input_dim: 256,
    //     hidden_dim: 256,
    //     output_dim: 768,
    //     num_heads: 1,
    //     activation: 'prelu',
    //     dropout: dropout,
    //     norm: 'batchnorm',
    //   },
    //   MPModelConfig: {
    //     edge_mask_rate: 0.5,
    //     feat_mask_rate: 0.5,
    //     edge_alpha_l: 3,
    //     feature_alpha_l: 2,
    //     hidden_dim: 64,
    //     dropout: 0.2,
    //   },
    //   ClassifierModelClass: 'LogReg',
    //   ClassifierModelConfig: {
    //     input_dim: 256,
    //     num_classes: 4,
    //   },
    //   additional: {
    //     loss_fn: 'sce',
    //     alpha_l: 3,
    //     replace_rate: 0.3,
    //     leave_unchanged: 0.2,
    //   },
    // },
    PretrainModelClass: 'HeCo',
    PretrainModelCkptPath: 'Experiments/heco_twitter_split_118_run1/train/saved_model/last.ckpt',
    PretrainModelConfig: {
      base_model: 'HeCo',
      ModelClass: 'HeCo',
      MappingModelClass: 'Linear',
      MappingModelConfig: {
        in_features: 768,
        out_features: 256,
        bias: true,
      },
      ContrastModelClass: 'Contrast',
      ContrastModelConfig: {
        hidden_dim: 256,
        tau: 0.5,
        lam: 0.2,
      },
      MPModelClass: 'MPEncoder',
      MPModelConfig: {
        num_metapaths: 3,
        hidden_dim: 256,
        attention_dropout: 0.2,
      },
      SCModelClass: 'SCEncoder',
      SCModelConfig: {
        hidden_dim: 256,
        sample_rate: [5, 5],
        num_neighbors: 2,
        attention_dropout: 0.4,
      },
      ClassifierModelClass: 'LogReg',
      ClassifierModelConfig: {
        input_dim: 256,
        num_classes: 4,
      },
      additional: {
        dropout: dropout,
      },
    },
    ModelClass: 'iHGT',
    ModelClassConfig: {
      node_types: ['user', 'tweet', 'keyword'],
      num_metapath_types: 3,
      batch_size: 512,
      class_token_dim: 256,
      num_classes: 2,
      num_node_type_tokens: 5,
      node_type_token_dim: 768,
      dropout: dropout,
      MLP_input_dim: 768,
      MLP_hidden_dim: 256,
      MLP_output_dim: 256,
      MLP_num_layers: 2,
      MLP_norm: 'bn',
      heads: 4,
      tau: 2,
    },

  },
  data_loader: {
    type: 'DataLoaderForGraph',
    dummy_data_loader: 0,
    additional: {},
    dataset_modules: {

      module_list: ['LoadTwitterData', 'LoadBinaryData', 'LoadSplits', 'LoadDataLoader'],
      module_dict:
        {
          LoadTwitterData: {
            type: 'LoadTwitterData',
            option: 'default',
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
          LoadBinaryData: {
            use_column: 'twitter',
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
                  dataset_type: 'iHGT_twitter',
                  split: 'train',

                },
              ],
              valid: [
                {
                  dataset_type: 'iHGT_twitter',
                  split: 'valid',
                },
              ],
              test: [
                {
                  dataset_type: 'iHGT_twitter',
                  split: 'valid',
                },
                {
                  dataset_type: 'iHGT_twitter',
                  split: 'test',
                },
              ],
            },
          },
        },
    },
  },
  train: {
    type: 'iHGTExecutor',
    epochs: train_epoch,
    batch_size: train_batch_size,
    lr: 0.05,
    wd: 0.0,
    scheduler: 'none',
    load_epoch: -1,
    load_model_path: '',
    load_best_model: true,
    save_interval: save_interval,
    additional: {
      // save_top_k_metric: 'valid/iHGT_twitter.valid/f1_macro',
      save_top_k_metric: 'valid/iHGT_twitter.valid/f1_macro',
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
