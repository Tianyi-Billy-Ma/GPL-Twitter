local base_env = import '../base_env.jsonnet';

local seed = 3;
local train_epoch = 100;
local train_batch_size = 128;
local valid_batch_size = 128;
local test_batch_size = 128;

local dropout = 0.2;
local save_interval = 1;

local override = {

  platform_type: 'pytorch',
  ignore_pretrained_weights: [],
  experiment_name: 'metapath2vec_twitter',
  seed: seed,
  model_config: {
    base_model: 'MetaPath2Vec',
    ModelClass: 'MP2Vec',
    EncoderModelClass: 'MetaPath2Vec',
    EncoderModelConfig: {
      embedding_dim: 64,
      walk_length: 10,
      walks_per_node: 3,
      context_size: 5,
      num_negative_samples: 3,
      sparse: true,
      metapath: [
        // uu
        ['user', 'to', 'user'],
        // uku
        ['user', 'to', 'keyword'],
        ['keyword', 'to', 'user'],
        // utu
        ['user', 'to', 'tweet'],
        ['tweet', 'to', 'user'],
        // uktku
        ['user', 'to', 'tweet'],
        ['tweet', 'to', 'keyword'],
        ['keyword', 'to', 'tweet'],
        ['tweet', 'to', 'user'],
      ],
    },
    DecoderModelClass: 'MLP',
    DecoderModelConfig: {
      num_layers: 2,
      input_dim: 64,
      hidden_dim: 256,
      output_dim: 4,
      dropout: dropout,
    },
  },
  data_loader: {
    type: 'DataLoaderForGraph',
    dummy_data_loader: 0,
    additional: {},
    dataset_modules: [
      {
        type: 'LoadTwitterData',
        option: 'default',
        config: {
          preprocess: ['build_metapath_for_MetaPath2Vec'],
          name: 'metapath2vec_twitter',
          path: 'TwitterData/',
        },

      },
      {
        type: 'LoadDataLoader',
        option: 'default',
        path: 'TwitterData/processed/split.pt',
        use_column: 'metapath2vec_twitter',
        split_ratio: {
          train: 0.1,
          valid: 0.1,
          test: 0.8,
        },
        config: {
          train: [
            {
              dataset_type: 'MetaPath2Vec_twitter',
              split: 'train',

            },
          ],
          valid: [
            {
              dataset_type: 'MetaPath2Vec_twitter',
              split: 'valid',
            },
          ],
          test: [
            {
              dataset_type: 'MetaPath2Vec_twitter',
              split: 'valid',
            },
            {
              dataset_type: 'MetaPath2Vec_twitter',
              split: 'test',
            },
          ],
        },
      },
    ],
  },
  train: {
    type: 'MP2VecExecutor',
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
      save_top_k_metric: 'valid/MetaPath2Vec_twitter.valid/accuracy',
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
