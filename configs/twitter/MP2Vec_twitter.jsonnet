local base_env = import '../base_env.jsonnet';

local seed = 3;

local save_interval = 1;

local override = {
  platform_type: 'pytorch',
  ignore_pretrained_weights: [],
  experiment_name: 'create_twitter_data',
  seed: seed,
  model_config: {
    base_model: 'MetaPath2Vec',
    ModelClass: 'MP2Vec',
    ModelConfig: {
      embedding_dim: 64,
      walk_length: 10,
      walks_per_node: 3,
      context_size: 5,
      num_negative_samples: 3,
      sparse: true,
      metapath: [
        ['user', 'keyword', 'user'],
        ['user', 'tweet', 'user'],
        ['user', 'user'],
        ['user', 'tweet', 'keyword', 'tweet', 'user'],
      ],
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
          name: ['twitter'],
          path: 'TwitterData/',
        },

      },
    ],
  },
  train: {
    type: 'MP2VecExecutor',
    epochs: 20,
    batch_size: 128,
    lr: 0.005,
    scheduler: 'none',
    load_epoch: -1,
    load_model_path: '',
    load_best_model: 0,
    save_interval: save_interval,
    additional: {},
  },
  validation: {},
  test: {},
  metrics: [],
};

std.mergePatch(base_env, override)
