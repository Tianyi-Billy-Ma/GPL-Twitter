local base_env = import '../base_env.jsonnet';


local seed = 3;

local override = {
  platform_type: 'pytorch',
  ignore_pretrained_weights: [],
  experiment_name: 'create_twitter_data',
  seed: seed,
  data_loader: {
    type: 'DataLoaderForGraph',
    dummy_data_loader: 0,
    additional: {},
    dataset_modules: [
      {
        type: 'LoadTwitterData',
        option: 'default',
        config: {
          preprocess: [],
          name: ['twitter'],
          path: 'TwitterData/',
        },
      },
    ],
  },
};

std.mergePatch(base_env, override)
