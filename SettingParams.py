import math

import numpy as np

mock_params = {'num_aps': 30,
               'num_users': 12,
               'num_antennas': 4,
               'ap_dist': 'Uniform',
               'users_dist': 'Uniform',
               'coverage_area_len': 500,
               'channel_model': 'Correlated Rayleigh',
               'block_len': 200,
               'pilot_len': 5,
               'pilot_alloc_alg': 'random',
               'pilot_power_control_alg': 'max',
               'uplink_power_control_alg': 'max',
               'downlink_power_alloc_alg': 'fractional',
               'user_max_power': 100,
               'ap_max_power': 200,
               'uplink_noise_power': -94,
               'downlink_noise_power': -94,
               'clustering_alg': 'canonical'
               }

journal_params = {'num_aps': 30,
                  'num_users': 12,
                  'num_antennas': 4,
                  'ap_dist': 'Uniform',
                  'users_dist': 'Uniform',
                  'coverage_area_len': 500,  # ~0.25 km ** 2 area
                  'channel_model': 'Correlated Rayleigh',
                  'block_len': 200,
                  'pilot_len': 4,
                  'pilot_alloc_alg': 'random',
                  'pilot_power_control_alg': 'max',
                  'uplink_power_control_alg': 'max',
                  'downlink_power_alloc_alg': 'fractional',
                  'user_max_power': 100,
                  'ap_max_power': 200,
                  'uplink_noise_power': -94,
                  'downlink_noise_power': -94,
                  'clustering_alg': 'canonical'
                  }

journal_params_massive = {'num_aps': 30,
                          'num_users': 24,
                          'num_antennas': 4,
                          'ap_dist': 'Uniform',
                          'users_dist': 'Uniform',
                          'coverage_area_len': 1000/math.sqrt(12),  # ~ 0.125 km ** 2 area
                          'channel_model': 'Correlated Rayleigh',
                          'block_len': 200,
                          'pilot_len': 4,
                          'pilot_alloc_alg': 'random',
                          'pilot_power_control_alg': 'max',
                          'uplink_power_control_alg': 'max',
                          'downlink_power_alloc_alg': 'fractional',
                          'user_max_power': 100,
                          'ap_max_power': 200,
                          'uplink_noise_power': -94,
                          'downlink_noise_power': -94,
                          'clustering_alg': 'canonical'
                          }