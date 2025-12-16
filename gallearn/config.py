import os
import configparser

env_str = os.getenv('CONDA_DEFAULT_ENV', 'base')
if env_str == 'base':
    env_str = ''
    env_prefix = ''
else:
    env_str = '_' + env_str
    env_prefix = env_str + '_'
home = os.path.expanduser(os.path.join(
    '~/'
))
config_fname = 'config' + env_str + '.ini'

def get_path():
    config_path = os.path.join(
        home, 
        config_fname
    )
    return config_path

def ensure_user_config():
    config = configparser.ConfigParser()
    config_path = get_path()

    if os.path.isfile(config_path):
        config.read(config_path)
    else:
        print(
            'Before importing gallearn for the first time, you were'
            ' supposed to create a'
            ' config file for your environment and/or add a gallearn_paths'
            ' section to your existing config file. However, no config file'
            ' existed up to now.'
            ' This code has created one for you in your home directory called' 
            ' {0}.'
            ' If you want to customize anything in the config file,'
            ' you can do so safely.'
            .format(config_fname)
        )
    
    if not config.has_section(f'{__package__}_paths'):
        config.add_section(f'{__package__}_paths')
        print(
            f'gallearn_paths section added to {config_fname}'
            '\n\nNOTE: Anything this code adds to gallearn_paths assumes'
            ' you are on UC Irvine\'s Greenplanet cluster in the cosmo'
            ' group. If you are not, your code will not work with these paths,'
            ' and you must properly configure {config_fname}.\n'
        )

    if not config.has_option(f'{__package__}_paths', 'host_2d_shapes'): 
        config.set(
            f'{__package__}_paths',
            'host_2d_shapes',
            '/DFS-L/DATA/cosmo/kleinca/data/'
                'AstroPhot_NewHost_bandr_Rerun_Sersic.csv',
        )
        print(f'host_2d_shapes added to {__package__}_paths')

    if not config.has_option(f'{__package__}_paths', 'sat_2d_shapes'):
        config.set(
            f'{__package__}_paths',
            'sat_2d_shapes',
            '/DFS-L/DATA/cosmo/kleinca/data/'
                'DataWithMockImagesWithBadExtinction/'
                'AstroPhot_Sate_Sersic_AllMeasure.csv'
        )
        print(f'sat_2d_shapes added to {__package__}_paths')

    ensure_key(
        config,
        'project_data_dir', 
        os.path.join(home, env_previx + 'data'),
               
    if not config.has_option(f'{__package__}_paths', 'firebox_data_dir'):
        config.set(
            f'{__package__}_paths',
            'firebox_data_dir',
            "/DFS-L/DATA/cosmo/jgmoren1/FIREbox/FB15N1024/"
        )
        print(f'firebox_data_dir added to {__package__}_paths')

    if not config.has_option(f'{__package__}_paths', 'sat_image_dir'):
        config.set(
            f'{__package__}_paths',
            'sat_image_dir',
            "/DFS-L/DATA/cosmo/kleinca/FIREBox_Images/satellite/band_ugr"
        )
        print(f'sat_image_dir added to {__package__}_paths')

    if not config.has_option(f'{__package__}_paths', 'host_image_dir'):
        config.set(
            f'{__package__}_paths',
            'host_image_dir',
            "/DFS-L/DATA/cosmo/kleinca/FIREBox_Images/host/band_ugr"
        )
        print(f'host_image_dir added to {__package__}_paths')

    if not config.has_option(f'{__package__}_paths', 'vmaps_dir'):
        config.set(
            f'{__package__}_paths',
            'host_image_dir',
            "/DFS-L/DATA/cosmo/pstaudt/gallearn/vmaps_res256_min_cden1.4e+1"
        )
        print(f'vmaps_dir added to {__package__}_paths')

    with open(config_path, 'w') as f:
        config.write(f)
    
    return config_path

def ensure_key(config, key, path, ensure_exists=False):
    if not config.has_option(f'{__package__}_paths', key):
        if ensure_exists and not os.path.isdir(path):
            os.makedirs(path)
            print(f'{path} created.')
        config.set(f'{__package__}_paths', key, path)
        print(f'{key} added to {__package__}_paths')

def load_config():
    '''
    Check to see if something has created an environment variable specifying
    a config path (GitHub will do this when it runs workflows). If not, look
    for the user defined config. If that doesn't exist, create a default in the
    home dir.
    '''
    # If CI set the path to the config_ci.ini, the following will be something
    # other than `False`.
    config_path = os.environ.get('MYPACKAGE_CONFIG')
    if config_path:
        # In that case, go ahead and get the full path the .ini
        config_path = os.path.expanduser(config_path)
    else:
        # Otherwise, check that the user has a config_<environment_name>.ini.
        # Create one for them if they don't.
        config_path = ensure_user_config()

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )
    config.read(config_path)
    return config

config = load_config()
