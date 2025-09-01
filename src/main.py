import importlib

from config import Config


def main(**kwargs):
    # load the module based on function argument
    module = importlib.import_module('app.'+kwargs['module'])
    # call the function based on function argument
    config = Config()
    getattr(module, kwargs['function'])(config, **kwargs)


main(module='train', function='process_train', gen_img=False, pato='cataract', filter='raw')
