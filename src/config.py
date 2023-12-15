import os

from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(os.getcwd(), '.env'))


class Config(object):
    action = 'train'
    secre_key = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    ds = 'mixed'  # 'unifesp_classificadas'
    n_layers = 9
    source = 'data/'+ds+'/'
    dest = 'data/'+ds+'/algo/'
    filesn = os.listdir(source+'normais')
    filesp = os.listdir(source+'alteradas/rd')
    proportion = 2.82   # 2.82
    # imgn = len(filesn)
    path = os.getcwd()
    # Define data path
    data_path = './' + dest
    epochs = 30
    batch_size = 32
    target_size = 299
    thresholds = 200
    # eyepacs 2.42
    # DEVICES NAMES
    hq = ['20sus', '021sus', '60sus', '70sus', '80sus', '30_', '60ses', '70ses', '80ses']
    # '30_', '60ses', '70ses', '80ses'
    lq = ['50_']

    type_train = 'n'  # 'ft' -> fine tunning, or 'n' -> normal or 'tl' -> transfer learning
    type_img = 'h'  # 'h' -> high reso, 'l' -> low_reso
