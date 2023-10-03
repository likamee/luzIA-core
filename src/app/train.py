from tensorflow.keras import callbacks

from app.utilities import cam_models, filter_img
from app.utilities.generic_funcs import remove_files_dir
from app.utilities.load_data import gen_data


def process_train(cfg, **kwargs):
    """ Train the model
    :param cfg: Config class
    :param kwargs:
        gen_img: if True, the images are generated
        pato: pathology to be trained
        filter: filter to be applied to the images
    :return: None
    """

    gen_img = kwargs['gen_img']
    pato = kwargs['pato']
    filter = kwargs['filter']

    if gen_img:
        remove_files_dir(cfg.data_path+'/normais')
        remove_files_dir(cfg.data_path+'/'+pato)
        remove_files_dir(cfg.source+'/excluidas')
        filter_img.apply_filter(filter, pato, cfg.source, cfg.filesn, cfg.filesp,
                                cfg.proportion, cfg.hq, cfg.lq, cfg.type_img)

    train_gen, val_gen = gen_data(cfg, pato)

    model_name = pato+'_'+filter+'_'+str(cfg.n_layers)
    model, model_name = cam_models.build_vgg16_GAP(cfg.n_layers, cfg.type_train, model_name)  # best 9 layers15
    filename = model_name+'.csv'
    csv_log = callbacks.CSVLogger('results/'+cfg.ds+'/'+filename, separator=',', append=False)
    # early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')
    file_path = 'models/'+cfg.ds+'/'+model_name+'.hdf5'
    checkpoint = callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callb_list = [csv_log, checkpoint]

    model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // cfg.batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // cfg.batch_size,
        epochs=cfg.epochs,
        callbacks=[callb_list])
    # Evaluating the MODEL.
    del model  # deletes the existing MODEL
