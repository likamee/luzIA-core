import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from tensorflow.keras import callbacks

from app.utilities import cam_models, filter_img
from app.utilities.generic_funcs import remove_files_dir
from app.utilities.load_data import __gen_df, gen_data, gen_data_kfold


def process_train(cfg, **kwargs):
    """Train the model
    :param cfg: Config class
    :param kwargs:
        gen_img: if True, the images are generated
        pato: pathology to be trained
        filter: filter to be applied to the images
    :return: None
    """

    gen_img = kwargs["gen_img"]
    pato = kwargs["pato"]
    filter = kwargs["filter"]

    if gen_img:
        remove_files_dir(cfg.data_path + "normais")
        remove_files_dir(cfg.data_path + "/" + pato)
        remove_files_dir(cfg.source + "excluidas")
        filter_img.apply_filter(
            filter,
            pato,
            cfg.source,
            cfg.filesn,
            cfg.filesp,
            cfg.proportion,
            cfg.hq,
            cfg.lq,
            cfg.type_img,
        )

    df_pato = __gen_df(pato, cfg.dest + pato)
    df_norm = __gen_df("normais", cfg.dest + "normais")
    df = pd.concat([df_pato, df_norm], ignore_index=True)

    # df = df.groupby('label').apply(lambda x: x.sample(n=17000, random_state=1)).reset_index(drop=True)

    # Extract labels for stratification
    labels = df["label"].values

    # Define the K-fold cross-validator
    n_splits = 2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []

    # Start the K-fold cross-validation loop
    fold_var = 1
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        print(f"Training on fold {fold_var}")

        # Generate data for the current fold
        train_gen, val_gen = gen_data_kfold(cfg, df, train_idx, val_idx)

        # Build and compile a fresh model for each fold

        if cfg.type_train == "n":
            model_name = f"{pato}_{filter}_{str(cfg.n_layers)}_fold{fold_var}"
            model, _ = cam_models.build_vgg16_GAP(
                cfg.n_layers, cfg.type_train, model_name
            )
        else:
            model_name = cfg.model_name
            model, _ = cam_models.build_vgg16_GAP(
                cfg.n_layers, cfg.type_train, model_name
            )
            model_name = f"{pato}_{filter}_{str(cfg.n_layers)}_fold{fold_var}"

        filename = model_name + ".csv"
        csv_log = callbacks.CSVLogger(
            "results/" + cfg.ds + "/" + filename, separator=",", append=False
        )
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=3, verbose=0, mode="min"
        )
        file_path = "models/" + cfg.ds + "/" + model_name + ".keras"
        checkpoint = callbacks.ModelCheckpoint(
            file_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
        )
        callb_list = [csv_log, checkpoint, early_stopping]

        # Compute class weights for the current fold
        class_weights = class_weight.compute_class_weight(
            "balanced", classes=np.unique(train_gen.classes), y=train_gen.classes
        )
        class_weights_dict = dict(enumerate(class_weights))

        # Fit the model
        model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // cfg.batch_size,
            validation_data=val_gen,
            validation_steps=val_gen.samples // cfg.batch_size,
            epochs=cfg.epochs,
            callbacks=[callb_list],
            class_weight=class_weights_dict,
        )

        # Evaluate the model on the validation set and store metrics if needed
        result = model.evaluate(val_gen, steps=val_gen.samples // cfg.batch_size)
        results.append(result)

        # Clean up after each fold to save memory
        del model

        fold_var += 1

    # Calculate average result
    average_result = np.mean(results, axis=0)
    print(f"Average result: {average_result}")
