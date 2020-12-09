import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from tensorflow import keras
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from skimage import io
import os
import keract


def append_ext(fn):
    return fn+".jpg"


def back_image_loading_for_control():
    for i in range(3):
        x1, y1 = next(train_generator)
        y1_int = np.argmax(y1, axis=-1)
    plt.figure(figsize=(20, 20))
    idx = 1
    for i in range(8):
        plt.subplot(4, 4, idx)
        idx += 1
        plt.imshow(x1[i].reshape(60, 60, 3))
        plt.subplot(4, 4, idx)
        plt.imshow(io.imread(os.path.join(train_generator.directory,
                                          train_generator.filenames[(train_generator.batch_index - 1) * 32 + i])))
        idx += 1
    plt.savefig('visual_original_comp.png', bbox_inches='tight')


def show_accuracy():
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=history.epoch, y=history.history["loss"],
        mode='lines',
        name='Training Loss'))

    fig2.add_trace(go.Scatter(
        x=history.epoch, y=history.history["val_loss"],
        mode='lines',
        name='Validation Loss'))

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=history.epoch, y=history.history["accuracy"],
        mode='lines',
        name='Training Accuracy'))

    fig3.add_trace(go.Scatter(
        x=history.epoch, y=history.history["val_accuracy"],
        mode='lines',
        name='Validation Accuracy'))
    fig2.show()
    fig3.show()


def show_test_data_results():
    test_generator.reset()
    pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)
    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    filenames = test_generator.filenames
    # niekde sa mi stratilo 10 zaznamov
    results = pd.DataFrame({"filename": filenames[:11000], "predictions": predictions[:11000]})
    # results.to_csv("results.csv", index=False)

    resultny_result = pd.merge(results, test_df[['id', main_column_name]], left_on='filename', right_on='id')

    print(classification_report(resultny_result[main_column_name], resultny_result.predictions))
    confusion_df = pd.DataFrame(confusion_matrix(resultny_result[main_column_name], resultny_result.predictions))
    labels_cf = test_df[main_column_name].unique()
    fig4 = go.Figure()
    fig4.add_trace(go.Heatmap(
        z=confusion_df,
        x=labels_cf,
        y=labels_cf,
        colorscale='Electric'))
    fig4.show()


def show_own_test_data_results():
    STEP_SIZE_OWN_TEST = own_test_generator.n // own_test_generator.batch_size
    own_test_generator.reset()
    pred = model.predict_generator(own_test_generator, steps=STEP_SIZE_OWN_TEST, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)
    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    filenames = own_test_generator.filenames
    # niekde sa mi stratilo 10 zaznamov
    results = pd.DataFrame({"filename": filenames, "predictions": predictions})

    for i in range(5):
        fig5 = make_subplots(rows=2, cols=1)
        print(i)
        full_path = os.path.join('own_images', results.filename[i])
        image = keras.preprocessing.image.load_img(full_path)
        image_arr = keras.preprocessing.image.img_to_array(image)
        fig5.add_trace(
            go.Image(z=image_arr),
            row=1, col=1
        )
        fig5.add_trace(
            go.Bar(x=list(labels.values()), y=pred[i]),
            row=2, col=1
        )
        fig5.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_df = pd.read_csv('styles.csv', dtype=str, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(train_df.shape)
    print('gender', train_df['gender'].nunique())
    print('masterCategory', train_df['masterCategory'].nunique())
    print('subCategory', train_df['subCategory'].nunique())
    print('articleType', train_df['articleType'].nunique())
    train_df, test_df = train_test_split(train_df, shuffle=False)
    train_df['id'] = train_df['id'].apply(append_ext)
    test_df['id'] = test_df['id'].apply(append_ext)

    print('train_df size', train_df.shape)
    print('test_df size', test_df.shape)
    data_gen = ImageDataGenerator(rescale=1. / 255., validation_split=0.25)
    main_column_name = 'subCategory'
    count_of_categories = train_df[main_column_name].nunique()
    image_size_def = 60

    train_generator = data_gen.flow_from_dataframe(
        dataframe=train_df,
        directory="images",
        x_col="id",
        y_col=main_column_name,
        subset="training",
        batch_size=image_size_def,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(image_size_def, image_size_def))

    train_generator.reset()  # resets the generator to the first batch
    back_image_loading_for_control()

    valid_generator = data_gen.flow_from_dataframe(
        dataframe=train_df,
        directory="images",
        x_col="id",
        y_col=main_column_name,
        subset="validation",
        batch_size=image_size_def,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(image_size_def, image_size_def))

    test_datagen = ImageDataGenerator(rescale=1. / 255.)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory="images",
        x_col="id",
        y_col=None,
        batch_size=image_size_def,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(image_size_def, image_size_def))

    df = pd.DataFrame(data={'id': ['01.jpg', '02.jpg', '03.jpg', '04.jpg', '05.jpg']})
    own_test_generator = test_datagen.flow_from_dataframe(
        dataframe=df,
        directory="own_images",
        x_col="id",
        y_col=None,
        batch_size=5,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(image_size_def, image_size_def))

    checkpoint_filepath = 'model/checkpoint'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        node='max',
        save_best_only=True
    )

    model = Sequential()
    model.add(Conv2D(image_size_def, (3, 3), padding='same',
                     input_shape=(image_size_def, image_size_def, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(image_size_def, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(count_of_categories, activation='softmax'))
    model.compile(optimizers.RMSprop(lr=0.0001, decay=1e-6), loss="categorical_crossentropy", metrics=["accuracy"])

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)


    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=valid_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs=25,
        callbacks=[early_stopping, model_checkpoint_callback]
    )
    show_accuracy()

    # model.load_weights(checkpoint_filepath)
    # scores = model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_TEST)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    show_test_data_results()
    show_own_test_data_results()
