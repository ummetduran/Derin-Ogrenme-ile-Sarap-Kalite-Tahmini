import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Activation
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel() - 1] = 1  # -1 !!!
    return labels_one_hot

def run(batchSize, epoch):
    train_df=pd.read_csv("winequality_red.csv")

    train_X=train_df.drop(columns=['quality'])
    print(train_X.head())
    train_Y=train_df[['quality']]

    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.20, shuffle = True, random_state =100)





    train_labels_flat = y_train.values.ravel()
    train_labels_count = np.unique(train_labels_flat).shape[0]
    y_train = dense_to_one_hot(train_labels_flat-1, train_labels_count)
    y_train = y_train.astype(np.uint8)
    train_labels_flat = y_test.values.ravel()
    train_labels_count = np.unique(train_labels_flat).shape[0]
    y_test = dense_to_one_hot(train_labels_flat-1, train_labels_count)
    y_test = y_test.astype(np.uint8)


    model=Sequential()

    columns_count=train_X.shape[1]
    model.add(Dense(16, batch_input_shape=(None,columns_count)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(9))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(6))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    model.summary()

    #modelin derlenmesi

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])


    #modelin eğitilmesi

    history =model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    batch_size=batchSize,
                    shuffle=True,
                    verbose=1,
                    epochs=epoch)





    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    input("Yeni verinin sonucunu görmek için Enter'a basın.\n")

    result1=model.predict(X_test.iloc[316:317])
    quality_points=['3','4','5','6','7','8']
    ind=0.1*0.3*np.arange(len(quality_points))
    width=0.01
    color_list=['blue','pink','red','limegreen','purple','green']
    for i in range(len(quality_points)):
        plt.bar(ind[i],result1[0][i], width, color=color_list[i])
    plt.title("Sınıflandırma sonuçları ", fontsize=20)
    plt.ylabel("Sınınflandırma skoru ", fontsize=16)
    plt.xticks(ind, quality_points, rotation=0, fontsize=14)
    plt.show()
    print("Tahmini sonuç: ", quality_points[np.argmax(result1)])
    print("Gerçek sonuç: ", quality_points[np.argmax(y_test[316:317])])

    input("KNN Hata Matirisi için Enter'a basın.\n")



    # KNN (K nearest neighborhood, en yakın k komşu) algoritması incelemesi
    print("KNN (K nearest neighborhood, en yakın k komşu) algoritması sınıflandırmasına göre, ")
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_tahmin = knn.predict(X_test)
    knn_cm = confusion_matrix(y_test.argmax(axis=1), knn_tahmin.argmax(axis=1))
    print()
    print("Hata (Confusion) Matrisi:")
    print(knn_cm)
    print()
    print("Sınıflandırması Raporlaması : ")
    print(classification_report(y_test, knn_tahmin))
    plt.matshow(knn_cm)
    plt.title('KNN Hata (Confusion) Matrisi')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    input("Yeni verinin sonucunu görmek için Enter'a basın.\n")

    result2=model.predict(X_test.iloc[74:75])
    for i in range(len(quality_points)):
        plt.bar(ind[i],result2[0][i], width, color=color_list[i])
    plt.title("Sınıflandırma sonuçları ", fontsize=20)
    plt.ylabel("Sınınflandırma skoru ", fontsize=16)
    plt.xticks(ind, quality_points, rotation=0, fontsize=14)
    plt.show()
    print("Tahmini sonuç: ", quality_points[np.argmax(result2)])
    print("Gerçek sonuç: ", quality_points[np.argmax(y_test[74:75])])



run(128,1024)
input("Epochs ve batch_size değerlerini değiştirerek restart yapmak için Enter'a basın.\n")
run(64,512)