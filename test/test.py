from sklearn import preprocessing

if __name__ == '__main__':
    labelList = ['no', 'yes', 'ok']

    lb = preprocessing.LabelBinarizer()

    dummY = lb.fit_transform(labelList)
    print(dummY)
