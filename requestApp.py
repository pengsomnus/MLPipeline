from flask import Flask, request, redirect, session, url_for
import json
from featureselection import FeatureSelection
from preprocess import DataPreprocessor
from dataloader import DataLoader
import storeMeg as st
app = Flask(__name__)
app.secret_key = 'pengsomnus'
# dataloader = DataLoader("房颤和冠心病预后", "Resource\\dataset\\318dataNew.xlsx", 0.1, -1)


@app.route('/')
def hello_world():
    return 'Welcome'


@app.route('/UploadData', methods=['GET', 'POST'])
def upload_data():
    dataset = request.files["dataset"]
    data_name = request.form.get("name") or "args没有参数"
    label_index = request.form.get("labelIndex") or "-1"
    test_ratio = request.form.get("testRatio") or "0.2"
    dataset.save(".\\Resource\\dataset\\"+str(data_name)+".csv")

    print(".\\Resource\\dataset\\"+str(data_name)+".csv")
    dataloader = DataLoader(name=str(data_name), path=".\\Resource\\dataset\\"+str(data_name),
                            test_ratio=float(test_ratio), labelindex=int(label_index))
    # TODO 这里还需要完成将数据集的信息使用如JSON的形式进行具体描述
    st.store_dataset(dataloader)
    session['dataloader'] = dataloader
    dict1 = {'name': data_name, 'dataset': dataloader.dataset_X.to_dict(), 'label': dataloader.dataset_label.to_dict()}
    return json.dumps(dict1)


@app.route('/selectFeature', methods=['GET', 'POST'])
def select_feature():
    feature_index = request.form.get("index")
    # data_name = request.form.get("name") or "args没有参数"
    # label_index = request.form.get("labelIndex") or "-1"
    # test_ratio = request.form.get("testRatio") or "0.2"
    # data_set = DataLoader(name=str(data_name), path=".\\Resource\\dataset\\"+str(data_name),
    #                       test_ratio=float(test_ratio), labelindex=int(label_index))

    data_set = session['dataloader']
    data_name = data_set.name
    feature_selection = FeatureSelection(data_set)
    feature_selection.manualselect(indexs=feature_index)
    dataloader = feature_selection.dataset
    session['dataloader'] = dataloader
    dict1 = {'name': data_name, 'dataset': dataloader.dataset_X.to_dict(), 'label': dataloader.dataset_label.to_dict()}
    return json.dumps(dict1)


@app.route('/getMissingNum', methods=['GET', 'POST'])
def get_missing_num():
    data_set = session['dataloaer']
    preprocessor = DataPreprocessor(data_set)
    missing_num = preprocessor.get_missing_num()
    dict1 = {'missingNum': missing_num}
    return json.dumps(dict1)


@app.route('/fillMissingByMedian', methods=['GET', 'POST'])
def fill_missing_median():
    indexes = request.form.getlist("indexes")
    data_set = session['dataloader']
    data_name = data_set.name
    preprocessor = DataPreprocessor(data_set)
    preprocessor.fillmissingvaluebymedian(indexes)
    dataloader = preprocessor.getdataset()
    session['dataloader'] = dataloader
    dict1 = {'name': data_name, 'dataset': dataloader.dataset_X.to_dict(), 'label': dataloader.dataset_label.to_dict()}
    return json.dumps(dict1)


@app.route('/fillMissingByMean', methods=['GET', 'POST'])
def fill_missing_mean():
    indexes = request.form.getlist("indexes")
    data_set = session['dataloader']
    data_name = data_set.name
    preprocessor = DataPreprocessor(data_set)
    preprocessor.fillmissingvaluebymean(indexes)
    dataloader = preprocessor.getdataset()
    session['dataloader'] = dataloader
    dict1 = {'name': data_name, 'dataset': dataloader.dataset_X.to_dict(), 'label': dataloader.dataset_label.to_dict()}
    return json.dumps(dict1)


@app.route('/fillMissingByMost', methods=['GET', 'POST'])
def fill_missing_most():
    indexes = request.form.getlist("indexes")
    data_set = session['dataloader']
    data_name = data_set.name
    preprocessor = DataPreprocessor(data_set)
    preprocessor.fillmissingvaluebymost(indexes)
    dataloader = preprocessor.getdataset()
    session['dataloader'] = dataloader
    dict1 = {'name': data_name, 'dataset': dataloader.dataset_X.to_dict(), 'label': dataloader.dataset_label.to_dict()}
    return json.dumps(dict1)


@app.route('/getSkew', methods=['GET', 'POST'])
def get_skew():
    data_set = session['dataloaer']
    preprocessor = DataPreprocessor(data_set)
    skew = preprocessor.getskew()
    dict1 = {'skew': skew}
    return json.dumps(dict1)


@app.route('/getKurt', methods=['GET', 'POST'])
def get_kurt():
    data_set = session['dataloaer']
    preprocessor = DataPreprocessor(data_set)
    kurt = preprocessor.getkurt()
    dict1 = {'kurt': kurt}
    return json.dumps(dict1)


@app.route('/getCorr', methods=['GET', 'POST'])
def get_kurt():
    index1 = request.form.get('index1')
    index2 = request.form.get('index2')
    data_set = session['dataloaer']
    preprocessor = DataPreprocessor(data_set)
    corr = preprocessor.getcorr(index1, index2)
    dict1 = {'corr': corr}
    return json.dumps(dict1)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5080)