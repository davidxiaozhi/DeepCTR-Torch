import pandas as pd
from pip import main
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# 使用pandas 读取上面介绍的数据，并进行简单的缺失值填充
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.models import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
import torch

if __name__ == "__main__":
    data = pd.read_csv('../data/criteo_sample.txt')
    print(data.head())
    #特征列添加标志27 个稀疏特征, 14 个稠密特征, 我们给稀疏特征 添加 C 的标志,给稠密特征添加 I 的标志
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
 
    # 特征缺失值处理 稀疏值 补充-1,稠密特征缺失值处理 填充0
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']
    
    # 这里我们使用的 sklearn 的 LabelEncoder 对类别特征进行编码,使用的 MinMaxScaler 讲数值特征压缩到0~1之间
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    
    # 这里是比较关键的一步，因为我们需要对类别特征进行Embedding，所以需要告诉模型每一个特征组有多少个embbedding向量，我们通过pandas的nunique()方法统计。
    # SparseFeat(特征名称,特征维度,embedding_size,embedding_name) 这里采用的是命名元组实现的
    # DensFeat(特征名称,特征维度)
    # 核心逻辑将 稀疏特征 和 稠密特征进行拼接 
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features] + [ DenseFeat(feat, 1,) for feat in dense_features]

    # 深度学习模型的特征输入 
    dnn_feature_columns = fixlen_feature_columns
    # 线性模型的输入
    linear_feature_columns = fixlen_feature_columns

    # 输入特征的名称
    fixlen_feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    fixlen_feature_names[20:30]
    
    #最后，我们按照上一步生成的特征列拼接数据
    # 8 2 原则拆分训练集和验证集
    train, test = train_test_split(data, test_size=0.2)

    train_model_input = [train[name] for name in fixlen_feature_names]
    test_model_input = [test[name] for name in fixlen_feature_names]
    
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
        
    # 初始化模型，进行训练和预测

    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, 
                   dnn_hidden_units=[1024,128],
                   dnn_use_bn=True,
                   #dnn_dropout=0.1,
                   task='binary',
                l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                metrics=["binary_crossentropy", "auc"],)
    model.fit(train_model_input, train[target].values,
            batch_size=128, epochs=500, validation_split=0.2, verbose=2)

    pred_ans = model.predict(test_model_input, 128)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))