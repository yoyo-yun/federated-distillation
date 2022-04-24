import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

SEED = 222
np.random.seed(SEED)
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('input.csv')

def get_train_test():   # 数据处理

    y = 1 * (df.cand_pty_affiliation == "REP")
    x = df.drop(['cand_pty_affiliation'],axis=1)
    x = pd.get_dummies(x,sparse=True)
    x.drop(x.columns[x.std()==0],axis=1,inplace=True)
    return train_test_split(x,y,test_size=0.95,random_state=SEED)

def get_models():   # 模型定义
    nb = GaussianNB()
    svc = SVC(C=100,probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
    lr = LogisticRegression(C=100,random_state=SEED)
    nn = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators =100, random_state = SEED)
    rf = RandomForestClassifier(n_estimators=1,max_depth=3,random_state=SEED)

    models = {'svm':svc,
              'knn':knn,
              'naive bayes':nb,
              'mlp-nn':nn,
              'random forest':rf,
              'gbm':gb,
              'logistic':lr,
              }
    return models

def train_base_learnres(base_learners,inp,out,verbose=True):    # 训练基本模型
    if verbose:print("fitting models.")
    for i,(name,m) in enumerate(base_learners.items()):
        if verbose:print("%s..." % name,end=" ",flush=False)
        m.fit(inp,out)
        if verbose:print("done")

def predict_base_learners(pred_base_learners,inp,verbose=True): # 把基本学习器的输出作为融合学习的特征，这里计算特征
    p = np.zeros((inp.shape[0],len(pred_base_learners)))
    if verbose:print("Generating base learner predictions.")
    for i,(name,m) in enumerate(pred_base_learners.items()):
        if verbose:print("%s..." % name,end=" ",flush=False)
        p_ = m.predict_proba(inp)
        p[:,i] = p_[:,1]
        if verbose:print("done")
    return p

def ensemble_predict(base_learners,meta_learner,inp,verbose=True):  # 融合学习进行预测
    p_pred = predict_base_learners(base_learners,inp,verbose=verbose)    # 测试数据必须先经过基本学习器计算特征
    return p_pred,meta_learner.predict_proba(p_pred)[:,1]

def ensenmble_by_blend():   # blend融合
    xtrain_base, xpred_base, ytrain_base, ypred_base = train_test_split(
        xtrain, ytrain, test_size=0.5, random_state=SEED
    )   # 把数据切分成两部分

    train_base_learnres(base_learners, xtrain_base, ytrain_base)  # 训练基本模型

    p_base = predict_base_learners(base_learners, xpred_base) # 把基本学习器的输出作为融合学习的特征，这里计算特征
    meta_learner.fit(p_base, ypred_base)    # 融合学习器的训练
    p_pred, p = ensemble_predict(base_learners, meta_learner, xtest)  # 融合学习进行预测
    print("\nEnsemble ROC-AUC score: %.3f" % roc_auc_score(ytest, p))


from sklearn.base import clone
def stacking(base_learners,meta_learner,X,y,generator): # stacking进行融合
    print("Fitting final base learners...",end="")
    train_base_learnres(base_learners,X,y,verbose=False)
    print("done")

    print("Generating cross-validated predictions...")
    cv_preds,cv_y = [],[]
    for i,(train_inx,test_idx) in enumerate(generator.split(X)):
        fold_xtrain,fold_ytrain = X[train_inx,:],y[train_inx]
        fold_xtest,fold_ytest = X[test_idx,:],y[test_idx]

        fold_base_learners = {name:clone(model)
                              for name,model in base_learners.items()}
        train_base_learnres(fold_base_learners,fold_xtrain,fold_ytrain,verbose=False)
        fold_P_base = predict_base_learners(fold_base_learners,fold_xtest,verbose=False)

        cv_preds.append(fold_P_base)
        cv_y.append(fold_ytest)

        print("Fold %i done" %(i+1))
    print("CV-predictions done")
    cv_preds = np.vstack(cv_preds)
    cv_y = np.hstack(cv_y)

    print("Fitting meta learner...",end="")
    meta_learner.fit(cv_preds,cv_y)
    print("done")

    return base_learners,meta_learner

def ensemble_by_stack():
    from sklearn.model_selection import KFold

    cv_base_learners,cv_meta_learner = stacking(
        get_models(),clone(meta_learner),xtrain.values,ytrain.values,KFold(2))
    P_pred,p = ensemble_predict(cv_base_learners,cv_meta_learner,xtest,verbose=False)
    print("\nEnsemble ROC-AUC score: %.3f" %roc_auc_score(ytest,p))

def plot_roc_curve(ytest,p_base_learners,p_ensemble,labels,ens_label):
    plt.figure(figsize=(10,8))
    plt.plot([0,1],[0,1],'k--')
    cm = [plt.cm.rainbow(i)
        for i in np.linspace(0,1.0, p_base_learners.shape[1] +1)]
    for i in range(p_base_learners.shape[1]):
        p = p_base_learners[:,i]
        fpr,tpr,_ = roc_curve(ytest,p)
        plt.plot(fpr,tpr,label = labels[i],c=cm[i+1])
    fpr, tpr, _ = roc_curve(ytest, p_ensemble)
    plt.plot(fpr, tpr, label=ens_label, c=cm[0])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(frameon=False)
    plt.show()

from mlens.ensemble import SuperLearner
def use_pack():
    sl =SuperLearner(
        folds=10,random_state=SEED,verbose=2,
        # backend="multiprocessing"
    )
    # Add the base learners and the meta learner
    sl.add(list(base_learners.values()),proba=True)
    sl.add_meta(meta_learner,proba=True)
    # Train the ensemble
    sl.fit(xtrain,ytrain)
    # Predict the test set
    p_sl=sl.predict_proba(xtest)

    print("\nSuper Learner ROC-AUC score: %.3f" % roc_auc_score(ytest,p_sl[:,1]))

if __name__ == "__main__":
    xtrain, xtest, ytrain, ytest = get_train_test()
    base_learners = get_models()

    meta_learner = GradientBoostingClassifier(
        n_estimators=1000,
        loss="exponential",
        max_depth=4,
        subsample=0.5,
        learning_rate=0.005,
        random_state=SEED
    )

    # ensenmble_by_blend() # blend进行融合
    # ensemble_by_stack()   # stack进行融合
    use_pack()  # 调用包进行融合