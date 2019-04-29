from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np

from startup import Ui_MainWindow as Startup_MainWindow 
from preprocessing import Ui_MainWindow as Preprocessing_MainWindow
from classification import Ui_Dialog as Classification_Dialog


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import datasets

from functools import partial
import datetime

DEFAULT_DATA = '2D Iris 1'
    
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

iris_X_2d = X[:, (2,3)]
iris_X_3d = X[:, (1,2,3)]

# 自带的数据集
iris_X_2d_1 = np.concatenate((iris_X_2d[y==0], iris_X_2d[y==1]))
iris_X_2d_2 = np.concatenate((iris_X_2d[y==1], iris_X_2d[y==2]))
iris_X_3d_1 = np.concatenate((iris_X_3d[y==0], iris_X_3d[y==2]))
iris_X_3d_2 = np.concatenate((iris_X_3d[y==1], iris_X_3d[y==2]))
iris_y_1 = np.concatenate((y[y==0], y[y==1]))
iris_y_2 = np.concatenate((y[y==1], y[y==2]))




class StartWindow(QMainWindow, Startup_MainWindow):

    def __init__(self):
        super(StartWindow, self).__init__()
        self.setupUi(self)

        self.nextbtn.clicked.connect(self.nextWindow)

    def nextWindow(self):
        self.next_window = MainWindow()
        self.hide()
        self.next_window.show()


class MainWindow(QMainWindow, Preprocessing_MainWindow):
    
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)    
        

        self.data = DEFAULT_DATA

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)

        self.data_tableview = QTableView()

        self.dataList = QComboBox()
        self.space = QLabel()
        self.space.setFixedWidth(430)
        
        self.toolBar.addWidget(self.dataList)
        self.toolBar.addWidget(self.space)
        self.import_btn = QPushButton('导入')
        self.toolBar.addWidget(self.import_btn)
        
        self.init_comboBox()

        self.dataList.currentTextChanged.connect(self.change_data)
        self.import_btn.clicked.connect(self.showDialog)
        self.std_chk.stateChanged.connect(self.plot_std)
        self.reset.clicked.connect(self.reset_click)
        self.next.clicked.connect(self.next_window)

        # 程序初始化的时候展现的数据
        self.plot_custom(iris_X_2d_1, iris_y_1)


    def reset_click(self):
        """
        重置checkBox
        后续添加新的重置
        """
        self.std_chk.setCheckState(0)
        self.impute_chk.setCheckState(0)


    def info_display(self, feature, target):
        '''显示数据的信息'''
        dimension = len(feature[0])
        unique, counts = np.unique(target, return_counts=True)
        samples = ', '.join([str(i) for i in counts])
        classes = len(unique)

        self.class_val.setText(str(classes))
        self.dimesion_val.setText(str(dimension))
        self.sample_val.setText(samples)
        
    def info_clear(self):
        '''清除数据信息'''
        self.class_val.setText('')
        self.dimesion_val.setText('')
        self.sample_val.setText('')

    
    def init_comboBox(self):
        ''''comboBox的初始化'''
        self.dataList.addItem('2D Iris 1', (iris_X_2d_1, iris_y_1))
        self.dataList.addItem('3D Iris 1', (iris_X_3d_1, iris_y_1))
        self.dataList.addItem('2D Iris 2', (iris_X_2d_2, iris_y_2))
        self.dataList.addItem('3D Iris 2', (iris_X_3d_2, iris_y_2))
        self.dataList.addItem('Custom')


    def change_data(self, data):
        '''comboBox改变事件'''
        self.data = data

        # 重置checkbutton
        self.reset_click()
    
        if self.dataList.currentData():
            feature, target = self.dataList.currentData()
            self.plot_custom(feature, target)
            self.info_display(feature, target)
        else:
            self.figure.clear()
            self.canvas.draw()
            self.info_clear()

    def showDialog(self):
        """导入数据"""
        # filename, _ = QFileDialog.getOpenFileName(self, "Open file", "datasets",
        # "Csv files(.csv);;", "All files(*.*)")
        filename, _ = QFileDialog.getOpenFileName(self, "Open file", ".", "Csv files(*.csv);;All files (*.*)")

        if filename:
            feature, target = self.load(filename)
            index = self.dataList.findText('Custom')
            self.dataList.setItemData(index, (feature, target))
            
            self.dataList.setCurrentText('Custom')
        
    
    def load(self, filename):
        '''csv文件的读入'''
        data = np.loadtxt(filename, delimiter=',')
        dimension = data.shape[1]
        cat = np.hsplit(data, [dimension-1])
        feature = cat[0]
        target = cat[1].flatten()
        target = target.astype('int')

        return feature, target
    
    
    def plot_custom(self, feature, target):
        '''自制数据的绘图'''
        self.figure.clear()

        # 由于可能重复触发plot_custom方法，为了避免重复地创造tableView,
        # 以及覆盖导致新的tableView没有加入到plot_layout中
        # 所以每次创建之前检查是否已存在，如果存在就删除
        if self.plot_layout.indexOf(self.canvas) == -1:
            self.plot_layout.replaceWidget(self.data_tableview, self.canvas)
            self.data_tableview.deleteLater()

        unique = np.unique(target)
        y0 = unique[0]
        y1 = unique[1]
        
        if feature.shape[1] == 2:
            ax = self.figure.add_subplot(111)
            ax.plot(feature[:,0][target==y0], feature[:,1][target==y0], 'g^')
            ax.plot(feature[:,0][target==y1], feature[:,1][target==y1], 'bs')
            
        elif feature.shape[1] == 3:
            ax = self.figure.add_subplot(111, projection='3d')
            ax.plot(feature[:,0][target==y0], feature[:,1][target==y0], feature[:,2][target==y0], 'g^')
            ax.plot(feature[:,0][target==y1], feature[:,1][target==y1], feature[:,2][target==y1], 'bs')

        elif feature.shape[1] > 3:
            self.data_tableview = QTableView()
            model = self.create_model(feature, target)
            self.data_tableview.setModel(model)

            self.plot_layout.replaceWidget(self.canvas, self.data_tableview)

        self.canvas.draw()


    def create_model(self, feature, target):
        '''创建tableview所需的模型'''

        rows = feature.shape[0]
        columns = feature.shape[1]
        
        model = QStandardItemModel(rows, columns+1)
        model.setHorizontalHeaderLabels(['X' + str(i) for i in range(columns)] + ['y'])

        for row in range(rows):
            for column in range(columns):
                item = QStandardItem(str(feature[row, column]))
                item.setEditable(False)
                # 设置每个位置的文本值
                model.setItem(row, column, item)
            model.setItem(row, column+1, QStandardItem(str(target[row])))
        
        return model


    def plot_std(self, state):
        '''标准化数据'''

        if not self.dataList.currentData():
            return
        feature, target = self.dataList.currentData()
        if state:
            stander = StandardScaler()
            feature_std = stander.fit_transform(feature)
            self.plot_custom(feature_std, target)
        else:
            self.plot_custom(feature, target)

    def next_window(self):
        if not self.dataList.currentData():
            return 
        feature, target = self.dataList.currentData()
        self.second_window = ChildWindow2(feature, target, self)
        self.hide()
        self.second_window.show()
        self.second_window.pre_btn1.clicked.connect(self.click_pre_btn)
        self.second_window.pre_btn2.clicked.connect(self.click_pre_btn)

    def click_pre_btn(self):
        if self.second_window:
            self.second_window.close()
        if not self.isVisible():
            self.show()
    


class ChildWindow2(QDialog, Classification_Dialog):

    def __init__(self, feature=None, target=None, parent=None):
        super(ChildWindow2, self).__init__()
        self.setupUi(self)

        # 训练的数据
        self.feature = feature
        self.target = target
        self.isStd = parent.std_chk.isChecked()

        self.gamma_val.currentTextChanged.connect(self.gamma_change)
        self.svm_start.clicked.connect(self.start)
        self.adaboost_start.clicked.connect(self.start)
        self.svm_results.itemSelectionChanged.connect(self.list_change)
        self.adaboost_results.itemSelectionChanged.connect(self.list_change)
        self.svm_results.setContextMenuPolicy(Qt.CustomContextMenu) # 开放右键策略
        self.svm_results.customContextMenuRequested[QPoint].connect(self.svm_menu)
        self.adaboost_results.setContextMenuPolicy(Qt.CustomContextMenu) # 开放右键策略
        self.adaboost_results.customContextMenuRequested[QPoint].connect(self.adaboost_menu)
    
    def gamma_change(self, text):
        if text == 'float':
            self.gamma_float.setEnabled(True)
        else:
            self.gamma_float.setDisabled(True)

    def start(self):
        '''开始分类'''
        if self.sender() == self.svm_start:
            clf_name = 'SVM'
            train_method = self.svm_train
            results = self.svm_results
        else:
            clf_name = 'Adaboost'
            train_method = self.adaboost_train
            results = self.adaboost_results

        # 测试数据    
        X_train, y_train = self.feature, self.target

        # 训练分类器
        clf = train_method(X_train, y_train)
        
        # 预测结果
        predict = cross_val_predict(clf, X_train, y_train, cv=3)

        # 定义指标
        metrics = self.score(y_train, predict)
        conf_mx = self.matrix(y_train, predict)

        # 保存参数
        if self.sender() == self.svm_start:
            params = self.kernel, self.gamma, self.C
        else:
            params = self.base_esti, self.n_esti, self.learning_rate

        # 生成Item
        item = QListWidgetItem(datetime.datetime.now().strftime('%H:%M:%S') + ' - ' + clf_name)
        item.setData(Qt.UserRole, metrics)
        item.setData(Qt.UserRole + 1, conf_mx)
        item.setData(Qt.UserRole + 2, params)
        item.setData(Qt.UserRole + 3, clf)
        
        results.addItem(item)
        results.setCurrentItem(item)


    def list_change(self):
        if self.sender() == self.svm_results:
            clf = 'SVM'
            results = self.svm_results
            output = self.svm_output
        else:
            clf = 'Adaboost'
            results = self.adaboost_results
            output = self.adaboost_output
        
        current_item = results.currentItem()

        # 删除掉所有item时current_item取空
        if current_item:
            metrics = current_item.data(Qt.UserRole)
            conf_mx = current_item.data(Qt.UserRole + 1)
            params = current_item.data(Qt.UserRole + 2)
            text = self.printout(clf, params, metrics, conf_mx)
            output.setText(text)

    def printout(self, clf, params, metrics, confusion_matrix):
        """输出分类结果信息"""

        head = ''' === 运行信息 ===
        数据集样本数 :      {0}
        特征数       :      {1}
        分类器       :      {2}
        '''.format(len(self.feature), self.feature.shape[1], clf)

        label = ('kernel', 'gamma', 'C') if clf == 'SVM' else ('base_estimator', 'n_estimators', 'learning_rate')
        params = '''\n === 分类器参数 ===
        {0:14} :      {3}
        {1:14} :      {4}
        {2:14} :      {5}
        '''.format(*label, *params)

        predict = '''\n === 预测评估 ===
        precision :         {0}
        recall    :         {1}
        accuracy  :         {2}
        f1        :         {3}
        '''.format(*metrics)

        matrix = '''\n === 混淆矩阵 ===
        {:3} {:3}    <-- classified as
        {:3} {:3}   |    in fact 0
        {:3} {:3}   |    in fact 1
        '''.format(0, 1, *confusion_matrix)
        
        return head + params + predict + matrix

    def score(self, y_train, predict):
        """计算评估指标"""

        precision = "%.5f" % precision_score(y_train, predict)
        recall = "%.5f" % recall_score(y_train, predict)
        accuracy = "%.5f" % accuracy_score(y_train, predict)
        f1 = "%.5f" % f1_score(y_train, predict)

        return precision, recall, accuracy, f1
    
    def matrix(self, y_train, pred):
        """计算混淆矩阵"""

        matrix  = confusion_matrix(y_train, pred)
        true_negatives = matrix[0,0]
        false_negatives = matrix[0,1]
        true_positives = matrix[1,1]
        false_positives = matrix[1,0]

        return true_negatives, false_positives, false_negatives, true_positives


    def svm_train(self, X_train, y_train):
        """svm的训练，返回k折预测结果"""

        self.kernel = self.kernel_val.currentText()
        if self.gamma_val.currentText() == 'float':
            self.gamma = self.gamma_float.value()
        else:
            self.gamma = self.gamma_val.currentText()
        self.C = self.C_val.value()

        self.svc = SVC(kernel=self.kernel, gamma=self.gamma, C=self.C)
        
        if self.isStd:
            svm_clf = Pipeline((
                ("scaler", StandardScaler()),
                ("svc", self.svc),
            ))
        else:
            svm_clf = Pipeline((
                ('svc', self.svc),
            ))
        svm_clf.fit(X_train, y_train)
        
        return svm_clf

    def adaboost_train(self, X_train, y_train):
        """adaboost的训练，返回k折预测结果"""

        self.base_esti = self.base_esti_val.currentText()
        self.n_esti = self.n_esti_val.value()
        self.learning_rate = self.learning_rate_val.value()

        if self.isStd:
            adaboost_clf = Pipeline((
                ("scaler", StandardScaler()),
                ('adaboost', AdaBoostClassifier(base_estimator=None, n_estimators=self.n_esti, learning_rate=self.learning_rate))
            ))
        else:
            adaboost_clf = Pipeline((
                ('adaboost', AdaBoostClassifier(base_estimator=None, n_estimators=self.n_esti, learning_rate=self.learning_rate)),
            ))
        adaboost_clf.fit(X_train, y_train)

        return adaboost_clf

    def svm_menu(self, point):
        item = self.svm_results.itemAt(point)
        if item: 
            popMenu = QMenu()
            delete_menu = QAction('删除', self)
            save_menu = QAction('保存', self)
            plot_menu = QAction('绘图', self)
            predict_menu = QAction('预测', self)
            popMenu.addAction(save_menu)
            popMenu.addAction(delete_menu)
            popMenu.addAction(plot_menu)
            popMenu.addAction(predict_menu)
            if self.feature.shape[1] > 2:
                plot_menu.setDisabled(True)

            delete_menu.triggered.connect(partial(self.delete_result, item))
            save_menu.triggered.connect(partial(self.save_result, item, 'SVM'))
            plot_menu.triggered.connect(partial(self.plot_clf, item))
            predict_menu.triggered.connect(partial(self.predict, item))

            popMenu.exec_(QCursor.pos())

    def adaboost_menu(self, point):
        item = self.adaboost_results.itemAt(point)
        if item: 
            popMenu = QMenu()
            delete_menu = QAction('删除', self)
            save_menu = QAction('保存', self)
            plot_menu = QAction('绘图', self)
            predict_menu = QAction('预测', self)
            popMenu.addAction(save_menu)
            popMenu.addAction(delete_menu)
            popMenu.addAction(plot_menu)
            popMenu.addAction(predict_menu)
            if self.feature.shape[1] > 2:
                plot_menu.setDisabled(True)

            delete_menu.triggered.connect(partial(self.delete_result, item))
            save_menu.triggered.connect(partial(self.save_result, item, 'AdaBoost'))
            plot_menu.triggered.connect(partial(self.plot_clf, item))
            predict_menu.triggered.connect(partial(self.predict, item))

            popMenu.exec_(QCursor.pos())

    def predict(self, item):
        import re
        features = self.feature.shape[1]
        clf = item.data(Qt.UserRole + 3)
        regInt='0|[1-9]\d*'
        regFloat='0\.\d+|[1-9]\d*\.\d+'
        regIntOrFloat = '\s*(' + regInt + '|' + regFloat + ')\s*'
        reg = '^' + ('\,'.join([regIntOrFloat] * features)) + '$'
        
        value, ok = QInputDialog.getText(self, "预测", "请输入特征向量:", QLineEdit.Normal, ",".join(['f' + str(x) for x in range(features)]))
        result = re.match(reg, value)
        if result:
            arr = list(map(float, result.groups()))
            test = [np.array(arr)]
            predict_result = clf.predict(test)[0]
            QMessageBox.about(self, "", "预测分类类别为: " + ('正类' if predict_result == 1 else '负类'))
        elif ok:
            QMessageBox.information(self, "", "请输入正确的格式")

    def delete_result(self, item):
        results = item.listWidget()
        results.removeItemWidget(results.takeItem(results.row(item)))
    
    def save_result(self, item, clf):
        filename, _ = QFileDialog.getSaveFileName(self,'save file','.')
        if filename:
            with open(filename, 'w') as f:
                metrics = item.data(Qt.UserRole)
                conf_mx = item.data(Qt.UserRole + 1)
                params = item.data(Qt.UserRole + 2)
                text = self.printout(clf, params, metrics, conf_mx)
                f.write(text)

    def plot_clf(self, item):
        X, y = self.feature, self.target
        clf = item.data(Qt.UserRole + 3)

        figure = plt.figure()
        canvas = FigureCanvas(figure)
        ax = figure.add_subplot(111)

        self.plot_dataset(ax, X, y)
        axes = ax.axis()
        self.plot_prediction(ax, clf, axes)
        
        svc = clf.named_steps.get('svc')
        if svc:
            scaler = clf.named_steps.get('scaler')
            self.plot_support_vector(ax, svc, scaler, axes)
        
        grid = QGridLayout()
        grid.addWidget(canvas)
        plot_widget = QDialog() 
        plot_widget.resize(550, 450)
        rect = self.frameGeometry()
        plot_widget.move(rect.left() + rect.width(), rect.top())
        plot_widget.setWindowTitle('figure')
        plot_widget.setLayout(grid)
        plot_widget.show()
        plot_widget.exec_()

    def plot_dataset(self, ax, X, y, axes=None):
        unique = np.unique(y)
        y0 = unique[0]
        y1 = unique[1]
        #ax.grid(True, which='both')
        ax.plot(X[:, 0][y==y0], X[:, 1][y==y0], 'bs')
        ax.plot(X[:, 0][y==y1], X[:, 1][y==y1], 'g^')

    def plot_prediction(self, ax, clf, axes=None):
        x0s = np.linspace(axes[0], axes[1], 100)
        x1s = np.linspace(axes[2], axes[3], 100)
        x0, x1 = np.meshgrid(x0s, x1s)
        X = np.c_[x0.ravel(), x1.ravel()]
        y_pred = clf.predict(X).reshape(x0.shape)
        y_decision = clf.decision_function(X).reshape(x0.shape)
        ax.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
        ax.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

    def plot_support_vector(self, ax, clf, scaler=None, axes=None):
        sv = clf.support_vectors_

        sv = scaler.inverse_transform(sv) if scaler else sv
        ax.scatter(sv[:, 0], sv[:, 1], s=75,
                c='',marker='o', edgecolors='r')
        
        if clf.kernel == 'linear' and not scaler:
            w = clf.coef_[0]
            b = clf.intercept_[0]
            xx = np.linspace(axes[0], axes[1], 100)
            yy_up = (w[0]*xx + b - 1) / -w[1]
            yy_down = (w[0]*xx + b + 1) / -w[1]
            plt.ylim(axes[2], axes[3])
            ax.plot(xx, yy_up, 'k--')
            ax.plot(xx, yy_down, 'k--')
            
        print(axes)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    startup = StartWindow()
    startup.show()
    
    sys.exit(app.exec_())