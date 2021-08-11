# [AI训练营]PaddleX实现目标检测baseline

手把手教你基于PaddleX实现目标检测。你需要实现以下任务：

> 1. 配置数据集（数据集选择、数据处理）
> 2. 配置模型并训练
> 3. 项目跑通即可达到结业要求

# 一、数据集说明

本项目使用的数据集是：[[AI训练营]目标检测数据集合集](https://aistudio.baidu.com/aistudio/datasetdetail/103743)，包含口罩识别 、交通标志识别、火焰检测、锥桶识别以及中秋元素识别。

该数据集已加载至本环境中，位于：**data/data103743/objDataset.zip**


```python
# 解压数据集（解压一次即可，请勿重复解压）
!unzip -oq /home/aistudio/data/data103743/objDataset.zip
```

解压完成后，左侧文件夹处会多一个名为**objDataset**的文件夹，该文件夹下有5个子文件夹：
- **barricade**——Gazebo锥桶检测
- **facemask**——口罩检测
- **fire**——火焰检测
- **MidAutumn**——中秋元素检测
- **roadsign_voc**——交通路标检测

每个子文件夹下有2个文件夹，分别存放着图像（**JPEGImages**）和标注文件（**Annotations**），如下所示：


```python
# 查看数据集文件结构
!tree objDataset -L 2
```

    objDataset
    ├── barricade
    │   ├── Annotations
    │   └── JPEGImages
    ├── facemask
    │   ├── Annotations
    │   └── JPEGImages
    ├── fire
    │   ├── Annotations
    │   └── JPEGImages
    ├── MidAutumn
    │   ├── Annotations
    │   └── JPEGImages
    └── roadsign_voc
        ├── Annotations
        └── JPEGImages
    
    15 directories, 0 files


# 二、数据准备

本基线系统使用的数据格式是PascalVOC格式，开发者基于PaddleX开发目标检测模型时，无需对数据格式进行转换，开箱即用。

但为了进行训练，还需要将数据划分为训练集、验证集和测试集。划分之前首先需要**安装PaddleX**。


```python
# 安装PaddleX
!pip install paddlex
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting paddlex
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d6/a2/07435f4aa1e51fe22bdf06c95d03bf1b78b7bc6625adbb51e35dc0804cc7/paddlex-1.3.11-py3-none-any.whl (516kB)
    [K     |████████████████████████████████| 522kB 16.6MB/s eta 0:00:01
    [?25hRequirement already satisfied: flask-cors in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (3.0.8)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.1.2)
    Collecting paddleslim==1.1.1 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d1/77/e257227bed9a70ff0d35a4a3c4e70ac2d2362c803834c4c52018f7c4b762/paddleslim-1.1.1-py2.py3-none-any.whl (145kB)
    [K     |████████████████████████████████| 153kB 28.0MB/s eta 0:00:01
    [?25hCollecting paddlehub==2.1.0 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/7a/29/3bd0ca43c787181e9c22fe44b944b64d7fcb14ce66d3bf4602d9ad2ac76c/paddlehub-2.1.0-py3-none-any.whl (211kB)
    [K     |████████████████████████████████| 215kB 30.0MB/s eta 0:00:01
    [?25hRequirement already satisfied: sklearn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.0)
    Requirement already satisfied: visualdl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.2.0)
    Collecting shapely>=1.7.0 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/98/f8/db4d3426a1aba9d5dfcc83ed5a3e2935d2b1deb73d350642931791a61c37/Shapely-1.7.1-cp37-cp37m-manylinux1_x86_64.whl (1.0MB)
    [K     |████████████████████████████████| 1.0MB 14.2MB/s eta 0:00:01    |███▏                            | 102kB 14.2MB/s eta 0:00:01
    [?25hCollecting xlwt (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/44/48/def306413b25c3d01753603b1a222a011b8621aed27cd7f89cbc27e6b0f4/xlwt-1.3.0-py2.py3-none-any.whl (99kB)
    [K     |████████████████████████████████| 102kB 35.9MB/s ta 0:00:01
    [?25hRequirement already satisfied: psutil in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.7.2)
    Collecting pycocotools; platform_system != "Windows" (from paddlex)
      Downloading https://mirror.baidu.com/pypi/packages/de/df/056875d697c45182ed6d2ae21f62015896fdb841906fe48e7268e791c467/pycocotools-2.0.2.tar.gz
    Requirement already satisfied: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.4.4)
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.36.1)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.1.1.26)
    Requirement already satisfied: Flask>=0.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask-cors->paddlex) (1.1.1)
    Requirement already satisfied: Six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask-cors->paddlex) (1.15.0)
    Requirement already satisfied: pyzmq in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddleslim==1.1.1->paddlex) (18.1.1)
    Requirement already satisfied: gitpython in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1.14)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (1.20.3)
    Requirement already satisfied: paddlenlp>=2.0.0rc5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (2.0.1)
    Requirement already satisfied: Pillow in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (7.1.2)
    Collecting paddle2onnx>=0.5.1 (from paddlehub==2.1.0->paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/37/80/aa6134b5f36aea45dc1b363e7af941dccabe4d7e167ac391ff046f34baf1/paddle2onnx-0.7-py3-none-any.whl (94kB)
    [K     |████████████████████████████████| 102kB 42.7MB/s ta 0:00:01
    [?25hRequirement already satisfied: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (4.1.0)
    Requirement already satisfied: gunicorn>=19.10.0; sys_platform != "win32" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.0.4)
    Requirement already satisfied: rarfile in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1)
    Requirement already satisfied: packaging in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.9)
    Requirement already satisfied: easydict in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (1.9)
    Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (2.2.3)
    Requirement already satisfied: filelock in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.0.12)
    Requirement already satisfied: scikit-learn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from sklearn->paddlex) (0.24.2)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.21.0)
    Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.1.5)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.8.53)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.14.0)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.8.2)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (2.22.0)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.0.0)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.7.1.1)
    Requirement already satisfied: setuptools>=18.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (56.2.0)
    Requirement already satisfied: cython>=0.27.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (0.29)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask>=0.9->flask-cors->paddlex) (2.10.1)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask>=0.9->flask-cors->paddlex) (0.16.0)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask>=0.9->flask-cors->paddlex) (7.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask>=0.9->flask-cors->paddlex) (1.1.0)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitpython->paddlehub==2.1.0->paddlex) (4.0.5)
    Requirement already satisfied: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.70.11.1)
    Requirement already satisfied: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (1.2.2)
    Requirement already satisfied: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.42.1)
    Requirement already satisfied: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (2.9.0)
    Requirement already satisfied: pyparsing>=2.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from packaging->paddlehub==2.1.0->paddlex) (2.4.2)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddlehub==2.1.0->paddlex) (0.10.0)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddlehub==2.1.0->paddlex) (2.8.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddlehub==2.1.0->paddlex) (2019.3)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddlehub==2.1.0->paddlex) (1.1.0)
    Requirement already satisfied: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (1.6.3)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (2.1.0)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (0.14.1)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (0.10.0)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (16.7.9)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (0.23)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.4)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (2.0.1)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.0)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.4.10)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (0.18.0)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (3.9.9)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.2.0)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.6.0)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.6.1)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2019.9.11)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (1.25.6)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2.8)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (2.8.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->Flask>=0.9->flask-cors->paddlex) (1.1.1)
    Requirement already satisfied: smmap<4,>=3.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitdb<5,>=4.0.1->gitpython->paddlehub==2.1.0->paddlex) (3.0.5)
    Requirement already satisfied: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.3.3)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->pre-commit->visualdl>=2.0.0->paddlex) (0.6.0)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->visualdl>=2.0.0->paddlex) (7.2.0)
    Building wheels for collected packages: pycocotools
      Building wheel for pycocotools (setup.py) ... [?25ldone
    [?25h  Created wheel for pycocotools: filename=pycocotools-2.0.2-cp37-cp37m-linux_x86_64.whl size=278364 sha256=fc2f9cf7c02aba354c80b5a53180fd3ffc0be46bf1da801071a241ebbe18b805
      Stored in directory: /home/aistudio/.cache/pip/wheels/fb/44/67/8baa69040569b1edbd7776ec6f82c387663e724908aaa60963
    Successfully built pycocotools
    Installing collected packages: paddleslim, paddle2onnx, paddlehub, shapely, xlwt, pycocotools, paddlex
      Found existing installation: paddlehub 2.0.4
        Uninstalling paddlehub-2.0.4:
          Successfully uninstalled paddlehub-2.0.4
    Successfully installed paddle2onnx-0.7 paddlehub-2.1.0 paddleslim-1.1.1 paddlex-1.3.11 pycocotools-2.0.2 shapely-1.7.1 xlwt-1.3.0


使用如下命令即可将数据划分为70%训练集，20%验证集和10%的测试集。


```python
# 划分数据集，按照train：eval：tes=7：2：1来划分
!paddlex --split_dataset --format VOC --dataset_dir objDataset/facemask --val_value 0.2 --test_value 0.1
```

    Dataset Split Done.[0m
    [0mTrain samples: 598[0m
    [0mEval samples: 170[0m
    [0mTest samples: 85[0m
    [0mSplit files saved in objDataset/facemask[0m
    [0m[0m

划分完成后，该数据集下会生成**labels.txt**, **train_list.txt**, **val_list.txt**和**test_list.txt**，分别存储类别信息，训练样本列表，验证样本列表，测试样本列表。如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/d29a92b4cfc34b0097ef46dbbc8562af824387889f224948ae49283e0adee19d)

在这里，**你需要将path to dataset部分替换成你选择的数据集路径**。在左侧文件夹处，将鼠标放到你想选择的数据集文件夹上，会出现三个图标，第一个图标表示复制该文件夹路径，点击即可获得该文件夹路径，用这个路径替换path to dataset即可。

![](https://ai-studio-static-online.cdn.bcebos.com/c28ed88586644f64b34709a592fea0b97ec80470c0e041fd9aa6b8da21c8e283)



```python
#使用GPU
import matplotlib
matplotlib.use('Agg') 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

# 三、数据预处理

在训练模型之前，对目标检测任务的数据进行操作，从而提升模型效果。可用于数据处理的API有：
- **Normalize**：对图像进行归一化
- **ResizeByShort**：根据图像的短边调整图像大小
- **RandomHorizontalFlip**：以一定的概率对图像进行随机水平翻转
- **RandomDistort**：以一定的概率对图像进行随机像素内容变换

更多关于数据处理的API及使用说明可查看文档：
[https://paddlex.readthedocs.io/zh_CN/release-1.3/apis/transforms/det_transforms.html](https://paddlex.readthedocs.io/zh_CN/release-1.3/apis/transforms/det_transforms.html)


```python
#定义数据增强与前处理
from paddlex.det import transforms
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250),
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(target_size=608, interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    transforms.Resize(target_size=608, interp='CUBIC'),
    transforms.Normalize(),
])
```

读取PascalVOC格式的检测数据集，并对样本进行相应的处理。


```python
import paddlex as pdx

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/facemask',
    file_list='objDataset/facemask/train_list.txt',
    label_list='objDataset/facemask/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/facemask',
    file_list='objDataset/facemask/val_list.txt',
    label_list='objDataset/facemask/labels.txt',
    transforms=eval_transforms)
```

    2021-08-11 14:41:44 [INFO]	Starting to read file list from dataset...
    2021-08-11 14:41:45 [INFO]	598 samples in file objDataset/facemask/train_list.txt
    creating index...
    index created!
    2021-08-11 14:41:45 [INFO]	Starting to read file list from dataset...
    2021-08-11 14:41:46 [INFO]	170 samples in file objDataset/facemask/val_list.txt
    creating index...
    index created!


需要注意的是：
- **data_dir** (str): 数据集所在的目录路径。
- **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对data_dir的相对路径）。
- **label_list** (str): 描述数据集包含的类别信息文件路径。

需要将第二步数据准备时生成的labels.txt, train_list.txt, val_list.txt和test_list.txt配置到以上变量中，如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/6462f7811da3436290948e5dde0c497d6ae51bcfe1904e0a863ff032363a4448)


# 四、模型训练

PaddleX目前提供了FasterRCNN和YOLOv3两种检测结构，多种backbone模型。本基线系统以骨干网络为MobileNetV1的YOLOv3算法为例。


```python
# 初始化模型
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-yolov3

# 此处需要补充目标检测模型代码
model = pdx.det.YOLOv3(num_classes=len(train_dataset.labels), backbone='DarkNet53')
```


```python
# 模型训练
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#id1
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html

# 此处需要补充模型训练参数
model.train(
    num_epochs=270,#训练轮数
    train_dataset=train_dataset,
    train_batch_size=8,#每轮训练中每batch次训练个数
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],#在第210轮之后以及240轮之后调低学习率
    save_interval_epochs=20,#每20轮保留一次模型参数
    save_dir='output/yolov3_DarkNet53')
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py:689: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      elif dtype == np.bool:


    2021-08-11 14:43:38 [INFO]	Downloading DarkNet53_ImageNet1k_pretrained.tar from https://paddle-imagenet-models-name.bj.bcebos.com/DarkNet53_ImageNet1k_pretrained.tar


      0%|          | 0/162940 [00:00<?, ?KB/s]100%|██████████| 162940/162940 [00:02<00:00, 54754.11KB/s]


    2021-08-11 14:43:41 [INFO]	Decompressing output/yolov3_DarkNet53/pretrain/DarkNet53_ImageNet1k_pretrained.tar...
    2021-08-11 14:43:47 [INFO]	Load pretrain weights from output/yolov3_DarkNet53/pretrain/DarkNet53_ImageNet1k_pretrained.
    2021-08-11 14:43:47 [INFO]	There are 260 varaibles in output/yolov3_DarkNet53/pretrain/DarkNet53_ImageNet1k_pretrained are loaded.
    2021-08-11 14:43:54 [INFO]	[TRAIN] Epoch=1/270, Step=2/74, loss=14410.99707, lr=0.0, time_each_step=3.39s, eta=19:4:31
    2021-08-11 14:43:55 [INFO]	[TRAIN] Epoch=1/270, Step=4/74, loss=17659.929688, lr=0.0, time_each_step=1.92s, eta=10:48:16
    2021-08-11 14:43:56 [INFO]	[TRAIN] Epoch=1/270, Step=6/74, loss=9926.530273, lr=1e-06, time_each_step=1.43s, eta=8:3:9
    2021-08-11 14:43:57 [INFO]	[TRAIN] Epoch=1/270, Step=8/74, loss=4361.056641, lr=1e-06, time_each_step=1.17s, eta=6:35:52
    2021-08-11 14:43:57 [INFO]	[TRAIN] Epoch=1/270, Step=10/74, loss=2769.319336, lr=1e-06, time_each_step=1.01s, eta=5:40:31
    2021-08-11 14:43:58 [INFO]	[TRAIN] Epoch=1/270, Step=12/74, loss=2325.603516, lr=1e-06, time_each_step=0.9s, eta=5:3:17
    2021-08-11 14:43:58 [INFO]	[TRAIN] Epoch=1/270, Step=14/74, loss=1598.766357, lr=2e-06, time_each_step=0.81s, eta=4:32:1
    2021-08-11 14:43:59 [INFO]	[TRAIN] Epoch=1/270, Step=16/74, loss=793.332275, lr=2e-06, time_each_step=0.75s, eta=4:13:12
    2021-08-11 14:44:00 [INFO]	[TRAIN] Epoch=1/270, Step=18/74, loss=496.705963, lr=2e-06, time_each_step=0.71s, eta=3:58:10
    2021-08-11 14:44:00 [INFO]	[TRAIN] Epoch=1/270, Step=20/74, loss=654.687988, lr=2e-06, time_each_step=0.66s, eta=3:44:16
    2021-08-11 14:44:01 [INFO]	[TRAIN] Epoch=1/270, Step=22/74, loss=763.41272, lr=3e-06, time_each_step=0.37s, eta=2:5:7
    2021-08-11 14:44:02 [INFO]	[TRAIN] Epoch=1/270, Step=24/74, loss=153.001862, lr=3e-06, time_each_step=0.36s, eta=1:59:53
    2021-08-11 14:44:03 [INFO]	[TRAIN] Epoch=1/270, Step=26/74, loss=248.127533, lr=3e-06, time_each_step=0.34s, eta=1:54:43
    2021-08-11 14:44:04 [INFO]	[TRAIN] Epoch=1/270, Step=28/74, loss=220.734818, lr=3e-06, time_each_step=0.35s, eta=1:58:11
    2021-08-11 14:44:05 [INFO]	[TRAIN] Epoch=1/270, Step=30/74, loss=197.248871, lr=4e-06, time_each_step=0.36s, eta=2:3:8
    2021-08-11 14:44:05 [INFO]	[TRAIN] Epoch=1/270, Step=32/74, loss=94.269745, lr=4e-06, time_each_step=0.35s, eta=1:59:26
    2021-08-11 14:44:06 [INFO]	[TRAIN] Epoch=1/270, Step=34/74, loss=114.914581, lr=4e-06, time_each_step=0.37s, eta=2:3:19
    2021-08-11 14:44:07 [INFO]	[TRAIN] Epoch=1/270, Step=36/74, loss=104.519501, lr=4e-06, time_each_step=0.37s, eta=2:6:14
    2021-08-11 14:44:08 [INFO]	[TRAIN] Epoch=1/270, Step=38/74, loss=64.73391, lr=5e-06, time_each_step=0.38s, eta=2:9:36
    2021-08-11 14:44:08 [INFO]	[TRAIN] Epoch=1/270, Step=40/74, loss=50.950668, lr=5e-06, time_each_step=0.4s, eta=2:15:7
    2021-08-11 14:44:09 [INFO]	[TRAIN] Epoch=1/270, Step=42/74, loss=94.230598, lr=5e-06, time_each_step=0.39s, eta=2:10:56
    2021-08-11 14:44:10 [INFO]	[TRAIN] Epoch=1/270, Step=44/74, loss=61.718361, lr=5e-06, time_each_step=0.38s, eta=2:8:15
    2021-08-11 14:44:10 [INFO]	[TRAIN] Epoch=1/270, Step=46/74, loss=62.598331, lr=6e-06, time_each_step=0.37s, eta=2:3:21
    2021-08-11 14:44:10 [INFO]	[TRAIN] Epoch=1/270, Step=48/74, loss=52.224197, lr=6e-06, time_each_step=0.34s, eta=1:54:27
    2021-08-11 14:44:11 [INFO]	[TRAIN] Epoch=1/270, Step=50/74, loss=43.955223, lr=6e-06, time_each_step=0.31s, eta=1:42:57
    2021-08-11 14:44:11 [INFO]	[TRAIN] Epoch=1/270, Step=52/74, loss=74.982948, lr=6e-06, time_each_step=0.3s, eta=1:41:17
    2021-08-11 14:44:11 [INFO]	[TRAIN] Epoch=1/270, Step=54/74, loss=64.809601, lr=7e-06, time_each_step=0.28s, eta=1:35:43
    2021-08-11 14:44:12 [INFO]	[TRAIN] Epoch=1/270, Step=56/74, loss=57.71294, lr=7e-06, time_each_step=0.25s, eta=1:25:3
    2021-08-11 14:44:12 [INFO]	[TRAIN] Epoch=1/270, Step=58/74, loss=51.144264, lr=7e-06, time_each_step=0.22s, eta=1:13:59
    2021-08-11 14:44:12 [INFO]	[TRAIN] Epoch=1/270, Step=60/74, loss=54.220737, lr=7e-06, time_each_step=0.19s, eta=1:5:8
    2021-08-11 14:44:13 [INFO]	[TRAIN] Epoch=1/270, Step=62/74, loss=61.444141, lr=8e-06, time_each_step=0.17s, eta=0:58:46
    2021-08-11 14:44:13 [INFO]	[TRAIN] Epoch=1/270, Step=64/74, loss=82.467171, lr=8e-06, time_each_step=0.18s, eta=1:0:34
    2021-08-11 14:44:14 [INFO]	[TRAIN] Epoch=1/270, Step=66/74, loss=50.46299, lr=8e-06, time_each_step=0.2s, eta=1:6:41
    2021-08-11 14:44:14 [INFO]	[TRAIN] Epoch=1/270, Step=68/74, loss=102.558907, lr=8e-06, time_each_step=0.19s, eta=1:5:17
    2021-08-11 14:44:15 [INFO]	[TRAIN] Epoch=1/270, Step=70/74, loss=79.900467, lr=9e-06, time_each_step=0.19s, eta=1:5:23
    2021-08-11 14:44:15 [INFO]	[TRAIN] Epoch=1/270, Step=72/74, loss=55.490906, lr=9e-06, time_each_step=0.19s, eta=1:2:43
    2021-08-11 14:44:15 [INFO]	[TRAIN] Epoch=1/270, Step=74/74, loss=74.614159, lr=9e-06, time_each_step=0.2s, eta=1:6:40
    2021-08-11 14:44:15 [INFO]	[TRAIN] Epoch 1 finished, loss=1549.713867, lr=5e-06 .
    2021-08-11 14:44:19 [INFO]	[TRAIN] Epoch=2/270, Step=2/74, loss=54.317619, lr=9e-06, time_each_step=0.36s, eta=2:8:20
    2021-08-11 14:44:20 [INFO]	[TRAIN] Epoch=2/270, Step=4/74, loss=45.229874, lr=1e-05, time_each_step=0.41s, eta=2:8:36
    2021-08-11 14:44:21 [INFO]	[TRAIN] Epoch=2/270, Step=6/74, loss=69.985336, lr=1e-05, time_each_step=0.43s, eta=2:8:44
    2021-08-11 14:44:22 [INFO]	[TRAIN] Epoch=2/270, Step=8/74, loss=79.371643, lr=1e-05, time_each_step=0.47s, eta=2:8:56
    2021-08-11 14:44:23 [INFO]	[TRAIN] Epoch=2/270, Step=10/74, loss=60.446053, lr=1e-05, time_each_step=0.48s, eta=2:8:58
    2021-08-11 14:44:24 [INFO]	[TRAIN] Epoch=2/270, Step=12/74, loss=109.934372, lr=1.1e-05, time_each_step=0.49s, eta=2:9:2
    2021-08-11 14:44:24 [INFO]	[TRAIN] Epoch=2/270, Step=14/74, loss=32.776985, lr=1.1e-05, time_each_step=0.5s, eta=2:9:5
    2021-08-11 14:44:25 [INFO]	[TRAIN] Epoch=2/270, Step=16/74, loss=51.631783, lr=1.1e-05, time_each_step=0.52s, eta=2:9:11
    2021-08-11 14:44:26 [INFO]	[TRAIN] Epoch=2/270, Step=18/74, loss=73.967567, lr=1.1e-05, time_each_step=0.54s, eta=2:9:19
    2021-08-11 14:44:26 [INFO]	[TRAIN] Epoch=2/270, Step=20/74, loss=66.292297, lr=1.2e-05, time_each_step=0.55s, eta=2:9:22
    2021-08-11 14:44:27 [INFO]	[TRAIN] Epoch=2/270, Step=22/74, loss=66.620651, lr=1.2e-05, time_each_step=0.42s, eta=2:8:33
    2021-08-11 14:44:28 [INFO]	[TRAIN] Epoch=2/270, Step=24/74, loss=58.901566, lr=1.2e-05, time_each_step=0.4s, eta=2:8:26
    2021-08-11 14:44:29 [INFO]	[TRAIN] Epoch=2/270, Step=26/74, loss=43.813828, lr=1.2e-05, time_each_step=0.4s, eta=2:8:24
    2021-08-11 14:44:30 [INFO]	[TRAIN] Epoch=2/270, Step=28/74, loss=89.828026, lr=1.3e-05, time_each_step=0.38s, eta=2:8:16
    2021-08-11 14:44:31 [INFO]	[TRAIN] Epoch=2/270, Step=30/74, loss=73.175293, lr=1.3e-05, time_each_step=0.4s, eta=2:8:21
    2021-08-11 14:44:31 [INFO]	[TRAIN] Epoch=2/270, Step=32/74, loss=37.136646, lr=1.3e-05, time_each_step=0.4s, eta=2:8:20
    2021-08-11 14:44:32 [INFO]	[TRAIN] Epoch=2/270, Step=34/74, loss=45.477249, lr=1.3e-05, time_each_step=0.41s, eta=2:8:25
    2021-08-11 14:44:33 [INFO]	[TRAIN] Epoch=2/270, Step=36/74, loss=52.921265, lr=1.4e-05, time_each_step=0.41s, eta=2:8:23
    2021-08-11 14:44:34 [INFO]	[TRAIN] Epoch=2/270, Step=38/74, loss=45.687725, lr=1.4e-05, time_each_step=0.4s, eta=2:8:20
    2021-08-11 14:44:34 [INFO]	[TRAIN] Epoch=2/270, Step=40/74, loss=41.396381, lr=1.4e-05, time_each_step=0.39s, eta=2:8:14
    2021-08-11 14:44:35 [INFO]	[TRAIN] Epoch=2/270, Step=42/74, loss=77.004822, lr=1.4e-05, time_each_step=0.36s, eta=2:8:5
    2021-08-11 14:44:35 [INFO]	[TRAIN] Epoch=2/270, Step=44/74, loss=22.742336, lr=1.5e-05, time_each_step=0.34s, eta=2:7:56
    2021-08-11 14:44:35 [INFO]	[TRAIN] Epoch=2/270, Step=46/74, loss=121.994659, lr=1.5e-05, time_each_step=0.32s, eta=2:7:48
    2021-08-11 14:44:36 [INFO]	[TRAIN] Epoch=2/270, Step=48/74, loss=33.847248, lr=1.5e-05, time_each_step=0.3s, eta=2:7:42
    2021-08-11 14:44:36 [INFO]	[TRAIN] Epoch=2/270, Step=50/74, loss=32.663006, lr=1.5e-05, time_each_step=0.27s, eta=2:7:30
    2021-08-11 14:44:36 [INFO]	[TRAIN] Epoch=2/270, Step=52/74, loss=59.22818, lr=1.6e-05, time_each_step=0.24s, eta=2:7:20
    2021-08-11 14:44:37 [INFO]	[TRAIN] Epoch=2/270, Step=54/74, loss=37.058487, lr=1.6e-05, time_each_step=0.21s, eta=2:7:10
    2021-08-11 14:44:37 [INFO]	[TRAIN] Epoch=2/270, Step=56/74, loss=76.560539, lr=1.6e-05, time_each_step=0.19s, eta=2:7:5
    2021-08-11 14:44:37 [INFO]	[TRAIN] Epoch=2/270, Step=58/74, loss=60.508751, lr=1.6e-05, time_each_step=0.19s, eta=2:7:3
    2021-08-11 14:44:38 [INFO]	[TRAIN] Epoch=2/270, Step=60/74, loss=59.038124, lr=1.7e-05, time_each_step=0.18s, eta=2:6:58
    2021-08-11 14:44:38 [INFO]	[TRAIN] Epoch=2/270, Step=62/74, loss=49.616859, lr=1.7e-05, time_each_step=0.17s, eta=2:6:56
    2021-08-11 14:44:39 [INFO]	[TRAIN] Epoch=2/270, Step=64/74, loss=60.330517, lr=1.7e-05, time_each_step=0.18s, eta=2:7:0
    2021-08-11 14:44:39 [INFO]	[TRAIN] Epoch=2/270, Step=66/74, loss=58.362625, lr=1.7e-05, time_each_step=0.19s, eta=2:7:2
    2021-08-11 14:44:39 [INFO]	[TRAIN] Epoch=2/270, Step=68/74, loss=51.179668, lr=1.8e-05, time_each_step=0.2s, eta=2:7:3
    2021-08-11 14:44:40 [INFO]	[TRAIN] Epoch=2/270, Step=70/74, loss=43.426147, lr=1.8e-05, time_each_step=0.2s, eta=2:7:4
    2021-08-11 14:44:40 [INFO]	[TRAIN] Epoch=2/270, Step=72/74, loss=36.550461, lr=1.8e-05, time_each_step=0.21s, eta=2:7:6
    2021-08-11 14:44:41 [INFO]	[TRAIN] Epoch=2/270, Step=74/74, loss=60.121029, lr=1.8e-05, time_each_step=0.21s, eta=2:7:8
    2021-08-11 14:44:41 [INFO]	[TRAIN] Epoch 2 finished, loss=58.631138, lr=1.4e-05 .
    2021-08-11 14:44:46 [INFO]	[TRAIN] Epoch=3/270, Step=2/74, loss=57.895691, lr=1.9e-05, time_each_step=0.45s, eta=1:55:58
    2021-08-11 14:44:47 [INFO]	[TRAIN] Epoch=3/270, Step=4/74, loss=32.310928, lr=1.9e-05, time_each_step=0.46s, eta=1:56:0
    2021-08-11 14:44:47 [INFO]	[TRAIN] Epoch=3/270, Step=6/74, loss=58.231457, lr=1.9e-05, time_each_step=0.48s, eta=1:56:8
    2021-08-11 14:44:48 [INFO]	[TRAIN] Epoch=3/270, Step=8/74, loss=50.934246, lr=1.9e-05, time_each_step=0.52s, eta=1:56:20
    2021-08-11 14:44:49 [INFO]	[TRAIN] Epoch=3/270, Step=10/74, loss=109.711952, lr=2e-05, time_each_step=0.54s, eta=1:56:27
    2021-08-11 14:44:50 [INFO]	[TRAIN] Epoch=3/270, Step=12/74, loss=44.132351, lr=2e-05, time_each_step=0.56s, eta=1:56:34
    2021-08-11 14:44:51 [INFO]	[TRAIN] Epoch=3/270, Step=14/74, loss=53.319176, lr=2e-05, time_each_step=0.59s, eta=1:56:44
    2021-08-11 14:44:52 [INFO]	[TRAIN] Epoch=3/270, Step=16/74, loss=49.972969, lr=2e-05, time_each_step=0.61s, eta=1:56:51
    2021-08-11 14:44:53 [INFO]	[TRAIN] Epoch=3/270, Step=18/74, loss=58.468678, lr=2.1e-05, time_each_step=0.61s, eta=1:56:50
    2021-08-11 14:44:53 [INFO]	[TRAIN] Epoch=3/270, Step=20/74, loss=62.700485, lr=2.1e-05, time_each_step=0.63s, eta=1:56:55
    2021-08-11 14:44:54 [INFO]	[TRAIN] Epoch=3/270, Step=22/74, loss=35.567478, lr=2.1e-05, time_each_step=0.41s, eta=1:55:34
    2021-08-11 14:44:55 [INFO]	[TRAIN] Epoch=3/270, Step=24/74, loss=42.68235, lr=2.1e-05, time_each_step=0.42s, eta=1:55:38
    2021-08-11 14:44:56 [INFO]	[TRAIN] Epoch=3/270, Step=26/74, loss=48.657825, lr=2.2e-05, time_each_step=0.41s, eta=1:55:33
    2021-08-11 14:44:57 [INFO]	[TRAIN] Epoch=3/270, Step=28/74, loss=50.484474, lr=2.2e-05, time_each_step=0.41s, eta=1:55:31
    2021-08-11 14:44:57 [INFO]	[TRAIN] Epoch=3/270, Step=30/74, loss=35.080395, lr=2.2e-05, time_each_step=0.4s, eta=1:55:27
    2021-08-11 14:44:58 [INFO]	[TRAIN] Epoch=3/270, Step=32/74, loss=54.04628, lr=2.2e-05, time_each_step=0.4s, eta=1:55:26
    2021-08-11 14:44:59 [INFO]	[TRAIN] Epoch=3/270, Step=34/74, loss=51.664825, lr=2.3e-05, time_each_step=0.4s, eta=1:55:24
    2021-08-11 14:45:00 [INFO]	[TRAIN] Epoch=3/270, Step=36/74, loss=51.325676, lr=2.3e-05, time_each_step=0.4s, eta=1:55:26
    2021-08-11 14:45:01 [INFO]	[TRAIN] Epoch=3/270, Step=38/74, loss=37.564884, lr=2.3e-05, time_each_step=0.41s, eta=1:55:29
    2021-08-11 14:45:02 [INFO]	[TRAIN] Epoch=3/270, Step=40/74, loss=52.594509, lr=2.3e-05, time_each_step=0.41s, eta=1:55:26
    2021-08-11 14:45:02 [INFO]	[TRAIN] Epoch=3/270, Step=42/74, loss=40.385563, lr=2.4e-05, time_each_step=0.4s, eta=1:55:21
    2021-08-11 14:45:03 [INFO]	[TRAIN] Epoch=3/270, Step=44/74, loss=31.101995, lr=2.4e-05, time_each_step=0.38s, eta=1:55:14
    2021-08-11 14:45:03 [INFO]	[TRAIN] Epoch=3/270, Step=46/74, loss=49.836739, lr=2.4e-05, time_each_step=0.37s, eta=1:55:10
    2021-08-11 14:45:03 [INFO]	[TRAIN] Epoch=3/270, Step=48/74, loss=35.4011, lr=2.4e-05, time_each_step=0.33s, eta=1:54:57
    2021-08-11 14:45:03 [INFO]	[TRAIN] Epoch=3/270, Step=50/74, loss=31.788248, lr=2.5e-05, time_each_step=0.31s, eta=1:54:48
    2021-08-11 14:45:04 [INFO]	[TRAIN] Epoch=3/270, Step=52/74, loss=66.4039, lr=2.5e-05, time_each_step=0.28s, eta=1:54:39
    2021-08-11 14:45:04 [INFO]	[TRAIN] Epoch=3/270, Step=54/74, loss=42.99749, lr=2.5e-05, time_each_step=0.25s, eta=1:54:28
    2021-08-11 14:45:05 [INFO]	[TRAIN] Epoch=3/270, Step=56/74, loss=36.415039, lr=2.5e-05, time_each_step=0.21s, eta=1:54:16
    2021-08-11 14:45:05 [INFO]	[TRAIN] Epoch=3/270, Step=58/74, loss=52.390247, lr=2.6e-05, time_each_step=0.2s, eta=1:54:11
    2021-08-11 14:45:05 [INFO]	[TRAIN] Epoch=3/270, Step=60/74, loss=26.578602, lr=2.6e-05, time_each_step=0.19s, eta=1:54:7
    2021-08-11 14:45:06 [INFO]	[TRAIN] Epoch=3/270, Step=62/74, loss=43.236897, lr=2.6e-05, time_each_step=0.18s, eta=1:54:5
    2021-08-11 14:45:06 [INFO]	[TRAIN] Epoch=3/270, Step=64/74, loss=72.259155, lr=2.6e-05, time_each_step=0.18s, eta=1:54:4
    2021-08-11 14:45:07 [INFO]	[TRAIN] Epoch=3/270, Step=66/74, loss=53.341053, lr=2.7e-05, time_each_step=0.2s, eta=1:54:8
    2021-08-11 14:45:07 [INFO]	[TRAIN] Epoch=3/270, Step=68/74, loss=118.488907, lr=2.7e-05, time_each_step=0.22s, eta=1:54:14
    2021-08-11 14:45:08 [INFO]	[TRAIN] Epoch=3/270, Step=70/74, loss=32.338326, lr=2.7e-05, time_each_step=0.23s, eta=1:54:17
    2021-08-11 14:45:08 [INFO]	[TRAIN] Epoch=3/270, Step=72/74, loss=45.970261, lr=2.7e-05, time_each_step=0.22s, eta=1:54:15
    2021-08-11 14:45:09 [INFO]	[TRAIN] Epoch=3/270, Step=74/74, loss=41.968788, lr=2.8e-05, time_each_step=0.22s, eta=1:54:14
    2021-08-11 14:45:09 [INFO]	[TRAIN] Epoch 3 finished, loss=51.567272, lr=2.3e-05 .
    2021-08-11 14:45:13 [INFO]	[TRAIN] Epoch=4/270, Step=2/74, loss=52.670967, lr=2.8e-05, time_each_step=0.44s, eta=2:6:3
    2021-08-11 14:45:14 [INFO]	[TRAIN] Epoch=4/270, Step=4/74, loss=53.593773, lr=2.8e-05, time_each_step=0.46s, eta=2:6:11
    2021-08-11 14:45:15 [INFO]	[TRAIN] Epoch=4/270, Step=6/74, loss=47.603634, lr=2.8e-05, time_each_step=0.49s, eta=2:6:19
    2021-08-11 14:45:16 [INFO]	[TRAIN] Epoch=4/270, Step=8/74, loss=55.723846, lr=2.9e-05, time_each_step=0.5s, eta=2:6:22
    2021-08-11 14:45:16 [INFO]	[TRAIN] Epoch=4/270, Step=10/74, loss=39.095528, lr=2.9e-05, time_each_step=0.51s, eta=2:6:24
    2021-08-11 14:45:17 [INFO]	[TRAIN] Epoch=4/270, Step=12/74, loss=42.579521, lr=2.9e-05, time_each_step=0.52s, eta=2:6:27
    2021-08-11 14:45:18 [INFO]	[TRAIN] Epoch=4/270, Step=14/74, loss=48.729294, lr=2.9e-05, time_each_step=0.52s, eta=2:6:27
    2021-08-11 14:45:19 [INFO]	[TRAIN] Epoch=4/270, Step=16/74, loss=36.745918, lr=3e-05, time_each_step=0.55s, eta=2:6:36
    2021-08-11 14:45:20 [INFO]	[TRAIN] Epoch=4/270, Step=18/74, loss=24.420815, lr=3e-05, time_each_step=0.57s, eta=2:6:44
    2021-08-11 14:45:20 [INFO]	[TRAIN] Epoch=4/270, Step=20/74, loss=55.524315, lr=3e-05, time_each_step=0.59s, eta=2:6:47
    2021-08-11 14:45:21 [INFO]	[TRAIN] Epoch=4/270, Step=22/74, loss=61.28754, lr=3e-05, time_each_step=0.39s, eta=2:5:38
    2021-08-11 14:45:22 [INFO]	[TRAIN] Epoch=4/270, Step=24/74, loss=46.974072, lr=3.1e-05, time_each_step=0.39s, eta=2:5:37
    2021-08-11 14:45:23 [INFO]	[TRAIN] Epoch=4/270, Step=26/74, loss=54.782036, lr=3.1e-05, time_each_step=0.37s, eta=2:5:29
    2021-08-11 14:45:23 [INFO]	[TRAIN] Epoch=4/270, Step=28/74, loss=27.610533, lr=3.1e-05, time_each_step=0.38s, eta=2:5:31
    2021-08-11 14:45:24 [INFO]	[TRAIN] Epoch=4/270, Step=30/74, loss=63.515823, lr=3.1e-05, time_each_step=0.39s, eta=2:5:32
    2021-08-11 14:45:25 [INFO]	[TRAIN] Epoch=4/270, Step=32/74, loss=42.333115, lr=3.2e-05, time_each_step=0.39s, eta=2:5:30
    2021-08-11 14:45:26 [INFO]	[TRAIN] Epoch=4/270, Step=34/74, loss=28.415445, lr=3.2e-05, time_each_step=0.38s, eta=2:5:28
    2021-08-11 14:45:26 [INFO]	[TRAIN] Epoch=4/270, Step=36/74, loss=33.144051, lr=3.2e-05, time_each_step=0.35s, eta=2:5:16
    2021-08-11 14:45:26 [INFO]	[TRAIN] Epoch=4/270, Step=38/74, loss=47.511482, lr=3.2e-05, time_each_step=0.33s, eta=2:5:8
    2021-08-11 14:45:27 [INFO]	[TRAIN] Epoch=4/270, Step=40/74, loss=48.250042, lr=3.3e-05, time_each_step=0.32s, eta=2:5:5
    2021-08-11 14:45:28 [INFO]	[TRAIN] Epoch=4/270, Step=42/74, loss=35.860065, lr=3.3e-05, time_each_step=0.34s, eta=2:5:9
    2021-08-11 14:45:29 [INFO]	[TRAIN] Epoch=4/270, Step=44/74, loss=44.266987, lr=3.3e-05, time_each_step=0.35s, eta=2:5:13
    2021-08-11 14:45:30 [INFO]	[TRAIN] Epoch=4/270, Step=46/74, loss=41.342575, lr=3.3e-05, time_each_step=0.36s, eta=2:5:17
    2021-08-11 14:45:30 [INFO]	[TRAIN] Epoch=4/270, Step=48/74, loss=50.899918, lr=3.4e-05, time_each_step=0.35s, eta=2:5:12
    2021-08-11 14:45:31 [INFO]	[TRAIN] Epoch=4/270, Step=50/74, loss=35.425102, lr=3.4e-05, time_each_step=0.34s, eta=2:5:9
    2021-08-11 14:45:31 [INFO]	[TRAIN] Epoch=4/270, Step=52/74, loss=53.756256, lr=3.4e-05, time_each_step=0.31s, eta=2:4:59
    2021-08-11 14:45:32 [INFO]	[TRAIN] Epoch=4/270, Step=54/74, loss=49.557606, lr=3.4e-05, time_each_step=0.3s, eta=2:4:54
    2021-08-11 14:45:32 [INFO]	[TRAIN] Epoch=4/270, Step=56/74, loss=61.459229, lr=3.5e-05, time_each_step=0.3s, eta=2:4:53
    2021-08-11 14:45:32 [INFO]	[TRAIN] Epoch=4/270, Step=58/74, loss=38.402, lr=3.5e-05, time_each_step=0.31s, eta=2:4:56
    2021-08-11 14:45:33 [INFO]	[TRAIN] Epoch=4/270, Step=60/74, loss=71.505219, lr=3.5e-05, time_each_step=0.31s, eta=2:4:55
    2021-08-11 14:45:33 [INFO]	[TRAIN] Epoch=4/270, Step=62/74, loss=33.65419, lr=3.5e-05, time_each_step=0.28s, eta=2:4:44
    2021-08-11 14:45:34 [INFO]	[TRAIN] Epoch=4/270, Step=64/74, loss=37.138187, lr=3.6e-05, time_each_step=0.26s, eta=2:4:38
    2021-08-11 14:45:35 [INFO]	[TRAIN] Epoch=4/270, Step=66/74, loss=56.960114, lr=3.6e-05, time_each_step=0.24s, eta=2:4:31
    2021-08-11 14:45:35 [INFO]	[TRAIN] Epoch=4/270, Step=68/74, loss=29.319147, lr=3.6e-05, time_each_step=0.24s, eta=2:4:32
    2021-08-11 14:45:36 [INFO]	[TRAIN] Epoch=4/270, Step=70/74, loss=28.044542, lr=3.6e-05, time_each_step=0.23s, eta=2:4:27
    2021-08-11 14:45:36 [INFO]	[TRAIN] Epoch=4/270, Step=72/74, loss=28.936241, lr=3.7e-05, time_each_step=0.25s, eta=2:4:32
    2021-08-11 14:45:37 [INFO]	[TRAIN] Epoch=4/270, Step=74/74, loss=52.288506, lr=3.7e-05, time_each_step=0.25s, eta=2:4:33
    2021-08-11 14:45:37 [INFO]	[TRAIN] Epoch 4 finished, loss=47.213413, lr=3.2e-05 .
    2021-08-11 14:45:45 [INFO]	[TRAIN] Epoch=5/270, Step=2/74, loss=55.845367, lr=3.7e-05, time_each_step=0.63s, eta=2:7:33
    2021-08-11 14:45:46 [INFO]	[TRAIN] Epoch=5/270, Step=4/74, loss=26.372948, lr=3.7e-05, time_each_step=0.65s, eta=2:7:40
    2021-08-11 14:45:46 [INFO]	[TRAIN] Epoch=5/270, Step=6/74, loss=28.596756, lr=3.8e-05, time_each_step=0.67s, eta=2:7:46
    2021-08-11 14:45:47 [INFO]	[TRAIN] Epoch=5/270, Step=8/74, loss=37.340927, lr=3.8e-05, time_each_step=0.7s, eta=2:7:55
    2021-08-11 14:45:48 [INFO]	[TRAIN] Epoch=5/270, Step=10/74, loss=69.335114, lr=3.8e-05, time_each_step=0.7s, eta=2:7:54
    2021-08-11 14:45:49 [INFO]	[TRAIN] Epoch=5/270, Step=12/74, loss=40.826302, lr=3.8e-05, time_each_step=0.73s, eta=2:8:2
    2021-08-11 14:45:50 [INFO]	[TRAIN] Epoch=5/270, Step=14/74, loss=31.477718, lr=3.9e-05, time_each_step=0.75s, eta=2:8:8
    2021-08-11 14:45:51 [INFO]	[TRAIN] Epoch=5/270, Step=16/74, loss=10.141375, lr=3.9e-05, time_each_step=0.77s, eta=2:8:14
    2021-08-11 14:45:52 [INFO]	[TRAIN] Epoch=5/270, Step=18/74, loss=33.647568, lr=3.9e-05, time_each_step=0.78s, eta=2:8:17
    2021-08-11 14:45:52 [INFO]	[TRAIN] Epoch=5/270, Step=20/74, loss=42.849834, lr=3.9e-05, time_each_step=0.79s, eta=2:8:21
    2021-08-11 14:45:53 [INFO]	[TRAIN] Epoch=5/270, Step=22/74, loss=58.8106, lr=4e-05, time_each_step=0.44s, eta=2:6:12
    2021-08-11 14:45:54 [INFO]	[TRAIN] Epoch=5/270, Step=24/74, loss=37.618118, lr=4e-05, time_each_step=0.42s, eta=2:6:4
    2021-08-11 14:45:55 [INFO]	[TRAIN] Epoch=5/270, Step=26/74, loss=66.488136, lr=4e-05, time_each_step=0.42s, eta=2:6:1
    2021-08-11 14:45:55 [INFO]	[TRAIN] Epoch=5/270, Step=28/74, loss=45.538273, lr=4e-05, time_each_step=0.4s, eta=2:5:55
    2021-08-11 14:45:56 [INFO]	[TRAIN] Epoch=5/270, Step=30/74, loss=35.960457, lr=4.1e-05, time_each_step=0.4s, eta=2:5:56
    2021-08-11 14:45:57 [INFO]	[TRAIN] Epoch=5/270, Step=32/74, loss=45.083977, lr=4.1e-05, time_each_step=0.38s, eta=2:5:46
    2021-08-11 14:45:57 [INFO]	[TRAIN] Epoch=5/270, Step=34/74, loss=31.365946, lr=4.1e-05, time_each_step=0.36s, eta=2:5:40
    2021-08-11 14:45:58 [INFO]	[TRAIN] Epoch=5/270, Step=36/74, loss=36.854828, lr=4.1e-05, time_each_step=0.38s, eta=2:5:45
    2021-08-11 14:45:59 [INFO]	[TRAIN] Epoch=5/270, Step=38/74, loss=37.309158, lr=4.2e-05, time_each_step=0.37s, eta=2:5:40
    2021-08-11 14:46:00 [INFO]	[TRAIN] Epoch=5/270, Step=40/74, loss=53.365814, lr=4.2e-05, time_each_step=0.37s, eta=2:5:40
    2021-08-11 14:46:00 [INFO]	[TRAIN] Epoch=5/270, Step=42/74, loss=24.750828, lr=4.2e-05, time_each_step=0.35s, eta=2:5:34
    2021-08-11 14:46:01 [INFO]	[TRAIN] Epoch=5/270, Step=44/74, loss=48.942436, lr=4.2e-05, time_each_step=0.35s, eta=2:5:33
    2021-08-11 14:46:02 [INFO]	[TRAIN] Epoch=5/270, Step=46/74, loss=23.199919, lr=4.3e-05, time_each_step=0.35s, eta=2:5:29
    2021-08-11 14:46:02 [INFO]	[TRAIN] Epoch=5/270, Step=48/74, loss=46.425385, lr=4.3e-05, time_each_step=0.33s, eta=2:5:22
    2021-08-11 14:46:02 [INFO]	[TRAIN] Epoch=5/270, Step=50/74, loss=37.631622, lr=4.3e-05, time_each_step=0.31s, eta=2:5:17
    2021-08-11 14:46:03 [INFO]	[TRAIN] Epoch=5/270, Step=52/74, loss=33.56744, lr=4.3e-05, time_each_step=0.32s, eta=2:5:18
    2021-08-11 14:46:04 [INFO]	[TRAIN] Epoch=5/270, Step=54/74, loss=85.179565, lr=4.4e-05, time_each_step=0.31s, eta=2:5:15
    2021-08-11 14:46:04 [INFO]	[TRAIN] Epoch=5/270, Step=56/74, loss=29.424147, lr=4.4e-05, time_each_step=0.29s, eta=2:5:6
    2021-08-11 14:46:04 [INFO]	[TRAIN] Epoch=5/270, Step=58/74, loss=44.626442, lr=4.4e-05, time_each_step=0.27s, eta=2:5:1
    2021-08-11 14:46:05 [INFO]	[TRAIN] Epoch=5/270, Step=60/74, loss=29.596378, lr=4.4e-05, time_each_step=0.26s, eta=2:4:57
    2021-08-11 14:46:05 [INFO]	[TRAIN] Epoch=5/270, Step=62/74, loss=34.111408, lr=4.5e-05, time_each_step=0.25s, eta=2:4:54
    2021-08-11 14:46:06 [INFO]	[TRAIN] Epoch=5/270, Step=64/74, loss=29.249613, lr=4.5e-05, time_each_step=0.23s, eta=2:4:48
    2021-08-11 14:46:06 [INFO]	[TRAIN] Epoch=5/270, Step=66/74, loss=50.677132, lr=4.5e-05, time_each_step=0.23s, eta=2:4:47
    2021-08-11 14:46:07 [INFO]	[TRAIN] Epoch=5/270, Step=68/74, loss=46.478043, lr=4.5e-05, time_each_step=0.24s, eta=2:4:48
    2021-08-11 14:46:07 [INFO]	[TRAIN] Epoch=5/270, Step=70/74, loss=48.585087, lr=4.6e-05, time_each_step=0.24s, eta=2:4:49
    2021-08-11 14:46:08 [INFO]	[TRAIN] Epoch=5/270, Step=72/74, loss=34.075165, lr=4.6e-05, time_each_step=0.23s, eta=2:4:45
    2021-08-11 14:46:08 [INFO]	[TRAIN] Epoch=5/270, Step=74/74, loss=42.624599, lr=4.6e-05, time_each_step=0.22s, eta=2:4:41
    2021-08-11 14:46:08 [INFO]	[TRAIN] Epoch 5 finished, loss=42.73925, lr=4.2e-05 .
    2021-08-11 14:46:13 [INFO]	[TRAIN] Epoch=6/270, Step=2/74, loss=24.583963, lr=4.6e-05, time_each_step=0.46s, eta=2:21:18
    2021-08-11 14:46:14 [INFO]	[TRAIN] Epoch=6/270, Step=4/74, loss=50.135056, lr=4.7e-05, time_each_step=0.47s, eta=2:21:22
    2021-08-11 14:46:15 [INFO]	[TRAIN] Epoch=6/270, Step=6/74, loss=29.847364, lr=4.7e-05, time_each_step=0.48s, eta=2:21:25
    2021-08-11 14:46:16 [INFO]	[TRAIN] Epoch=6/270, Step=8/74, loss=62.956921, lr=4.7e-05, time_each_step=0.5s, eta=2:21:32
    2021-08-11 14:46:16 [INFO]	[TRAIN] Epoch=6/270, Step=10/74, loss=42.594017, lr=4.7e-05, time_each_step=0.53s, eta=2:21:41
    2021-08-11 14:46:17 [INFO]	[TRAIN] Epoch=6/270, Step=12/74, loss=47.185272, lr=4.8e-05, time_each_step=0.54s, eta=2:21:44
    2021-08-11 14:46:18 [INFO]	[TRAIN] Epoch=6/270, Step=14/74, loss=22.139139, lr=4.8e-05, time_each_step=0.56s, eta=2:21:50
    2021-08-11 14:46:19 [INFO]	[TRAIN] Epoch=6/270, Step=16/74, loss=68.419044, lr=4.8e-05, time_each_step=0.56s, eta=2:21:50
    2021-08-11 14:46:20 [INFO]	[TRAIN] Epoch=6/270, Step=18/74, loss=30.618046, lr=4.8e-05, time_each_step=0.59s, eta=2:21:59
    2021-08-11 14:46:20 [INFO]	[TRAIN] Epoch=6/270, Step=20/74, loss=44.433716, lr=4.9e-05, time_each_step=0.6s, eta=2:22:1
    2021-08-11 14:46:21 [INFO]	[TRAIN] Epoch=6/270, Step=22/74, loss=37.793739, lr=4.9e-05, time_each_step=0.37s, eta=2:20:36
    2021-08-11 14:46:21 [INFO]	[TRAIN] Epoch=6/270, Step=24/74, loss=41.862278, lr=4.9e-05, time_each_step=0.37s, eta=2:20:36
    2021-08-11 14:46:22 [INFO]	[TRAIN] Epoch=6/270, Step=26/74, loss=60.305984, lr=4.9e-05, time_each_step=0.37s, eta=2:20:34
    2021-08-11 14:46:23 [INFO]	[TRAIN] Epoch=6/270, Step=28/74, loss=33.057613, lr=5e-05, time_each_step=0.38s, eta=2:20:37
    2021-08-11 14:46:24 [INFO]	[TRAIN] Epoch=6/270, Step=30/74, loss=29.729515, lr=5e-05, time_each_step=0.38s, eta=2:20:38
    2021-08-11 14:46:25 [INFO]	[TRAIN] Epoch=6/270, Step=32/74, loss=47.514259, lr=5e-05, time_each_step=0.39s, eta=2:20:42
    2021-08-11 14:46:26 [INFO]	[TRAIN] Epoch=6/270, Step=34/74, loss=45.578339, lr=5e-05, time_each_step=0.4s, eta=2:20:41
    2021-08-11 14:46:27 [INFO]	[TRAIN] Epoch=6/270, Step=36/74, loss=32.71579, lr=5.1e-05, time_each_step=0.4s, eta=2:20:40
    2021-08-11 14:46:28 [INFO]	[TRAIN] Epoch=6/270, Step=38/74, loss=32.726936, lr=5.1e-05, time_each_step=0.41s, eta=2:20:44
    2021-08-11 14:46:28 [INFO]	[TRAIN] Epoch=6/270, Step=40/74, loss=39.391155, lr=5.1e-05, time_each_step=0.41s, eta=2:20:42
    2021-08-11 14:46:29 [INFO]	[TRAIN] Epoch=6/270, Step=42/74, loss=49.354912, lr=5.1e-05, time_each_step=0.4s, eta=2:20:39
    2021-08-11 14:46:29 [INFO]	[TRAIN] Epoch=6/270, Step=44/74, loss=33.878281, lr=5.2e-05, time_each_step=0.39s, eta=2:20:35
    2021-08-11 14:46:29 [INFO]	[TRAIN] Epoch=6/270, Step=46/74, loss=47.626358, lr=5.2e-05, time_each_step=0.37s, eta=2:20:28
    2021-08-11 14:46:30 [INFO]	[TRAIN] Epoch=6/270, Step=48/74, loss=17.750969, lr=5.2e-05, time_each_step=0.33s, eta=2:20:14
    2021-08-11 14:46:30 [INFO]	[TRAIN] Epoch=6/270, Step=50/74, loss=92.052635, lr=5.2e-05, time_each_step=0.3s, eta=2:20:4
    2021-08-11 14:46:30 [INFO]	[TRAIN] Epoch=6/270, Step=52/74, loss=31.757599, lr=5.3e-05, time_each_step=0.28s, eta=2:19:55
    2021-08-11 14:46:31 [INFO]	[TRAIN] Epoch=6/270, Step=54/74, loss=47.448963, lr=5.3e-05, time_each_step=0.25s, eta=2:19:44
    2021-08-11 14:46:31 [INFO]	[TRAIN] Epoch=6/270, Step=56/74, loss=44.638214, lr=5.3e-05, time_each_step=0.23s, eta=2:19:38
    2021-08-11 14:46:32 [INFO]	[TRAIN] Epoch=6/270, Step=58/74, loss=43.665443, lr=5.3e-05, time_each_step=0.19s, eta=2:19:26
    2021-08-11 14:46:32 [INFO]	[TRAIN] Epoch=6/270, Step=60/74, loss=44.137993, lr=5.4e-05, time_each_step=0.2s, eta=2:19:28
    2021-08-11 14:46:33 [INFO]	[TRAIN] Epoch=6/270, Step=62/74, loss=42.466278, lr=5.4e-05, time_each_step=0.2s, eta=2:19:26
    2021-08-11 14:46:33 [INFO]	[TRAIN] Epoch=6/270, Step=64/74, loss=37.050007, lr=5.4e-05, time_each_step=0.2s, eta=2:19:26
    2021-08-11 14:46:33 [INFO]	[TRAIN] Epoch=6/270, Step=66/74, loss=32.0396, lr=5.4e-05, time_each_step=0.2s, eta=2:19:26
    2021-08-11 14:46:34 [INFO]	[TRAIN] Epoch=6/270, Step=68/74, loss=75.84005, lr=5.5e-05, time_each_step=0.21s, eta=2:19:31
    2021-08-11 14:46:34 [INFO]	[TRAIN] Epoch=6/270, Step=70/74, loss=33.08519, lr=5.5e-05, time_each_step=0.22s, eta=2:19:31
    2021-08-11 14:46:35 [INFO]	[TRAIN] Epoch=6/270, Step=72/74, loss=34.998352, lr=5.5e-05, time_each_step=0.22s, eta=2:19:33
    2021-08-11 14:46:35 [INFO]	[TRAIN] Epoch=6/270, Step=74/74, loss=20.746492, lr=5.5e-05, time_each_step=0.24s, eta=2:19:36
    2021-08-11 14:46:35 [INFO]	[TRAIN] Epoch 6 finished, loss=41.710644, lr=5.1e-05 .
    2021-08-11 14:46:47 [INFO]	[TRAIN] Epoch=7/270, Step=2/74, loss=24.431557, lr=5.6e-05, time_each_step=0.79s, eta=2:5:19
    2021-08-11 14:46:48 [INFO]	[TRAIN] Epoch=7/270, Step=4/74, loss=27.373938, lr=5.6e-05, time_each_step=0.81s, eta=2:5:28
    2021-08-11 14:46:49 [INFO]	[TRAIN] Epoch=7/270, Step=6/74, loss=43.91333, lr=5.6e-05, time_each_step=0.84s, eta=2:5:37
    2021-08-11 14:46:50 [INFO]	[TRAIN] Epoch=7/270, Step=8/74, loss=55.010845, lr=5.6e-05, time_each_step=0.86s, eta=2:5:41
    2021-08-11 14:46:51 [INFO]	[TRAIN] Epoch=7/270, Step=10/74, loss=32.787434, lr=5.7e-05, time_each_step=0.88s, eta=2:5:49
    2021-08-11 14:46:52 [INFO]	[TRAIN] Epoch=7/270, Step=12/74, loss=55.349308, lr=5.7e-05, time_each_step=0.91s, eta=2:5:57
    2021-08-11 14:46:52 [INFO]	[TRAIN] Epoch=7/270, Step=14/74, loss=21.595016, lr=5.7e-05, time_each_step=0.92s, eta=2:5:58
    2021-08-11 14:46:53 [INFO]	[TRAIN] Epoch=7/270, Step=16/74, loss=44.866554, lr=5.7e-05, time_each_step=0.94s, eta=2:6:4
    2021-08-11 14:46:54 [INFO]	[TRAIN] Epoch=7/270, Step=18/74, loss=25.460064, lr=5.8e-05, time_each_step=0.96s, eta=2:6:10
    2021-08-11 14:46:55 [INFO]	[TRAIN] Epoch=7/270, Step=20/74, loss=28.4415, lr=5.8e-05, time_each_step=0.97s, eta=2:6:13
    2021-08-11 14:46:56 [INFO]	[TRAIN] Epoch=7/270, Step=22/74, loss=32.650452, lr=5.8e-05, time_each_step=0.45s, eta=2:3:5
    2021-08-11 14:46:57 [INFO]	[TRAIN] Epoch=7/270, Step=24/74, loss=55.410591, lr=5.8e-05, time_each_step=0.44s, eta=2:2:58
    2021-08-11 14:46:57 [INFO]	[TRAIN] Epoch=7/270, Step=26/74, loss=39.606968, lr=5.9e-05, time_each_step=0.4s, eta=2:2:45
    2021-08-11 14:46:58 [INFO]	[TRAIN] Epoch=7/270, Step=28/74, loss=39.905304, lr=5.9e-05, time_each_step=0.42s, eta=2:2:51
    2021-08-11 14:46:59 [INFO]	[TRAIN] Epoch=7/270, Step=30/74, loss=58.040146, lr=5.9e-05, time_each_step=0.42s, eta=2:2:50
    2021-08-11 14:47:00 [INFO]	[TRAIN] Epoch=7/270, Step=32/74, loss=30.255444, lr=5.9e-05, time_each_step=0.43s, eta=2:2:52
    2021-08-11 14:47:01 [INFO]	[TRAIN] Epoch=7/270, Step=34/74, loss=59.359734, lr=6e-05, time_each_step=0.43s, eta=2:2:52
    2021-08-11 14:47:02 [INFO]	[TRAIN] Epoch=7/270, Step=36/74, loss=25.888134, lr=6e-05, time_each_step=0.44s, eta=2:2:51
    2021-08-11 14:47:02 [INFO]	[TRAIN] Epoch=7/270, Step=38/74, loss=47.966743, lr=6e-05, time_each_step=0.42s, eta=2:2:44
    2021-08-11 14:47:03 [INFO]	[TRAIN] Epoch=7/270, Step=40/74, loss=28.811333, lr=6e-05, time_each_step=0.41s, eta=2:2:41
    2021-08-11 14:47:04 [INFO]	[TRAIN] Epoch=7/270, Step=42/74, loss=23.067989, lr=6.1e-05, time_each_step=0.39s, eta=2:2:34
    2021-08-11 14:47:04 [INFO]	[TRAIN] Epoch=7/270, Step=44/74, loss=32.915848, lr=6.1e-05, time_each_step=0.37s, eta=2:2:27
    2021-08-11 14:47:04 [INFO]	[TRAIN] Epoch=7/270, Step=46/74, loss=27.785446, lr=6.1e-05, time_each_step=0.37s, eta=2:2:25
    2021-08-11 14:47:05 [INFO]	[TRAIN] Epoch=7/270, Step=48/74, loss=30.824408, lr=6.1e-05, time_each_step=0.33s, eta=2:2:11
    2021-08-11 14:47:05 [INFO]	[TRAIN] Epoch=7/270, Step=50/74, loss=50.537346, lr=6.2e-05, time_each_step=0.3s, eta=2:2:1
    2021-08-11 14:47:06 [INFO]	[TRAIN] Epoch=7/270, Step=52/74, loss=37.663811, lr=6.2e-05, time_each_step=0.27s, eta=2:1:51
    2021-08-11 14:47:06 [INFO]	[TRAIN] Epoch=7/270, Step=54/74, loss=47.334862, lr=6.2e-05, time_each_step=0.25s, eta=2:1:44
    2021-08-11 14:47:07 [INFO]	[TRAIN] Epoch=7/270, Step=56/74, loss=64.124847, lr=6.2e-05, time_each_step=0.23s, eta=2:1:37
    2021-08-11 14:47:07 [INFO]	[TRAIN] Epoch=7/270, Step=58/74, loss=33.322697, lr=6.3e-05, time_each_step=0.23s, eta=2:1:35
    2021-08-11 14:47:07 [INFO]	[TRAIN] Epoch=7/270, Step=60/74, loss=36.075054, lr=6.3e-05, time_each_step=0.21s, eta=2:1:28
    2021-08-11 14:47:08 [INFO]	[TRAIN] Epoch=7/270, Step=62/74, loss=33.472103, lr=6.3e-05, time_each_step=0.19s, eta=2:1:22
    2021-08-11 14:47:08 [INFO]	[TRAIN] Epoch=7/270, Step=64/74, loss=41.988152, lr=6.3e-05, time_each_step=0.19s, eta=2:1:22
    2021-08-11 14:47:08 [INFO]	[TRAIN] Epoch=7/270, Step=66/74, loss=26.589249, lr=6.4e-05, time_each_step=0.19s, eta=2:1:20
    2021-08-11 14:47:09 [INFO]	[TRAIN] Epoch=7/270, Step=68/74, loss=24.452745, lr=6.4e-05, time_each_step=0.19s, eta=2:1:19
    2021-08-11 14:47:09 [INFO]	[TRAIN] Epoch=7/270, Step=70/74, loss=40.269592, lr=6.4e-05, time_each_step=0.18s, eta=2:1:17
    2021-08-11 14:47:09 [INFO]	[TRAIN] Epoch=7/270, Step=72/74, loss=45.134758, lr=6.4e-05, time_each_step=0.19s, eta=2:1:19
    2021-08-11 14:47:10 [INFO]	[TRAIN] Epoch=7/270, Step=74/74, loss=33.266941, lr=6.5e-05, time_each_step=0.18s, eta=2:1:17
    2021-08-11 14:47:10 [INFO]	[TRAIN] Epoch 7 finished, loss=40.581078, lr=6e-05 
    2021-08-11 14:59:48 [INFO]	[TRAIN] Epoch=31/270, Step=2/74, loss=38.447529, lr=0.000125, time_each_step=0.49s, eta=2:13:8
    2021-08-11 14:59:49 [INFO]	[TRAIN] Epoch=31/270, Step=4/74, loss=30.710838, lr=0.000125, time_each_step=0.52s, eta=2:13:9
    2021-08-11 14:59:50 [INFO]	[TRAIN] Epoch=31/270, Step=6/74, loss=15.109745, lr=0.000125, time_each_step=0.55s, eta=2:13:10
    2021-08-11 14:59:51 [INFO]	[TRAIN] Epoch=31/270, Step=8/74, loss=25.216272, lr=0.000125, time_each_step=0.57s, eta=2:13:10
    2021-08-11 14:59:52 [INFO]	[TRAIN] Epoch=31/270, Step=10/74, loss=20.093672, lr=0.000125, time_each_step=0.6s, eta=2:13:11
    2021-08-11 14:59:53 [INFO]	[TRAIN] Epoch=31/270, Step=12/74, loss=17.580746, lr=0.000125, time_each_step=0.62s, eta=2:13:11
    2021-08-11 14:59:53 [INFO]	[TRAIN] Epoch=31/270, Step=14/74, loss=27.159307, lr=0.000125, time_each_step=0.64s, eta=2:13:11
    2021-08-11 14:59:54 [INFO]	[TRAIN] Epoch=31/270, Step=16/74, loss=21.805767, lr=0.000125, time_each_step=0.66s, eta=2:13:11
    2021-08-11 14:59:55 [INFO]	[TRAIN] Epoch=31/270, Step=18/74, loss=22.647041, lr=0.000125, time_each_step=0.67s, eta=2:13:10
    2021-08-11 14:59:56 [INFO]	[TRAIN] Epoch=31/270, Step=20/74, loss=32.316292, lr=0.000125, time_each_step=0.68s, eta=2:13:10
    2021-08-11 14:59:57 [INFO]	[TRAIN] Epoch=31/270, Step=22/74, loss=22.887379, lr=0.000125, time_each_step=0.43s, eta=2:12:55
    2021-08-11 14:59:57 [INFO]	[TRAIN] Epoch=31/270, Step=24/74, loss=20.246067, lr=0.000125, time_each_step=0.41s, eta=2:12:53
    2021-08-11 14:59:58 [INFO]	[TRAIN] Epoch=31/270, Step=26/74, loss=22.286407, lr=0.000125, time_each_step=0.4s, eta=2:12:52
    2021-08-11 14:59:58 [INFO]	[TRAIN] Epoch=31/270, Step=28/74, loss=22.630087, lr=0.000125, time_each_step=0.38s, eta=2:12:50
    2021-08-11 14:59:59 [INFO]	[TRAIN] Epoch=31/270, Step=30/74, loss=32.134781, lr=0.000125, time_each_step=0.38s, eta=2:12:49
    2021-08-11 15:00:00 [INFO]	[TRAIN] Epoch=31/270, Step=32/74, loss=25.680773, lr=0.000125, time_each_step=0.38s, eta=2:12:49
    2021-08-11 15:00:01 [INFO]	[TRAIN] Epoch=31/270, Step=34/74, loss=41.472183, lr=0.000125, time_each_step=0.39s, eta=2:12:48
    2021-08-11 15:00:02 [INFO]	[TRAIN] Epoch=31/270, Step=36/74, loss=35.500969, lr=0.000125, time_each_step=0.39s, eta=2:12:47
    2021-08-11 15:00:03 [INFO]	[TRAIN] Epoch=31/270, Step=38/74, loss=26.507149, lr=0.000125, time_each_step=0.39s, eta=2:12:47
    2021-08-11 15:00:04 [INFO]	[TRAIN] Epoch=31/270, Step=40/74, loss=13.570386, lr=0.000125, time_each_step=0.4s, eta=2:12:46
    2021-08-11 15:00:04 [INFO]	[TRAIN] Epoch=31/270, Step=42/74, loss=30.455368, lr=0.000125, time_each_step=0.38s, eta=2:12:45
    2021-08-11 15:00:05 [INFO]	[TRAIN] Epoch=31/270, Step=44/74, loss=13.230737, lr=0.000125, time_each_step=0.36s, eta=2:12:44
    2021-08-11 15:00:05 [INFO]	[TRAIN] Epoch=31/270, Step=46/74, loss=36.538174, lr=0.000125, time_each_step=0.35s, eta=2:12:43
    2021-08-11 15:00:06 [INFO]	[TRAIN] Epoch=31/270, Step=48/74, loss=19.804485, lr=0.000125, time_each_step=0.36s, eta=2:12:42
    2021-08-11 15:00:06 [INFO]	[TRAIN] Epoch=31/270, Step=50/74, loss=12.795017, lr=0.000125, time_each_step=0.35s, eta=2:12:41
    2021-08-11 15:00:07 [INFO]	[TRAIN] Epoch=31/270, Step=52/74, loss=40.323792, lr=0.000125, time_each_step=0.32s, eta=2:12:40
    2021-08-11 15:00:07 [INFO]	[TRAIN] Epoch=31/270, Step=54/74, loss=24.400478, lr=0.000125, time_each_step=0.3s, eta=2:12:39
    2021-08-11 15:00:07 [INFO]	[TRAIN] Epoch=31/270, Step=56/74, loss=23.133568, lr=0.000125, time_each_step=0.27s, eta=2:12:38
    2021-08-11 15:00:08 [INFO]	[TRAIN] Epoch=31/270, Step=58/74, loss=17.667137, lr=0.000125, time_each_step=0.24s, eta=2:12:37
    2021-08-11 15:00:08 [INFO]	[TRAIN] Epoch=31/270, Step=60/74, loss=28.19463, lr=0.000125, time_each_step=0.22s, eta=2:12:36
    2021-08-11 15:00:08 [INFO]	[TRAIN] Epoch=31/270, Step=62/74, loss=17.607494, lr=0.000125, time_each_step=0.21s, eta=2:12:35
    2021-08-11 15:00:09 [INFO]	[TRAIN] Epoch=31/270, Step=64/74, loss=35.475128, lr=0.000125, time_each_step=0.22s, eta=2:12:35
    2021-08-11 15:00:09 [INFO]	[TRAIN] Epoch=31/270, Step=66/74, loss=11.182123, lr=0.000125, time_each_step=0.22s, eta=2:12:34
    2021-08-11 15:00:10 [INFO]	[TRAIN] Epoch=31/270, Step=68/74, loss=21.226387, lr=0.000125, time_each_step=0.22s, eta=2:12:34
    2021-08-11 15:00:10 [INFO]	[TRAIN] Epoch=31/270, Step=70/74, loss=23.388151, lr=0.000125, time_each_step=0.21s, eta=2:12:34
    2021-08-11 15:00:11 [INFO]	[TRAIN] Epoch=31/270, Step=72/74, loss=18.485092, lr=0.000125, time_each_step=0.2s, eta=2:12:33
    2021-08-11 15:00:11 [INFO]	[TRAIN] Epoch=31/270, Step=74/74, loss=21.092745, lr=0.000125, time_each_step=0.19s, eta=2:12:33
    2021-08-11 15:00:11 [INFO]	[TRAIN] Epoch 31 finished, loss=25.706875, lr=0.000125 .
    2021-08-11 15:00:15 [INFO]	[TRAIN] Epoch=32/270, Step=2/74, loss=21.26026, lr=0.000125, time_each_step=0.4s, eta=2:1:50
    2021-08-11 15:00:16 [INFO]	[TRAIN] Epoch=32/270, Step=4/74, loss=28.621546, lr=0.000125, time_each_step=0.43s, eta=2:1:51
    2021-08-11 15:00:17 [INFO]	[TRAIN] Epoch=32/270, Step=6/74, loss=17.444874, lr=0.000125, time_each_step=0.45s, eta=2:1:52
    2021-08-11 15:00:18 [INFO]	[TRAIN] Epoch=32/270, Step=8/74, loss=23.700159, lr=0.000125, time_each_step=0.49s, eta=2:1:54
    2021-08-11 15:00:19 [INFO]	[TRAIN] Epoch=32/270, Step=10/74, loss=9.409878, lr=0.000125, time_each_step=0.51s, eta=2:1:54
    2021-08-11 15:00:20 [INFO]	[TRAIN] Epoch=32/270, Step=12/74, loss=24.19537, lr=0.000125, time_each_step=0.53s, eta=2:1:54
    2021-08-11 15:00:21 [INFO]	[TRAIN] Epoch=32/270, Step=14/74, loss=29.992529, lr=0.000125, time_each_step=0.54s, eta=2:1:54
    2021-08-11 15:00:22 [INFO]	[TRAIN] Epoch=32/270, Step=16/74, loss=46.151802, lr=0.000125, time_each_step=0.57s, eta=2:1:54
    2021-08-11 15:00:23 [INFO]	[TRAIN] Epoch=32/270, Step=18/74, loss=32.840538, lr=0.000125, time_each_step=0.6s, eta=2:1:55
    2021-08-11 15:00:24 [INFO]	[TRAIN] Epoch=32/270, Step=20/74, loss=27.362764, lr=0.000125, time_each_step=0.64s, eta=2:1:56
    2021-08-11 15:00:24 [INFO]	[TRAIN] Epoch=32/270, Step=22/74, loss=15.924627, lr=0.000125, time_each_step=0.45s, eta=2:1:45
    2021-08-11 15:00:25 [INFO]	[TRAIN] Epoch=32/270, Step=24/74, loss=26.502319, lr=0.000125, time_each_step=0.44s, eta=2:1:44
    2021-08-11 15:00:26 [INFO]	[TRAIN] Epoch=32/270, Step=26/74, loss=30.434, lr=0.000125, time_each_step=0.43s, eta=2:1:42
    2021-08-11 15:00:26 [INFO]	[TRAIN] Epoch=32/270, Step=28/74, loss=19.038603, lr=0.000125, time_each_step=0.41s, eta=2:1:40
    2021-08-11 15:00:27 [INFO]	[TRAIN] Epoch=32/270, Step=30/74, loss=25.728106, lr=0.000125, time_each_step=0.41s, eta=2:1:39
    2021-08-11 15:00:28 [INFO]	[TRAIN] Epoch=32/270, Step=32/74, loss=23.620161, lr=0.000125, time_each_step=0.41s, eta=2:1:39
    2021-08-11 15:00:29 [INFO]	[TRAIN] Epoch=32/270, Step=34/74, loss=23.823351, lr=0.000125, time_each_step=0.42s, eta=2:1:38
    2021-08-11 15:00:30 [INFO]	[TRAIN] Epoch=32/270, Step=36/74, loss=19.561668, lr=0.000125, time_each_step=0.41s, eta=2:1:37
    2021-08-11 15:00:31 [INFO]	[TRAIN] Epoch=32/270, Step=38/74, loss=16.14249, lr=0.000125, time_each_step=0.41s, eta=2:1:36
    2021-08-11 15:00:32 [INFO]	[TRAIN] Epoch=32/270, Step=40/74, loss=14.523209, lr=0.000125, time_each_step=0.4s, eta=2:1:35
    2021-08-11 15:00:32 [INFO]	[TRAIN] Epoch=32/270, Step=42/74, loss=23.492128, lr=0.000125, time_each_step=0.39s, eta=2:1:34
    2021-08-11 15:00:33 [INFO]	[TRAIN] Epoch=32/270, Step=44/74, loss=31.216061, lr=0.000125, time_each_step=0.38s, eta=2:1:33
    2021-08-11 15:00:33 [INFO]	[TRAIN] Epoch=32/270, Step=46/74, loss=30.011248, lr=0.000125, time_each_step=0.36s, eta=2:1:32
    2021-08-11 15:00:34 [INFO]	[TRAIN] Epoch=32/270, Step=48/74, loss=19.56802, lr=0.000125, time_each_step=0.35s, eta=2:1:31
    2021-08-11 15:00:34 [INFO]	[TRAIN] Epoch=32/270, Step=50/74, loss=27.014217, lr=0.000125, time_each_step=0.33s, eta=2:1:29
    2021-08-11 15:00:34 [INFO]	[TRAIN] Epoch=32/270, Step=52/74, loss=24.50264, lr=0.000125, time_each_step=0.3s, eta=2:1:28
    2021-08-11 15:00:34 [INFO]	[TRAIN] Epoch=32/270, Step=54/74, loss=21.003559, lr=0.000125, time_each_step=0.27s, eta=2:1:27
    2021-08-11 15:00:35 [INFO]	[TRAIN] Epoch=32/270, Step=56/74, loss=37.838947, lr=0.000125, time_each_step=0.24s, eta=2:1:26
    2021-08-11 15:00:35 [INFO]	[TRAIN] Epoch=32/270, Step=58/74, loss=11.15711, lr=0.000125, time_each_step=0.22s, eta=2:1:25
    2021-08-11 15:00:36 [INFO]	[TRAIN] Epoch=32/270, Step=60/74, loss=14.119827, lr=0.000125, time_each_step=0.19s, eta=2:1:24
    2021-08-11 15:00:36 [INFO]	[TRAIN] Epoch=32/270, Step=62/74, loss=17.845152, lr=0.000125, time_each_step=0.18s, eta=2:1:24
    2021-08-11 15:00:36 [INFO]	[TRAIN] Epoch=32/270, Step=64/74, loss=15.890081, lr=0.000125, time_each_step=0.18s, eta=2:1:23
    2021-08-11 15:00:37 [INFO]	[TRAIN] Epoch=32/270, Step=66/74, loss=30.38567, lr=0.000125, time_each_step=0.19s, eta=2:1:23
    2021-08-11 15:00:37 [INFO]	[TRAIN] Epoch=32/270, Step=68/74, loss=18.766399, lr=0.000125, time_each_step=0.18s, eta=2:1:23
    2021-08-11 15:00:38 [INFO]	[TRAIN] Epoch=32/270, Step=70/74, loss=35.892132, lr=0.000125, time_each_step=0.18s, eta=2:1:22
    2021-08-11 15:00:38 [INFO]	[TRAIN] Epoch=32/270, Step=72/74, loss=32.557632, lr=0.000125, time_each_step=0.18s, eta=2:1:22
    2021-08-11 15:00:38 [INFO]	[TRAIN] Epoch=32/270, Step=74/74, loss=20.320999, lr=0.000125, time_each_step=0.2s, eta=2:1:22
    2021-08-11 15:00:38 [INFO]	[TRAIN] Epoch 32 finished, loss=24.407364, lr=0.000125 .
    2021-08-11 15:00:43 [INFO]	[TRAIN] Epoch=33/270, Step=2/74, loss=33.54118, lr=0.000125, time_each_step=0.42s, eta=1:56:11
    2021-08-11 15:00:44 [INFO]	[TRAIN] Epoch=33/270, Step=4/74, loss=17.036211, lr=0.000125, time_each_step=0.45s, eta=1:56:12
    2021-08-11 15:00:45 [INFO]	[TRAIN] Epoch=33/270, Step=6/74, loss=21.084444, lr=0.000125, time_each_step=0.46s, eta=1:56:12
    2021-08-11 15:00:45 [INFO]	[TRAIN] Epoch=33/270, Step=8/74, loss=33.292152, lr=0.000125, time_each_step=0.47s, eta=1:56:12
    2021-08-11 15:00:46 [INFO]	[TRAIN] Epoch=33/270, Step=10/74, loss=21.595881, lr=0.000125, time_each_step=0.49s, eta=1:56:13
    2021-08-11 15:00:47 [INFO]	[TRAIN] Epoch=33/270, Step=12/74, loss=27.176903, lr=0.000125, time_each_step=0.52s, eta=1:56:14
    2021-08-11 15:00:48 [INFO]	[TRAIN] Epoch=33/270, Step=14/74, loss=40.798492, lr=0.000125, time_each_step=0.55s, eta=1:56:14
    2021-08-11 15:00:49 [INFO]	[TRAIN] Epoch=33/270, Step=16/74, loss=18.852875, lr=0.000125, time_each_step=0.55s, eta=1:56:13
    2021-08-11 15:00:49 [INFO]	[TRAIN] Epoch=33/270, Step=18/74, loss=19.167494, lr=0.000125, time_each_step=0.57s, eta=1:56:13
    2021-08-11 15:00:50 [INFO]	[TRAIN] Epoch=33/270, Step=20/74, loss=16.599005, lr=0.000125, time_each_step=0.59s, eta=1:56:13
    2021-08-11 15:00:51 [INFO]	[TRAIN] Epoch=33/270, Step=22/74, loss=30.965143, lr=0.000125, time_each_step=0.38s, eta=1:56:1
    2021-08-11 15:00:51 [INFO]	[TRAIN] Epoch=33/270, Step=24/74, loss=30.915304, lr=0.000125, time_each_step=0.36s, eta=1:55:59
    2021-08-11 15:00:52 [INFO]	[TRAIN] Epoch=33/270, Step=26/74, loss=26.04731, lr=0.000125, time_each_step=0.37s, eta=1:55:59
    2021-08-11 15:00:53 [INFO]	[TRAIN] Epoch=33/270, Step=28/74, loss=24.553349, lr=0.000125, time_each_step=0.37s, eta=1:55:58
    2021-08-11 15:00:54 [INFO]	[TRAIN] Epoch=33/270, Step=30/74, loss=15.343371, lr=0.000125, time_each_step=0.37s, eta=1:55:57
    2021-08-11 15:00:55 [INFO]	[TRAIN] Epoch=33/270, Step=32/74, loss=17.724308, lr=0.000125, time_each_step=0.36s, eta=1:55:56
    2021-08-11 15:00:55 [INFO]	[TRAIN] Epoch=33/270, Step=34/74, loss=27.035297, lr=0.000125, time_each_step=0.36s, eta=1:55:55
    2021-08-11 15:00:56 [INFO]	[TRAIN] Epoch=33/270, Step=36/74, loss=20.221983, lr=0.000125, time_each_step=0.38s, eta=1:55:56
    2021-08-11 15:00:57 [INFO]	[TRAIN] Epoch=33/270, Step=38/74, loss=19.754301, lr=0.000125, time_each_step=0.37s, eta=1:55:54
    2021-08-11 15:00:57 [INFO]	[TRAIN] Epoch=33/270, Step=40/74, loss=27.859856, lr=0.000125, time_each_step=0.36s, eta=1:55:53
    2021-08-11 15:00:58 [INFO]	[TRAIN] Epoch=33/270, Step=42/74, loss=13.601071, lr=0.000125, time_each_step=0.35s, eta=1:55:52
    2021-08-11 15:00:59 [INFO]	[TRAIN] Epoch=33/270, Step=44/74, loss=55.420685, lr=0.000125, time_each_step=0.36s, eta=1:55:52
    2021-08-11 15:00:59 [INFO]	[TRAIN] Epoch=33/270, Step=46/74, loss=28.049173, lr=0.000125, time_each_step=0.36s, eta=1:55:51
    2021-08-11 15:01:00 [INFO]	[TRAIN] Epoch=33/270, Step=48/74, loss=23.394184, lr=0.000125, time_each_step=0.36s, eta=1:55:50
    2021-08-11 15:01:01 [INFO]	[TRAIN] Epoch=33/270, Step=50/74, loss=48.718605, lr=0.000125, time_each_step=0.34s, eta=1:55:49
    2021-08-11 15:01:01 [INFO]	[TRAIN] Epoch=33/270, Step=52/74, loss=42.506001, lr=0.000125, time_each_step=0.33s, eta=1:55:48
    2021-08-11 15:01:01 [INFO]	[TRAIN] Epoch=33/270, Step=54/74, loss=21.536423, lr=0.000125, time_each_step=0.3s, eta=1:55:47
    2021-08-11 15:01:02 [INFO]	[TRAIN] Epoch=33/270, Step=56/74, loss=17.354839, lr=0.000125, time_each_step=0.28s, eta=1:55:46
    2021-08-11 15:01:02 [INFO]	[TRAIN] Epoch=33/270, Step=58/74, loss=33.61298, lr=0.000125, time_each_step=0.28s, eta=1:55:45
    2021-08-11 15:01:03 [INFO]	[TRAIN] Epoch=33/270, Step=60/74, loss=13.329172, lr=0.000125, time_each_step=0.27s, eta=1:55:45
    2021-08-11 15:01:03 [INFO]	[TRAIN] Epoch=33/270, Step=62/74, loss=33.308399, lr=0.000125, time_each_step=0.27s, eta=1:55:44
    2021-08-11 15:01:04 [INFO]	[TRAIN] Epoch=33/270, Step=64/74, loss=31.233553, lr=0.000125, time_each_step=0.26s, eta=1:55:44
    2021-08-11 15:01:04 [INFO]	[TRAIN] Epoch=33/270, Step=66/74, loss=16.280062, lr=0.000125, time_each_step=0.25s, eta=1:55:43
    2021-08-11 15:01:05 [INFO]	[TRAIN] Epoch=33/270, Step=68/74, loss=16.91604, lr=0.000125, time_each_step=0.23s, eta=1:55:42
    2021-08-11 15:01:05 [INFO]	[TRAIN] Epoch=33/270, Step=70/74, loss=25.265749, lr=0.000125, time_each_step=0.23s, eta=1:55:42
    2021-08-11 15:01:05 [INFO]	[TRAIN] Epoch=33/270, Step=72/74, loss=27.464447, lr=0.000125, time_each_step=0.22s, eta=1:55:41
    2021-08-11 15:01:06 [INFO]	[TRAIN] Epoch=33/270, Step=74/74, loss=18.48625, lr=0.000125, time_each_step=0.24s, eta=1:55:41
    2021-08-11 15:01:06 [INFO]	[TRAIN] Epoch 33 finished, loss=27.545856, lr=0.000125 .
    2021-08-11 15:01:19 [INFO]	[TRAIN] Epoch=34/270, Step=2/74, loss=28.585033, lr=0.000125, time_each_step=0.88s, eta=1:56:55
    2021-08-11 15:01:20 [INFO]	[TRAIN] Epoch=34/270, Step=4/74, loss=20.41663, lr=0.000125, time_each_step=0.9s, eta=1:56:55
    2021-08-11 15:01:21 [INFO]	[TRAIN] Epoch=34/270, Step=6/74, loss=27.784836, lr=0.000125, time_each_step=0.91s, eta=1:56:54
    2021-08-11 15:01:22 [INFO]	[TRAIN] Epoch=34/270, Step=8/74, loss=29.522018, lr=0.000125, time_each_step=0.93s, eta=1:56:53
    2021-08-11 15:01:23 [INFO]	[TRAIN] Epoch=34/270, Step=10/74, loss=21.671829, lr=0.000125, time_each_step=0.95s, eta=1:56:53
    2021-08-11 15:01:23 [INFO]	[TRAIN] Epoch=34/270, Step=12/74, loss=22.175873, lr=0.000125, time_each_step=0.96s, eta=1:56:51
    2021-08-11 15:01:24 [INFO]	[TRAIN] Epoch=34/270, Step=14/74, loss=26.715626, lr=0.000125, time_each_step=0.99s, eta=1:56:51
    2021-08-11 15:01:25 [INFO]	[TRAIN] Epoch=34/270, Step=16/74, loss=14.527718, lr=0.000125, time_each_step=1.0s, eta=1:56:50
    2021-08-11 15:01:26 [INFO]	[TRAIN] Epoch=34/270, Step=18/74, loss=49.47411, lr=0.000125, time_each_step=1.01s, eta=1:56:49
    2021-08-11 15:01:26 [INFO]	[TRAIN] Epoch=34/270, Step=20/74, loss=25.769678, lr=0.000125, time_each_step=1.01s, eta=1:56:47
    2021-08-11 15:01:27 [INFO]	[TRAIN] Epoch=34/270, Step=22/74, loss=13.365525, lr=0.000125, time_each_step=0.38s, eta=1:56:12
    2021-08-11 15:01:28 [INFO]	[TRAIN] Epoch=34/270, Step=24/74, loss=29.028776, lr=0.000125, time_each_step=0.38s, eta=1:56:11
    2021-08-11 15:01:29 [INFO]	[TRAIN] Epoch=34/270, Step=26/74, loss=16.462605, lr=0.000125, time_each_step=0.38s, eta=1:56:10
    2021-08-11 15:01:30 [INFO]	[TRAIN] Epoch=34/270, Step=28/74, loss=23.901331, lr=0.000125, time_each_step=0.39s, eta=1:56:10
    2021-08-11 15:01:30 [INFO]	[TRAIN] Epoch=34/270, Step=30/74, loss=40.180313, lr=0.000125, time_each_step=0.38s, eta=1:56:9
    2021-08-11 15:01:31 [INFO]	[TRAIN] Epoch=34/270, Step=32/74, loss=22.363157, lr=0.000125, time_each_step=0.38s, eta=1:56:8
    2021-08-11 15:01:32 [INFO]	[TRAIN] Epoch=34/270, Step=34/74, loss=23.711111, lr=0.000125, time_each_step=0.38s, eta=1:56:7
    2021-08-11 15:01:33 [INFO]	[TRAIN] Epoch=34/270, Step=36/74, loss=17.173901, lr=0.000125, time_each_step=0.37s, eta=1:56:6
    2021-08-11 15:01:33 [INFO]	[TRAIN] Epoch=34/270, Step=38/74, loss=30.125423, lr=0.000125, time_each_step=0.37s, eta=1:56:5
    2021-08-11 15:01:34 [INFO]	[TRAIN] Epoch=34/270, Step=40/74, loss=12.076602, lr=0.000125, time_each_step=0.39s, eta=1:56:5
    2021-08-11 15:01:35 [INFO]	[TRAIN] Epoch=34/270, Step=42/74, loss=13.372695, lr=0.000125, time_each_step=0.38s, eta=1:56:4
    2021-08-11 15:01:35 [INFO]	[TRAIN] Epoch=34/270, Step=44/74, loss=26.826311, lr=0.000125, time_each_step=0.38s, eta=1:56:3
    2021-08-11 15:01:36 [INFO]	[TRAIN] Epoch=34/270, Step=46/74, loss=22.063587, lr=0.000125, time_each_step=0.37s, eta=1:56:2
    2021-08-11 15:01:36 [INFO]	[TRAIN] Epoch=34/270, Step=48/74, loss=15.609911, lr=0.000125, time_each_step=0.34s, eta=1:56:1
    2021-08-11 15:01:37 [INFO]	[TRAIN] Epoch=34/270, Step=50/74, loss=26.32979, lr=0.000125, time_each_step=0.33s, eta=1:56:0
    2021-08-11 15:01:37 [INFO]	[TRAIN] Epoch=34/270, Step=52/74, loss=12.867619, lr=0.000125, time_each_step=0.31s, eta=1:55:59
    2021-08-11 15:01:38 [INFO]	[TRAIN] Epoch=34/270, Step=54/74, loss=22.348738, lr=0.000125, time_each_step=0.28s, eta=1:55:58
    2021-08-11 15:01:38 [INFO]	[TRAIN] Epoch=34/270, Step=56/74, loss=39.04538, lr=0.000125, time_each_step=0.27s, eta=1:55:57
    2021-08-11 15:01:38 [INFO]	[TRAIN] Epoch=34/270, Step=58/74, loss=29.178177, lr=0.000125, time_each_step=0.25s, eta=1:55:56
    2021-08-11 15:01:39 [INFO]	[TRAIN] Epoch=34/270, Step=60/74, loss=18.113625, lr=0.000125, time_each_step=0.22s, eta=1:55:55
    2021-08-11 15:01:39 [INFO]	[TRAIN] Epoch=34/270, Step=62/74, loss=27.098743, lr=0.000125, time_each_step=0.22s, eta=1:55:55
    2021-08-11 15:01:39 [INFO]	[TRAIN] Epoch=34/270, Step=64/74, loss=34.369068, lr=0.000125, time_each_step=0.2s, eta=1:55:54
    2021-08-11 15:01:40 [INFO]	[TRAIN] Epoch=34/270, Step=66/74, loss=74.285118, lr=0.000125, time_each_step=0.18s, eta=1:55:53
    2021-08-11 15:01:40 [INFO]	[TRAIN] Epoch=34/270, Step=68/74, loss=48.206215, lr=0.000125, time_each_step=0.19s, eta=1:55:53
    2021-08-11 15:01:40 [INFO]	[TRAIN] Epoch=34/270, Step=70/74, loss=22.976385, lr=0.000125, time_each_step=0.18s, eta=1:55:53
    2021-08-11 15:01:41 [INFO]	[TRAIN] Epoch=34/270, Step=72/74, loss=15.093455, lr=0.000125, time_each_step=0.19s, eta=1:55:52
    2021-08-11 15:01:41 [INFO]	[TRAIN] Epoch=34/270, Step=74/74, loss=35.443645, lr=0.000125, time_each_step=0.18s, eta=1:55:52
    2021-08-11 15:01:41 [INFO]	[TRAIN] Epoch 34 finished, loss=26.290224, lr=0.000125 .
    2021-08-11 15:01:47 [INFO]	[TRAIN] Epoch=35/270, Step=2/74, loss=20.913904, lr=0.000125, time_each_step=0.47s, eta=2:24:42
    2021-08-11 15:01:48 [INFO]	[TRAIN] Epoch=35/270, Step=4/74, loss=56.330502, lr=0.000125, time_each_step=0.5s, eta=2:24:44
    2021-08-11 15:01:49 [INFO]	[TRAIN] Epoch=35/270, Step=6/74, loss=13.341489, lr=0.000125, time_each_step=0.52s, eta=2:24:43
    2021-08-11 15:01:50 [INFO]	[TRAIN] Epoch=35/270, Step=8/74, loss=22.095032, lr=0.000125, time_each_step=0.53s, eta=2:24:43
    2021-08-11 15:01:51 [INFO]	[TRAIN] Epoch=35/270, Step=10/74, loss=40.358047, lr=0.000125, time_each_step=0.56s, eta=2:24:44
    2021-08-11 15:01:51 [INFO]	[TRAIN] Epoch=35/270, Step=12/74, loss=37.379532, lr=0.000125, time_each_step=0.59s, eta=2:24:45
    2021-08-11 15:01:52 [INFO]	[TRAIN] Epoch=35/270, Step=14/74, loss=23.633347, lr=0.000125, time_each_step=0.6s, eta=2:24:45
    2021-08-11 15:01:53 [INFO]	[TRAIN] Epoch=35/270, Step=16/74, loss=17.038136, lr=0.000125, time_each_step=0.62s, eta=2:24:44
    2021-08-11 15:01:54 [INFO]	[TRAIN] Epoch=35/270, Step=18/74, loss=23.940805, lr=0.000125, time_each_step=0.64s, eta=2:24:44
    2021-08-11 15:01:55 [INFO]	[TRAIN] Epoch=35/270, Step=20/74, loss=23.255352, lr=0.000125, time_each_step=0.67s, eta=2:24:45
    2021-08-11 15:01:55 [INFO]	[TRAIN] Epoch=35/270, Step=22/74, loss=22.999332, lr=0.000125, time_each_step=0.41s, eta=2:24:29
    2021-08-11 15:01:56 [INFO]	[TRAIN] Epoch=35/270, Step=24/74, loss=67.293396, lr=0.000125, time_each_step=0.39s, eta=2:24:28
    2021-08-11 15:01:57 [INFO]	[TRAIN] Epoch=35/270, Step=26/74, loss=25.094484, lr=0.000125, time_each_step=0.39s, eta=2:24:27
    2021-08-11 15:01:57 [INFO]	[TRAIN] Epoch=35/270, Step=28/74, loss=26.740685, lr=0.000125, time_each_step=0.39s, eta=2:24:26
    2021-08-11 15:01:58 [INFO]	[TRAIN] Epoch=35/270, Step=30/74, loss=31.351782, lr=0.000125, time_each_step=0.37s, eta=2:24:25
    2021-08-11 15:01:59 [INFO]	[TRAIN] Epoch=35/270, Step=32/74, loss=23.275833, lr=0.000125, time_each_step=0.35s, eta=2:24:23
    2021-08-11 15:01:59 [INFO]	[TRAIN] Epoch=35/270, Step=34/74, loss=26.543888, lr=0.000125, time_each_step=0.36s, eta=2:24:23
    2021-08-11 15:02:00 [INFO]	[TRAIN] Epoch=35/270, Step=36/74, loss=17.005257, lr=0.000125, time_each_step=0.36s, eta=2:24:22
    2021-08-11 15:02:01 [INFO]	[TRAIN] Epoch=35/270, Step=38/74, loss=21.540815, lr=0.000125, time_each_step=0.35s, eta=2:24:21
    2021-08-11 15:02:01 [INFO]	[TRAIN] Epoch=35/270, Step=40/74, loss=14.916183, lr=0.000125, time_each_step=0.33s, eta=2:24:20
    2021-08-11 15:02:02 [INFO]	[TRAIN] Epoch=35/270, Step=42/74, loss=20.20306, lr=0.000125, time_each_step=0.33s, eta=2:24:19
    2021-08-11 15:02:02 [INFO]	[TRAIN] Epoch=35/270, Step=44/74, loss=26.844503, lr=0.000125, time_each_step=0.31s, eta=2:24:18
    2021-08-11 15:02:03 [INFO]	[TRAIN] Epoch=35/270, Step=46/74, loss=11.083918, lr=0.000125, time_each_step=0.31s, eta=2:24:17
    2021-08-11 15:02:03 [INFO]	[TRAIN] Epoch=35/270, Step=48/74, loss=33.960693, lr=0.000125, time_each_step=0.3s, eta=2:24:16
    2021-08-11 15:02:04 [INFO]	[TRAIN] Epoch=35/270, Step=50/74, loss=18.859205, lr=0.000125, time_each_step=0.28s, eta=2:24:15
    2021-08-11 15:02:04 [INFO]	[TRAIN] Epoch=35/270, Step=52/74, loss=30.525871, lr=0.000125, time_each_step=0.27s, eta=2:24:14
    2021-08-11 15:02:04 [INFO]	[TRAIN] Epoch=35/270, Step=54/74, loss=30.53413, lr=0.000125, time_each_step=0.25s, eta=2:24:13
    2021-08-11 15:02:05 [INFO]	[TRAIN] Epoch=35/270, Step=56/74, loss=12.591213, lr=0.000125, time_each_step=0.24s, eta=2:24:13
    2021-08-11 15:02:05 [INFO]	[TRAIN] Epoch=35/270, Step=58/74, loss=26.760929, lr=0.000125, time_each_step=0.22s, eta=2:24:12
    2021-08-11 15:02:06 [INFO]	[TRAIN] Epoch=35/270, Step=60/74, loss=16.152237, lr=0.000125, time_each_step=0.21s, eta=2:24:11
    2021-08-11 15:02:06 [INFO]	[TRAIN] Epoch=35/270, Step=62/74, loss=38.776085, lr=0.000125, time_each_step=0.21s, eta=2:24:11
    2021-08-11 15:02:06 [INFO]	[TRAIN] Epoch=35/270, Step=64/74, loss=24.712765, lr=0.000125, time_each_step=0.2s, eta=2:24:10
    2021-08-11 15:02:07 [INFO]	[TRAIN] Epoch=35/270, Step=66/74, loss=15.85234, lr=0.000125, time_each_step=0.2s, eta=2:24:10
    2021-08-11 15:02:07 [INFO]	[TRAIN] Epoch=35/270, Step=68/74, loss=39.042416, lr=0.000125, time_each_step=0.2s, eta=2:24:10
    2021-08-11 15:02:08 [INFO]	[TRAIN] Epoch=35/270, Step=70/74, loss=17.884399, lr=0.000125, time_each_step=0.21s, eta=2:24:9
    2021-08-11 15:02:08 [INFO]	[TRAIN] Epoch=35/270, Step=72/74, loss=12.680005, lr=0.000125, time_each_step=0.22s, eta=2:24:9
    2021-08-11 15:02:09 [INFO]	[TRAIN] Epoch=35/270, Step=74/74, loss=22.08609, lr=0.000125, time_each_step=0.21s, eta=2:24:8
    2021-08-11 15:02:09 [INFO]	[TRAIN] Epoch 35 finished, loss=26.584044, lr=0.000125 .
    2021-08-11 15:02:14 [INFO]	[TRAIN] Epoch=36/270, Step=2/74, loss=20.845078, lr=0.000125, time_each_step=0.47s, eta=1:55:4
    2021-08-11 15:02:15 [INFO]	[TRAIN] Epoch=36/270, Step=4/74, loss=19.341686, lr=0.000125, time_each_step=0.5s, eta=1:55:5
    2021-08-11 15:02:16 [INFO]	[TRAIN] Epoch=36/270, Step=6/74, loss=27.927689, lr=0.000125, time_each_step=0.51s, eta=1:55:4
    2021-08-11 15:02:16 [INFO]	[TRAIN] Epoch=36/270, Step=8/74, loss=16.065603, lr=0.000125, time_each_step=0.51s, eta=1:55:4
    2021-08-11 15:02:17 [INFO]	[TRAIN] Epoch=36/270, Step=10/74, loss=25.360603, lr=0.000125, time_each_step=0.54s, eta=1:55:4
    2021-08-11 15:02:18 [INFO]	[TRAIN] Epoch=36/270, Step=12/74, loss=39.550148, lr=0.000125, time_each_step=0.57s, eta=1:55:5
    2021-08-11 15:02:19 [INFO]	[TRAIN] Epoch=36/270, Step=14/74, loss=11.711065, lr=0.000125, time_each_step=0.59s, eta=1:55:5
    2021-08-11 15:02:20 [INFO]	[TRAIN] Epoch=36/270, Step=16/74, loss=27.044018, lr=0.000125, time_each_step=0.59s, eta=1:55:4
    2021-08-11 15:02:21 [INFO]	[TRAIN] Epoch=36/270, Step=18/74, loss=61.445, lr=0.000125, time_each_step=0.61s, eta=1:55:4
    2021-08-11 15:02:21 [INFO]	[TRAIN] Epoch=36/270, Step=20/74, loss=31.695332, lr=0.000125, time_each_step=0.62s, eta=1:55:3
    2021-08-11 15:02:22 [INFO]	[TRAIN] Epoch=36/270, Step=22/74, loss=24.462925, lr=0.000125, time_each_step=0.38s, eta=1:54:49
    2021-08-11 15:02:22 [INFO]	[TRAIN] Epoch=36/270, Step=24/74, loss=41.180134, lr=0.000125, time_each_step=0.37s, eta=1:54:48
    2021-08-11 15:02:23 [INFO]	[TRAIN] Epoch=36/270, Step=26/74, loss=18.314596, lr=0.000125, time_each_step=0.37s, eta=1:54:47
    2021-08-11 15:02:24 [INFO]	[TRAIN] Epoch=36/270, Step=28/74, loss=19.055548, lr=0.000125, time_each_step=0.37s, eta=1:54:47
    2021-08-11 15:02:25 [INFO]	[TRAIN] Epoch=36/270, Step=30/74, loss=16.852697, lr=0.000125, time_each_step=0.38s, eta=1:54:46
    2021-08-11 15:02:26 [INFO]	[TRAIN] Epoch=36/270, Step=32/74, loss=11.633168, lr=0.000125, time_each_step=0.38s, eta=1:54:45
    2021-08-11 15:02:27 [INFO]	[TRAIN] Epoch=36/270, Step=34/74, loss=11.980305, lr=0.000125, time_each_step=0.38s, eta=1:54:45
    2021-08-11 15:02:28 [INFO]	[TRAIN] Epoch=36/270, Step=36/74, loss=15.541656, lr=0.000125, time_each_step=0.39s, eta=1:54:45
    2021-08-11 15:02:28 [INFO]	[TRAIN] Epoch=36/270, Step=38/74, loss=16.168394, lr=0.000125, time_each_step=0.38s, eta=1:54:43
    2021-08-11 15:02:29 [INFO]	[TRAIN] Epoch=36/270, Step=40/74, loss=61.739563, lr=0.000125, time_each_step=0.41s, eta=1:54:44
    2021-08-11 15:02:30 [INFO]	[TRAIN] Epoch=36/270, Step=42/74, loss=19.76001, lr=0.000125, time_each_step=0.4s, eta=1:54:43
    2021-08-11 15:02:30 [INFO]	[TRAIN] Epoch=36/270, Step=44/74, loss=44.189636, lr=0.000125, time_each_step=0.39s, eta=1:54:41
    2021-08-11 15:02:30 [INFO]	[TRAIN] Epoch=36/270, Step=46/74, loss=13.438208, lr=0.000125, time_each_step=0.37s, eta=1:54:40
    2021-08-11 15:02:31 [INFO]	[TRAIN] Epoch=36/270, Step=48/74, loss=31.61581, lr=0.000125, time_each_step=0.35s, eta=1:54:39
    2021-08-11 15:02:31 [INFO]	[TRAIN] Epoch=36/270, Step=50/74, loss=20.552542, lr=0.000125, time_each_step=0.33s, eta=1:54:38
    2021-08-11 15:02:32 [INFO]	[TRAIN] Epoch=36/270, Step=52/74, loss=31.71953, lr=0.000125, time_each_step=0.29s, eta=1:54:36
    2021-08-11 15:02:32 [INFO]	[TRAIN] Epoch=36/270, Step=54/74, loss=31.226082, lr=0.000125, time_each_step=0.26s, eta=1:54:35
    2021-08-11 15:02:32 [INFO]	[TRAIN] Epoch=36/270, Step=56/74, loss=34.64592, lr=0.000125, time_each_step=0.23s, eta=1:54:34
    2021-08-11 15:02:33 [INFO]	[TRAIN] Epoch=36/270, Step=58/74, loss=28.515156, lr=0.000125, time_each_step=0.21s, eta=1:54:33
    2021-08-11 15:02:33 [INFO]	[TRAIN] Epoch=36/270, Step=60/74, loss=18.733974, lr=0.000125, time_each_step=0.2s, eta=1:54:32
    2021-08-11 15:02:33 [INFO]	[TRAIN] Epoch=36/270, Step=62/74, loss=35.339325, lr=0.000125, time_each_step=0.19s, eta=1:54:32
    2021-08-11 15:02:34 [INFO]	[TRAIN] Epoch=36/270, Step=64/74, loss=26.1492, lr=0.000125, time_each_step=0.18s, eta=1:54:32
    2021-08-11 15:02:34 [INFO]	[TRAIN] Epoch=36/270, Step=66/74, loss=17.788609, lr=0.000125, time_each_step=0.19s, eta=1:54:31
    2021-08-11 15:02:35 [INFO]	[TRAIN] Epoch=36/270, Step=68/74, loss=21.730602, lr=0.000125, time_each_step=0.2s, eta=1:54:31
    2021-08-11 15:02:35 [INFO]	[TRAIN] Epoch=36/270, Step=70/74, loss=27.729614, lr=0.000125, time_each_step=0.19s, eta=1:54:30
    2021-08-11 15:02:36 [INFO]	[TRAIN] Epoch=36/270, Step=72/74, loss=23.775925, lr=0.000125, time_each_step=0.2s, eta=1:54:30
    2021-08-11 15:02:36 [INFO]	[TRAIN] Epoch=36/270, Step=74/74, loss=18.866581, lr=0.000125, time_each_step=0.21s, eta=1:54:30
    2021-08-11 15:02:36 [INFO]	[TRAIN] Epoch 36 finished, loss=24.679832, lr=0.000125 .
    2021-08-11 15:02:42 [INFO]	[TRAIN] Epoch=37/270, Step=2/74, loss=16.315273, lr=0.000125, time_each_step=0.51s, eta=1:53:53
    2021-08-11 15:02:43 [INFO]	[TRAIN] Epoch=37/270, Step=4/74, loss=18.265261, lr=0.000125, time_each_step=0.52s, eta=1:53:53
    2021-08-11 15:02:44 [INFO]	[TRAIN] Epoch=37/270, Step=6/74, loss=15.989603, lr=0.000125, time_each_step=0.54s, eta=1:53:53
    2021-08-11 15:02:45 [INFO]	[TRAIN] Epoch=37/270, Step=8/74, loss=20.400013, lr=0.000125, time_each_step=0.56s, eta=1:53:53
    2021-08-11 15:02:46 [INFO]	[TRAIN] Epoch=37/270, Step=10/74, loss=26.548021, lr=0.000125, time_each_step=0.58s, eta=1:53:54
    2021-08-11 15:02:46 [INFO]	[TRAIN] Epoch=37/270, Step=12/74, loss=29.439186, lr=0.000125, time_each_step=0.59s, eta=1:53:53
    2021-08-11 15:02:47 [INFO]	[TRAIN] Epoch=37/270, Step=14/74, loss=17.709547, lr=0.000125, time_each_step=0.62s, eta=1:53:54
    2021-08-11 15:02:48 [INFO]	[TRAIN] Epoch=37/270, Step=16/74, loss=32.081448, lr=0.000125, time_each_step=0.64s, eta=1:53:54
    2021-08-11 15:02:49 [INFO]	[TRAIN] Epoch=37/270, Step=18/74, loss=23.055744, lr=0.000125, time_each_step=0.66s, eta=1:53:53
    2021-08-11 15:02:50 [INFO]	[TRAIN] Epoch=37/270, Step=20/74, loss=57.8876, lr=0.000125, time_each_step=0.68s, eta=1:53:53
    2021-08-11 15:02:51 [INFO]	[TRAIN] Epoch=37/270, Step=22/74, loss=31.425735, lr=0.000125, time_each_step=0.43s, eta=1:53:39
    2021-08-11 15:02:52 [INFO]	[TRAIN] Epoch=37/270, Step=24/74, loss=20.331898, lr=0.000125, time_each_step=0.44s, eta=1:53:38
    2021-08-11 15:02:53 [INFO]	[TRAIN] Epoch=37/270, Step=26/74, loss=37.624084, lr=0.000125, time_each_step=0.43s, eta=1:53:37
    2021-08-11 15:02:53 [INFO]	[TRAIN] Epoch=37/270, Step=28/74, loss=46.25927, lr=0.000125, time_each_step=0.42s, eta=1:53:36
    2021-08-11 15:02:53 [INFO]	[TRAIN] Epoch=37/270, Step=30/74, loss=40.056057, lr=0.000125, time_each_step=0.4s, eta=1:53:34
    2021-08-11 15:02:54 [INFO]	[TRAIN] Epoch=37/270, Step=32/74, loss=19.527637, lr=0.000125, time_each_step=0.38s, eta=1:53:33
    2021-08-11 15:02:54 [INFO]	[TRAIN] Epoch=37/270, Step=34/74, loss=20.82338, lr=0.000125, time_each_step=0.35s, eta=1:53:30
    2021-08-11 15:02:55 [INFO]	[TRAIN] Epoch=37/270, Step=36/74, loss=22.798531, lr=0.000125, time_each_step=0.35s, eta=1:53:30
    2021-08-11 15:02:56 [INFO]	[TRAIN] Epoch=37/270, Step=38/74, loss=17.348789, lr=0.000125, time_each_step=0.37s, eta=1:53:30
    2021-08-11 15:02:57 [INFO]	[TRAIN] Epoch=37/270, Step=40/74, loss=24.921062, lr=0.000125, time_each_step=0.35s, eta=1:53:29
    2021-08-11 15:02:58 [INFO]	[TRAIN] Epoch=37/270, Step=42/74, loss=37.724026, lr=0.000125, time_each_step=0.35s, eta=1:53:28
    2021-08-11 15:02:59 [INFO]	[TRAIN] Epoch=37/270, Step=44/74, loss=23.933584, lr=0.000125, time_each_step=0.34s, eta=1:53:27
    2021-08-11 15:02:59 [INFO]	[TRAIN] Epoch=37/270, Step=46/74, loss=18.684259, lr=0.000125, time_each_step=0.32s, eta=1:53:26
    2021-08-11 15:03:00 [INFO]	[TRAIN] Epoch=37/270, Step=48/74, loss=22.919239, lr=0.000125, time_each_step=0.33s, eta=1:53:25
    2021-08-11 15:03:00 [INFO]	[TRAIN] Epoch=37/270, Step=50/74, loss=30.298021, lr=0.000125, time_each_step=0.33s, eta=1:53:24
    2021-08-11 15:03:00 [INFO]	[TRAIN] Epoch=37/270, Step=52/74, loss=22.819431, lr=0.000125, time_each_step=0.33s, eta=1:53:24
    2021-08-11 15:03:01 [INFO]	[TRAIN] Epoch=37/270, Step=54/74, loss=29.683643, lr=0.000125, time_each_step=0.33s, eta=1:53:23
    2021-08-11 15:03:01 [INFO]	[TRAIN] Epoch=37/270, Step=56/74, loss=20.56415, lr=0.000125, time_each_step=0.3s, eta=1:53:22
    2021-08-11 15:03:02 [INFO]	[TRAIN] Epoch=37/270, Step=58/74, loss=31.314898, lr=0.000125, time_each_step=0.27s, eta=1:53:21
    2021-08-11 15:03:02 [INFO]	[TRAIN] Epoch=37/270, Step=60/74, loss=22.476923, lr=0.000125, time_each_step=0.26s, eta=1:53:20
    2021-08-11 15:03:02 [INFO]	[TRAIN] Epoch=37/270, Step=62/74, loss=16.822701, lr=0.000125, time_each_step=0.23s, eta=1:53:19
    2021-08-11 15:03:03 [INFO]	[TRAIN] Epoch=37/270, Step=64/74, loss=21.466036, lr=0.000125, time_each_step=0.22s, eta=1:53:19
    2021-08-11 15:03:03 [INFO]	[TRAIN] Epoch=37/270, Step=66/74, loss=15.369672, lr=0.000125, time_each_step=0.21s, eta=1:53:18
    2021-08-11 15:03:04 [INFO]	[TRAIN] Epoch=37/270, Step=68/74, loss=29.520388, lr=0.000125, time_each_step=0.2s, eta=1:53:18
    2021-08-11 15:03:04 [INFO]	[TRAIN] Epoch=37/270, Step=70/74, loss=23.93655, lr=0.000125, time_each_step=0.2s, eta=1:53:17
    2021-08-11 15:03:04 [INFO]	[TRAIN] Epoch=37/270, Step=72/74, loss=15.922249, lr=0.000125, time_each_step=0.2s, eta=1:53:17
    2021-08-11 15:03:05 [INFO]	[TRAIN] Epoch=37/270, Step=74/74, loss=25.872498, lr=0.000125, time_each_step=0.2s, eta=1:53:17
    2021-08-11 15:03:05 [INFO]	[TRAIN] Epoch 37 finished, loss=26.53253, lr=0.000125 .
    2021-08-11 15:03:16 [INFO]	[TRAIN] Epoch=38/270, Step=2/74, loss=61.713821, lr=0.000125, time_each_step=0.72s, eta=1:58:41
    2021-08-11 15:03:17 [INFO]	[TRAIN] Epoch=38/270, Step=4/74, loss=13.762304, lr=0.000125, time_each_step=0.75s, eta=1:58:41
    2021-08-11 15:03:17 [INFO]	[TRAIN] Epoch=38/270, Step=6/74, loss=18.277803, lr=0.000125, time_each_step=0.76s, eta=1:58:40
    2021-08-11 15:03:18 [INFO]	[TRAIN] Epoch=38/270, Step=8/74, loss=43.48468, lr=0.000125, time_each_step=0.78s, eta=1:58:40
    2021-08-11 15:03:19 [INFO]	[TRAIN] Epoch=38/270, Step=10/74, loss=14.579135, lr=0.000125, time_each_step=0.8s, eta=1:58:40
    2021-08-11 15:03:20 [INFO]	[TRAIN] Epoch=38/270, Step=12/74, loss=42.897354, lr=0.000125, time_each_step=0.84s, eta=1:58:40
    2021-08-11 15:03:21 [INFO]	[TRAIN] Epoch=38/270, Step=14/74, loss=17.632631, lr=0.000125, time_each_step=0.85s, eta=1:58:39
    2021-08-11 15:03:21 [INFO]	[TRAIN] Epoch=38/270, Step=16/74, loss=21.402607, lr=0.000125, time_each_step=0.87s, eta=1:58:39
    2021-08-11 15:03:22 [INFO]	[TRAIN] Epoch=38/270, Step=18/74, loss=20.616604, lr=0.000125, time_each_step=0.89s, eta=1:58:39
    2021-08-11 15:03:23 [INFO]	[TRAIN] Epoch=38/270, Step=20/74, loss=12.421739, lr=0.000125, time_each_step=0.91s, eta=1:58:38
    2021-08-11 15:03:24 [INFO]	[TRAIN] Epoch=38/270, Step=22/74, loss=17.156403, lr=0.000125, time_each_step=0.4s, eta=1:58:9
    2021-08-11 15:03:24 [INFO]	[TRAIN] Epoch=38/270, Step=24/74, loss=41.687111, lr=0.000125, time_each_step=0.39s, eta=1:58:8
    2021-08-11 15:03:25 [INFO]	[TRAIN] Epoch=38/270, Step=26/74, loss=43.341244, lr=0.000125, time_each_step=0.4s, eta=1:58:8
    2021-08-11 15:03:26 [INFO]	[TRAIN] Epoch=38/270, Step=28/74, loss=29.420271, lr=0.000125, time_each_step=0.39s, eta=1:58:7
    2021-08-11 15:03:26 [INFO]	[TRAIN] Epoch=38/270, Step=30/74, loss=19.452921, lr=0.000125, time_each_step=0.37s, eta=1:58:5
    2021-08-11 15:03:27 [INFO]	[TRAIN] Epoch=38/270, Step=32/74, loss=27.846409, lr=0.000125, time_each_step=0.36s, eta=1:58:4
    2021-08-11 15:03:28 [INFO]	[TRAIN] Epoch=38/270, Step=34/74, loss=18.065872, lr=0.000125, time_each_step=0.38s, eta=1:58:4
    2021-08-11 15:03:29 [INFO]	[TRAIN] Epoch=38/270, Step=36/74, loss=35.532181, lr=0.000125, time_each_step=0.39s, eta=1:58:3
    2021-08-11 15:03:30 [INFO]	[TRAIN] Epoch=38/270, Step=38/74, loss=20.751095, lr=0.000125, time_each_step=0.39s, eta=1:58:3
    2021-08-11 15:03:31 [INFO]	[TRAIN] Epoch=38/270, Step=40/74, loss=28.169968, lr=0.000125, time_each_step=0.41s, eta=1:58:2
    2021-08-11 15:03:32 [INFO]	[TRAIN] Epoch=38/270, Step=42/74, loss=27.85981, lr=0.000125, time_each_step=0.41s, eta=1:58:2
    2021-08-11 15:03:32 [INFO]	[TRAIN] Epoch=38/270, Step=44/74, loss=33.029671, lr=0.000125, time_each_step=0.39s, eta=1:58:0
    2021-08-11 15:03:33 [INFO]	[TRAIN] Epoch=38/270, Step=46/74, loss=33.079895, lr=0.000125, time_each_step=0.38s, eta=1:57:59
    2021-08-11 15:03:33 [INFO]	[TRAIN] Epoch=38/270, Step=48/74, loss=25.225912, lr=0.000125, time_each_step=0.36s, eta=1:57:58
    2021-08-11 15:03:34 [INFO]	[TRAIN] Epoch=38/270, Step=50/74, loss=32.969929, lr=0.000125, time_each_step=0.36s, eta=1:57:57
    2021-08-11 15:03:34 [INFO]	[TRAIN] Epoch=38/270, Step=52/74, loss=27.610575, lr=0.000125, time_each_step=0.35s, eta=1:57:56
    2021-08-11 15:03:34 [INFO]	[TRAIN] Epoch=38/270, Step=54/74, loss=31.073559, lr=0.000125, time_each_step=0.32s, eta=1:57:55
    2021-08-11 15:03:35 [INFO]	[TRAIN] Epoch=38/270, Step=56/74, loss=12.861688, lr=0.000125, time_each_step=0.28s, eta=1:57:54
    2021-08-11 15:03:35 [INFO]	[TRAIN] Epoch=38/270, Step=58/74, loss=23.451523, lr=0.000125, time_each_step=0.26s, eta=1:57:53
    2021-08-11 15:03:36 [INFO]	[TRAIN] Epoch=38/270, Step=60/74, loss=48.467388, lr=0.000125, time_each_step=0.23s, eta=1:57:52
    2021-08-11 15:03:36 [INFO]	[TRAIN] Epoch=38/270, Step=62/74, loss=17.340225, lr=0.000125, time_each_step=0.21s, eta=1:57:51
    2021-08-11 15:03:36 [INFO]	[TRAIN] Epoch=38/270, Step=64/74, loss=24.792376, lr=0.000125, time_each_step=0.21s, eta=1:57:51
    2021-08-11 15:03:37 [INFO]	[TRAIN] Epoch=38/270, Step=66/74, loss=19.185764, lr=0.000125, time_each_step=0.19s, eta=1:57:50
    2021-08-11 15:03:37 [INFO]	[TRAIN] Epoch=38/270, Step=68/74, loss=19.516817, lr=0.000125, time_each_step=0.19s, eta=1:57:50
    2021-08-11 15:03:38 [INFO]	[TRAIN] Epoch=38/270, Step=70/74, loss=18.976048, lr=0.000125, time_each_step=0.2s, eta=1:57:49
    2021-08-11 15:03:38 [INFO]	[TRAIN] Epoch=38/270, Step=72/74, loss=17.626408, lr=0.000125, time_each_step=0.19s, eta=1:57:49
    2021-08-11 15:03:38 [INFO]	[TRAIN] Epoch=38/270, Step=74/74, loss=13.908101, lr=0.000125, time_each_step=0.2s, eta=1:57:49
    2021-08-11 15:03:38 [INFO]	[TRAIN] Epoch 38 finished, loss=24.085888, lr=0.000125 .
    2021-08-11 15:03:51 [INFO]	[TRAIN] Epoch=39/270, Step=2/74, loss=24.202784, lr=0.000125, time_each_step=0.78s, eta=2:17:29
    2021-08-11 15:03:52 [INFO]	[TRAIN] Epoch=39/270, Step=4/74, loss=21.429823, lr=0.000125, time_each_step=0.81s, eta=2:17:29
    2021-08-11 15:03:52 [INFO]	[TRAIN] Epoch=39/270, Step=6/74, loss=24.955059, lr=0.000125, time_each_step=0.82s, eta=2:17:28
    2021-08-11 15:03:53 [INFO]	[TRAIN] Epoch=39/270, Step=8/74, loss=29.884417, lr=0.000125, time_each_step=0.85s, eta=2:17:28
    2021-08-11 15:03:54 [INFO]	[TRAIN] Epoch=39/270, Step=10/74, loss=18.06675, lr=0.000125, time_each_step=0.88s, eta=2:17:29
    2021-08-11 15:03:55 [INFO]	[TRAIN] Epoch=39/270, Step=12/74, loss=17.772926, lr=0.000125, time_each_step=0.91s, eta=2:17:29
    2021-08-11 15:03:55 [INFO]	[TRAIN] Epoch=39/270, Step=14/74, loss=18.695032, lr=0.000125, time_each_step=0.92s, eta=2:17:27
    2021-08-11 15:03:56 [INFO]	[TRAIN] Epoch=39/270, Step=16/74, loss=31.242077, lr=0.000125, time_each_step=0.93s, eta=2:17:26
    2021-08-11 15:03:57 [INFO]	[TRAIN] Epoch=39/270, Step=18/74, loss=42.111977, lr=0.000125, time_each_step=0.94s, eta=2:17:25
    2021-08-11 15:03:58 [INFO]	[TRAIN] Epoch=39/270, Step=20/74, loss=23.330227, lr=0.000125, time_each_step=0.96s, eta=2:17:24
    2021-08-11 15:03:58 [INFO]	[TRAIN] Epoch=39/270, Step=22/74, loss=13.423571, lr=0.000125, time_each_step=0.39s, eta=2:16:52
    2021-08-11 15:03:59 [INFO]	[TRAIN] Epoch=39/270, Step=24/74, loss=23.151699, lr=0.000125, time_each_step=0.38s, eta=2:16:51
    2021-08-11 15:04:00 [INFO]	[TRAIN] Epoch=39/270, Step=26/74, loss=42.169373, lr=0.000125, time_each_step=0.38s, eta=2:16:50
    2021-08-11 15:04:01 [INFO]	[TRAIN] Epoch=39/270, Step=28/74, loss=14.644203, lr=0.000125, time_each_step=0.37s, eta=2:16:49
    2021-08-11 15:04:01 [INFO]	[TRAIN] Epoch=39/270, Step=30/74, loss=26.227652, lr=0.000125, time_each_step=0.36s, eta=2:16:48
    2021-08-11 15:04:02 [INFO]	[TRAIN] Epoch=39/270, Step=32/74, loss=17.345024, lr=0.000125, time_each_step=0.36s, eta=2:16:47
    2021-08-11 15:04:03 [INFO]	[TRAIN] Epoch=39/270, Step=34/74, loss=26.734283, lr=0.000125, time_each_step=0.38s, eta=2:16:47
    2021-08-11 15:04:04 [INFO]	[TRAIN] Epoch=39/270, Step=36/74, loss=38.693855, lr=0.000125, time_each_step=0.4s, eta=2:16:47
    2021-08-11 15:04:05 [INFO]	[TRAIN] Epoch=39/270, Step=38/74, loss=19.825018, lr=0.000125, time_each_step=0.42s, eta=2:16:47
    2021-08-11 15:04:06 [INFO]	[TRAIN] Epoch=39/270, Step=40/74, loss=24.155378, lr=0.000125, time_each_step=0.42s, eta=2:16:46
    2021-08-11 15:04:06 [INFO]	[TRAIN] Epoch=39/270, Step=42/74, loss=26.226816, lr=0.000125, time_each_step=0.4s, eta=2:16:45
    2021-08-11 15:04:07 [INFO]	[TRAIN] Epoch=39/270, Step=44/74, loss=18.7272, lr=0.000125, time_each_step=0.37s, eta=2:16:43
    2021-08-11 15:04:07 [INFO]	[TRAIN] Epoch=39/270, Step=46/74, loss=39.418983, lr=0.000125, time_each_step=0.36s, eta=2:16:42
    2021-08-11 15:04:07 [INFO]	[TRAIN] Epoch=39/270, Step=48/74, loss=34.034637, lr=0.000125, time_each_step=0.34s, eta=2:16:41
    2021-08-11 15:04:08 [INFO]	[TRAIN] Epoch=39/270, Step=50/74, loss=31.672438, lr=0.000125, time_each_step=0.32s, eta=2:16:40
    2021-08-11 15:04:08 [INFO]	[TRAIN] Epoch=39/270, Step=52/74, loss=16.304928, lr=0.000125, time_each_step=0.3s, eta=2:16:39
    2021-08-11 15:04:08 [INFO]	[TRAIN] Epoch=39/270, Step=54/74, loss=23.218542, lr=0.000125, time_each_step=0.27s, eta=2:16:38
    2021-08-11 15:04:09 [INFO]	[TRAIN] Epoch=39/270, Step=56/74, loss=57.846046, lr=0.000125, time_each_step=0.24s, eta=2:16:36
    2021-08-11 15:04:09 [INFO]	[TRAIN] Epoch=39/270, Step=58/74, loss=22.670326, lr=0.000125, time_each_step=0.21s, eta=2:16:35
    2021-08-11 15:04:10 [INFO]	[TRAIN] Epoch=39/270, Step=60/74, loss=26.128668, lr=0.000125, time_each_step=0.19s, eta=2:16:35
    2021-08-11 15:04:10 [INFO]	[TRAIN] Epoch=39/270, Step=62/74, loss=29.645052, lr=0.000125, time_each_step=0.2s, eta=2:16:35
    2021-08-11 15:04:11 [INFO]	[TRAIN] Epoch=39/270, Step=64/74, loss=22.953688, lr=0.000125, time_each_step=0.21s, eta=2:16:34
    2021-08-11 15:04:11 [INFO]	[TRAIN] Epoch=39/270, Step=66/74, loss=21.419842, lr=0.000125, time_each_step=0.21s, eta=2:16:34
    2021-08-11 15:04:11 [INFO]	[TRAIN] Epoch=39/270, Step=68/74, loss=14.445843, lr=0.000125, time_each_step=0.21s, eta=2:16:33
    2021-08-11 15:04:12 [INFO]	[TRAIN] Epoch=39/270, Step=70/74, loss=27.387794, lr=0.000125, time_each_step=0.21s, eta=2:16:33
    2021-08-11 15:04:12 [INFO]	[TRAIN] Epoch=39/270, Step=72/74, loss=16.704132, lr=0.000125, time_each_step=0.21s, eta=2:16:33
    2021-08-11 15:04:13 [INFO]	[TRAIN] Epoch=39/270, Step=74/74, loss=20.604801, lr=0.000125, time_each_step=0.2s, eta=2:16:32
    2021-08-11 15:04:13 [INFO]	[TRAIN] Epoch 39 finished, loss=24.527138, lr=0.000125 .
    2021-08-11 15:04:31 [INFO]	[TRAIN] Epoch=40/270, Step=2/74, loss=23.142214, lr=0.000125, time_each_step=1.08s, eta=2:19:8
    2021-08-11 15:04:32 [INFO]	[TRAIN] Epoch=40/270, Step=4/74, loss=18.960772, lr=0.000125, time_each_step=1.11s, eta=2:19:7
    2021-08-11 15:04:32 [INFO]	[TRAIN] Epoch=40/270, Step=6/74, loss=40.904839, lr=0.000125, time_each_step=1.12s, eta=2:19:6
    2021-08-11 15:04:33 [INFO]	[TRAIN] Epoch=40/270, Step=8/74, loss=32.592182, lr=0.000125, time_each_step=1.11s, eta=2:19:3
    2021-08-11 15:04:33 [INFO]	[TRAIN] Epoch=40/270, Step=10/74, loss=22.952431, lr=0.000125, time_each_step=1.1s, eta=2:19:0
    2021-08-11 15:04:34 [INFO]	[TRAIN] Epoch=40/270, Step=12/74, loss=39.257217, lr=0.000125, time_each_step=1.14s, eta=2:19:1
    2021-08-11 15:04:35 [INFO]	[TRAIN] Epoch=40/270, Step=14/74, loss=17.326483, lr=0.000125, time_each_step=1.18s, eta=2:19:1
    2021-08-11 15:04:37 [INFO]	[TRAIN] Epoch=40/270, Step=16/74, loss=47.541145, lr=0.000125, time_each_step=1.23s, eta=2:19:1
    2021-08-11 15:04:37 [INFO]	[TRAIN] Epoch=40/270, Step=18/74, loss=18.728561, lr=0.000125, time_each_step=1.26s, eta=2:19:0
    2021-08-11 15:04:38 [INFO]	[TRAIN] Epoch=40/270, Step=20/74, loss=27.534309, lr=0.000125, time_each_step=1.28s, eta=2:18:59
    2021-08-11 15:04:39 [INFO]	[TRAIN] Epoch=40/270, Step=22/74, loss=20.162769, lr=0.000125, time_each_step=0.41s, eta=2:18:11
    2021-08-11 15:04:40 [INFO]	[TRAIN] Epoch=40/270, Step=24/74, loss=26.266899, lr=0.000125, time_each_step=0.41s, eta=2:18:10
    2021-08-11 15:04:41 [INFO]	[TRAIN] Epoch=40/270, Step=26/74, loss=19.021393, lr=0.000125, time_each_step=0.44s, eta=2:18:11
    2021-08-11 15:04:42 [INFO]	[TRAIN] Epoch=40/270, Step=28/74, loss=39.447891, lr=0.000125, time_each_step=0.46s, eta=2:18:11
    2021-08-11 15:04:43 [INFO]	[TRAIN] Epoch=40/270, Step=30/74, loss=36.257668, lr=0.000125, time_each_step=0.5s, eta=2:18:12
    2021-08-11 15:04:44 [INFO]	[TRAIN] Epoch=40/270, Step=32/74, loss=44.571384, lr=0.000125, time_each_step=0.48s, eta=2:18:10
    2021-08-11 15:04:44 [INFO]	[TRAIN] Epoch=40/270, Step=34/74, loss=33.692265, lr=0.000125, time_each_step=0.45s, eta=2:18:8
    2021-08-11 15:04:45 [INFO]	[TRAIN] Epoch=40/270, Step=36/74, loss=17.56411, lr=0.000125, time_each_step=0.41s, eta=2:18:6
    2021-08-11 15:04:45 [INFO]	[TRAIN] Epoch=40/270, Step=38/74, loss=21.946148, lr=0.000125, time_each_step=0.4s, eta=2:18:4
    2021-08-11 15:04:46 [INFO]	[TRAIN] Epoch=40/270, Step=40/74, loss=24.756111, lr=0.000125, time_each_step=0.4s, eta=2:18:4
    2021-08-11 15:04:47 [INFO]	[TRAIN] Epoch=40/270, Step=42/74, loss=26.260798, lr=0.000125, time_each_step=0.39s, eta=2:18:2
    2021-08-11 15:04:47 [INFO]	[TRAIN] Epoch=40/270, Step=44/74, loss=22.443787, lr=0.000125, time_each_step=0.36s, eta=2:18:1
    2021-08-11 15:04:48 [INFO]	[TRAIN] Epoch=40/270, Step=46/74, loss=47.99707, lr=0.000125, time_each_step=0.33s, eta=2:17:59
    2021-08-11 15:04:48 [INFO]	[TRAIN] Epoch=40/270, Step=48/74, loss=19.220545, lr=0.000125, time_each_step=0.31s, eta=2:17:58
    2021-08-11 15:04:48 [INFO]	[TRAIN] Epoch=40/270, Step=50/74, loss=19.201748, lr=0.000125, time_each_step=0.28s, eta=2:17:56
    2021-08-11 15:04:49 [INFO]	[TRAIN] Epoch=40/270, Step=52/74, loss=32.474754, lr=0.000125, time_each_step=0.26s, eta=2:17:56
    2021-08-11 15:04:49 [INFO]	[TRAIN] Epoch=40/270, Step=54/74, loss=21.442453, lr=0.000125, time_each_step=0.26s, eta=2:17:55
    2021-08-11 15:04:50 [INFO]	[TRAIN] Epoch=40/270, Step=56/74, loss=16.639502, lr=0.000125, time_each_step=0.24s, eta=2:17:54
    2021-08-11 15:04:50 [INFO]	[TRAIN] Epoch=40/270, Step=58/74, loss=27.74048, lr=0.000125, time_each_step=0.23s, eta=2:17:54
    2021-08-11 15:04:51 [INFO]	[TRAIN] Epoch=40/270, Step=60/74, loss=12.290872, lr=0.000125, time_each_step=0.22s, eta=2:17:53
    2021-08-11 15:04:51 [INFO]	[TRAIN] Epoch=40/270, Step=62/74, loss=23.915573, lr=0.000125, time_each_step=0.23s, eta=2:17:53
    2021-08-11 15:04:52 [INFO]	[TRAIN] Epoch=40/270, Step=64/74, loss=20.229313, lr=0.000125, time_each_step=0.23s, eta=2:17:52
    2021-08-11 15:04:52 [INFO]	[TRAIN] Epoch=40/270, Step=66/74, loss=36.33392, lr=0.000125, time_each_step=0.22s, eta=2:17:52
    2021-08-11 15:04:52 [INFO]	[TRAIN] Epoch=40/270, Step=68/74, loss=21.503414, lr=0.000125, time_each_step=0.22s, eta=2:17:51
    2021-08-11 15:04:53 [INFO]	[TRAIN] Epoch=40/270, Step=70/74, loss=21.697731, lr=0.000125, time_each_step=0.22s, eta=2:17:51
    2021-08-11 15:04:53 [INFO]	[TRAIN] Epoch=40/270, Step=72/74, loss=21.9021, lr=0.000125, time_each_step=0.21s, eta=2:17:50
    2021-08-11 15:04:53 [INFO]	[TRAIN] Epoch=40/270, Step=74/74, loss=46.799339, lr=0.000125, time_each_step=0.2s, eta=2:17:50
    2021-08-11 15:04:53 [INFO]	[TRAIN] Epoch 40 finished, loss=25.946478, lr=0.000125 .
    2021-08-11 15:04:53 [INFO]	Start to evaluating(total_samples=170, total_steps=22)...


    100%|██████████| 22/22 [00:08<00:00,  2.52it/s]


    2021-08-11 15:05:02 [INFO]	[EVAL] Finished, Epoch=40, bbox_map=49.058375 .
    2021-08-11 15:05:08 [INFO]	Model saved in output/yolov3_DarkNet53/best_model.
    2021-08-11 15:05:14 [INFO]	Model saved in output/yolov3_DarkNet53/epoch_40.
    2021-08-11 15:05:14 [INFO]	Current evaluated best model in eval_dataset is epoch_40, bbox_map=49.05837500109879
    2021-08-11 15:27:03 [INFO]	[TRAIN] Epoch=79/270, Step=42/74, loss=12.301958, lr=0.000125, time_each_step=0.39s, eta=1:54:40
    2021-08-11 15:27:03 [INFO]	[TRAIN] Epoch=79/270, Step=44/74, loss=41.615589, lr=0.000125, time_each_step=0.38s, eta=1:54:39
    2021-08-11 15:27:04 [INFO]	[TRAIN] Epoch=79/270, Step=46/74, loss=14.294861, lr=0.000125, time_each_step=0.38s, eta=1:54:38
    2021-08-11 15:27:05 [INFO]	[TRAIN] Epoch=79/270, Step=48/74, loss=17.687487, lr=0.000125, time_each_step=0.37s, eta=1:54:37
    2021-08-11 15:27:05 [INFO]	[TRAIN] Epoch=79/270, Step=50/74, loss=25.448942, lr=0.000125, time_each_step=0.36s, eta=1:54:36
    2021-08-11 15:27:06 [INFO]	[TRAIN] Epoch=79/270, Step=52/74, loss=20.617859, lr=0.000125, time_each_step=0.35s, eta=1:54:35
    2021-08-11 15:27:06 [INFO]	[TRAIN] Epoch=79/270, Step=54/74, loss=12.241684, lr=0.000125, time_each_step=0.32s, eta=1:54:34
    2021-08-11 15:27:07 [INFO]	[TRAIN] Epoch=79/270, Step=56/74, loss=26.019966, lr=0.000125, time_each_step=0.3s, eta=1:54:33
    2021-08-11 15:27:07 [INFO]	[TRAIN] Epoch=79/270, Step=58/74, loss=28.120443, lr=0.000125, time_each_step=0.27s, eta=1:54:32
    2021-08-11 15:27:07 [INFO]	[TRAIN] Epoch=79/270, Step=60/74, loss=28.597652, lr=0.000125, time_each_step=0.26s, eta=1:54:31
    2021-08-11 15:27:08 [INFO]	[TRAIN] Epoch=79/270, Step=62/74, loss=25.840469, lr=0.000125, time_each_step=0.24s, eta=1:54:30
    2021-08-11 15:27:08 [INFO]	[TRAIN] Epoch=79/270, Step=64/74, loss=17.490973, lr=0.000125, time_each_step=0.24s, eta=1:54:30
    2021-08-11 15:27:09 [INFO]	[TRAIN] Epoch=79/270, Step=66/74, loss=54.776783, lr=0.000125, time_each_step=0.23s, eta=1:54:29
    2021-08-11 15:27:09 [INFO]	[TRAIN] Epoch=79/270, Step=68/74, loss=14.990078, lr=0.000125, time_each_step=0.21s, eta=1:54:29
    2021-08-11 15:27:09 [INFO]	[TRAIN] Epoch=79/270, Step=70/74, loss=21.487331, lr=0.000125, time_each_step=0.19s, eta=1:54:28
    2021-08-11 15:27:10 [INFO]	[TRAIN] Epoch=79/270, Step=72/74, loss=14.205408, lr=0.000125, time_each_step=0.18s, eta=1:54:28
    2021-08-11 15:27:10 [INFO]	[TRAIN] Epoch=79/270, Step=74/74, loss=20.269062, lr=0.000125, time_each_step=0.19s, eta=1:54:27
    2021-08-11 15:27:10 [INFO]	[TRAIN] Epoch 79 finished, loss=23.095877, lr=0.000125 .
    2021-08-11 15:27:16 [INFO]	[TRAIN] Epoch=80/270, Step=2/74, loss=17.042706, lr=0.000125, time_each_step=0.46s, eta=1:32:23
    2021-08-11 15:27:17 [INFO]	[TRAIN] Epoch=80/270, Step=4/74, loss=22.15415, lr=0.000125, time_each_step=0.5s, eta=1:32:24
    2021-08-11 15:27:18 [INFO]	[TRAIN] Epoch=80/270, Step=6/74, loss=15.202286, lr=0.000125, time_each_step=0.52s, eta=1:32:25
    2021-08-11 15:27:19 [INFO]	[TRAIN] Epoch=80/270, Step=8/74, loss=35.769417, lr=0.000125, time_each_step=0.55s, eta=1:32:25
    2021-08-11 15:27:19 [INFO]	[TRAIN] Epoch=80/270, Step=10/74, loss=32.911304, lr=0.000125, time_each_step=0.56s, eta=1:32:25
    2021-08-11 15:27:20 [INFO]	[TRAIN] Epoch=80/270, Step=12/74, loss=19.531046, lr=0.000125, time_each_step=0.55s, eta=1:32:24
    2021-08-11 15:27:21 [INFO]	[TRAIN] Epoch=80/270, Step=14/74, loss=22.993843, lr=0.000125, time_each_step=0.59s, eta=1:32:25
    2021-08-11 15:27:22 [INFO]	[TRAIN] Epoch=80/270, Step=16/74, loss=20.417568, lr=0.000125, time_each_step=0.62s, eta=1:32:25
    2021-08-11 15:27:22 [INFO]	[TRAIN] Epoch=80/270, Step=18/74, loss=15.344674, lr=0.000125, time_each_step=0.64s, eta=1:32:25
    2021-08-11 15:27:23 [INFO]	[TRAIN] Epoch=80/270, Step=20/74, loss=21.050362, lr=0.000125, time_each_step=0.65s, eta=1:32:25
    2021-08-11 15:27:24 [INFO]	[TRAIN] Epoch=80/270, Step=22/74, loss=9.379828, lr=0.000125, time_each_step=0.41s, eta=1:32:10
    2021-08-11 15:27:25 [INFO]	[TRAIN] Epoch=80/270, Step=24/74, loss=31.818069, lr=0.000125, time_each_step=0.38s, eta=1:32:9
    2021-08-11 15:27:25 [INFO]	[TRAIN] Epoch=80/270, Step=26/74, loss=23.921095, lr=0.000125, time_each_step=0.37s, eta=1:32:7
    2021-08-11 15:27:26 [INFO]	[TRAIN] Epoch=80/270, Step=28/74, loss=17.752522, lr=0.000125, time_each_step=0.35s, eta=1:32:5
    2021-08-11 15:27:26 [INFO]	[TRAIN] Epoch=80/270, Step=30/74, loss=29.093307, lr=0.000125, time_each_step=0.35s, eta=1:32:5
    2021-08-11 15:27:27 [INFO]	[TRAIN] Epoch=80/270, Step=32/74, loss=12.308991, lr=0.000125, time_each_step=0.37s, eta=1:32:5
    2021-08-11 15:27:28 [INFO]	[TRAIN] Epoch=80/270, Step=34/74, loss=24.065111, lr=0.000125, time_each_step=0.37s, eta=1:32:4
    2021-08-11 15:27:29 [INFO]	[TRAIN] Epoch=80/270, Step=36/74, loss=15.67263, lr=0.000125, time_each_step=0.38s, eta=1:32:4
    2021-08-11 15:27:30 [INFO]	[TRAIN] Epoch=80/270, Step=38/74, loss=14.540319, lr=0.000125, time_each_step=0.37s, eta=1:32:3
    2021-08-11 15:27:30 [INFO]	[TRAIN] Epoch=80/270, Step=40/74, loss=38.52417, lr=0.000125, time_each_step=0.36s, eta=1:32:2
    2021-08-11 15:27:31 [INFO]	[TRAIN] Epoch=80/270, Step=42/74, loss=17.060579, lr=0.000125, time_each_step=0.35s, eta=1:32:1
    2021-08-11 15:27:31 [INFO]	[TRAIN] Epoch=80/270, Step=44/74, loss=37.864952, lr=0.000125, time_each_step=0.34s, eta=1:32:0
    2021-08-11 15:27:32 [INFO]	[TRAIN] Epoch=80/270, Step=46/74, loss=18.777164, lr=0.000125, time_each_step=0.33s, eta=1:31:59
    2021-08-11 15:27:32 [INFO]	[TRAIN] Epoch=80/270, Step=48/74, loss=27.737745, lr=0.000125, time_each_step=0.34s, eta=1:31:58
    2021-08-11 15:27:33 [INFO]	[TRAIN] Epoch=80/270, Step=50/74, loss=20.735155, lr=0.000125, time_each_step=0.33s, eta=1:31:57
    2021-08-11 15:27:33 [INFO]	[TRAIN] Epoch=80/270, Step=52/74, loss=22.869843, lr=0.000125, time_each_step=0.31s, eta=1:31:56
    2021-08-11 15:27:34 [INFO]	[TRAIN] Epoch=80/270, Step=54/74, loss=23.236549, lr=0.000125, time_each_step=0.29s, eta=1:31:55
    2021-08-11 15:27:34 [INFO]	[TRAIN] Epoch=80/270, Step=56/74, loss=19.325306, lr=0.000125, time_each_step=0.26s, eta=1:31:54
    2021-08-11 15:27:35 [INFO]	[TRAIN] Epoch=80/270, Step=58/74, loss=20.096006, lr=0.000125, time_each_step=0.25s, eta=1:31:53
    2021-08-11 15:27:35 [INFO]	[TRAIN] Epoch=80/270, Step=60/74, loss=10.842567, lr=0.000125, time_each_step=0.24s, eta=1:31:53
    2021-08-11 15:27:36 [INFO]	[TRAIN] Epoch=80/270, Step=62/74, loss=23.766228, lr=0.000125, time_each_step=0.24s, eta=1:31:52
    2021-08-11 15:27:36 [INFO]	[TRAIN] Epoch=80/270, Step=64/74, loss=21.181334, lr=0.000125, time_each_step=0.23s, eta=1:31:52
    2021-08-11 15:27:36 [INFO]	[TRAIN] Epoch=80/270, Step=66/74, loss=22.511557, lr=0.000125, time_each_step=0.22s, eta=1:31:51
    2021-08-11 15:27:37 [INFO]	[TRAIN] Epoch=80/270, Step=68/74, loss=14.397124, lr=0.000125, time_each_step=0.22s, eta=1:31:51
    2021-08-11 15:27:37 [INFO]	[TRAIN] Epoch=80/270, Step=70/74, loss=11.603191, lr=0.000125, time_each_step=0.2s, eta=1:31:50
    2021-08-11 15:27:37 [INFO]	[TRAIN] Epoch=80/270, Step=72/74, loss=28.138184, lr=0.000125, time_each_step=0.19s, eta=1:31:50
    2021-08-11 15:27:38 [INFO]	[TRAIN] Epoch=80/270, Step=74/74, loss=12.942713, lr=0.000125, time_each_step=0.17s, eta=1:31:49
    2021-08-11 15:27:38 [INFO]	[TRAIN] Epoch 80 finished, loss=21.933819, lr=0.000125 .
    2021-08-11 15:27:38 [INFO]	Start to evaluating(total_samples=170, total_steps=22)...


    100%|██████████| 22/22 [00:13<00:00,  1.61it/s]


    2021-08-11 15:27:51 [INFO]	[EVAL] Finished, Epoch=80, bbox_map=60.814431 .
    2021-08-11 15:27:58 [INFO]	Model saved in output/yolov3_DarkNet53/best_model.
    2021-08-11 15:28:03 [INFO]	Model saved in output/yolov3_DarkNet53/epoch_80.
    2021-08-11 15:28:03 [INFO]	Current evaluated best model in eval_dataset is epoch_80, bbox_map=60.814430758171355
    2021-08-11 15:28:14 [INFO]	[TRAIN] Epoch=81/270, Step=2/74, loss=12.145984, lr=0.000125, time_each_step=0.69s, eta=1:31:34
    2021-08-11 15:28:15 [INFO]	[TRAIN] Epoch=81/270, Step=4/74, loss=13.19843, lr=0.000125, time_each_step=0.73s, eta=1:31:35
    2021-08-11 15:28:16 [INFO]	[TRAIN] Epoch=81/270, Step=6/74, loss=26.974857, lr=0.000125, time_each_step=0.75s, eta=1:31:35
    2021-08-11 15:28:17 [INFO]	[TRAIN] Epoch=81/270, Step=8/74, loss=22.62273, lr=0.000125, time_each_step=0.77s, eta=1:31:34
    2021-08-11 15:28:17 [INFO]	[TRAIN] Epoch=81/270, Step=10/74, loss=14.297921, lr=0.000125, time_each_step=0.78s, eta=1:31:34
    2021-08-11 15:28:18 [INFO]	[TRAIN] Epoch=81/270, Step=12/74, loss=30.015312, lr=0.000125, time_each_step=0.82s, eta=1:31:34
    2021-08-11 15:28:19 [INFO]	[TRAIN] Epoch=81/270, Step=14/74, loss=15.902565, lr=0.000125, time_each_step=0.84s, eta=1:31:34
    2021-08-11 15:28:20 [INFO]	[TRAIN] Epoch=81/270, Step=16/74, loss=18.470806, lr=0.000125, time_each_step=0.86s, eta=1:31:34
    2021-08-11 15:28:21 [INFO]	[TRAIN] Epoch=81/270, Step=18/74, loss=25.691807, lr=0.000125, time_each_step=0.88s, eta=1:31:33
    2021-08-11 15:28:22 [INFO]	[TRAIN] Epoch=81/270, Step=20/74, loss=10.839764, lr=0.000125, time_each_step=0.93s, eta=1:31:34
    2021-08-11 15:28:23 [INFO]	[TRAIN] Epoch=81/270, Step=22/74, loss=25.675213, lr=0.000125, time_each_step=0.44s, eta=1:31:7
    2021-08-11 15:28:23 [INFO]	[TRAIN] Epoch=81/270, Step=24/74, loss=27.59827, lr=0.000125, time_each_step=0.42s, eta=1:31:5
    2021-08-11 15:28:24 [INFO]	[TRAIN] Epoch=81/270, Step=26/74, loss=19.293005, lr=0.000125, time_each_step=0.41s, eta=1:31:3
    2021-08-11 15:28:25 [INFO]	[TRAIN] Epoch=81/270, Step=28/74, loss=28.399118, lr=0.000125, time_each_step=0.41s, eta=1:31:3
    2021-08-11 15:28:25 [INFO]	[TRAIN] Epoch=81/270, Step=30/74, loss=21.249996, lr=0.000125, time_each_step=0.41s, eta=1:31:2
    2021-08-11 15:28:26 [INFO]	[TRAIN] Epoch=81/270, Step=32/74, loss=9.46666, lr=0.000125, time_each_step=0.39s, eta=1:31:0
    2021-08-11 15:28:27 [INFO]	[TRAIN] Epoch=81/270, Step=34/74, loss=27.610592, lr=0.000125, time_each_step=0.4s, eta=1:31:0
    2021-08-11 15:28:28 [INFO]	[TRAIN] Epoch=81/270, Step=36/74, loss=17.982817, lr=0.000125, time_each_step=0.4s, eta=1:30:59
    2021-08-11 15:28:28 [INFO]	[TRAIN] Epoch=81/270, Step=38/74, loss=15.546053, lr=0.000125, time_each_step=0.38s, eta=1:30:58
    2021-08-11 15:28:29 [INFO]	[TRAIN] Epoch=81/270, Step=40/74, loss=20.27882, lr=0.000125, time_each_step=0.34s, eta=1:30:55
    2021-08-11 15:28:29 [INFO]	[TRAIN] Epoch=81/270, Step=42/74, loss=22.421539, lr=0.000125, time_each_step=0.33s, eta=1:30:54
    2021-08-11 15:28:30 [INFO]	[TRAIN] Epoch=81/270, Step=44/74, loss=15.912359, lr=0.000125, time_each_step=0.32s, eta=1:30:53
    2021-08-11 15:28:30 [INFO]	[TRAIN] Epoch=81/270, Step=46/74, loss=13.513418, lr=0.000125, time_each_step=0.31s, eta=1:30:52
    2021-08-11 15:28:30 [INFO]	[TRAIN] Epoch=81/270, Step=48/74, loss=59.334736, lr=0.000125, time_each_step=0.28s, eta=1:30:51
    2021-08-11 15:28:31 [INFO]	[TRAIN] Epoch=81/270, Step=50/74, loss=20.71826, lr=0.000125, time_each_step=0.27s, eta=1:30:50
    2021-08-11 15:28:31 [INFO]	[TRAIN] Epoch=81/270, Step=52/74, loss=32.922638, lr=0.000125, time_each_step=0.25s, eta=1:30:49
    2021-08-11 15:28:31 [INFO]	[TRAIN] Epoch=81/270, Step=54/74, loss=16.272541, lr=0.000125, time_each_step=0.21s, eta=1:30:48
    2021-08-11 15:28:32 [INFO]	[TRAIN] Epoch=81/270, Step=56/74, loss=16.355276, lr=0.000125, time_each_step=0.2s, eta=1:30:47
    2021-08-11 15:28:32 [INFO]	[TRAIN] Epoch=81/270, Step=58/74, loss=10.28528, lr=0.000125, time_each_step=0.21s, eta=1:30:47
    2021-08-11 15:28:33 [INFO]	[TRAIN] Epoch=81/270, Step=60/74, loss=21.01667, lr=0.000125, time_each_step=0.21s, eta=1:30:47
    2021-08-11 15:28:33 [INFO]	[TRAIN] Epoch=81/270, Step=62/74, loss=17.463778, lr=0.000125, time_each_step=0.2s, eta=1:30:46
    2021-08-11 15:28:34 [INFO]	[TRAIN] Epoch=81/270, Step=64/74, loss=19.63438, lr=0.000125, time_each_step=0.21s, eta=1:30:46
    2021-08-11 15:28:34 [INFO]	[TRAIN] Epoch=81/270, Step=66/74, loss=17.147411, lr=0.000125, time_each_step=0.2s, eta=1:30:45
    2021-08-11 15:28:35 [INFO]	[TRAIN] Epoch=81/270, Step=68/74, loss=18.461842, lr=0.000125, time_each_step=0.21s, eta=1:30:45
    2021-08-11 15:28:35 [INFO]	[TRAIN] Epoch=81/270, Step=70/74, loss=33.25573, lr=0.000125, time_each_step=0.2s, eta=1:30:45
    2021-08-11 15:28:35 [INFO]	[TRAIN] Epoch=81/270, Step=72/74, loss=39.10952, lr=0.000125, time_each_step=0.2s, eta=1:30:44
    2021-08-11 15:28:36 [INFO]	[TRAIN] Epoch=81/270, Step=74/74, loss=17.70801, lr=0.000125, time_each_step=0.21s, eta=1:30:44
    2021-08-11 15:28:36 [INFO]	[TRAIN] Epoch 81 finished, loss=21.177896, lr=0.000125 .
    2021-08-11 15:28:40 [INFO]	[TRAIN] Epoch=82/270, Step=2/74, loss=25.845747, lr=0.000125, time_each_step=0.41s, eta=1:46:36
    2021-08-11 15:28:41 [INFO]	[TRAIN] Epoch=82/270, Step=4/74, loss=28.769094, lr=0.000125, time_each_step=0.43s, eta=1:46:36
    2021-08-11 15:28:42 [INFO]	[TRAIN] Epoch=82/270, Step=6/74, loss=20.155956, lr=0.000125, time_each_step=0.46s, eta=1:46:37
    2021-08-11 15:28:43 [INFO]	[TRAIN] Epoch=82/270, Step=8/74, loss=13.924913, lr=0.000125, time_each_step=0.47s, eta=1:46:37
    2021-08-11 15:28:43 [INFO]	[TRAIN] Epoch=82/270, Step=10/74, loss=17.285336, lr=0.000125, time_each_step=0.48s, eta=1:46:37
    2021-08-11 15:28:44 [INFO]	[TRAIN] Epoch=82/270, Step=12/74, loss=18.229191, lr=0.000125, time_each_step=0.5s, eta=1:46:37
    2021-08-11 15:28:45 [INFO]	[TRAIN] Epoch=82/270, Step=14/74, loss=33.898849, lr=0.000125, time_each_step=0.53s, eta=1:46:37
    2021-08-11 15:28:46 [INFO]	[TRAIN] Epoch=82/270, Step=16/74, loss=16.096725, lr=0.000125, time_each_step=0.56s, eta=1:46:38
    2021-08-11 15:28:47 [INFO]	[TRAIN] Epoch=82/270, Step=18/74, loss=20.700207, lr=0.000125, time_each_step=0.58s, eta=1:46:38
    2021-08-11 15:28:47 [INFO]	[TRAIN] Epoch=82/270, Step=20/74, loss=11.321684, lr=0.000125, time_each_step=0.59s, eta=1:46:38
    2021-08-11 15:28:49 [INFO]	[TRAIN] Epoch=82/270, Step=22/74, loss=19.080647, lr=0.000125, time_each_step=0.42s, eta=1:46:28
    2021-08-11 15:28:49 [INFO]	[TRAIN] Epoch=82/270, Step=24/74, loss=10.61104, lr=0.000125, time_each_step=0.41s, eta=1:46:27
    2021-08-11 15:28:50 [INFO]	[TRAIN] Epoch=82/270, Step=26/74, loss=28.570461, lr=0.000125, time_each_step=0.4s, eta=1:46:25
    2021-08-11 15:28:51 [INFO]	[TRAIN] Epoch=82/270, Step=28/74, loss=19.378754, lr=0.000125, time_each_step=0.39s, eta=1:46:24
    2021-08-11 15:28:51 [INFO]	[TRAIN] Epoch=82/270, Step=30/74, loss=21.467731, lr=0.000125, time_each_step=0.38s, eta=1:46:23
    2021-08-11 15:28:52 [INFO]	[TRAIN] Epoch=82/270, Step=32/74, loss=8.093724, lr=0.000125, time_each_step=0.38s, eta=1:46:22
    2021-08-11 15:28:53 [INFO]	[TRAIN] Epoch=82/270, Step=34/74, loss=10.285525, lr=0.000125, time_each_step=0.38s, eta=1:46:21
    2021-08-11 15:28:54 [INFO]	[TRAIN] Epoch=82/270, Step=36/74, loss=22.101418, lr=0.000125, time_each_step=0.37s, eta=1:46:20
    2021-08-11 15:28:54 [INFO]	[TRAIN] Epoch=82/270, Step=38/74, loss=25.922617, lr=0.000125, time_each_step=0.37s, eta=1:46:19
    2021-08-11 15:28:55 [INFO]	[TRAIN] Epoch=82/270, Step=40/74, loss=21.880062, lr=0.000125, time_each_step=0.38s, eta=1:46:19
    2021-08-11 15:28:56 [INFO]	[TRAIN] Epoch=82/270, Step=42/74, loss=17.78511, lr=0.000125, time_each_step=0.35s, eta=1:46:17
    2021-08-11 15:28:56 [INFO]	[TRAIN] Epoch=82/270, Step=44/74, loss=34.385464, lr=0.000125, time_each_step=0.35s, eta=1:46:16
    2021-08-11 15:28:57 [INFO]	[TRAIN] Epoch=82/270, Step=46/74, loss=31.596678, lr=0.000125, time_each_step=0.34s, eta=1:46:16
    2021-08-11 15:28:57 [INFO]	[TRAIN] Epoch=82/270, Step=48/74, loss=23.46973, lr=0.000125, time_each_step=0.34s, eta=1:46:15
    2021-08-11 15:28:58 [INFO]	[TRAIN] Epoch=82/270, Step=50/74, loss=44.226204, lr=0.000125, time_each_step=0.34s, eta=1:46:14
    2021-08-11 15:28:58 [INFO]	[TRAIN] Epoch=82/270, Step=52/74, loss=27.984383, lr=0.000125, time_each_step=0.33s, eta=1:46:13
    2021-08-11 15:28:59 [INFO]	[TRAIN] Epoch=82/270, Step=54/74, loss=19.520294, lr=0.000125, time_each_step=0.29s, eta=1:46:12
    2021-08-11 15:28:59 [INFO]	[TRAIN] Epoch=82/270, Step=56/74, loss=14.043155, lr=0.000125, time_each_step=0.28s, eta=1:46:11
    2021-08-11 15:29:00 [INFO]	[TRAIN] Epoch=82/270, Step=58/74, loss=24.372932, lr=0.000125, time_each_step=0.27s, eta=1:46:10
    2021-08-11 15:29:00 [INFO]	[TRAIN] Epoch=82/270, Step=60/74, loss=10.863235, lr=0.000125, time_each_step=0.25s, eta=1:46:9
    2021-08-11 15:29:00 [INFO]	[TRAIN] Epoch=82/270, Step=62/74, loss=25.095295, lr=0.000125, time_each_step=0.23s, eta=1:46:9
    2021-08-11 15:29:01 [INFO]	[TRAIN] Epoch=82/270, Step=64/74, loss=31.260658, lr=0.000125, time_each_step=0.22s, eta=1:46:8
    2021-08-11 15:29:01 [INFO]	[TRAIN] Epoch=82/270, Step=66/74, loss=24.054676, lr=0.000125, time_each_step=0.2s, eta=1:46:8
    2021-08-11 15:29:01 [INFO]	[TRAIN] Epoch=82/270, Step=68/74, loss=20.768501, lr=0.000125, time_each_step=0.19s, eta=1:46:7
    2021-08-11 15:29:02 [INFO]	[TRAIN] Epoch=82/270, Step=70/74, loss=33.681396, lr=0.000125, time_each_step=0.19s, eta=1:46:7
    2021-08-11 15:29:02 [INFO]	[TRAIN] Epoch=82/270, Step=72/74, loss=26.086864, lr=0.000125, time_each_step=0.19s, eta=1:46:6
    2021-08-11 15:29:03 [INFO]	[TRAIN] Epoch=82/270, Step=74/74, loss=11.652936, lr=0.000125, time_each_step=0.2s, eta=1:46:6
    2021-08-11 15:29:03 [INFO]	[TRAIN] Epoch 82 finished, loss=20.882753, lr=0.000125 .
    2021-08-11 15:29:18 [INFO]	[TRAIN] Epoch=83/270, Step=2/74, loss=9.288946, lr=0.000125, time_each_step=0.94s, eta=1:29:21
    2021-08-11 15:29:19 [INFO]	[TRAIN] Epoch=83/270, Step=4/74, loss=17.913214, lr=0.000125, time_each_step=0.97s, eta=1:29:21
    2021-08-11 15:29:20 [INFO]	[TRAIN] Epoch=83/270, Step=6/74, loss=18.293091, lr=0.000125, time_each_step=0.99s, eta=1:29:21
    2021-08-11 15:29:21 [INFO]	[TRAIN] Epoch=83/270, Step=8/74, loss=21.32, lr=0.000125, time_each_step=1.04s, eta=1:29:22
    2021-08-11 15:29:22 [INFO]	[TRAIN] Epoch=83/270, Step=10/74, loss=21.579535, lr=0.000125, time_each_step=1.06s, eta=1:29:21
    2021-08-11 15:29:23 [INFO]	[TRAIN] Epoch=83/270, Step=12/74, loss=34.294163, lr=0.000125, time_each_step=1.08s, eta=1:29:20
    2021-08-11 15:29:23 [INFO]	[TRAIN] Epoch=83/270, Step=14/74, loss=19.910305, lr=0.000125, time_each_step=1.11s, eta=1:29:20
    2021-08-11 15:29:24 [INFO]	[TRAIN] Epoch=83/270, Step=16/74, loss=27.280354, lr=0.000125, time_each_step=1.12s, eta=1:29:18
    2021-08-11 15:29:25 [INFO]	[TRAIN] Epoch=83/270, Step=18/74, loss=32.826897, lr=0.000125, time_each_step=1.13s, eta=1:29:17
    2021-08-11 15:29:25 [INFO]	[TRAIN] Epoch=83/270, Step=20/74, loss=23.511326, lr=0.000125, time_each_step=1.14s, eta=1:29:15
    2021-08-11 15:29:26 [INFO]	[TRAIN] Epoch=83/270, Step=22/74, loss=9.106371, lr=0.000125, time_each_step=0.4s, eta=1:28:34
    2021-08-11 15:29:26 [INFO]	[TRAIN] Epoch=83/270, Step=24/74, loss=24.300156, lr=0.000125, time_each_step=0.37s, eta=1:28:32
    2021-08-11 15:29:27 [INFO]	[TRAIN] Epoch=83/270, Step=26/74, loss=37.101757, lr=0.000125, time_each_step=0.36s, eta=1:28:30
    2021-08-11 15:29:28 [INFO]	[TRAIN] Epoch=83/270, Step=28/74, loss=8.117881, lr=0.000125, time_each_step=0.33s, eta=1:28:29
    2021-08-11 15:29:28 [INFO]	[TRAIN] Epoch=83/270, Step=30/74, loss=22.959127, lr=0.000125, time_each_step=0.33s, eta=1:28:28
    2021-08-11 15:29:29 [INFO]	[TRAIN] Epoch=83/270, Step=32/74, loss=44.146832, lr=0.000125, time_each_step=0.33s, eta=1:28:27
    2021-08-11 15:29:30 [INFO]	[TRAIN] Epoch=83/270, Step=34/74, loss=39.165642, lr=0.000125, time_each_step=0.33s, eta=1:28:26
    2021-08-11 15:29:31 [INFO]	[TRAIN] Epoch=83/270, Step=36/74, loss=14.968621, lr=0.000125, time_each_step=0.33s, eta=1:28:26
    2021-08-11 15:29:32 [INFO]	[TRAIN] Epoch=83/270, Step=38/74, loss=24.588949, lr=0.000125, time_each_step=0.33s, eta=1:28:25
    2021-08-11 15:29:33 [INFO]	[TRAIN] Epoch=83/270, Step=40/74, loss=19.756163, lr=0.000125, time_each_step=0.36s, eta=1:28:25
    2021-08-11 15:29:33 [INFO]	[TRAIN] Epoch=83/270, Step=42/74, loss=27.112581, lr=0.000125, time_each_step=0.37s, eta=1:28:25
    2021-08-11 15:29:34 [INFO]	[TRAIN] Epoch=83/270, Step=44/74, loss=9.313912, lr=0.000125, time_each_step=0.37s, eta=1:28:24
    2021-08-11 15:29:34 [INFO]	[TRAIN] Epoch=83/270, Step=46/74, loss=18.590752, lr=0.000125, time_each_step=0.37s, eta=1:28:24
    2021-08-11 15:29:35 [INFO]	[TRAIN] Epoch=83/270, Step=48/74, loss=17.056969, lr=0.000125, time_each_step=0.35s, eta=1:28:22
    2021-08-11 15:29:35 [INFO]	[TRAIN] Epoch=83/270, Step=50/74, loss=10.306386, lr=0.000125, time_each_step=0.34s, eta=1:28:21
    2021-08-11 15:29:36 [INFO]	[TRAIN] Epoch=83/270, Step=52/74, loss=37.25333, lr=0.000125, time_each_step=0.32s, eta=1:28:20
    2021-08-11 15:29:36 [INFO]	[TRAIN] Epoch=83/270, Step=54/74, loss=26.91787, lr=0.000125, time_each_step=0.3s, eta=1:28:19
    2021-08-11 15:29:36 [INFO]	[TRAIN] Epoch=83/270, Step=56/74, loss=19.186687, lr=0.000125, time_each_step=0.29s, eta=1:28:18
    2021-08-11 15:29:37 [INFO]	[TRAIN] Epoch=83/270, Step=58/74, loss=29.420017, lr=0.000125, time_each_step=0.27s, eta=1:28:18
    2021-08-11 15:29:38 [INFO]	[TRAIN] Epoch=83/270, Step=60/74, loss=10.812683, lr=0.000125, time_each_step=0.25s, eta=1:28:17
    2021-08-11 15:29:38 [INFO]	[TRAIN] Epoch=83/270, Step=62/74, loss=27.251923, lr=0.000125, time_each_step=0.25s, eta=1:28:16
    2021-08-11 15:29:39 [INFO]	[TRAIN] Epoch=83/270, Step=64/74, loss=15.097791, lr=0.000125, time_each_step=0.24s, eta=1:28:16
    2021-08-11 15:29:39 [INFO]	[TRAIN] Epoch=83/270, Step=66/74, loss=23.021635, lr=0.000125, time_each_step=0.23s, eta=1:28:15
    2021-08-11 15:29:40 [INFO]	[TRAIN] Epoch=83/270, Step=68/74, loss=15.62751, lr=0.000125, time_each_step=0.24s, eta=1:28:15
    2021-08-11 15:29:40 [INFO]	[TRAIN] Epoch=83/270, Step=70/74, loss=15.081932, lr=0.000125, time_each_step=0.23s, eta=1:28:14
    2021-08-11 15:29:40 [INFO]	[TRAIN] Epoch=83/270, Step=72/74, loss=23.615555, lr=0.000125, time_each_step=0.23s, eta=1:28:14
    2021-08-11 15:29:41 [INFO]	[TRAIN] Epoch=83/270, Step=74/74, loss=13.471607, lr=0.000125, time_each_step=0.23s, eta=1:28:13
    2021-08-11 15:29:41 [INFO]	[TRAIN] Epoch 83 finished, loss=22.021238, lr=0.000125 .
    2021-08-11 15:29:47 [INFO]	[TRAIN] Epoch=84/270, Step=2/74, loss=29.312813, lr=0.000125, time_each_step=0.51s, eta=2:2:37
    2021-08-11 15:29:48 [INFO]	[TRAIN] Epoch=84/270, Step=4/74, loss=19.475424, lr=0.000125, time_each_step=0.53s, eta=2:2:37
    2021-08-11 15:29:48 [INFO]	[TRAIN] Epoch=84/270, Step=6/74, loss=28.830952, lr=0.000125, time_each_step=0.54s, eta=2:2:37
    2021-08-11 15:29:49 [INFO]	[TRAIN] Epoch=84/270, Step=8/74, loss=12.278261, lr=0.000125, time_each_step=0.54s, eta=2:2:36
    2021-08-11 15:29:50 [INFO]	[TRAIN] Epoch=84/270, Step=10/74, loss=22.018578, lr=0.000125, time_each_step=0.56s, eta=2:2:36
    2021-08-11 15:29:51 [INFO]	[TRAIN] Epoch=84/270, Step=12/74, loss=15.06217, lr=0.000125, time_each_step=0.57s, eta=2:2:36
    2021-08-11 15:29:51 [INFO]	[TRAIN] Epoch=84/270, Step=14/74, loss=17.478773, lr=0.000125, time_each_step=0.59s, eta=2:2:36
    2021-08-11 15:29:52 [INFO]	[TRAIN] Epoch=84/270, Step=16/74, loss=20.082726, lr=0.000125, time_each_step=0.61s, eta=2:2:36
    2021-08-11 15:29:53 [INFO]	[TRAIN] Epoch=84/270, Step=18/74, loss=15.133399, lr=0.000125, time_each_step=0.62s, eta=2:2:35
    2021-08-11 15:29:53 [INFO]	[TRAIN] Epoch=84/270, Step=20/74, loss=15.10759, lr=0.000125, time_each_step=0.63s, eta=2:2:34
    2021-08-11 15:29:54 [INFO]	[TRAIN] Epoch=84/270, Step=22/74, loss=27.43012, lr=0.000125, time_each_step=0.37s, eta=2:2:19
    2021-08-11 15:29:55 [INFO]	[TRAIN] Epoch=84/270, Step=24/74, loss=17.766787, lr=0.000125, time_each_step=0.36s, eta=2:2:18
    2021-08-11 15:29:56 [INFO]	[TRAIN] Epoch=84/270, Step=26/74, loss=27.0868, lr=0.000125, time_each_step=0.36s, eta=2:2:18
    2021-08-11 15:29:57 [INFO]	[TRAIN] Epoch=84/270, Step=28/74, loss=20.358765, lr=0.000125, time_each_step=0.38s, eta=2:2:18
    2021-08-11 15:29:57 [INFO]	[TRAIN] Epoch=84/270, Step=30/74, loss=12.166598, lr=0.000125, time_each_step=0.38s, eta=2:2:17
    2021-08-11 15:29:58 [INFO]	[TRAIN] Epoch=84/270, Step=32/74, loss=17.207897, lr=0.000125, time_each_step=0.4s, eta=2:2:17
    2021-08-11 15:29:59 [INFO]	[TRAIN] Epoch=84/270, Step=34/74, loss=28.328804, lr=0.000125, time_each_step=0.39s, eta=2:2:16
    2021-08-11 15:30:00 [INFO]	[TRAIN] Epoch=84/270, Step=36/74, loss=19.106411, lr=0.000125, time_each_step=0.38s, eta=2:2:15
    2021-08-11 15:30:00 [INFO]	[TRAIN] Epoch=84/270, Step=38/74, loss=23.420376, lr=0.000125, time_each_step=0.39s, eta=2:2:14
    2021-08-11 15:30:01 [INFO]	[TRAIN] Epoch=84/270, Step=40/74, loss=16.626167, lr=0.000125, time_each_step=0.38s, eta=2:2:13
    2021-08-11 15:30:01 [INFO]	[TRAIN] Epoch=84/270, Step=42/74, loss=26.4613, lr=0.000125, time_each_step=0.37s, eta=2:2:12
    2021-08-11 15:30:02 [INFO]	[TRAIN] Epoch=84/270, Step=44/74, loss=19.632191, lr=0.000125, time_each_step=0.35s, eta=2:2:11
    2021-08-11 15:30:02 [INFO]	[TRAIN] Epoch=84/270, Step=46/74, loss=23.457844, lr=0.000125, time_each_step=0.33s, eta=2:2:10
    2021-08-11 15:30:03 [INFO]	[TRAIN] Epoch=84/270, Step=48/74, loss=16.654217, lr=0.000125, time_each_step=0.31s, eta=2:2:8
    2021-08-11 15:30:03 [INFO]	[TRAIN] Epoch=84/270, Step=50/74, loss=22.86459, lr=0.000125, time_each_step=0.28s, eta=2:2:7
    2021-08-11 15:30:04 [INFO]	[TRAIN] Epoch=84/270, Step=52/74, loss=16.556652, lr=0.000125, time_each_step=0.26s, eta=2:2:6
    2021-08-11 15:30:04 [INFO]	[TRAIN] Epoch=84/270, Step=54/74, loss=15.011932, lr=0.000125, time_each_step=0.26s, eta=2:2:6
    2021-08-11 15:30:04 [INFO]	[TRAIN] Epoch=84/270, Step=56/74, loss=20.24921, lr=0.000125, time_each_step=0.24s, eta=2:2:5
    2021-08-11 15:30:05 [INFO]	[TRAIN] Epoch=84/270, Step=58/74, loss=26.706812, lr=0.000125, time_each_step=0.22s, eta=2:2:4
    2021-08-11 15:30:05 [INFO]	[TRAIN] Epoch=84/270, Step=60/74, loss=8.680601, lr=0.000125, time_each_step=0.22s, eta=2:2:3
    2021-08-11 15:30:06 [INFO]	[TRAIN] Epoch=84/270, Step=62/74, loss=25.989779, lr=0.000125, time_each_step=0.22s, eta=2:2:3
    2021-08-11 15:30:06 [INFO]	[TRAIN] Epoch=84/270, Step=64/74, loss=28.403034, lr=0.000125, time_each_step=0.21s, eta=2:2:2
    2021-08-11 15:30:06 [INFO]	[TRAIN] Epoch=84/270, Step=66/74, loss=36.555428, lr=0.000125, time_each_step=0.2s, eta=2:2:2
    2021-08-11 15:30:07 [INFO]	[TRAIN] Epoch=84/270, Step=68/74, loss=33.390705, lr=0.000125, time_each_step=0.2s, eta=2:2:2
    2021-08-11 15:30:07 [INFO]	[TRAIN] Epoch=84/270, Step=70/74, loss=54.924244, lr=0.000125, time_each_step=0.2s, eta=2:2:1
    2021-08-11 15:30:07 [INFO]	[TRAIN] Epoch=84/270, Step=72/74, loss=40.825092, lr=0.000125, time_each_step=0.19s, eta=2:2:1
    2021-08-11 15:30:08 [INFO]	[TRAIN] Epoch=84/270, Step=74/74, loss=35.071266, lr=0.000125, time_each_step=0.18s, eta=2:2:0
    2021-08-11 15:30:08 [INFO]	[TRAIN] Epoch 84 finished, loss=22.284374, lr=0.000125 .
    2021-08-11 15:30:11 [INFO]	[TRAIN] Epoch=85/270, Step=2/74, loss=13.391454, lr=0.000125, time_each_step=0.35s, eta=1:28:45
    2021-08-11 15:30:13 [INFO]	[TRAIN] Epoch=85/270, Step=4/74, loss=16.214096, lr=0.000125, time_each_step=0.4s, eta=1:28:47
    2021-08-11 15:30:14 [INFO]	[TRAIN] Epoch=85/270, Step=6/74, loss=17.453367, lr=0.000125, time_each_step=0.42s, eta=1:28:48
    2021-08-11 15:30:14 [INFO]	[TRAIN] Epoch=85/270, Step=8/74, loss=22.989017, lr=0.000125, time_each_step=0.44s, eta=1:28:49
    2021-08-11 15:30:15 [INFO]	[TRAIN] Epoch=85/270, Step=10/74, loss=26.452009, lr=0.000125, time_each_step=0.46s, eta=1:28:49
    2021-08-11 15:30:16 [INFO]	[TRAIN] Epoch=85/270, Step=12/74, loss=12.388235, lr=0.000125, time_each_step=0.48s, eta=1:28:49
    2021-08-11 15:30:16 [INFO]	[TRAIN] Epoch=85/270, Step=14/74, loss=16.461637, lr=0.000125, time_each_step=0.49s, eta=1:28:49
    2021-08-11 15:30:17 [INFO]	[TRAIN] Epoch=85/270, Step=16/74, loss=42.236702, lr=0.000125, time_each_step=0.51s, eta=1:28:49
    2021-08-11 15:30:18 [INFO]	[TRAIN] Epoch=85/270, Step=18/74, loss=21.919312, lr=0.000125, time_each_step=0.53s, eta=1:28:49
    2021-08-11 15:30:19 [INFO]	[TRAIN] Epoch=85/270, Step=20/74, loss=14.511343, lr=0.000125, time_each_step=0.56s, eta=1:28:50
    2021-08-11 15:30:20 [INFO]	[TRAIN] Epoch=85/270, Step=22/74, loss=17.071514, lr=0.000125, time_each_step=0.41s, eta=1:28:41
    2021-08-11 15:30:20 [INFO]	[TRAIN] Epoch=85/270, Step=24/74, loss=20.113045, lr=0.000125, time_each_step=0.38s, eta=1:28:38
    2021-08-11 15:30:21 [INFO]	[TRAIN] Epoch=85/270, Step=26/74, loss=40.961102, lr=0.000125, time_each_step=0.37s, eta=1:28:37
    2021-08-11 15:30:21 [INFO]	[TRAIN] Epoch=85/270, Step=28/74, loss=31.90855, lr=0.000125, time_each_step=0.35s, eta=1:28:36
    2021-08-11 15:30:22 [INFO]	[TRAIN] Epoch=85/270, Step=30/74, loss=10.770481, lr=0.000125, time_each_step=0.33s, eta=1:28:34
    2021-08-11 15:30:22 [INFO]	[TRAIN] Epoch=85/270, Step=32/74, loss=24.662785, lr=0.000125, time_each_step=0.32s, eta=1:28:33
    2021-08-11 15:30:23 [INFO]	[TRAIN] Epoch=85/270, Step=34/74, loss=15.741051, lr=0.000125, time_each_step=0.31s, eta=1:28:32
    2021-08-11 15:30:24 [INFO]	[TRAIN] Epoch=85/270, Step=36/74, loss=12.40731, lr=0.000125, time_each_step=0.31s, eta=1:28:31
    2021-08-11 15:30:24 [INFO]	[TRAIN] Epoch=85/270, Step=38/74, loss=45.504105, lr=0.000125, time_each_step=0.31s, eta=1:28:30
    2021-08-11 15:30:25 [INFO]	[TRAIN] Epoch=85/270, Step=40/74, loss=12.211182, lr=0.000125, time_each_step=0.3s, eta=1:28:29
    2021-08-11 15:30:26 [INFO]	[TRAIN] Epoch=85/270, Step=42/74, loss=20.377008, lr=0.000125, time_each_step=0.3s, eta=1:28:29
    2021-08-11 15:30:26 [INFO]	[TRAIN] Epoch=85/270, Step=44/74, loss=7.642914, lr=0.000125, time_each_step=0.3s, eta=1:28:29
    2021-08-11 15:30:27 [INFO]	[TRAIN] Epoch=85/270, Step=46/74, loss=25.795879, lr=0.000125, time_each_step=0.3s, eta=1:28:28
    2021-08-11 15:30:27 [INFO]	[TRAIN] Epoch=85/270, Step=48/74, loss=28.076809, lr=0.000125, time_each_step=0.29s, eta=1:28:27
    2021-08-11 15:30:28 [INFO]	[TRAIN] Epoch=85/270, Step=50/74, loss=16.564487, lr=0.000125, time_each_step=0.31s, eta=1:28:27
    2021-08-11 15:30:28 [INFO]	[TRAIN] Epoch=85/270, Step=52/74, loss=12.108864, lr=0.000125, time_each_step=0.31s, eta=1:28:26
    2021-08-11 15:30:29 [INFO]	[TRAIN] Epoch=85/270, Step=54/74, loss=40.574554, lr=0.000125, time_each_step=0.3s, eta=1:28:26
    2021-08-11 15:30:29 [INFO]	[TRAIN] Epoch=85/270, Step=56/74, loss=12.246788, lr=0.000125, time_each_step=0.29s, eta=1:28:25
    2021-08-11 15:30:30 [INFO]	[TRAIN] Epoch=85/270, Step=58/74, loss=16.465382, lr=0.000125, time_each_step=0.27s, eta=1:28:24
    2021-08-11 15:30:30 [INFO]	[TRAIN] Epoch=85/270, Step=60/74, loss=10.873485, lr=0.000125, time_each_step=0.25s, eta=1:28:23
    2021-08-11 15:30:30 [INFO]	[TRAIN] Epoch=85/270, Step=62/74, loss=35.315742, lr=0.000125, time_each_step=0.24s, eta=1:28:22
    2021-08-11 15:30:31 [INFO]	[TRAIN] Epoch=85/270, Step=64/74, loss=11.307827, lr=0.000125, time_each_step=0.23s, eta=1:28:22
    2021-08-11 15:30:32 [INFO]	[TRAIN] Epoch=85/270, Step=66/74, loss=24.526463, lr=0.000125, time_each_step=0.24s, eta=1:28:21
    2021-08-11 15:30:32 [INFO]	[TRAIN] Epoch=85/270, Step=68/74, loss=19.600691, lr=0.000125, time_each_step=0.24s, eta=1:28:21
    2021-08-11 15:30:32 [INFO]	[TRAIN] Epoch=85/270, Step=70/74, loss=11.186939, lr=0.000125, time_each_step=0.22s, eta=1:28:20
    2021-08-11 15:30:33 [INFO]	[TRAIN] Epoch=85/270, Step=72/74, loss=33.125141, lr=0.000125, time_each_step=0.23s, eta=1:28:20
    2021-08-11 15:30:33 [INFO]	[TRAIN] Epoch=85/270, Step=74/74, loss=23.005587, lr=0.000125, time_each_step=0.23s, eta=1:28:19
    2021-08-11 15:30:33 [INFO]	[TRAIN] Epoch 85 finished, loss=22.07988, lr=0.000125 .
    2021-08-11 15:30:38 [INFO]	[TRAIN] Epoch=86/270, Step=2/74, loss=25.079391, lr=0.000125, time_each_step=0.44s, eta=1:23:9
    2021-08-11 15:30:39 [INFO]	[TRAIN] Epoch=86/270, Step=4/74, loss=29.497116, lr=0.000125, time_each_step=0.46s, eta=1:23:10
    2021-08-11 15:30:40 [INFO]	[TRAIN] Epoch=86/270, Step=6/74, loss=34.132057, lr=0.000125, time_each_step=0.49s, eta=1:23:11
    2021-08-11 15:30:40 [INFO]	[TRAIN] Epoch=86/270, Step=8/74, loss=32.750832, lr=0.000125, time_each_step=0.5s, eta=1:23:10
    2021-08-11 15:30:41 [INFO]	[TRAIN] Epoch=86/270, Step=10/74, loss=9.659463, lr=0.000125, time_each_step=0.49s, eta=1:23:9
    2021-08-11 15:30:42 [INFO]	[TRAIN] Epoch=86/270, Step=12/74, loss=32.159363, lr=0.000125, time_each_step=0.5s, eta=1:23:8
    2021-08-11 15:30:42 [INFO]	[TRAIN] Epoch=86/270, Step=14/74, loss=16.225197, lr=0.000125, time_each_step=0.51s, eta=1:23:8
    2021-08-11 15:30:43 [INFO]	[TRAIN] Epoch=86/270, Step=16/74, loss=46.935188, lr=0.000125, time_each_step=0.55s, eta=1:23:9
    2021-08-11 15:30:44 [INFO]	[TRAIN] Epoch=86/270, Step=18/74, loss=18.807106, lr=0.000125, time_each_step=0.55s, eta=1:23:8
    2021-08-11 15:30:45 [INFO]	[TRAIN] Epoch=86/270, Step=20/74, loss=20.698055, lr=0.000125, time_each_step=0.57s, eta=1:23:8
    2021-08-11 15:30:45 [INFO]	[TRAIN] Epoch=86/270, Step=22/74, loss=14.768574, lr=0.000125, time_each_step=0.37s, eta=1:22:57
    2021-08-11 15:30:46 [INFO]	[TRAIN] Epoch=86/270, Step=24/74, loss=11.639118, lr=0.000125, time_each_step=0.36s, eta=1:22:55
    2021-08-11 15:30:47 [INFO]	[TRAIN] Epoch=86/270, Step=26/74, loss=13.149733, lr=0.000125, time_each_step=0.36s, eta=1:22:54
    2021-08-11 15:30:47 [INFO]	[TRAIN] Epoch=86/270, Step=28/74, loss=9.143066, lr=0.000125, time_each_step=0.35s, eta=1:22:53
    2021-08-11 15:30:48 [INFO]	[TRAIN] Epoch=86/270, Step=30/74, loss=27.454771, lr=0.000125, time_each_step=0.37s, eta=1:22:54
    2021-08-11 15:30:49 [INFO]	[TRAIN] Epoch=86/270, Step=32/74, loss=35.578804, lr=0.000125, time_each_step=0.38s, eta=1:22:53
    2021-08-11 15:30:50 [INFO]	[TRAIN] Epoch=86/270, Step=34/74, loss=19.317665, lr=0.000125, time_each_step=0.37s, eta=1:22:52
    2021-08-11 15:30:51 [INFO]	[TRAIN] Epoch=86/270, Step=36/74, loss=21.335596, lr=0.000125, time_each_step=0.36s, eta=1:22:51
    2021-08-11 15:30:51 [INFO]	[TRAIN] Epoch=86/270, Step=38/74, loss=18.115324, lr=0.000125, time_each_step=0.37s, eta=1:22:50
    2021-08-11 15:30:52 [INFO]	[TRAIN] Epoch=86/270, Step=40/74, loss=15.193987, lr=0.000125, time_each_step=0.37s, eta=1:22:50
    2021-08-11 15:30:53 [INFO]	[TRAIN] Epoch=86/270, Step=42/74, loss=15.290554, lr=0.000125, time_each_step=0.36s, eta=1:22:49
    2021-08-11 15:30:53 [INFO]	[TRAIN] Epoch=86/270, Step=44/74, loss=17.4538, lr=0.000125, time_each_step=0.37s, eta=1:22:48
    2021-08-11 15:30:54 [INFO]	[TRAIN] Epoch=86/270, Step=46/74, loss=15.019451, lr=0.000125, time_each_step=0.35s, eta=1:22:47
    2021-08-11 15:30:54 [INFO]	[TRAIN] Epoch=86/270, Step=48/74, loss=32.914345, lr=0.000125, time_each_step=0.33s, eta=1:22:46
    2021-08-11 15:30:54 [INFO]	[TRAIN] Epoch=86/270, Step=50/74, loss=18.231144, lr=0.000125, time_each_step=0.31s, eta=1:22:45
    2021-08-11 15:30:55 [INFO]	[TRAIN] Epoch=86/270, Step=52/74, loss=14.705347, lr=0.000125, time_each_step=0.28s, eta=1:22:43
    2021-08-11 15:30:55 [INFO]	[TRAIN] Epoch=86/270, Step=54/74, loss=16.203541, lr=0.000125, time_each_step=0.26s, eta=1:22:43
    2021-08-11 15:30:55 [INFO]	[TRAIN] Epoch=86/270, Step=56/74, loss=17.523785, lr=0.000125, time_each_step=0.24s, eta=1:22:42
    2021-08-11 15:30:56 [INFO]	[TRAIN] Epoch=86/270, Step=58/74, loss=14.160797, lr=0.000125, time_each_step=0.23s, eta=1:22:41
    2021-08-11 15:30:56 [INFO]	[TRAIN] Epoch=86/270, Step=60/74, loss=14.232485, lr=0.000125, time_each_step=0.21s, eta=1:22:40
    2021-08-11 15:30:57 [INFO]	[TRAIN] Epoch=86/270, Step=62/74, loss=11.181065, lr=0.000125, time_each_step=0.21s, eta=1:22:40
    2021-08-11 15:30:57 [INFO]	[TRAIN] Epoch=86/270, Step=64/74, loss=23.865294, lr=0.000125, time_each_step=0.2s, eta=1:22:39
    2021-08-11 15:30:58 [INFO]	[TRAIN] Epoch=86/270, Step=66/74, loss=37.439144, lr=0.000125, time_each_step=0.2s, eta=1:22:39
    2021-08-11 15:30:58 [INFO]	[TRAIN] Epoch=86/270, Step=68/74, loss=14.609062, lr=0.000125, time_each_step=0.21s, eta=1:22:39
    2021-08-11 15:30:58 [INFO]	[TRAIN] Epoch=86/270, Step=70/74, loss=17.211027, lr=0.000125, time_each_step=0.2s, eta=1:22:38
    2021-08-11 15:30:59 [INFO]	[TRAIN] Epoch=86/270, Step=72/74, loss=15.69144, lr=0.000125, time_each_step=0.2s, eta=1:22:38
    2021-08-11 15:30:59 [INFO]	[TRAIN] Epoch=86/270, Step=74/74, loss=35.870979, lr=0.000125, time_each_step=0.2s, eta=1:22:37
    2021-08-11 15:30:59 [INFO]	[TRAIN] Epoch 86 finished, loss=21.426342, lr=0.000125 .
    2021-08-11 15:31:03 [INFO]	[TRAIN] Epoch=87/270, Step=2/74, loss=32.255787, lr=0.000125, time_each_step=0.37s, eta=1:23:7
    2021-08-11 15:31:04 [INFO]	[TRAIN] Epoch=87/270, Step=4/74, loss=13.604059, lr=0.000125, time_each_step=0.39s, eta=1:23:8
    2021-08-11 15:31:05 [INFO]	[TRAIN] Epoch=87/270, Step=6/74, loss=48.439812, lr=0.000125, time_each_step=0.43s, eta=1:23:9
    2021-08-11 15:31:06 [INFO]	[TRAIN] Epoch=87/270, Step=8/74, loss=20.816156, lr=0.000125, time_each_step=0.44s, eta=1:23:9
    2021-08-11 15:31:07 [INFO]	[TRAIN] Epoch=87/270, Step=10/74, loss=16.752022, lr=0.000125, time_each_step=0.46s, eta=1:23:10
    2021-08-11 15:31:07 [INFO]	[TRAIN] Epoch=87/270, Step=12/74, loss=25.247849, lr=0.000125, time_each_step=0.49s, eta=1:23:10
    2021-08-11 15:31:08 [INFO]	[TRAIN] Epoch=87/270, Step=14/74, loss=17.893854, lr=0.000125, time_each_step=0.49s, eta=1:23:10
    2021-08-11 15:31:09 [INFO]	[TRAIN] Epoch=87/270, Step=16/74, loss=29.103168, lr=0.000125, time_each_step=0.51s, eta=1:23:10
    2021-08-11 15:31:10 [INFO]	[TRAIN] Epoch=87/270, Step=18/74, loss=31.15806, lr=0.000125, time_each_step=0.53s, eta=1:23:10
    2021-08-11 15:31:10 [INFO]	[TRAIN] Epoch=87/270, Step=20/74, loss=12.579012, lr=0.000125, time_each_step=0.56s, eta=1:23:10
    2021-08-11 15:31:11 [INFO]	[TRAIN] Epoch=87/270, Step=22/74, loss=11.621316, lr=0.000125, time_each_step=0.39s, eta=1:23:1
    2021-08-11 15:31:12 [INFO]	[TRAIN] Epoch=87/270, Step=24/74, loss=22.014418, lr=0.000125, time_each_step=0.4s, eta=1:23:0
    2021-08-11 15:31:12 [INFO]	[TRAIN] Epoch=87/270, Step=26/74, loss=31.863235, lr=0.000125, time_each_step=0.37s, eta=1:22:58
    2021-08-11 15:31:13 [INFO]	[TRAIN] Epoch=87/270, Step=28/74, loss=25.363041, lr=0.000125, time_each_step=0.37s, eta=1:22:57
    2021-08-11 15:31:14 [INFO]	[TRAIN] Epoch=87/270, Step=30/74, loss=48.890038, lr=0.000125, time_each_step=0.36s, eta=1:22:56
    2021-08-11 15:31:15 [INFO]	[TRAIN] Epoch=87/270, Step=32/74, loss=27.162357, lr=0.000125, time_each_step=0.36s, eta=1:22:55
    2021-08-11 15:31:15 [INFO]	[TRAIN] Epoch=87/270, Step=34/74, loss=42.527843, lr=0.000125, time_each_step=0.37s, eta=1:22:55
    2021-08-11 15:31:16 [INFO]	[TRAIN] Epoch=87/270, Step=36/74, loss=28.093143, lr=0.000125, time_each_step=0.38s, eta=1:22:54
    2021-08-11 15:31:17 [INFO]	[TRAIN] Epoch=87/270, Step=38/74, loss=22.755438, lr=0.000125, time_each_step=0.38s, eta=1:22:54
    2021-08-11 15:31:18 [INFO]	[TRAIN] Epoch=87/270, Step=40/74, loss=15.593246, lr=0.000125, time_each_step=0.37s, eta=1:22:53
    2021-08-11 15:31:18 [INFO]	[TRAIN] Epoch=87/270, Step=42/74, loss=24.042885, lr=0.000125, time_each_step=0.38s, eta=1:22:52
    2021-08-11 15:31:19 [INFO]	[TRAIN] Epoch=87/270, Step=44/74, loss=33.987411, lr=0.000125, time_each_step=0.36s, eta=1:22:51
    2021-08-11 15:31:19 [INFO]	[TRAIN] Epoch=87/270, Step=46/74, loss=12.329744, lr=0.000125, time_each_step=0.34s, eta=1:22:50
    2021-08-11 15:31:20 [INFO]	[TRAIN] Epoch=87/270, Step=48/74, loss=12.717602, lr=0.000125, time_each_step=0.32s, eta=1:22:48
    2021-08-11 15:31:20 [INFO]	[TRAIN] Epoch=87/270, Step=50/74, loss=36.619751, lr=0.000125, time_each_step=0.31s, eta=1:22:47
    2021-08-11 15:31:20 [INFO]	[TRAIN] Epoch=87/270, Step=52/74, loss=45.438915, lr=0.000125, time_each_step=0.28s, eta=1:22:46
    2021-08-11 15:31:21 [INFO]	[TRAIN] Epoch=87/270, Step=54/74, loss=43.815533, lr=0.000125, time_each_step=0.26s, eta=1:22:45
    2021-08-11 15:31:21 [INFO]	[TRAIN] Epoch=87/270, Step=56/74, loss=12.068146, lr=0.000125, time_each_step=0.24s, eta=1:22:44
    2021-08-11 15:31:21 [INFO]	[TRAIN] Epoch=87/270, Step=58/74, loss=17.745735, lr=0.000125, time_each_step=0.22s, eta=1:22:44
    2021-08-11 15:31:22 [INFO]	[TRAIN] Epoch=87/270, Step=60/74, loss=25.441465, lr=0.000125, time_each_step=0.21s, eta=1:22:43
    2021-08-11 15:31:22 [INFO]	[TRAIN] Epoch=87/270, Step=62/74, loss=32.809246, lr=0.000125, time_each_step=0.19s, eta=1:22:42
    2021-08-11 15:31:22 [INFO]	[TRAIN] Epoch=87/270, Step=64/74, loss=9.512187, lr=0.000125, time_each_step=0.18s, eta=1:22:42
    2021-08-11 15:31:23 [INFO]	[TRAIN] Epoch=87/270, Step=66/74, loss=18.762135, lr=0.000125, time_each_step=0.19s, eta=1:22:42
    2021-08-11 15:31:23 [INFO]	[TRAIN] Epoch=87/270, Step=68/74, loss=36.733555, lr=0.000125, time_each_step=0.18s, eta=1:22:41
    2021-08-11 15:31:24 [INFO]	[TRAIN] Epoch=87/270, Step=70/74, loss=15.574787, lr=0.000125, time_each_step=0.18s, eta=1:22:41
    2021-08-11 15:31:24 [INFO]	[TRAIN] Epoch=87/270, Step=72/74, loss=32.902317, lr=0.000125, time_each_step=0.18s, eta=1:22:40
    2021-08-11 15:31:24 [INFO]	[TRAIN] Epoch=87/270, Step=74/74, loss=12.926945, lr=0.000125, time_each_step=0.19s, eta=1:22:40
    2021-08-11 15:31:24 [INFO]	[TRAIN] Epoch 87 finished, loss=23.048689, lr=0.000125 .
    2021-08-11 15:31:31 [INFO]	[TRAIN] Epoch=88/270, Step=2/74, loss=14.560598, lr=0.000125, time_each_step=0.48s, eta=1:21:19
    2021-08-11 15:31:31 [INFO]	[TRAIN] Epoch=88/270, Step=4/74, loss=32.71394, lr=0.000125, time_each_step=0.5s, eta=1:21:19
    2021-08-11 15:31:33 [INFO]	[TRAIN] Epoch=88/270, Step=6/74, loss=22.522144, lr=0.000125, time_each_step=0.54s, eta=1:21:21
    2021-08-11 15:31:33 [INFO]	[TRAIN] Epoch=88/270, Step=8/74, loss=13.222681, lr=0.000125, time_each_step=0.56s, eta=1:21:21
    2021-08-11 15:31:34 [INFO]	[TRAIN] Epoch=88/270, Step=10/74, loss=45.23053, lr=0.000125, time_each_step=0.58s, eta=1:21:21
    2021-08-11 15:31:34 [INFO]	[TRAIN] Epoch=88/270, Step=12/74, loss=40.223927, lr=0.000125, time_each_step=0.58s, eta=1:21:20
    2021-08-11 15:31:35 [INFO]	[TRAIN] Epoch=88/270, Step=14/74, loss=37.602398, lr=0.000125, time_each_step=0.59s, eta=1:21:20
    2021-08-11 15:31:36 [INFO]	[TRAIN] Epoch=88/270, Step=16/74, loss=38.879349, lr=0.000125, time_each_step=0.61s, eta=1:21:20
    2021-08-11 15:31:37 [INFO]	[TRAIN] Epoch=88/270, Step=18/74, loss=35.956413, lr=0.000125, time_each_step=0.64s, eta=1:21:20
    2021-08-11 15:31:38 [INFO]	[TRAIN] Epoch=88/270, Step=20/74, loss=17.8459, lr=0.000125, time_each_step=0.67s, eta=1:21:20
    2021-08-11 15:31:39 [INFO]	[TRAIN] Epoch=88/270, Step=22/74, loss=28.062893, lr=0.000125, time_each_step=0.39s, eta=1:21:5
    2021-08-11 15:31:40 [INFO]	[TRAIN] Epoch=88/270, Step=24/74, loss=18.757233, lr=0.000125, time_each_step=0.41s, eta=1:21:5
    2021-08-11 15:31:40 [INFO]	[TRAIN] Epoch=88/270, Step=26/74, loss=21.477592, lr=0.000125, time_each_step=0.39s, eta=1:21:3
    2021-08-11 15:31:41 [INFO]	[TRAIN] Epoch=88/270, Step=28/74, loss=36.352379, lr=0.000125, time_each_step=0.4s, eta=1:21:3
    2021-08-11 15:31:42 [INFO]	[TRAIN] Epoch=88/270, Step=30/74, loss=25.798588, lr=0.000125, time_each_step=0.41s, eta=1:21:3
    2021-08-11 15:31:43 [INFO]	[TRAIN] Epoch=88/270, Step=32/74, loss=15.976401, lr=0.000125, time_each_step=0.42s, eta=1:21:2
    2021-08-11 15:31:43 [INFO]	[TRAIN] Epoch=88/270, Step=34/74, loss=14.258514, lr=0.000125, time_each_step=0.43s, eta=1:21:1
    2021-08-11 15:31:44 [INFO]	[TRAIN] Epoch=88/270, Step=36/74, loss=16.871172, lr=0.000125, time_each_step=0.41s, eta=1:21:0
    2021-08-11 15:31:45 [INFO]	[TRAIN] Epoch=88/270, Step=38/74, loss=29.648399, lr=0.000125, time_each_step=0.39s, eta=1:20:58
    2021-08-11 15:31:45 [INFO]	[TRAIN] Epoch=88/270, Step=40/74, loss=23.889332, lr=0.000125, time_each_step=0.39s, eta=1:20:57
    2021-08-11 15:31:46 [INFO]	[TRAIN] Epoch=88/270, Step=42/74, loss=23.711187, lr=0.000125, time_each_step=0.39s, eta=1:20:57
    2021-08-11 15:31:47 [INFO]	[TRAIN] Epoch=88/270, Step=44/74, loss=18.485096, lr=0.000125, time_each_step=0.36s, eta=1:20:55
    2021-08-11 15:31:47 [INFO]	[TRAIN] Epoch=88/270, Step=46/74, loss=41.55658, lr=0.000125, time_each_step=0.34s, eta=1:20:54
    2021-08-11 15:31:47 [INFO]	[TRAIN] Epoch=88/270, Step=48/74, loss=30.119118, lr=0.000125, time_each_step=0.31s, eta=1:20:52
    2021-08-11 15:31:48 [INFO]	[TRAIN] Epoch=88/270, Step=50/74, loss=12.506306, lr=0.000125, time_each_step=0.29s, eta=1:20:51
    2021-08-11 15:31:49 [INFO]	[TRAIN] Epoch=88/270, Step=52/74, loss=38.389282, lr=0.000125, time_each_step=0.29s, eta=1:20:51
    2021-08-11 15:31:49 [INFO]	[TRAIN] Epoch=88/270, Step=54/74, loss=20.82003, lr=0.000125, time_each_step=0.28s, eta=1:20:50
    2021-08-11 15:31:49 [INFO]	[TRAIN] Epoch=88/270, Step=56/74, loss=26.750078, lr=0.000125, time_each_step=0.26s, eta=1:20:49
    2021-08-11 15:31:50 [INFO]	[TRAIN] Epoch=88/270, Step=58/74, loss=18.816441, lr=0.000125, time_each_step=0.26s, eta=1:20:48
    2021-08-11 15:31:50 [INFO]	[TRAIN] Epoch=88/270, Step=60/74, loss=20.02927, lr=0.000125, time_each_step=0.24s, eta=1:20:48
    2021-08-11 15:31:51 [INFO]	[TRAIN] Epoch=88/270, Step=62/74, loss=24.643219, lr=0.000125, time_each_step=0.21s, eta=1:20:47
    2021-08-11 15:31:51 [INFO]	[TRAIN] Epoch=88/270, Step=64/74, loss=31.724548, lr=0.000125, time_each_step=0.2s, eta=1:20:46
    2021-08-11 15:31:51 [INFO]	[TRAIN] Epoch=88/270, Step=66/74, loss=15.363716, lr=0.000125, time_each_step=0.21s, eta=1:20:46
    2021-08-11 15:31:52 [INFO]	[TRAIN] Epoch=88/270, Step=68/74, loss=15.570408, lr=0.000125, time_each_step=0.22s, eta=1:20:46
    2021-08-11 15:31:52 [INFO]	[TRAIN] Epoch=88/270, Step=70/74, loss=23.172695, lr=0.000125, time_each_step=0.22s, eta=1:20:45
    2021-08-11 15:31:53 [INFO]	[TRAIN] Epoch=88/270, Step=72/74, loss=23.966845, lr=0.000125, time_each_step=0.21s, eta=1:20:45
    2021-08-11 15:31:53 [INFO]	[TRAIN] Epoch=88/270, Step=74/74, loss=49.475693, lr=0.000125, time_each_step=0.21s, eta=1:20:44
    2021-08-11 15:31:53 [INFO]	[TRAIN] Epoch 88 finished, loss=23.054127, lr=0.000125 .
    2021-08-11 15:32:03 [INFO]	[TRAIN] Epoch=89/270, Step=2/74, loss=18.644733, lr=0.000125, time_each_step=0.68s, eta=1:32:25
    2021-08-11 15:32:04 [INFO]	[TRAIN] Epoch=89/270, Step=4/74, loss=14.771705, lr=0.000125, time_each_step=0.7s, eta=1:32:25
    2021-08-11 15:32:04 [INFO]	[TRAIN] Epoch=89/270, Step=6/74, loss=21.828371, lr=0.000125, time_each_step=0.71s, eta=1:32:25
    2021-08-11 15:32:05 [INFO]	[TRAIN] Epoch=89/270, Step=8/74, loss=28.04896, lr=0.000125, time_each_step=0.73s, eta=1:32:25
    2021-08-11 15:32:06 [INFO]	[TRAIN] Epoch=89/270, Step=10/74, loss=17.716148, lr=0.000125, time_each_step=0.75s, eta=1:32:24
    2021-08-11 15:32:06 [INFO]	[TRAIN] Epoch=89/270, Step=12/74, loss=15.6901, lr=0.000125, time_each_step=0.75s, eta=1:32:23
    2021-08-11 15:32:07 [INFO]	[TRAIN] Epoch=89/270, Step=14/74, loss=27.29561, lr=0.000125, time_each_step=0.76s, eta=1:32:22
    2021-08-11 15:32:08 [INFO]	[TRAIN] Epoch=89/270, Step=16/74, loss=18.25396, lr=0.000125, time_each_step=0.78s, eta=1:32:21
    2021-08-11 15:32:09 [INFO]	[TRAIN] Epoch=89/270, Step=18/74, loss=28.418894, lr=0.000125, time_each_step=0.79s, eta=1:32:20
    2021-08-11 15:32:09 [INFO]	[TRAIN] Epoch=89/270, Step=20/74, loss=15.015805, lr=0.000125, time_each_step=0.81s, eta=1:32:20
    2021-08-11 15:32:10 [INFO]	[TRAIN] Epoch=89/270, Step=22/74, loss=16.831381, lr=0.000125, time_each_step=0.36s, eta=1:31:55
    2021-08-11 15:32:11 [INFO]	[TRAIN] Epoch=89/270, Step=24/74, loss=16.422497, lr=0.000125, time_each_step=0.39s, eta=1:31:56
    2021-08-11 15:32:12 [INFO]	[TRAIN] Epoch=89/270, Step=26/74, loss=12.796261, lr=0.000125, time_each_step=0.37s, eta=1:31:54
    2021-08-11 15:32:13 [INFO]	[TRAIN] Epoch=89/270, Step=28/74, loss=15.763265, lr=0.000125, time_each_step=0.36s, eta=1:31:53
    2021-08-11 15:32:13 [INFO]	[TRAIN] Epoch=89/270, Step=30/74, loss=16.061069, lr=0.000125, time_each_step=0.37s, eta=1:31:53
    2021-08-11 15:32:14 [INFO]	[TRAIN] Epoch=89/270, Step=32/74, loss=34.250484, lr=0.000125, time_each_step=0.38s, eta=1:31:52
    2021-08-11 15:32:15 [INFO]	[TRAIN] Epoch=89/270, Step=34/74, loss=16.19088, lr=0.000125, time_each_step=0.39s, eta=1:31:52
    2021-08-11 15:32:16 [INFO]	[TRAIN] Epoch=89/270, Step=36/74, loss=19.401928, lr=0.000125, time_each_step=0.38s, eta=1:31:51
    2021-08-11 15:32:17 [INFO]	[TRAIN] Epoch=89/270, Step=38/74, loss=15.915834, lr=0.000125, time_each_step=0.4s, eta=1:31:51
    2021-08-11 15:32:17 [INFO]	[TRAIN] Epoch=89/270, Step=40/74, loss=17.40546, lr=0.000125, time_each_step=0.4s, eta=1:31:50
    2021-08-11 15:32:18 [INFO]	[TRAIN] Epoch=89/270, Step=42/74, loss=21.025368, lr=0.000125, time_each_step=0.39s, eta=1:31:49
    2021-08-11 15:32:19 [INFO]	[TRAIN] Epoch=89/270, Step=44/74, loss=19.413845, lr=0.000125, time_each_step=0.39s, eta=1:31:48
    2021-08-11 15:32:20 [INFO]	[TRAIN] Epoch=89/270, Step=46/74, loss=26.410341, lr=0.000125, time_each_step=0.41s, eta=1:31:48
    2021-08-11 15:32:20 [INFO]	[TRAIN] Epoch=89/270, Step=48/74, loss=17.787239, lr=0.000125, time_each_step=0.39s, eta=1:31:46
    2021-08-11 15:32:21 [INFO]	[TRAIN] Epoch=89/270, Step=50/74, loss=25.001242, lr=0.000125, time_each_step=0.37s, eta=1:31:45
    2021-08-11 15:32:21 [INFO]	[TRAIN] Epoch=89/270, Step=52/74, loss=31.22234, lr=0.000125, time_each_step=0.38s, eta=1:31:45
    2021-08-11 15:32:22 [INFO]	[TRAIN] Epoch=89/270, Step=54/74, loss=55.380249, lr=0.000125, time_each_step=0.34s, eta=1:31:43
    2021-08-11 15:32:22 [INFO]	[TRAIN] Epoch=89/270, Step=56/74, loss=20.522072, lr=0.000125, time_each_step=0.32s, eta=1:31:42
    2021-08-11 15:32:22 [INFO]	[TRAIN] Epoch=89/270, Step=58/74, loss=9.826811, lr=0.000125, time_each_step=0.29s, eta=1:31:41
    2021-08-11 15:32:23 [INFO]	[TRAIN] Epoch=89/270, Step=60/74, loss=28.950029, lr=0.000125, time_each_step=0.27s, eta=1:31:40
    2021-08-11 15:32:23 [INFO]	[TRAIN] Epoch=89/270, Step=62/74, loss=15.543924, lr=0.000125, time_each_step=0.26s, eta=1:31:39
    2021-08-11 15:32:24 [INFO]	[TRAIN] Epoch=89/270, Step=64/74, loss=24.373959, lr=0.000125, time_each_step=0.23s, eta=1:31:39
    2021-08-11 15:32:24 [INFO]	[TRAIN] Epoch=89/270, Step=66/74, loss=18.817478, lr=0.000125, time_each_step=0.21s, eta=1:31:38
    2021-08-11 15:32:25 [INFO]	[TRAIN] Epoch=89/270, Step=68/74, loss=9.777248, lr=0.000125, time_each_step=0.22s, eta=1:31:38
    2021-08-11 15:32:25 [INFO]	[TRAIN] Epoch=89/270, Step=70/74, loss=31.557108, lr=0.000125, time_each_step=0.21s, eta=1:31:37
    2021-08-11 15:32:25 [INFO]	[TRAIN] Epoch=89/270, Step=72/74, loss=26.560638, lr=0.000125, time_each_step=0.2s, eta=1:31:37
    2021-08-11 15:32:26 [INFO]	[TRAIN] Epoch=89/270, Step=74/74, loss=18.828842, lr=0.000125, time_each_step=0.2s, eta=1:31:36
    2021-08-11 15:32:26 [INFO]	[TRAIN] Epoch 89 finished, loss=21.484566, lr=0.000125 .
    2021-08-11 15:32:45 [INFO]	[TRAIN] Epoch=90/270, Step=2/74, loss=21.536894, lr=0.000125, time_each_step=1.12s, eta=1:42:46
    2021-08-11 15:32:45 [INFO]	[TRAIN] Epoch=90/270, Step=4/74, loss=27.233568, lr=0.000125, time_each_step=1.14s, eta=1:42:45
    2021-08-11 15:32:46 [INFO]	[TRAIN] Epoch=90/270, Step=6/74, loss=27.401615, lr=0.000125, time_each_step=1.17s, eta=1:42:45
    2021-08-11 15:32:47 [INFO]	[TRAIN] Epoch=90/270, Step=8/74, loss=20.933664, lr=0.000125, time_each_step=1.18s, eta=1:42:43
    2021-08-11 15:32:48 [INFO]	[TRAIN] Epoch=90/270, Step=10/74, loss=7.194006, lr=0.000125, time_each_step=1.2s, eta=1:42:42
    2021-08-11 15:32:48 [INFO]	[TRAIN] Epoch=90/270, Step=12/74, loss=22.162321, lr=0.000125, time_each_step=1.21s, eta=1:42:40
    2021-08-11 15:32:49 [INFO]	[TRAIN] Epoch=90/270, Step=14/74, loss=26.867262, lr=0.000125, time_each_step=1.22s, eta=1:42:38
    2021-08-11 15:32:50 [INFO]	[TRAIN] Epoch=90/270, Step=16/74, loss=28.060038, lr=0.000125, time_each_step=1.24s, eta=1:42:37
    2021-08-11 15:32:50 [INFO]	[TRAIN] Epoch=90/270, Step=18/74, loss=9.818938, lr=0.000125, time_each_step=1.25s, eta=1:42:35
    2021-08-11 15:32:51 [INFO]	[TRAIN] Epoch=90/270, Step=20/74, loss=37.262154, lr=0.000125, time_each_step=1.28s, eta=1:42:34
    2021-08-11 15:32:52 [INFO]	[TRAIN] Epoch=90/270, Step=22/74, loss=22.816736, lr=0.000125, time_each_step=0.38s, eta=1:41:45
    2021-08-11 15:32:53 [INFO]	[TRAIN] Epoch=90/270, Step=24/74, loss=18.329872, lr=0.000125, time_each_step=0.39s, eta=1:41:45
    2021-08-11 15:32:54 [INFO]	[TRAIN] Epoch=90/270, Step=26/74, loss=20.778614, lr=0.000125, time_each_step=0.39s, eta=1:41:44
    2021-08-11 15:32:55 [INFO]	[TRAIN] Epoch=90/270, Step=28/74, loss=29.042274, lr=0.000125, time_each_step=0.39s, eta=1:41:43
    2021-08-11 15:32:55 [INFO]	[TRAIN] Epoch=90/270, Step=30/74, loss=22.261448, lr=0.000125, time_each_step=0.39s, eta=1:41:42
    2021-08-11 15:32:56 [INFO]	[TRAIN] Epoch=90/270, Step=32/74, loss=17.227951, lr=0.000125, time_each_step=0.38s, eta=1:41:41
    2021-08-11 15:32:57 [INFO]	[TRAIN] Epoch=90/270, Step=34/74, loss=26.762711, lr=0.000125, time_each_step=0.39s, eta=1:41:41
    2021-08-11 15:32:58 [INFO]	[TRAIN] Epoch=90/270, Step=36/74, loss=8.331539, lr=0.000125, time_each_step=0.4s, eta=1:41:40
    2021-08-11 15:32:58 [INFO]	[TRAIN] Epoch=90/270, Step=38/74, loss=42.851013, lr=0.000125, time_each_step=0.39s, eta=1:41:39
    2021-08-11 15:32:59 [INFO]	[TRAIN] Epoch=90/270, Step=40/74, loss=22.808027, lr=0.000125, time_each_step=0.37s, eta=1:41:38
    2021-08-11 15:32:59 [INFO]	[TRAIN] Epoch=90/270, Step=42/74, loss=24.021017, lr=0.000125, time_each_step=0.35s, eta=1:41:36
    2021-08-11 15:33:00 [INFO]	[TRAIN] Epoch=90/270, Step=44/74, loss=34.873299, lr=0.000125, time_each_step=0.35s, eta=1:41:36
    2021-08-11 15:33:01 [INFO]	[TRAIN] Epoch=90/270, Step=46/74, loss=55.949753, lr=0.000125, time_each_step=0.34s, eta=1:41:35
    2021-08-11 15:33:01 [INFO]	[TRAIN] Epoch=90/270, Step=48/74, loss=12.386339, lr=0.000125, time_each_step=0.33s, eta=1:41:34
    2021-08-11 15:33:02 [INFO]	[TRAIN] Epoch=90/270, Step=50/74, loss=21.716799, lr=0.000125, time_each_step=0.31s, eta=1:41:33
    2021-08-11 15:33:02 [INFO]	[TRAIN] Epoch=90/270, Step=52/74, loss=29.166527, lr=0.000125, time_each_step=0.31s, eta=1:41:32
    2021-08-11 15:33:02 [INFO]	[TRAIN] Epoch=90/270, Step=54/74, loss=13.991469, lr=0.000125, time_each_step=0.27s, eta=1:41:31
    2021-08-11 15:33:03 [INFO]	[TRAIN] Epoch=90/270, Step=56/74, loss=24.701138, lr=0.000125, time_each_step=0.26s, eta=1:41:30
    2021-08-11 15:33:03 [INFO]	[TRAIN] Epoch=90/270, Step=58/74, loss=24.36982, lr=0.000125, time_each_step=0.26s, eta=1:41:29
    2021-08-11 15:33:04 [INFO]	[TRAIN] Epoch=90/270, Step=60/74, loss=14.488284, lr=0.000125, time_each_step=0.24s, eta=1:41:29
    2021-08-11 15:33:04 [INFO]	[TRAIN] Epoch=90/270, Step=62/74, loss=44.012733, lr=0.000125, time_each_step=0.24s, eta=1:41:28
    2021-08-11 15:33:04 [INFO]	[TRAIN] Epoch=90/270, Step=64/74, loss=25.730343, lr=0.000125, time_each_step=0.22s, eta=1:41:27
    2021-08-11 15:33:05 [INFO]	[TRAIN] Epoch=90/270, Step=66/74, loss=16.072411, lr=0.000125, time_each_step=0.21s, eta=1:41:27
    2021-08-11 15:33:05 [INFO]	[TRAIN] Epoch=90/270, Step=68/74, loss=12.717312, lr=0.000125, time_each_step=0.2s, eta=1:41:26
    2021-08-11 15:33:06 [INFO]	[TRAIN] Epoch=90/270, Step=70/74, loss=25.301266, lr=0.000125, time_each_step=0.21s, eta=1:41:26
    2021-08-11 15:33:06 [INFO]	[TRAIN] Epoch=90/270, Step=72/74, loss=25.407139, lr=0.000125, time_each_step=0.21s, eta=1:41:26
    2021-08-11 15:33:07 [INFO]	[TRAIN] Epoch=90/270, Step=74/74, loss=23.445301, lr=0.000125, time_each_step=0.21s, eta=1:41:25
    2021-08-11 15:33:07 [INFO]	[TRAIN] Epoch 90 finished, loss=23.245478, lr=0.000125 .
    2021-08-11 15:33:15 [INFO]	[TRAIN] Epoch=91/270, Step=2/74, loss=24.511339, lr=0.000125, time_each_step=0.58s, eta=2:6:57
    2021-08-11 15:33:15 [INFO]	[TRAIN] Epoch=91/270, Step=4/74, loss=12.257869, lr=0.000125, time_each_step=0.59s, eta=2:6:57
    2021-08-11 15:33:16 [INFO]	[TRAIN] Epoch=91/270, Step=6/74, loss=17.778744, lr=0.000125, time_each_step=0.62s, eta=2:6:57
    2021-08-11 15:33:17 [INFO]	[TRAIN] Epoch=91/270, Step=8/74, loss=28.828423, lr=0.000125, time_each_step=0.63s, eta=2:6:57
    2021-08-11 15:33:18 [INFO]	[TRAIN] Epoch=91/270, Step=10/74, loss=17.290607, lr=0.000125, time_each_step=0.66s, eta=2:6:57
    2021-08-11 15:33:19 [INFO]	[TRAIN] Epoch=91/270, Step=12/74, loss=23.375622, lr=0.000125, time_each_step=0.69s, eta=2:6:58
    2021-08-11 15:33:19 [INFO]	[TRAIN] Epoch=91/270, Step=14/74, loss=8.854414, lr=0.000125, time_each_step=0.7s, eta=2:6:57
    2021-08-11 15:33:20 [INFO]	[TRAIN] Epoch=91/270, Step=16/74, loss=37.664562, lr=0.000125, time_each_step=0.72s, eta=2:6:57
    2021-08-11 15:33:21 [INFO]	[TRAIN] Epoch=91/270, Step=18/74, loss=16.734062, lr=0.000125, time_each_step=0.75s, eta=2:6:57
    2021-08-11 15:33:22 [INFO]	[TRAIN] Epoch=91/270, Step=20/74, loss=20.301929, lr=0.000125, time_each_step=0.77s, eta=2:6:57
    2021-08-11 15:33:23 [INFO]	[TRAIN] Epoch=91/270, Step=22/74, loss=17.130516, lr=0.000125, time_each_step=0.42s, eta=2:6:37
    2021-08-11 15:33:23 [INFO]	[TRAIN] Epoch=91/270, Step=24/74, loss=17.884689, lr=0.000125, time_each_step=0.41s, eta=2:6:36
    2021-08-11 15:33:24 [INFO]	[TRAIN] Epoch=91/270, Step=26/74, loss=31.38501, lr=0.000125, time_each_step=0.42s, eta=2:6:35
    2021-08-11 15:33:25 [INFO]	[TRAIN] Epoch=91/270, Step=28/74, loss=12.059471, lr=0.000125, time_each_step=0.42s, eta=2:6:34
    2021-08-11 15:33:26 [INFO]	[TRAIN] Epoch=91/270, Step=30/74, loss=26.948895, lr=0.000125, time_each_step=0.4s, eta=2:6:33
    2021-08-11 15:33:26 [INFO]	[TRAIN] Epoch=91/270, Step=32/74, loss=21.262156, lr=0.000125, time_each_step=0.39s, eta=2:6:32
    2021-08-11 15:33:27 [INFO]	[TRAIN] Epoch=91/270, Step=34/74, loss=31.623047, lr=0.000125, time_each_step=0.37s, eta=2:6:30
    2021-08-11 15:33:27 [INFO]	[TRAIN] Epoch=91/270, Step=36/74, loss=27.973034, lr=0.000125, time_each_step=0.35s, eta=2:6:28
    2021-08-11 15:33:28 [INFO]	[TRAIN] Epoch=91/270, Step=38/74, loss=19.378345, lr=0.000125, time_each_step=0.34s, eta=2:6:27
    2021-08-11 15:33:29 [INFO]	[TRAIN] Epoch=91/270, Step=40/74, loss=17.3004, lr=0.000125, time_each_step=0.34s, eta=2:6:27
    2021-08-11 15:33:29 [INFO]	[TRAIN] Epoch=91/270, Step=42/74, loss=14.128635, lr=0.000125, time_each_step=0.33s, eta=2:6:26
    2021-08-11 15:33:30 [INFO]	[TRAIN] Epoch=91/270, Step=44/74, loss=25.581549, lr=0.000125, time_each_step=0.33s, eta=2:6:25
    2021-08-11 15:33:31 [INFO]	[TRAIN] Epoch=91/270, Step=46/74, loss=17.098986, lr=0.000125, time_each_step=0.31s, eta=2:6:24
    2021-08-11 15:33:31 [INFO]	[TRAIN] Epoch=91/270, Step=48/74, loss=26.417988, lr=0.000125, time_each_step=0.31s, eta=2:6:23
    2021-08-11 15:33:32 [INFO]	[TRAIN] Epoch=91/270, Step=50/74, loss=36.071732, lr=0.000125, time_each_step=0.3s, eta=2:6:22
    2021-08-11 15:33:32 [INFO]	[TRAIN] Epoch=91/270, Step=52/74, loss=17.04656, lr=0.000125, time_each_step=0.29s, eta=2:6:21
    2021-08-11 15:33:32 [INFO]	[TRAIN] Epoch=91/270, Step=54/74, loss=9.860714, lr=0.000125, time_each_step=0.29s, eta=2:6:21
    2021-08-11 15:33:33 [INFO]	[TRAIN] Epoch=91/270, Step=56/74, loss=33.610538, lr=0.000125, time_each_step=0.29s, eta=2:6:20
    2021-08-11 15:33:34 [INFO]	[TRAIN] Epoch=91/270, Step=58/74, loss=13.535924, lr=0.000125, time_each_step=0.28s, eta=2:6:20
    2021-08-11 15:33:34 [INFO]	[TRAIN] Epoch=91/270, Step=60/74, loss=14.307795, lr=0.000125, time_each_step=0.26s, eta=2:6:19
    2021-08-11 15:33:34 [INFO]	[TRAIN] Epoch=91/270, Step=62/74, loss=18.205214, lr=0.000125, time_each_step=0.25s, eta=2:6:18
    2021-08-11 15:33:35 [INFO]	[TRAIN] Epoch=91/270, Step=64/74, loss=27.281307, lr=0.000125, time_each_step=0.24s, eta=2:6:18
    2021-08-11 15:33:35 [INFO]	[TRAIN] Epoch=91/270, Step=66/74, loss=25.962996, lr=0.000125, time_each_step=0.24s, eta=2:6:17
    2021-08-11 15:33:36 [INFO]	[TRAIN] Epoch=91/270, Step=68/74, loss=13.046114, lr=0.000125, time_each_step=0.23s, eta=2:6:16
    2021-08-11 15:33:36 [INFO]	[TRAIN] Epoch=91/270, Step=70/74, loss=17.367599, lr=0.000125, time_each_step=0.22s, eta=2:6:16
    2021-08-11 15:33:36 [INFO]	[TRAIN] Epoch=91/270, Step=72/74, loss=23.39818, lr=0.000125, time_each_step=0.22s, eta=2:6:16
    2021-08-11 15:33:37 [INFO]	[TRAIN] Epoch=91/270, Step=74/74, loss=19.742542, lr=0.000125, time_each_step=0.21s, eta=2:6:15
    2021-08-11 15:33:37 [INFO]	[TRAIN] Epoch 91 finished, loss=21.244057, lr=0.000125 .
    2021-08-11 15:33:42 [INFO]	[TRAIN] Epoch=92/270, Step=2/74, loss=17.665066, lr=0.000125, time_each_step=0.43s, eta=1:33:59
    2021-08-11 15:33:42 [INFO]	[TRAIN] Epoch=92/270, Step=4/74, loss=14.251398, lr=0.000125, time_each_step=0.44s, eta=1:33:59
    2021-08-11 15:33:43 [INFO]	[TRAIN] Epoch=92/270, Step=6/74, loss=18.108479, lr=0.000125, time_each_step=0.46s, eta=1:34:0
    2021-08-11 15:33:44 [INFO]	[TRAIN] Epoch=92/270, Step=8/74, loss=17.563171, lr=0.000125, time_each_step=0.49s, eta=1:34:1
    2021-08-11 15:33:45 [INFO]	[TRAIN] Epoch=92/270, Step=10/74, loss=17.802551, lr=0.000125, time_each_step=0.52s, eta=1:34:1
    2021-08-11 15:33:46 [INFO]	[TRAIN] Epoch=92/270, Step=12/74, loss=26.824368, lr=0.000125, time_each_step=0.53s, eta=1:34:1
    2021-08-11 15:33:47 [INFO]	[TRAIN] Epoch=92/270, Step=14/74, loss=21.225449, lr=0.000125, time_each_step=0.55s, eta=1:34:2
    2021-08-11 15:33:48 [INFO]	[TRAIN] Epoch=92/270, Step=16/74, loss=29.387615, lr=0.000125, time_each_step=0.58s, eta=1:34:2
    2021-08-11 15:33:48 [INFO]	[TRAIN] Epoch=92/270, Step=18/74, loss=23.555016, lr=0.000125, time_each_step=0.59s, eta=1:34:1
    2021-08-11 15:33:49 [INFO]	[TRAIN] Epoch=92/270, Step=20/74, loss=24.147062, lr=0.000125, time_each_step=0.62s, eta=1:34:2
    2021-08-11 15:33:50 [INFO]	[TRAIN] Epoch=92/270, Step=22/74, loss=11.234073, lr=0.000125, time_each_step=0.41s, eta=1:33:50
    2021-08-11 15:33:51 [INFO]	[TRAIN] Epoch=92/270, Step=24/74, loss=24.078796, lr=0.000125, time_each_step=0.42s, eta=1:33:49
    2021-08-11 15:33:51 [INFO]	[TRAIN] Epoch=92/270, Step=26/74, loss=18.789955, lr=0.000125, time_each_step=0.41s, eta=1:33:48
    2021-08-11 15:33:52 [INFO]	[TRAIN] Epoch=92/270, Step=28/74, loss=36.873707, lr=0.000125, time_each_step=0.4s, eta=1:33:47
    2021-08-11 15:33:53 [INFO]	[TRAIN] Epoch=92/270, Step=30/74, loss=19.548803, lr=0.000125, time_each_step=0.39s, eta=1:33:46
    2021-08-11 15:33:54 [INFO]	[TRAIN] Epoch=92/270, Step=32/74, loss=21.034607, lr=0.000125, time_each_step=0.39s, eta=1:33:45
    2021-08-11 15:33:54 [INFO]	[TRAIN] Epoch=92/270, Step=34/74, loss=21.882896, lr=0.000125, time_each_step=0.38s, eta=1:33:43
    2021-08-11 15:33:55 [INFO]	[TRAIN] Epoch=92/270, Step=36/74, loss=45.62484, lr=0.000125, time_each_step=0.39s, eta=1:33:43
    2021-08-11 15:33:56 [INFO]	[TRAIN] Epoch=92/270, Step=38/74, loss=16.376759, lr=0.000125, time_each_step=0.39s, eta=1:33:42
    2021-08-11 15:33:56 [INFO]	[TRAIN] Epoch=92/270, Step=40/74, loss=16.103745, lr=0.000125, time_each_step=0.37s, eta=1:33:41
    2021-08-11 15:33:57 [INFO]	[TRAIN] Epoch=92/270, Step=42/74, loss=15.721189, lr=0.000125, time_each_step=0.37s, eta=1:33:40
    2021-08-11 15:33:58 [INFO]	[TRAIN] Epoch=92/270, Step=44/74, loss=29.553318, lr=0.000125, time_each_step=0.35s, eta=1:33:39
    2021-08-11 15:33:58 [INFO]	[TRAIN] Epoch=92/270, Step=46/74, loss=20.549612, lr=0.000125, time_each_step=0.35s, eta=1:33:38
    2021-08-11 15:33:59 [INFO]	[TRAIN] Epoch=92/270, Step=48/74, loss=24.493343, lr=0.000125, time_each_step=0.33s, eta=1:33:37
    2021-08-11 15:33:59 [INFO]	[TRAIN] Epoch=92/270, Step=50/74, loss=27.15901, lr=0.000125, time_each_step=0.31s, eta=1:33:36
    2021-08-11 15:34:00 [INFO]	[TRAIN] Epoch=92/270, Step=52/74, loss=20.728825, lr=0.000125, time_each_step=0.29s, eta=1:33:35
    2021-08-11 15:34:00 [INFO]	[TRAIN] Epoch=92/270, Step=54/74, loss=17.690107, lr=0.000125, time_each_step=0.28s, eta=1:33:34
    2021-08-11 15:34:01 [INFO]	[TRAIN] Epoch=92/270, Step=56/74, loss=21.3517, lr=0.000125, time_each_step=0.26s, eta=1:33:33
    2021-08-11 15:34:01 [INFO]	[TRAIN] Epoch=92/270, Step=58/74, loss=30.784355, lr=0.000125, time_each_step=0.24s, eta=1:33:32
    2021-08-11 15:34:01 [INFO]	[TRAIN] Epoch=92/270, Step=60/74, loss=13.792846, lr=0.000125, time_each_step=0.23s, eta=1:33:31
    2021-08-11 15:34:02 [INFO]	[TRAIN] Epoch=92/270, Step=62/74, loss=25.957592, lr=0.000125, time_each_step=0.22s, eta=1:33:31
    2021-08-11 15:34:02 [INFO]	[TRAIN] Epoch=92/270, Step=64/74, loss=15.705698, lr=0.000125, time_each_step=0.2s, eta=1:33:30
    2021-08-11 15:34:02 [INFO]	[TRAIN] Epoch=92/270, Step=66/74, loss=26.061594, lr=0.000125, time_each_step=0.2s, eta=1:33:30
    2021-08-11 15:34:02 [INFO]	[TRAIN] Epoch=92/270, Step=68/74, loss=17.761009, lr=0.000125, time_each_step=0.18s, eta=1:33:29
    2021-08-11 15:34:03 [INFO]	[TRAIN] Epoch=92/270, Step=70/74, loss=20.278053, lr=0.000125, time_each_step=0.18s, eta=1:33:29
    2021-08-11 15:34:03 [INFO]	[TRAIN] Epoch=92/270, Step=72/74, loss=14.57565, lr=0.000125, time_each_step=0.18s, eta=1:33:29
    2021-08-11 15:34:03 [INFO]	[TRAIN] Epoch=92/270, Step=74/74, loss=35.20121, lr=0.000125, time_each_step=0.18s, eta=1:33:28
    2021-08-11 15:34:03 [INFO]	[TRAIN] Epoch 92 finished, loss=21.151398, lr=0.000125 .
    2021-08-11 15:34:14 [INFO]	[TRAIN] Epoch=93/270, Step=2/74, loss=31.120457, lr=0.000125, time_each_step=0.7s, eta=1:24:7
    2021-08-11 15:34:15 [INFO]	[TRAIN] Epoch=93/270, Step=4/74, loss=12.64445, lr=0.000125, time_each_step=0.73s, eta=1:24:8
    2021-08-11 15:34:16 [INFO]	[TRAIN] Epoch=93/270, Step=6/74, loss=18.094921, lr=0.000125, time_each_step=0.75s, eta=1:24:8
    2021-08-11 15:34:17 [INFO]	[TRAIN] Epoch=93/270, Step=8/74, loss=19.516104, lr=0.000125, time_each_step=0.77s, eta=1:24:8
    2021-08-11 15:34:18 [INFO]	[TRAIN] Epoch=93/270, Step=10/74, loss=24.794884, lr=0.000125, time_each_step=0.79s, eta=1:24:8
    2021-08-11 15:34:19 [INFO]	[TRAIN] Epoch=93/270, Step=12/74, loss=18.134537, lr=0.000125, time_each_step=0.82s, eta=1:24:8
    2021-08-11 15:34:20 [INFO]	[TRAIN] Epoch=93/270, Step=14/74, loss=25.436861, lr=0.000125, time_each_step=0.85s, eta=1:24:8
    2021-08-11 15:34:20 [INFO]	[TRAIN] Epoch=93/270, Step=16/74, loss=13.993749, lr=0.000125, time_each_step=0.87s, eta=1:24:7
    2021-08-11 15:34:21 [INFO]	[TRAIN] Epoch=93/270, Step=18/74, loss=17.057119, lr=0.000125, time_each_step=0.89s, eta=1:24:7
    2021-08-11 15:34:22 [INFO]	[TRAIN] Epoch=93/270, Step=20/74, loss=20.495487, lr=0.000125, time_each_step=0.91s, eta=1:24:6
    2021-08-11 15:34:23 [INFO]	[TRAIN] Epoch=93/270, Step=22/74, loss=46.467209, lr=0.000125, time_each_step=0.4s, eta=1:23:38
    2021-08-11 15:34:23 [INFO]	[TRAIN] Epoch=93/270, Step=24/74, loss=8.979901, lr=0.000125, time_each_step=0.41s, eta=1:23:37
    2021-08-11 15:34:25 [INFO]	[TRAIN] Epoch=93/270, Step=26/74, loss=19.793144, lr=0.000125, time_each_step=0.42s, eta=1:23:37
    2021-08-11 15:34:25 [INFO]	[TRAIN] Epoch=93/270, Step=28/74, loss=30.028236, lr=0.000125, time_each_step=0.42s, eta=1:23:36
    2021-08-11 15:34:26 [INFO]	[TRAIN] Epoch=93/270, Step=30/74, loss=12.360947, lr=0.000125, time_each_step=0.42s, eta=1:23:35
    2021-08-11 15:34:27 [INFO]	[TRAIN] Epoch=93/270, Step=32/74, loss=45.559647, lr=0.000125, time_each_step=0.42s, eta=1:23:34
    2021-08-11 15:34:28 [INFO]	[TRAIN] Epoch=93/270, Step=34/74, loss=13.57477, lr=0.000125, time_each_step=0.42s, eta=1:23:33
    2021-08-11 15:34:29 [INFO]	[TRAIN] Epoch=93/270, Step=36/74, loss=28.376314, lr=0.000125, time_each_step=0.42s, eta=1:23:33
    2021-08-11 15:34:29 [INFO]	[TRAIN] Epoch=93/270, Step=38/74, loss=15.116152, lr=0.000125, time_each_step=0.42s, eta=1:23:32
    2021-08-11 15:34:30 [INFO]	[TRAIN] Epoch=93/270, Step=40/74, loss=23.358793, lr=0.000125, time_each_step=0.42s, eta=1:23:31
    2021-08-11 15:34:31 [INFO]	[TRAIN] Epoch=93/270, Step=42/74, loss=15.915207, lr=0.000125, time_each_step=0.42s, eta=1:23:30
    2021-08-11 15:34:31 [INFO]	[TRAIN] Epoch=93/270, Step=44/74, loss=39.401257, lr=0.000125, time_each_step=0.39s, eta=1:23:29
    2021-08-11 15:34:32 [INFO]	[TRAIN] Epoch=93/270, Step=46/74, loss=25.083229, lr=0.000125, time_each_step=0.36s, eta=1:23:27
    2021-08-11 15:34:32 [INFO]	[TRAIN] Epoch=93/270, Step=48/74, loss=25.234842, lr=0.000125, time_each_step=0.36s, eta=1:23:26
    2021-08-11 15:34:33 [INFO]	[TRAIN] Epoch=93/270, Step=50/74, loss=25.420368, lr=0.000125, time_each_step=0.34s, eta=1:23:25
    2021-08-11 15:34:33 [INFO]	[TRAIN] Epoch=93/270, Step=52/74, loss=11.292576, lr=0.000125, time_each_step=0.31s, eta=1:23:24
    2021-08-11 15:34:34 [INFO]	[TRAIN] Epoch=93/270, Step=54/74, loss=28.94837, lr=0.000125, time_each_step=0.29s, eta=1:23:22
    2021-08-11 15:34:34 [INFO]	[TRAIN] Epoch=93/270, Step=56/74, loss=16.529079, lr=0.000125, time_each_step=0.27s, eta=1:23:22
    2021-08-11 15:34:34 [INFO]	[TRAIN] Epoch=93/270, Step=58/74, loss=26.528473, lr=0.000125, time_each_step=0.26s, eta=1:23:21
    2021-08-11 15:34:35 [INFO]	[TRAIN] Epoch=93/270, Step=60/74, loss=38.833286, lr=0.000125, time_each_step=0.24s, eta=1:23:20
    2021-08-11 15:34:35 [INFO]	[TRAIN] Epoch=93/270, Step=62/74, loss=19.142132, lr=0.000125, time_each_step=0.23s, eta=1:23:19
    2021-08-11 15:34:36 [INFO]	[TRAIN] Epoch=93/270, Step=64/74, loss=19.663403, lr=0.000125, time_each_step=0.23s, eta=1:23:19
    2021-08-11 15:34:36 [INFO]	[TRAIN] Epoch=93/270, Step=66/74, loss=20.782684, lr=0.000125, time_each_step=0.23s, eta=1:23:19
    2021-08-11 15:34:37 [INFO]	[TRAIN] Epoch=93/270, Step=68/74, loss=18.41725, lr=0.000125, time_each_step=0.22s, eta=1:23:18
    2021-08-11 15:34:37 [INFO]	[TRAIN] Epoch=93/270, Step=70/74, loss=14.734507, lr=0.000125, time_each_step=0.21s, eta=1:23:18
    2021-08-11 15:34:38 [INFO]	[TRAIN] Epoch=93/270, Step=72/74, loss=17.753874, lr=0.000125, time_each_step=0.22s, eta=1:23:17
    2021-08-11 15:34:38 [INFO]	[TRAIN] Epoch=93/270, Step=74/74, loss=27.903584, lr=0.000125, time_each_step=0.23s, eta=1:23:17
    2021-08-11 15:34:38 [INFO]	[TRAIN] Epoch 93 finished, loss=22.860846, lr=0.000125 .
    2021-08-11 15:34:44 [INFO]	[TRAIN] Epoch=94/270, Step=2/74, loss=23.493032, lr=0.000125, time_each_step=0.49s, eta=1:46:45
    2021-08-11 15:34:45 [INFO]	[TRAIN] Epoch=94/270, Step=4/74, loss=13.980455, lr=0.000125, time_each_step=0.53s, eta=1:46:47
    2021-08-11 15:34:46 [INFO]	[TRAIN] Epoch=94/270, Step=6/74, loss=15.00414, lr=0.000125, time_each_step=0.56s, eta=1:46:48
    2021-08-11 15:34:47 [INFO]	[TRAIN] Epoch=94/270, Step=8/74, loss=23.980356, lr=0.000125, time_each_step=0.58s, eta=1:46:48
    2021-08-11 15:34:48 [INFO]	[TRAIN] Epoch=94/270, Step=10/74, loss=27.177549, lr=0.000125, time_each_step=0.59s, eta=1:46:47
    2021-08-11 15:34:48 [INFO]	[TRAIN] Epoch=94/270, Step=12/74, loss=15.597982, lr=0.000125, time_each_step=0.59s, eta=1:46:47
    2021-08-11 15:34:49 [INFO]	[TRAIN] Epoch=94/270, Step=14/74, loss=22.563084, lr=0.000125, time_each_step=0.6s, eta=1:46:46
    2021-08-11 15:34:50 [INFO]	[TRAIN] Epoch=94/270, Step=16/74, loss=10.483778, lr=0.000125, time_each_step=0.63s, eta=1:46:46
    2021-08-11 15:34:51 [INFO]	[TRAIN] Epoch=94/270, Step=18/74, loss=35.106628, lr=0.000125, time_each_step=0.64s, eta=1:46:46
    2021-08-11 15:34:51 [INFO]	[TRAIN] Epoch=94/270, Step=20/74, loss=36.275417, lr=0.000125, time_each_step=0.66s, eta=1:46:45
    2021-08-11 15:34:52 [INFO]	[TRAIN] Epoch=94/270, Step=22/74, loss=59.576111, lr=0.000125, time_each_step=0.41s, eta=1:46:31
    2021-08-11 15:34:53 [INFO]	[TRAIN] Epoch=94/270, Step=24/74, loss=15.171636, lr=0.000125, time_each_step=0.38s, eta=1:46:29
    2021-08-11 15:34:54 [INFO]	[TRAIN] Epoch=94/270, Step=26/74, loss=19.430126, lr=0.000125, time_each_step=0.38s, eta=1:46:28
    2021-08-11 15:34:54 [INFO]	[TRAIN] Epoch=94/270, Step=28/74, loss=12.344836, lr=0.000125, time_each_step=0.37s, eta=1:46:27
    2021-08-11 15:34:55 [INFO]	[TRAIN] Epoch=94/270, Step=30/74, loss=24.88673, lr=0.000125, time_each_step=0.36s, eta=1:46:25
    2021-08-11 15:34:56 [INFO]	[TRAIN] Epoch=94/270, Step=32/74, loss=25.705683, lr=0.000125, time_each_step=0.36s, eta=1:46:25
    2021-08-11 15:34:56 [INFO]	[TRAIN] Epoch=94/270, Step=34/74, loss=13.950928, lr=0.000125, time_each_step=0.36s, eta=1:46:24
    2021-08-11 15:34:57 [INFO]	[TRAIN] Epoch=94/270, Step=36/74, loss=43.029785, lr=0.000125, time_each_step=0.35s, eta=1:46:23
    2021-08-11 15:34:58 [INFO]	[TRAIN] Epoch=94/270, Step=38/74, loss=17.151814, lr=0.000125, time_each_step=0.36s, eta=1:46:23
    2021-08-11 15:34:58 [INFO]	[TRAIN] Epoch=94/270, Step=40/74, loss=24.885315, lr=0.000125, time_each_step=0.35s, eta=1:46:21
    2021-08-11 15:34:59 [INFO]	[TRAIN] Epoch=94/270, Step=42/74, loss=12.654383, lr=0.000125, time_each_step=0.33s, eta=1:46:20
    2021-08-11 15:35:00 [INFO]	[TRAIN] Epoch=94/270, Step=44/74, loss=11.638393, lr=0.000125, time_each_step=0.35s, eta=1:46:20
    2021-08-11 15:35:00 [INFO]	[TRAIN] Epoch=94/270, Step=46/74, loss=20.385977, lr=0.000125, time_each_step=0.33s, eta=1:46:19
    2021-08-11 15:35:01 [INFO]	[TRAIN] Epoch=94/270, Step=48/74, loss=17.482941, lr=0.000125, time_each_step=0.32s, eta=1:46:18
    2021-08-11 15:35:01 [INFO]	[TRAIN] Epoch=94/270, Step=50/74, loss=16.34811, lr=0.000125, time_each_step=0.3s, eta=1:46:17
    2021-08-11 15:35:01 [INFO]	[TRAIN] Epoch=94/270, Step=52/74, loss=19.587704, lr=0.000125, time_each_step=0.3s, eta=1:46:16
    2021-08-11 15:35:02 [INFO]	[TRAIN] Epoch=94/270, Step=54/74, loss=16.234722, lr=0.000125, time_each_step=0.28s, eta=1:46:15
    2021-08-11 15:35:02 [INFO]	[TRAIN] Epoch=94/270, Step=56/74, loss=55.930355, lr=0.000125, time_each_step=0.26s, eta=1:46:14
    2021-08-11 15:35:03 [INFO]	[TRAIN] Epoch=94/270, Step=58/74, loss=43.616604, lr=0.000125, time_each_step=0.25s, eta=1:46:14
    2021-08-11 15:35:03 [INFO]	[TRAIN] Epoch=94/270, Step=60/74, loss=28.969217, lr=0.000125, time_each_step=0.23s, eta=1:46:13
    2021-08-11 15:35:03 [INFO]	[TRAIN] Epoch=94/270, Step=62/74, loss=14.981735, lr=0.000125, time_each_step=0.23s, eta=1:46:12
    2021-08-11 15:35:04 [INFO]	[TRAIN] Epoch=94/270, Step=64/74, loss=25.414152, lr=0.000125, time_each_step=0.2s, eta=1:46:12
    2021-08-11 15:35:04 [INFO]	[TRAIN] Epoch=94/270, Step=66/74, loss=17.604479, lr=0.000125, time_each_step=0.19s, eta=1:46:11
    2021-08-11 15:35:04 [INFO]	[TRAIN] Epoch=94/270, Step=68/74, loss=15.913408, lr=0.000125, time_each_step=0.19s, eta=1:46:11
    2021-08-11 15:35:05 [INFO]	[TRAIN] Epoch=94/270, Step=70/74, loss=16.808453, lr=0.000125, time_each_step=0.19s, eta=1:46:10
    2021-08-11 15:35:05 [INFO]	[TRAIN] Epoch=94/270, Step=72/74, loss=19.040184, lr=0.000125, time_each_step=0.2s, eta=1:46:10
    2021-08-11 15:35:06 [INFO]	[TRAIN] Epoch=94/270, Step=74/74, loss=17.678627, lr=0.000125, time_each_step=0.2s, eta=1:46:10
    2021-08-11 15:35:06 [INFO]	[TRAIN] Epoch 94 finished, loss=21.622393, lr=0.000125 .
    2021-08-11 15:35:26 [INFO]	[TRAIN] Epoch=95/270, Step=2/74, loss=30.620541, lr=0.000125, time_each_step=1.17s, eta=1:26:9
    2021-08-11 15:35:26 [INFO]	[TRAIN] Epoch=95/270, Step=4/74, loss=26.894157, lr=0.000125, time_each_step=1.18s, eta=1:26:7
    2021-08-11 15:35:27 [INFO]	[TRAIN] Epoch=95/270, Step=6/74, loss=19.815416, lr=0.000125, time_each_step=1.2s, eta=1:26:6
    2021-08-11 15:35:28 [INFO]	[TRAIN] Epoch=95/270, Step=8/74, loss=13.185664, lr=0.000125, time_each_step=1.21s, eta=1:26:4
    2021-08-11 15:35:28 [INFO]	[TRAIN] Epoch=95/270, Step=10/74, loss=37.304695, lr=0.000125, time_each_step=1.23s, eta=1:26:3
    2021-08-11 15:35:29 [INFO]	[TRAIN] Epoch=95/270, Step=12/74, loss=48.265137, lr=0.000125, time_each_step=1.26s, eta=1:26:3
    2021-08-11 15:35:30 [INFO]	[TRAIN] Epoch=95/270, Step=14/74, loss=9.22628, lr=0.000125, time_each_step=1.3s, eta=1:26:3
    2021-08-11 15:35:32 [INFO]	[TRAIN] Epoch=95/270, Step=16/74, loss=32.022484, lr=0.000125, time_each_step=1.34s, eta=1:26:2
    2021-08-11 15:35:33 [INFO]	[TRAIN] Epoch=95/270, Step=18/74, loss=47.04805, lr=0.000125, time_each_step=1.36s, eta=1:26:1
    2021-08-11 15:35:33 [INFO]	[TRAIN] Epoch=95/270, Step=20/74, loss=16.892262, lr=0.000125, time_each_step=1.38s, eta=1:25:59
    2021-08-11 15:35:34 [INFO]	[TRAIN] Epoch=95/270, Step=22/74, loss=19.792587, lr=0.000125, time_each_step=0.45s, eta=1:25:8
    2021-08-11 15:35:35 [INFO]	[TRAIN] Epoch=95/270, Step=24/74, loss=22.766277, lr=0.000125, time_each_step=0.46s, eta=1:25:8
    2021-08-11 15:35:36 [INFO]	[TRAIN] Epoch=95/270, Step=26/74, loss=18.467255, lr=0.000125, time_each_step=0.45s, eta=1:25:6
    2021-08-11 15:35:36 [INFO]	[TRAIN] Epoch=95/270, Step=28/74, loss=27.773615, lr=0.000125, time_each_step=0.44s, eta=1:25:5
    2021-08-11 15:35:37 [INFO]	[TRAIN] Epoch=95/270, Step=30/74, loss=11.076743, lr=0.000125, time_each_step=0.44s, eta=1:25:4
    2021-08-11 15:35:38 [INFO]	[TRAIN] Epoch=95/270, Step=32/74, loss=20.077511, lr=0.000125, time_each_step=0.45s, eta=1:25:3
    2021-08-11 15:35:39 [INFO]	[TRAIN] Epoch=95/270, Step=34/74, loss=30.06255, lr=0.000125, time_each_step=0.44s, eta=1:25:2
    2021-08-11 15:35:40 [INFO]	[TRAIN] Epoch=95/270, Step=36/74, loss=14.793769, lr=0.000125, time_each_step=0.42s, eta=1:25:1
    2021-08-11 15:35:41 [INFO]	[TRAIN] Epoch=95/270, Step=38/74, loss=47.326946, lr=0.000125, time_each_step=0.43s, eta=1:25:0
    2021-08-11 15:35:42 [INFO]	[TRAIN] Epoch=95/270, Step=40/74, loss=18.783974, lr=0.000125, time_each_step=0.41s, eta=1:24:59
    2021-08-11 15:35:42 [INFO]	[TRAIN] Epoch=95/270, Step=42/74, loss=27.037764, lr=0.000125, time_each_step=0.39s, eta=1:24:57
    2021-08-11 15:35:43 [INFO]	[TRAIN] Epoch=95/270, Step=44/74, loss=22.666361, lr=0.000125, time_each_step=0.36s, eta=1:24:55
    2021-08-11 15:35:43 [INFO]	[TRAIN] Epoch=95/270, Step=46/74, loss=49.673443, lr=0.000125, time_each_step=0.36s, eta=1:24:55
    2021-08-11 15:35:43 [INFO]	[TRAIN] Epoch=95/270, Step=48/74, loss=14.139326, lr=0.000125, time_each_step=0.35s, eta=1:24:54
    2021-08-11 15:35:44 [INFO]	[TRAIN] Epoch=95/270, Step=50/74, loss=40.442406, lr=0.000125, time_each_step=0.35s, eta=1:24:53
    2021-08-11 15:35:44 [INFO]	[TRAIN] Epoch=95/270, Step=52/74, loss=13.272435, lr=0.000125, time_each_step=0.31s, eta=1:24:51
    2021-08-11 15:35:45 [INFO]	[TRAIN] Epoch=95/270, Step=54/74, loss=20.273418, lr=0.000125, time_each_step=0.27s, eta=1:24:50
    2021-08-11 15:35:45 [INFO]	[TRAIN] Epoch=95/270, Step=56/74, loss=21.036427, lr=0.000125, time_each_step=0.25s, eta=1:24:49
    2021-08-11 15:35:46 [INFO]	[TRAIN] Epoch=95/270, Step=58/74, loss=17.462776, lr=0.000125, time_each_step=0.23s, eta=1:24:48
    2021-08-11 15:35:46 [INFO]	[TRAIN] Epoch=95/270, Step=60/74, loss=27.511292, lr=0.000125, time_each_step=0.22s, eta=1:24:48
    2021-08-11 15:35:46 [INFO]	[TRAIN] Epoch=95/270, Step=62/74, loss=29.681755, lr=0.000125, time_each_step=0.21s, eta=1:24:47
    2021-08-11 15:35:47 [INFO]	[TRAIN] Epoch=95/270, Step=64/74, loss=20.430584, lr=0.000125, time_each_step=0.21s, eta=1:24:47
    2021-08-11 15:35:47 [INFO]	[TRAIN] Epoch=95/270, Step=66/74, loss=19.925846, lr=0.000125, time_each_step=0.2s, eta=1:24:46
    2021-08-11 15:35:48 [INFO]	[TRAIN] Epoch=95/270, Step=68/74, loss=26.030617, lr=0.000125, time_each_step=0.21s, eta=1:24:46
    2021-08-11 15:35:48 [INFO]	[TRAIN] Epoch=95/270, Step=70/74, loss=11.157507, lr=0.000125, time_each_step=0.2s, eta=1:24:45
    2021-08-11 15:35:48 [INFO]	[TRAIN] Epoch=95/270, Step=72/74, loss=22.318226, lr=0.000125, time_each_step=0.2s, eta=1:24:45
    2021-08-11 15:35:49 [INFO]	[TRAIN] Epoch=95/270, Step=74/74, loss=19.035683, lr=0.000125, time_each_step=0.21s, eta=1:24:45
    2021-08-11 15:35:49 [INFO]	[TRAIN] Epoch 95 finished, loss=22.710106, lr=0.000125 .
    2021-08-11 15:36:08 [INFO]	[TRAIN] Epoch=96/270, Step=2/74, loss=11.157682, lr=0.000125, time_each_step=1.14s, eta=2:10:33
    2021-08-11 15:36:09 [INFO]	[TRAIN] Epoch=96/270, Step=4/74, loss=13.198305, lr=0.000125, time_each_step=1.15s, eta=2:10:31
    2021-08-11 15:36:09 [INFO]	[TRAIN] Epoch=96/270, Step=6/74, loss=17.879129, lr=0.000125, time_each_step=1.17s, eta=2:10:30
    2021-08-11 15:36:10 [INFO]	[TRAIN] Epoch=96/270, Step=8/74, loss=16.235325, lr=0.000125, time_each_step=1.2s, eta=2:10:30
    2021-08-11 15:36:11 [INFO]	[TRAIN] Epoch=96/270, Step=10/74, loss=16.678139, lr=0.000125, time_each_step=1.22s, eta=2:10:28
    2021-08-11 15:36:12 [INFO]	[TRAIN] Epoch=96/270, Step=12/74, loss=23.508224, lr=0.000125, time_each_step=1.23s, eta=2:10:27
    2021-08-11 15:36:13 [INFO]	[TRAIN] Epoch=96/270, Step=14/74, loss=36.380058, lr=0.000125, time_each_step=1.24s, eta=2:10:25
    2021-08-11 15:36:13 [INFO]	[TRAIN] Epoch=96/270, Step=16/74, loss=16.791943, lr=0.000125, time_each_step=1.27s, eta=2:10:24
    2021-08-11 15:36:14 [INFO]	[TRAIN] Epoch=96/270, Step=18/74, loss=31.633509, lr=0.000125, time_each_step=1.29s, eta=2:10:22
    2021-08-11 15:36:15 [INFO]	[TRAIN] Epoch=96/270, Step=20/74, loss=9.933943, lr=0.000125, time_each_step=1.3s, eta=2:10:21
    2021-08-11 15:36:16 [INFO]	[TRAIN] Epoch=96/270, Step=22/74, loss=35.30209, lr=0.000125, time_each_step=0.38s, eta=2:9:30
    2021-08-11 15:36:17 [INFO]	[TRAIN] Epoch=96/270, Step=24/74, loss=20.749651, lr=0.000125, time_each_step=0.39s, eta=2:9:30
    2021-08-11 15:36:17 [INFO]	[TRAIN] Epoch=96/270, Step=26/74, loss=19.448063, lr=0.000125, time_each_step=0.39s, eta=2:9:29
    2021-08-11 15:36:18 [INFO]	[TRAIN] Epoch=96/270, Step=28/74, loss=14.803411, lr=0.000125, time_each_step=0.39s, eta=2:9:28
    2021-08-11 15:36:19 [INFO]	[TRAIN] Epoch=96/270, Step=30/74, loss=27.334427, lr=0.000125, time_each_step=0.38s, eta=2:9:27
    2021-08-11 15:36:19 [INFO]	[TRAIN] Epoch=96/270, Step=32/74, loss=16.014076, lr=0.000125, time_each_step=0.38s, eta=2:9:26
    2021-08-11 15:36:20 [INFO]	[TRAIN] Epoch=96/270, Step=34/74, loss=30.59569, lr=0.000125, time_each_step=0.38s, eta=2:9:26
    2021-08-11 15:36:21 [INFO]	[TRAIN] Epoch=96/270, Step=36/74, loss=7.75388, lr=0.000125, time_each_step=0.38s, eta=2:9:25
    2021-08-11 15:36:21 [INFO]	[TRAIN] Epoch=96/270, Step=38/74, loss=22.912464, lr=0.000125, time_each_step=0.37s, eta=2:9:24
    2021-08-11 15:36:22 [INFO]	[TRAIN] Epoch=96/270, Step=40/74, loss=28.850874, lr=0.000125, time_each_step=0.36s, eta=2:9:23
    2021-08-11 15:36:23 [INFO]	[TRAIN] Epoch=96/270, Step=42/74, loss=20.091713, lr=0.000125, time_each_step=0.36s, eta=2:9:22
    2021-08-11 15:36:24 [INFO]	[TRAIN] Epoch=96/270, Step=44/74, loss=21.050512, lr=0.000125, time_each_step=0.35s, eta=2:9:21
    2021-08-11 15:36:24 [INFO]	[TRAIN] Epoch=96/270, Step=46/74, loss=37.033112, lr=0.000125, time_each_step=0.34s, eta=2:9:20
    2021-08-11 15:36:25 [INFO]	[TRAIN] Epoch=96/270, Step=48/74, loss=34.66539, lr=0.000125, time_each_step=0.33s, eta=2:9:19
    2021-08-11 15:36:25 [INFO]	[TRAIN] Epoch=96/270, Step=50/74, loss=13.661402, lr=0.000125, time_each_step=0.33s, eta=2:9:18
    2021-08-11 15:36:26 [INFO]	[TRAIN] Epoch=96/270, Step=52/74, loss=7.958707, lr=0.000125, time_each_step=0.32s, eta=2:9:18
    2021-08-11 15:36:26 [INFO]	[TRAIN] Epoch=96/270, Step=54/74, loss=18.575373, lr=0.000125, time_each_step=0.31s, eta=2:9:17
    2021-08-11 15:36:27 [INFO]	[TRAIN] Epoch=96/270, Step=56/74, loss=17.085306, lr=0.000125, time_each_step=0.28s, eta=2:9:16
    2021-08-11 15:36:27 [INFO]	[TRAIN] Epoch=96/270, Step=58/74, loss=26.569401, lr=0.000125, time_each_step=0.27s, eta=2:9:15
    2021-08-11 15:36:27 [INFO]	[TRAIN] Epoch=96/270, Step=60/74, loss=12.744633, lr=0.000125, time_each_step=0.27s, eta=2:9:14
    2021-08-11 15:36:28 [INFO]	[TRAIN] Epoch=96/270, Step=62/74, loss=26.53821, lr=0.000125, time_each_step=0.25s, eta=2:9:13
    2021-08-11 15:36:28 [INFO]	[TRAIN] Epoch=96/270, Step=64/74, loss=12.345298, lr=0.000125, time_each_step=0.23s, eta=2:9:13
    2021-08-11 15:36:28 [INFO]	[TRAIN] Epoch=96/270, Step=66/74, loss=27.71982, lr=0.000125, time_each_step=0.23s, eta=2:9:12
    2021-08-11 15:36:29 [INFO]	[TRAIN] Epoch=96/270, Step=68/74, loss=20.758226, lr=0.000125, time_each_step=0.22s, eta=2:9:12
    2021-08-11 15:36:29 [INFO]	[TRAIN] Epoch=96/270, Step=70/74, loss=18.383509, lr=0.000125, time_each_step=0.21s, eta=2:9:11
    2021-08-11 15:36:30 [INFO]	[TRAIN] Epoch=96/270, Step=72/74, loss=20.735672, lr=0.000125, time_each_step=0.2s, eta=2:9:11
    2021-08-11 15:36:30 [INFO]	[TRAIN] Epoch=96/270, Step=74/74, loss=25.873306, lr=0.000125, time_each_step=0.19s, eta=2:9:10
    2021-08-11 15:36:30 [INFO]	[TRAIN] Epoch 96 finished, loss=21.88756, lr=0.000125 .
    2021-08-11 15:36:49 [INFO]	[TRAIN] Epoch=97/270, Step=2/74, loss=10.39745, lr=0.000125, time_each_step=1.11s, eta=2:4:41
    2021-08-11 15:36:50 [INFO]	[TRAIN] Epoch=97/270, Step=4/74, loss=14.631416, lr=0.000125, time_each_step=1.14s, eta=2:4:41
    2021-08-11 15:36:51 [INFO]	[TRAIN] Epoch=97/270, Step=6/74, loss=25.188116, lr=0.000125, time_each_step=1.16s, eta=2:4:40
    2021-08-11 15:36:52 [INFO]	[TRAIN] Epoch=97/270, Step=8/74, loss=15.257844, lr=0.000125, time_each_step=1.19s, eta=2:4:39
    2021-08-11 15:36:52 [INFO]	[TRAIN] Epoch=97/270, Step=10/74, loss=29.427605, lr=0.000125, time_each_step=1.2s, eta=2:4:37
    2021-08-11 15:36:53 [INFO]	[TRAIN] Epoch=97/270, Step=12/74, loss=15.364227, lr=0.000125, time_each_step=1.22s, eta=2:4:36
    2021-08-11 15:36:54 [INFO]	[TRAIN] Epoch=97/270, Step=14/74, loss=16.299164, lr=0.000125, time_each_step=1.24s, eta=2:4:35
    2021-08-11 15:36:55 [INFO]	[TRAIN] Epoch=97/270, Step=16/74, loss=14.092315, lr=0.000125, time_each_step=1.26s, eta=2:4:34
    2021-08-11 15:36:55 [INFO]	[TRAIN] Epoch=97/270, Step=18/74, loss=17.924923, lr=0.000125, time_each_step=1.27s, eta=2:4:32
    2021-08-11 15:36:56 [INFO]	[TRAIN] Epoch=97/270, Step=20/74, loss=10.515421, lr=0.000125, time_each_step=1.3s, eta=2:4:31
    2021-08-11 15:36:57 [INFO]	[TRAIN] Epoch=97/270, Step=22/74, loss=40.267815, lr=0.000125, time_each_step=0.42s, eta=2:3:42
    2021-08-11 15:36:58 [INFO]	[TRAIN] Epoch=97/270, Step=24/74, loss=16.102264, lr=0.000125, time_each_step=0.41s, eta=2:3:41
    2021-08-11 15:36:59 [INFO]	[TRAIN] Epoch=97/270, Step=26/74, loss=17.485165, lr=0.000125, time_each_step=0.42s, eta=2:3:41
    2021-08-11 15:37:00 [INFO]	[TRAIN] Epoch=97/270, Step=28/74, loss=65.49482, lr=0.000125, time_each_step=0.41s, eta=2:3:39
    2021-08-11 15:37:00 [INFO]	[TRAIN] Epoch=97/270, Step=30/74, loss=24.405884, lr=0.000125, time_each_step=0.42s, eta=2:3:39
    2021-08-11 15:37:01 [INFO]	[TRAIN] Epoch=97/270, Step=32/74, loss=25.714931, lr=0.000125, time_each_step=0.41s, eta=2:3:38
    2021-08-11 15:37:02 [INFO]	[TRAIN] Epoch=97/270, Step=34/74, loss=10.831161, lr=0.000125, time_each_step=0.41s, eta=2:3:37
    2021-08-11 15:37:03 [INFO]	[TRAIN] Epoch=97/270, Step=36/74, loss=31.969742, lr=0.000125, time_each_step=0.41s, eta=2:3:36
    2021-08-11 15:37:03 [INFO]	[TRAIN] Epoch=97/270, Step=38/74, loss=24.690823, lr=0.000125, time_each_step=0.4s, eta=2:3:35
    2021-08-11 15:37:04 [INFO]	[TRAIN] Epoch=97/270, Step=40/74, loss=54.473434, lr=0.000125, time_each_step=0.38s, eta=2:3:34
    2021-08-11 15:37:04 [INFO]	[TRAIN] Epoch=97/270, Step=42/74, loss=26.263937, lr=0.000125, time_each_step=0.37s, eta=2:3:32
    2021-08-11 15:37:05 [INFO]	[TRAIN] Epoch=97/270, Step=44/74, loss=35.40704, lr=0.000125, time_each_step=0.35s, eta=2:3:31
    2021-08-11 15:37:06 [INFO]	[TRAIN] Epoch=97/270, Step=46/74, loss=18.870789, lr=0.000125, time_each_step=0.32s, eta=2:3:30
    2021-08-11 15:37:06 [INFO]	[TRAIN] Epoch=97/270, Step=48/74, loss=14.359077, lr=0.000125, time_each_step=0.33s, eta=2:3:29
    2021-08-11 15:37:07 [INFO]	[TRAIN] Epoch=97/270, Step=50/74, loss=18.170515, lr=0.000125, time_each_step=0.31s, eta=2:3:28
    2021-08-11 15:37:07 [INFO]	[TRAIN] Epoch=97/270, Step=52/74, loss=9.592016, lr=0.000125, time_each_step=0.29s, eta=2:3:27
    2021-08-11 15:37:07 [INFO]	[TRAIN] Epoch=97/270, Step=54/74, loss=22.244783, lr=0.000125, time_each_step=0.27s, eta=2:3:26
    2021-08-11 15:37:08 [INFO]	[TRAIN] Epoch=97/270, Step=56/74, loss=15.108416, lr=0.000125, time_each_step=0.26s, eta=2:3:25
    2021-08-11 15:37:09 [INFO]	[TRAIN] Epoch=97/270, Step=58/74, loss=17.784401, lr=0.000125, time_each_step=0.26s, eta=2:3:25
    2021-08-11 15:37:09 [INFO]	[TRAIN] Epoch=97/270, Step=60/74, loss=16.578869, lr=0.000125, time_each_step=0.25s, eta=2:3:24
    2021-08-11 15:37:09 [INFO]	[TRAIN] Epoch=97/270, Step=62/74, loss=20.296446, lr=0.000125, time_each_step=0.23s, eta=2:3:23
    2021-08-11 15:37:09 [INFO]	[TRAIN] Epoch=97/270, Step=64/74, loss=37.685699, lr=0.000125, time_each_step=0.22s, eta=2:3:23
    2021-08-11 15:37:10 [INFO]	[TRAIN] Epoch=97/270, Step=66/74, loss=21.222691, lr=0.000125, time_each_step=0.21s, eta=2:3:22
    2021-08-11 15:37:10 [INFO]	[TRAIN] Epoch=97/270, Step=68/74, loss=12.961777, lr=0.000125, time_each_step=0.2s, eta=2:3:22
    2021-08-11 15:37:11 [INFO]	[TRAIN] Epoch=97/270, Step=70/74, loss=21.621086, lr=0.000125, time_each_step=0.21s, eta=2:3:21
    2021-08-11 15:37:11 [INFO]	[TRAIN] Epoch=97/270, Step=72/74, loss=71.35424, lr=0.000125, time_each_step=0.21s, eta=2:3:21
    2021-08-11 15:37:11 [INFO]	[TRAIN] Epoch=97/270, Step=74/74, loss=36.159855, lr=0.000125, time_each_step=0.21s, eta=2:3:21
    2021-08-11 15:37:11 [INFO]	[TRAIN] Epoch 97 finished, loss=22.496767, lr=0.000125 .
    2021-08-11 15:37:17 [INFO]	[TRAIN] Epoch=98/270, Step=2/74, loss=32.830425, lr=0.000125, time_each_step=0.42s, eta=2:3:8
    2021-08-11 15:37:17 [INFO]	[TRAIN] Epoch=98/270, Step=4/74, loss=8.178135, lr=0.000125, time_each_step=0.42s, eta=2:3:7
    2021-08-11 15:37:18 [INFO]	[TRAIN] Epoch=98/270, Step=6/74, loss=19.636116, lr=0.000125, time_each_step=0.44s, eta=2:3:8
    2021-08-11 15:37:19 [INFO]	[TRAIN] Epoch=98/270, Step=8/74, loss=28.319527, lr=0.000125, time_each_step=0.47s, eta=2:3:9
    2021-08-11 15:37:19 [INFO]	[TRAIN] Epoch=98/270, Step=10/74, loss=21.076445, lr=0.000125, time_each_step=0.5s, eta=2:3:10
    2021-08-11 15:37:20 [INFO]	[TRAIN] Epoch=98/270, Step=12/74, loss=40.914883, lr=0.000125, time_each_step=0.53s, eta=2:3:11
    2021-08-11 15:37:22 [INFO]	[TRAIN] Epoch=98/270, Step=14/74, loss=22.192242, lr=0.000125, time_each_step=0.56s, eta=2:3:11
    2021-08-11 15:37:22 [INFO]	[TRAIN] Epoch=98/270, Step=16/74, loss=15.991963, lr=0.000125, time_each_step=0.58s, eta=2:3:11
    2021-08-11 15:37:23 [INFO]	[TRAIN] Epoch=98/270, Step=18/74, loss=39.103619, lr=0.000125, time_each_step=0.58s, eta=2:3:10
    2021-08-11 15:37:24 [INFO]	[TRAIN] Epoch=98/270, Step=20/74, loss=15.181162, lr=0.000125, time_each_step=0.61s, eta=2:3:11
    2021-08-11 15:37:24 [INFO]	[TRAIN] Epoch=98/270, Step=22/74, loss=21.59182, lr=0.000125, time_each_step=0.39s, eta=2:2:58
    2021-08-11 15:37:25 [INFO]	[TRAIN] Epoch=98/270, Step=24/74, loss=25.180412, lr=0.000125, time_each_step=0.42s, eta=2:2:58
    2021-08-11 15:37:27 [INFO]	[TRAIN] Epoch=98/270, Step=26/74, loss=19.494295, lr=0.000125, time_each_step=0.45s, eta=2:2:59
    2021-08-11 15:37:28 [INFO]	[TRAIN] Epoch=98/270, Step=28/74, loss=26.150208, lr=0.000125, time_each_step=0.45s, eta=2:2:58
    2021-08-11 15:37:28 [INFO]	[TRAIN] Epoch=98/270, Step=30/74, loss=21.207607, lr=0.000125, time_each_step=0.43s, eta=2:2:57
    2021-08-11 15:37:29 [INFO]	[TRAIN] Epoch=98/270, Step=32/74, loss=12.22757, lr=0.000125, time_each_step=0.42s, eta=2:2:55
    2021-08-11 15:37:30 [INFO]	[TRAIN] Epoch=98/270, Step=34/74, loss=28.622303, lr=0.000125, time_each_step=0.41s, eta=2:2:54
    2021-08-11 15:37:30 [INFO]	[TRAIN] Epoch=98/270, Step=36/74, loss=22.564257, lr=0.000125, time_each_step=0.4s, eta=2:2:53
    2021-08-11 15:37:31 [INFO]	[TRAIN] Epoch=98/270, Step=38/74, loss=35.013103, lr=0.000125, time_each_step=0.42s, eta=2:2:53
    2021-08-11 15:37:32 [INFO]	[TRAIN] Epoch=98/270, Step=40/74, loss=17.09874, lr=0.000125, time_each_step=0.41s, eta=2:2:51
    2021-08-11 15:37:32 [INFO]	[TRAIN] Epoch=98/270, Step=42/74, loss=15.097997, lr=0.000125, time_each_step=0.41s, eta=2:2:51
    2021-08-11 16:42:15 [INFO]	[TRAIN] Epoch=209/270, Step=72/74, loss=20.986595, lr=0.000125, time_each_step=0.22s, eta=0:31:43
    2021-08-11 16:42:16 [INFO]	[TRAIN] Epoch=209/270, Step=74/74, loss=22.81385, lr=0.000125, time_each_step=0.21s, eta=0:31:43
    2021-08-11 16:42:16 [INFO]	[TRAIN] Epoch 209 finished, loss=18.017061, lr=0.000125 .
    2021-08-11 16:42:20 [INFO]	[TRAIN] Epoch=210/270, Step=2/74, loss=14.865284, lr=0.000125, time_each_step=0.41s, eta=0:32:54
    2021-08-11 16:42:21 [INFO]	[TRAIN] Epoch=210/270, Step=4/74, loss=13.939883, lr=0.000125, time_each_step=0.42s, eta=0:32:54
    2021-08-11 16:42:22 [INFO]	[TRAIN] Epoch=210/270, Step=6/74, loss=25.542038, lr=0.000125, time_each_step=0.43s, eta=0:32:54
    2021-08-11 16:42:22 [INFO]	[TRAIN] Epoch=210/270, Step=8/74, loss=19.481401, lr=0.000125, time_each_step=0.44s, eta=0:32:54
    2021-08-11 16:42:23 [INFO]	[TRAIN] Epoch=210/270, Step=10/74, loss=18.471357, lr=0.000125, time_each_step=0.45s, eta=0:32:53
    2021-08-11 16:42:23 [INFO]	[TRAIN] Epoch=210/270, Step=12/74, loss=16.714258, lr=0.000125, time_each_step=0.45s, eta=0:32:53
    2021-08-11 16:42:24 [INFO]	[TRAIN] Epoch=210/270, Step=14/74, loss=18.347313, lr=0.000125, time_each_step=0.49s, eta=0:32:54
    2021-08-11 16:42:25 [INFO]	[TRAIN] Epoch=210/270, Step=16/74, loss=19.235186, lr=0.000125, time_each_step=0.51s, eta=0:32:54
    2021-08-11 16:42:26 [INFO]	[TRAIN] Epoch=210/270, Step=18/74, loss=22.620939, lr=0.000125, time_each_step=0.55s, eta=0:32:55
    2021-08-11 16:42:27 [INFO]	[TRAIN] Epoch=210/270, Step=20/74, loss=6.355416, lr=0.000125, time_each_step=0.57s, eta=0:32:56
    2021-08-11 16:42:28 [INFO]	[TRAIN] Epoch=210/270, Step=22/74, loss=19.177589, lr=0.000125, time_each_step=0.39s, eta=0:32:45
    2021-08-11 16:42:29 [INFO]	[TRAIN] Epoch=210/270, Step=24/74, loss=23.59774, lr=0.000125, time_each_step=0.4s, eta=0:32:45
    2021-08-11 16:42:29 [INFO]	[TRAIN] Epoch=210/270, Step=26/74, loss=11.652861, lr=0.000125, time_each_step=0.39s, eta=0:32:43
    2021-08-11 16:42:30 [INFO]	[TRAIN] Epoch=210/270, Step=28/74, loss=16.037863, lr=0.000125, time_each_step=0.4s, eta=0:32:43
    2021-08-11 16:42:31 [INFO]	[TRAIN] Epoch=210/270, Step=30/74, loss=21.436771, lr=0.000125, time_each_step=0.41s, eta=0:32:43
    2021-08-11 16:42:32 [INFO]	[TRAIN] Epoch=210/270, Step=32/74, loss=15.875498, lr=0.000125, time_each_step=0.43s, eta=0:32:43
    2021-08-11 16:42:33 [INFO]	[TRAIN] Epoch=210/270, Step=34/74, loss=16.578362, lr=0.000125, time_each_step=0.41s, eta=0:32:41
    2021-08-11 16:42:34 [INFO]	[TRAIN] Epoch=210/270, Step=36/74, loss=30.159298, lr=0.000125, time_each_step=0.42s, eta=0:32:41
    2021-08-11 16:42:34 [INFO]	[TRAIN] Epoch=210/270, Step=38/74, loss=12.023346, lr=0.000125, time_each_step=0.4s, eta=0:32:39
    2021-08-11 16:42:35 [INFO]	[TRAIN] Epoch=210/270, Step=40/74, loss=23.704615, lr=0.000125, time_each_step=0.38s, eta=0:32:38
    2021-08-11 16:42:35 [INFO]	[TRAIN] Epoch=210/270, Step=42/74, loss=25.385647, lr=0.000125, time_each_step=0.37s, eta=0:32:37
    2021-08-11 16:42:36 [INFO]	[TRAIN] Epoch=210/270, Step=44/74, loss=13.829526, lr=0.000125, time_each_step=0.37s, eta=0:32:36
    2021-08-11 16:42:37 [INFO]	[TRAIN] Epoch=210/270, Step=46/74, loss=21.73513, lr=0.000125, time_each_step=0.37s, eta=0:32:35
    2021-08-11 16:42:38 [INFO]	[TRAIN] Epoch=210/270, Step=48/74, loss=25.860155, lr=0.000125, time_each_step=0.38s, eta=0:32:35
    2021-08-11 16:42:39 [INFO]	[TRAIN] Epoch=210/270, Step=50/74, loss=16.628941, lr=0.000125, time_each_step=0.38s, eta=0:32:34
    2021-08-11 16:42:39 [INFO]	[TRAIN] Epoch=210/270, Step=52/74, loss=8.484032, lr=0.000125, time_each_step=0.37s, eta=0:32:33
    2021-08-11 16:42:40 [INFO]	[TRAIN] Epoch=210/270, Step=54/74, loss=18.519382, lr=0.000125, time_each_step=0.35s, eta=0:32:32
    2021-08-11 16:42:40 [INFO]	[TRAIN] Epoch=210/270, Step=56/74, loss=19.853119, lr=0.000125, time_each_step=0.33s, eta=0:32:31
    2021-08-11 16:42:41 [INFO]	[TRAIN] Epoch=210/270, Step=58/74, loss=8.87142, lr=0.000125, time_each_step=0.31s, eta=0:32:30
    2021-08-11 16:42:41 [INFO]	[TRAIN] Epoch=210/270, Step=60/74, loss=18.360439, lr=0.000125, time_each_step=0.31s, eta=0:32:29
    2021-08-11 16:42:41 [INFO]	[TRAIN] Epoch=210/270, Step=62/74, loss=20.619797, lr=0.000125, time_each_step=0.29s, eta=0:32:28
    2021-08-11 16:42:41 [INFO]	[TRAIN] Epoch=210/270, Step=64/74, loss=24.597622, lr=0.000125, time_each_step=0.26s, eta=0:32:27
    2021-08-11 16:42:42 [INFO]	[TRAIN] Epoch=210/270, Step=66/74, loss=18.033554, lr=0.000125, time_each_step=0.25s, eta=0:32:27
    2021-08-11 16:42:42 [INFO]	[TRAIN] Epoch=210/270, Step=68/74, loss=38.986526, lr=0.000125, time_each_step=0.22s, eta=0:32:26
    2021-08-11 16:42:43 [INFO]	[TRAIN] Epoch=210/270, Step=70/74, loss=19.866541, lr=0.000125, time_each_step=0.21s, eta=0:32:26
    2021-08-11 16:42:43 [INFO]	[TRAIN] Epoch=210/270, Step=72/74, loss=18.253372, lr=0.000125, time_each_step=0.19s, eta=0:32:25
    2021-08-11 16:42:43 [INFO]	[TRAIN] Epoch=210/270, Step=74/74, loss=14.260531, lr=0.000125, time_each_step=0.19s, eta=0:32:25
    2021-08-11 16:42:43 [INFO]	[TRAIN] Epoch 210 finished, loss=19.464113, lr=0.000125 .
    2021-08-11 16:42:51 [INFO]	[TRAIN] Epoch=211/270, Step=2/74, loss=22.705933, lr=1.2e-05, time_each_step=0.57s, eta=0:29:24
    2021-08-11 16:42:52 [INFO]	[TRAIN] Epoch=211/270, Step=4/74, loss=20.203716, lr=1.2e-05, time_each_step=0.58s, eta=0:29:24
    2021-08-11 16:42:53 [INFO]	[TRAIN] Epoch=211/270, Step=6/74, loss=12.08175, lr=1.2e-05, time_each_step=0.6s, eta=0:29:24
    2021-08-11 16:42:54 [INFO]	[TRAIN] Epoch=211/270, Step=8/74, loss=11.995147, lr=1.2e-05, time_each_step=0.62s, eta=0:29:24
    2021-08-11 16:42:54 [INFO]	[TRAIN] Epoch=211/270, Step=10/74, loss=19.601452, lr=1.2e-05, time_each_step=0.64s, eta=0:29:24
    2021-08-11 16:42:55 [INFO]	[TRAIN] Epoch=211/270, Step=12/74, loss=18.454376, lr=1.2e-05, time_each_step=0.66s, eta=0:29:24
    2021-08-11 16:42:56 [INFO]	[TRAIN] Epoch=211/270, Step=14/74, loss=25.086918, lr=1.2e-05, time_each_step=0.67s, eta=0:29:23
    2021-08-11 16:42:56 [INFO]	[TRAIN] Epoch=211/270, Step=16/74, loss=16.043926, lr=1.2e-05, time_each_step=0.68s, eta=0:29:23
    2021-08-11 16:42:57 [INFO]	[TRAIN] Epoch=211/270, Step=18/74, loss=25.454052, lr=1.2e-05, time_each_step=0.71s, eta=0:29:23
    2021-08-11 16:42:58 [INFO]	[TRAIN] Epoch=211/270, Step=20/74, loss=17.179657, lr=1.2e-05, time_each_step=0.73s, eta=0:29:22
    2021-08-11 16:42:59 [INFO]	[TRAIN] Epoch=211/270, Step=22/74, loss=9.657164, lr=1.2e-05, time_each_step=0.36s, eta=0:29:2
    2021-08-11 16:42:59 [INFO]	[TRAIN] Epoch=211/270, Step=24/74, loss=10.513961, lr=1.2e-05, time_each_step=0.36s, eta=0:29:1
    2021-08-11 16:43:00 [INFO]	[TRAIN] Epoch=211/270, Step=26/74, loss=28.111954, lr=1.2e-05, time_each_step=0.35s, eta=0:29:0
    2021-08-11 16:43:01 [INFO]	[TRAIN] Epoch=211/270, Step=28/74, loss=17.632145, lr=1.2e-05, time_each_step=0.35s, eta=0:28:59
    2021-08-11 16:43:01 [INFO]	[TRAIN] Epoch=211/270, Step=30/74, loss=16.750963, lr=1.2e-05, time_each_step=0.35s, eta=0:28:59
    2021-08-11 16:43:02 [INFO]	[TRAIN] Epoch=211/270, Step=32/74, loss=24.917927, lr=1.2e-05, time_each_step=0.33s, eta=0:28:57
    2021-08-11 16:43:03 [INFO]	[TRAIN] Epoch=211/270, Step=34/74, loss=19.426086, lr=1.2e-05, time_each_step=0.34s, eta=0:28:57
    2021-08-11 16:43:04 [INFO]	[TRAIN] Epoch=211/270, Step=36/74, loss=12.692455, lr=1.2e-05, time_each_step=0.37s, eta=0:28:57
    2021-08-11 16:43:04 [INFO]	[TRAIN] Epoch=211/270, Step=38/74, loss=33.817387, lr=1.2e-05, time_each_step=0.36s, eta=0:28:56
    2021-08-11 16:43:05 [INFO]	[TRAIN] Epoch=211/270, Step=40/74, loss=20.785973, lr=1.2e-05, time_each_step=0.36s, eta=0:28:55
    2021-08-11 16:43:06 [INFO]	[TRAIN] Epoch=211/270, Step=42/74, loss=23.805285, lr=1.2e-05, time_each_step=0.35s, eta=0:28:54
    2021-08-11 16:43:07 [INFO]	[TRAIN] Epoch=211/270, Step=44/74, loss=13.813204, lr=1.2e-05, time_each_step=0.36s, eta=0:28:54
    2021-08-11 16:43:07 [INFO]	[TRAIN] Epoch=211/270, Step=46/74, loss=33.733818, lr=1.2e-05, time_each_step=0.36s, eta=0:28:53
    2021-08-11 16:43:08 [INFO]	[TRAIN] Epoch=211/270, Step=48/74, loss=24.759342, lr=1.2e-05, time_each_step=0.35s, eta=0:28:52
    2021-08-11 16:43:08 [INFO]	[TRAIN] Epoch=211/270, Step=50/74, loss=13.610241, lr=1.2e-05, time_each_step=0.35s, eta=0:28:51
    2021-08-11 16:43:09 [INFO]	[TRAIN] Epoch=211/270, Step=52/74, loss=14.00459, lr=1.2e-05, time_each_step=0.35s, eta=0:28:51
    2021-08-11 16:43:09 [INFO]	[TRAIN] Epoch=211/270, Step=54/74, loss=20.902691, lr=1.2e-05, time_each_step=0.34s, eta=0:28:50
    2021-08-11 16:43:10 [INFO]	[TRAIN] Epoch=211/270, Step=56/74, loss=12.015361, lr=1.2e-05, time_each_step=0.3s, eta=0:28:48
    2021-08-11 16:43:10 [INFO]	[TRAIN] Epoch=211/270, Step=58/74, loss=17.90559, lr=1.2e-05, time_each_step=0.28s, eta=0:28:48
    2021-08-11 16:43:10 [INFO]	[TRAIN] Epoch=211/270, Step=60/74, loss=8.640396, lr=1.2e-05, time_each_step=0.27s, eta=0:28:47
    2021-08-11 16:43:11 [INFO]	[TRAIN] Epoch=211/270, Step=62/74, loss=10.620688, lr=1.2e-05, time_each_step=0.26s, eta=0:28:46
    2021-08-11 16:43:11 [INFO]	[TRAIN] Epoch=211/270, Step=64/74, loss=25.948067, lr=1.2e-05, time_each_step=0.25s, eta=0:28:46
    2021-08-11 16:43:12 [INFO]	[TRAIN] Epoch=211/270, Step=66/74, loss=13.576771, lr=1.2e-05, time_each_step=0.24s, eta=0:28:45
    2021-08-11 16:43:12 [INFO]	[TRAIN] Epoch=211/270, Step=68/74, loss=20.587952, lr=1.2e-05, time_each_step=0.22s, eta=0:28:44
    2021-08-11 16:43:13 [INFO]	[TRAIN] Epoch=211/270, Step=70/74, loss=15.768864, lr=1.2e-05, time_each_step=0.22s, eta=0:28:44
    2021-08-11 16:43:13 [INFO]	[TRAIN] Epoch=211/270, Step=72/74, loss=14.53038, lr=1.2e-05, time_each_step=0.22s, eta=0:28:44
    2021-08-11 16:43:14 [INFO]	[TRAIN] Epoch=211/270, Step=74/74, loss=47.635143, lr=1.2e-05, time_each_step=0.21s, eta=0:28:43
    2021-08-11 16:43:14 [INFO]	[TRAIN] Epoch 211 finished, loss=18.624046, lr=1.2e-05 .
    2021-08-11 16:43:23 [INFO]	[TRAIN] Epoch=212/270, Step=2/74, loss=30.177753, lr=1.2e-05, time_each_step=0.67s, eta=0:31:23
    2021-08-11 16:43:24 [INFO]	[TRAIN] Epoch=212/270, Step=4/74, loss=14.43529, lr=1.2e-05, time_each_step=0.7s, eta=0:31:24
    2021-08-11 16:43:25 [INFO]	[TRAIN] Epoch=212/270, Step=6/74, loss=14.992807, lr=1.2e-05, time_each_step=0.71s, eta=0:31:23
    2021-08-11 16:43:26 [INFO]	[TRAIN] Epoch=212/270, Step=8/74, loss=27.017532, lr=1.2e-05, time_each_step=0.74s, eta=0:31:24
    2021-08-11 16:43:26 [INFO]	[TRAIN] Epoch=212/270, Step=10/74, loss=18.807331, lr=1.2e-05, time_each_step=0.74s, eta=0:31:22
    2021-08-11 16:43:27 [INFO]	[TRAIN] Epoch=212/270, Step=12/74, loss=9.274878, lr=1.2e-05, time_each_step=0.76s, eta=0:31:22
    2021-08-11 16:43:28 [INFO]	[TRAIN] Epoch=212/270, Step=14/74, loss=17.72871, lr=1.2e-05, time_each_step=0.79s, eta=0:31:22
    2021-08-11 16:43:29 [INFO]	[TRAIN] Epoch=212/270, Step=16/74, loss=37.371586, lr=1.2e-05, time_each_step=0.82s, eta=0:31:23
    2021-08-11 16:43:30 [INFO]	[TRAIN] Epoch=212/270, Step=18/74, loss=14.285647, lr=1.2e-05, time_each_step=0.84s, eta=0:31:22
    2021-08-11 16:43:31 [INFO]	[TRAIN] Epoch=212/270, Step=20/74, loss=19.084644, lr=1.2e-05, time_each_step=0.89s, eta=0:31:23
    2021-08-11 16:43:32 [INFO]	[TRAIN] Epoch=212/270, Step=22/74, loss=21.090897, lr=1.2e-05, time_each_step=0.46s, eta=0:30:59
    2021-08-11 16:43:33 [INFO]	[TRAIN] Epoch=212/270, Step=24/74, loss=16.806162, lr=1.2e-05, time_each_step=0.45s, eta=0:30:57
    2021-08-11 16:43:34 [INFO]	[TRAIN] Epoch=212/270, Step=26/74, loss=8.612982, lr=1.2e-05, time_each_step=0.46s, eta=0:30:57
    2021-08-11 16:43:35 [INFO]	[TRAIN] Epoch=212/270, Step=28/74, loss=21.650373, lr=1.2e-05, time_each_step=0.46s, eta=0:30:56
    2021-08-11 16:43:36 [INFO]	[TRAIN] Epoch=212/270, Step=30/74, loss=14.481437, lr=1.2e-05, time_each_step=0.47s, eta=0:30:56
    2021-08-11 16:43:37 [INFO]	[TRAIN] Epoch=212/270, Step=32/74, loss=19.663919, lr=1.2e-05, time_each_step=0.47s, eta=0:30:55
    2021-08-11 16:43:37 [INFO]	[TRAIN] Epoch=212/270, Step=34/74, loss=10.571087, lr=1.2e-05, time_each_step=0.47s, eta=0:30:54
    2021-08-11 16:43:38 [INFO]	[TRAIN] Epoch=212/270, Step=36/74, loss=9.709825, lr=1.2e-05, time_each_step=0.45s, eta=0:30:52
    2021-08-11 16:43:38 [INFO]	[TRAIN] Epoch=212/270, Step=38/74, loss=10.661797, lr=1.2e-05, time_each_step=0.44s, eta=0:30:51
    2021-08-11 16:43:39 [INFO]	[TRAIN] Epoch=212/270, Step=40/74, loss=41.075359, lr=1.2e-05, time_each_step=0.39s, eta=0:30:48
    2021-08-11 16:43:40 [INFO]	[TRAIN] Epoch=212/270, Step=42/74, loss=30.406496, lr=1.2e-05, time_each_step=0.39s, eta=0:30:48
    2021-08-11 16:43:41 [INFO]	[TRAIN] Epoch=212/270, Step=44/74, loss=20.010021, lr=1.2e-05, time_each_step=0.38s, eta=0:30:46
    2021-08-11 16:43:41 [INFO]	[TRAIN] Epoch=212/270, Step=46/74, loss=14.100859, lr=1.2e-05, time_each_step=0.35s, eta=0:30:45
    2021-08-11 16:43:41 [INFO]	[TRAIN] Epoch=212/270, Step=48/74, loss=16.00004, lr=1.2e-05, time_each_step=0.32s, eta=0:30:43
    2021-08-11 16:43:42 [INFO]	[TRAIN] Epoch=212/270, Step=50/74, loss=16.823917, lr=1.2e-05, time_each_step=0.31s, eta=0:30:42
    2021-08-11 16:43:42 [INFO]	[TRAIN] Epoch=212/270, Step=52/74, loss=13.814589, lr=1.2e-05, time_each_step=0.29s, eta=0:30:41
    2021-08-11 16:43:43 [INFO]	[TRAIN] Epoch=212/270, Step=54/74, loss=10.65278, lr=1.2e-05, time_each_step=0.27s, eta=0:30:40
    2021-08-11 16:43:43 [INFO]	[TRAIN] Epoch=212/270, Step=56/74, loss=12.154355, lr=1.2e-05, time_each_step=0.26s, eta=0:30:40
    2021-08-11 16:43:44 [INFO]	[TRAIN] Epoch=212/270, Step=58/74, loss=14.077754, lr=1.2e-05, time_each_step=0.26s, eta=0:30:39
    2021-08-11 16:43:44 [INFO]	[TRAIN] Epoch=212/270, Step=60/74, loss=15.288761, lr=1.2e-05, time_each_step=0.25s, eta=0:30:39
    2021-08-11 16:43:45 [INFO]	[TRAIN] Epoch=212/270, Step=62/74, loss=19.992846, lr=1.2e-05, time_each_step=0.22s, eta=0:30:38
    2021-08-11 16:43:45 [INFO]	[TRAIN] Epoch=212/270, Step=64/74, loss=15.155432, lr=1.2e-05, time_each_step=0.22s, eta=0:30:37
    2021-08-11 16:43:45 [INFO]	[TRAIN] Epoch=212/270, Step=66/74, loss=27.765484, lr=1.2e-05, time_each_step=0.21s, eta=0:30:37
    2021-08-11 16:43:46 [INFO]	[TRAIN] Epoch=212/270, Step=68/74, loss=19.648918, lr=1.2e-05, time_each_step=0.22s, eta=0:30:36
    2021-08-11 16:43:46 [INFO]	[TRAIN] Epoch=212/270, Step=70/74, loss=14.485346, lr=1.2e-05, time_each_step=0.22s, eta=0:30:36
    2021-08-11 16:43:47 [INFO]	[TRAIN] Epoch=212/270, Step=72/74, loss=21.220039, lr=1.2e-05, time_each_step=0.22s, eta=0:30:35
    2021-08-11 16:43:47 [INFO]	[TRAIN] Epoch=212/270, Step=74/74, loss=17.637188, lr=1.2e-05, time_each_step=0.22s, eta=0:30:35
    2021-08-11 16:43:47 [INFO]	[TRAIN] Epoch 212 finished, loss=18.443367, lr=1.2e-05 .
    2021-08-11 16:44:01 [INFO]	[TRAIN] Epoch=213/270, Step=2/74, loss=11.241315, lr=1.2e-05, time_each_step=0.9s, eta=0:34:24
    2021-08-11 16:44:02 [INFO]	[TRAIN] Epoch=213/270, Step=4/74, loss=27.134453, lr=1.2e-05, time_each_step=0.93s, eta=0:34:24
    2021-08-11 16:44:03 [INFO]	[TRAIN] Epoch=213/270, Step=6/74, loss=19.659605, lr=1.2e-05, time_each_step=0.94s, eta=0:34:23
    2021-08-11 16:44:04 [INFO]	[TRAIN] Epoch=213/270, Step=8/74, loss=15.729311, lr=1.2e-05, time_each_step=0.97s, eta=0:34:23
    2021-08-11 16:44:05 [INFO]	[TRAIN] Epoch=213/270, Step=10/74, loss=15.655651, lr=1.2e-05, time_each_step=0.98s, eta=0:34:22
    2021-08-11 16:44:06 [INFO]	[TRAIN] Epoch=213/270, Step=12/74, loss=13.89613, lr=1.2e-05, time_each_step=1.03s, eta=0:34:23
    2021-08-11 16:44:07 [INFO]	[TRAIN] Epoch=213/270, Step=14/74, loss=11.767821, lr=1.2e-05, time_each_step=1.04s, eta=0:34:22
    2021-08-11 16:44:07 [INFO]	[TRAIN] Epoch=213/270, Step=16/74, loss=11.417698, lr=1.2e-05, time_each_step=1.04s, eta=0:34:20
    2021-08-11 16:44:08 [INFO]	[TRAIN] Epoch=213/270, Step=18/74, loss=19.28063, lr=1.2e-05, time_each_step=1.05s, eta=0:34:18
    2021-08-11 16:44:09 [INFO]	[TRAIN] Epoch=213/270, Step=20/74, loss=14.95976, lr=1.2e-05, time_each_step=1.08s, eta=0:34:17
    2021-08-11 16:44:10 [INFO]	[TRAIN] Epoch=213/270, Step=22/74, loss=40.364357, lr=1.2e-05, time_each_step=0.41s, eta=0:33:41
    2021-08-11 16:44:10 [INFO]	[TRAIN] Epoch=213/270, Step=24/74, loss=30.562922, lr=1.2e-05, time_each_step=0.41s, eta=0:33:40
    2021-08-11 16:44:11 [INFO]	[TRAIN] Epoch=213/270, Step=26/74, loss=11.553459, lr=1.2e-05, time_each_step=0.41s, eta=0:33:39
    2021-08-11 16:44:12 [INFO]	[TRAIN] Epoch=213/270, Step=28/74, loss=15.654613, lr=1.2e-05, time_each_step=0.41s, eta=0:33:38
    2021-08-11 16:44:13 [INFO]	[TRAIN] Epoch=213/270, Step=30/74, loss=29.687237, lr=1.2e-05, time_each_step=0.41s, eta=0:33:37
    2021-08-11 16:44:13 [INFO]	[TRAIN] Epoch=213/270, Step=32/74, loss=27.139923, lr=1.2e-05, time_each_step=0.37s, eta=0:33:35
    2021-08-11 16:44:14 [INFO]	[TRAIN] Epoch=213/270, Step=34/74, loss=13.771881, lr=1.2e-05, time_each_step=0.37s, eta=0:33:34
    2021-08-11 16:44:15 [INFO]	[TRAIN] Epoch=213/270, Step=36/74, loss=14.963207, lr=1.2e-05, time_each_step=0.38s, eta=0:33:34
    2021-08-11 16:44:15 [INFO]	[TRAIN] Epoch=213/270, Step=38/74, loss=14.06185, lr=1.2e-05, time_each_step=0.37s, eta=0:33:32
    2021-08-11 16:44:16 [INFO]	[TRAIN] Epoch=213/270, Step=40/74, loss=18.108122, lr=1.2e-05, time_each_step=0.37s, eta=0:33:32
    2021-08-11 16:44:17 [INFO]	[TRAIN] Epoch=213/270, Step=42/74, loss=33.639771, lr=1.2e-05, time_each_step=0.35s, eta=0:33:30
    2021-08-11 16:44:17 [INFO]	[TRAIN] Epoch=213/270, Step=44/74, loss=15.257897, lr=1.2e-05, time_each_step=0.35s, eta=0:33:30
    2021-08-11 16:44:18 [INFO]	[TRAIN] Epoch=213/270, Step=46/74, loss=16.582424, lr=1.2e-05, time_each_step=0.33s, eta=0:33:29
    2021-08-11 16:44:18 [INFO]	[TRAIN] Epoch=213/270, Step=48/74, loss=13.498061, lr=1.2e-05, time_each_step=0.31s, eta=0:33:27
    2021-08-11 16:44:19 [INFO]	[TRAIN] Epoch=213/270, Step=50/74, loss=28.201435, lr=1.2e-05, time_each_step=0.3s, eta=0:33:26
    2021-08-11 16:44:19 [INFO]	[TRAIN] Epoch=213/270, Step=52/74, loss=10.475401, lr=1.2e-05, time_each_step=0.29s, eta=0:33:26
    2021-08-11 16:44:19 [INFO]	[TRAIN] Epoch=213/270, Step=54/74, loss=16.019991, lr=1.2e-05, time_each_step=0.27s, eta=0:33:25
    2021-08-11 16:44:20 [INFO]	[TRAIN] Epoch=213/270, Step=56/74, loss=20.759954, lr=1.2e-05, time_each_step=0.26s, eta=0:33:24
    2021-08-11 16:44:20 [INFO]	[TRAIN] Epoch=213/270, Step=58/74, loss=13.999541, lr=1.2e-05, time_each_step=0.25s, eta=0:33:23
    2021-08-11 16:44:21 [INFO]	[TRAIN] Epoch=213/270, Step=60/74, loss=18.897736, lr=1.2e-05, time_each_step=0.23s, eta=0:33:22
    2021-08-11 16:44:21 [INFO]	[TRAIN] Epoch=213/270, Step=62/74, loss=17.349108, lr=1.2e-05, time_each_step=0.22s, eta=0:33:22
    2021-08-11 16:44:21 [INFO]	[TRAIN] Epoch=213/270, Step=64/74, loss=20.574039, lr=1.2e-05, time_each_step=0.21s, eta=0:33:21
    2021-08-11 16:44:22 [INFO]	[TRAIN] Epoch=213/270, Step=66/74, loss=5.758162, lr=1.2e-05, time_each_step=0.2s, eta=0:33:21
    2021-08-11 16:44:22 [INFO]	[TRAIN] Epoch=213/270, Step=68/74, loss=28.250692, lr=1.2e-05, time_each_step=0.2s, eta=0:33:20
    2021-08-11 16:44:23 [INFO]	[TRAIN] Epoch=213/270, Step=70/74, loss=23.788816, lr=1.2e-05, time_each_step=0.19s, eta=0:33:20
    2021-08-11 16:44:23 [INFO]	[TRAIN] Epoch=213/270, Step=72/74, loss=18.989351, lr=1.2e-05, time_each_step=0.19s, eta=0:33:20
    2021-08-11 16:44:24 [INFO]	[TRAIN] Epoch=213/270, Step=74/74, loss=31.838545, lr=1.2e-05, time_each_step=0.21s, eta=0:33:19
    2021-08-11 16:44:24 [INFO]	[TRAIN] Epoch 213 finished, loss=19.084501, lr=1.2e-05 .
    2021-08-11 16:44:44 [INFO]	[TRAIN] Epoch=214/270, Step=2/74, loss=11.496687, lr=1.2e-05, time_each_step=1.18s, eta=0:36:57
    2021-08-11 16:44:45 [INFO]	[TRAIN] Epoch=214/270, Step=4/74, loss=17.374201, lr=1.2e-05, time_each_step=1.23s, eta=0:36:58
    2021-08-11 16:44:46 [INFO]	[TRAIN] Epoch=214/270, Step=6/74, loss=18.900928, lr=1.2e-05, time_each_step=1.25s, eta=0:36:58
    2021-08-11 16:44:47 [INFO]	[TRAIN] Epoch=214/270, Step=8/74, loss=13.894676, lr=1.2e-05, time_each_step=1.3s, eta=0:36:58
    2021-08-11 16:44:47 [INFO]	[TRAIN] Epoch=214/270, Step=10/74, loss=14.266946, lr=1.2e-05, time_each_step=1.3s, eta=0:36:56
    2021-08-11 16:44:48 [INFO]	[TRAIN] Epoch=214/270, Step=12/74, loss=24.685862, lr=1.2e-05, time_each_step=1.32s, eta=0:36:54
    2021-08-11 16:44:49 [INFO]	[TRAIN] Epoch=214/270, Step=14/74, loss=14.96589, lr=1.2e-05, time_each_step=1.35s, eta=0:36:53
    2021-08-11 16:44:50 [INFO]	[TRAIN] Epoch=214/270, Step=16/74, loss=18.416645, lr=1.2e-05, time_each_step=1.37s, eta=0:36:52
    2021-08-11 16:44:51 [INFO]	[TRAIN] Epoch=214/270, Step=18/74, loss=26.793129, lr=1.2e-05, time_each_step=1.42s, eta=0:36:52
    2021-08-11 16:44:52 [INFO]	[TRAIN] Epoch=214/270, Step=20/74, loss=19.987705, lr=1.2e-05, time_each_step=1.43s, eta=0:36:50
    2021-08-11 16:44:53 [INFO]	[TRAIN] Epoch=214/270, Step=22/74, loss=29.834393, lr=1.2e-05, time_each_step=0.47s, eta=0:35:57
    2021-08-11 16:44:54 [INFO]	[TRAIN] Epoch=214/270, Step=24/74, loss=10.539841, lr=1.2e-05, time_each_step=0.44s, eta=0:35:54
    2021-08-11 16:44:55 [INFO]	[TRAIN] Epoch=214/270, Step=26/74, loss=19.09811, lr=1.2e-05, time_each_step=0.44s, eta=0:35:54
    2021-08-11 16:44:56 [INFO]	[TRAIN] Epoch=214/270, Step=28/74, loss=25.776516, lr=1.2e-05, time_each_step=0.43s, eta=0:35:52
    2021-08-11 16:44:57 [INFO]	[TRAIN] Epoch=214/270, Step=30/74, loss=14.205194, lr=1.2e-05, time_each_step=0.45s, eta=0:35:52
    2021-08-11 16:44:57 [INFO]	[TRAIN] Epoch=214/270, Step=32/74, loss=12.781091, lr=1.2e-05, time_each_step=0.45s, eta=0:35:51
    2021-08-11 16:44:58 [INFO]	[TRAIN] Epoch=214/270, Step=34/74, loss=18.761719, lr=1.2e-05, time_each_step=0.44s, eta=0:35:50
    2021-08-11 16:44:59 [INFO]	[TRAIN] Epoch=214/270, Step=36/74, loss=21.286615, lr=1.2e-05, time_each_step=0.44s, eta=0:35:49
    2021-08-11 16:44:59 [INFO]	[TRAIN] Epoch=214/270, Step=38/74, loss=32.491299, lr=1.2e-05, time_each_step=0.41s, eta=0:35:47
    2021-08-11 16:45:00 [INFO]	[TRAIN] Epoch=214/270, Step=40/74, loss=12.726534, lr=1.2e-05, time_each_step=0.4s, eta=0:35:46
    2021-08-11 16:45:01 [INFO]	[TRAIN] Epoch=214/270, Step=42/74, loss=18.417549, lr=1.2e-05, time_each_step=0.38s, eta=0:35:44
    2021-08-11 16:45:01 [INFO]	[TRAIN] Epoch=214/270, Step=44/74, loss=23.952972, lr=1.2e-05, time_each_step=0.37s, eta=0:35:43
    2021-08-11 16:45:01 [INFO]	[TRAIN] Epoch=214/270, Step=46/74, loss=8.307012, lr=1.2e-05, time_each_step=0.33s, eta=0:35:41
    2021-08-11 16:45:02 [INFO]	[TRAIN] Epoch=214/270, Step=48/74, loss=16.819622, lr=1.2e-05, time_each_step=0.3s, eta=0:35:40
    2021-08-11 16:45:02 [INFO]	[TRAIN] Epoch=214/270, Step=50/74, loss=17.779516, lr=1.2e-05, time_each_step=0.27s, eta=0:35:39
    2021-08-11 16:45:02 [INFO]	[TRAIN] Epoch=214/270, Step=52/74, loss=12.508502, lr=1.2e-05, time_each_step=0.25s, eta=0:35:38
    2021-08-11 16:45:03 [INFO]	[TRAIN] Epoch=214/270, Step=54/74, loss=10.249863, lr=1.2e-05, time_each_step=0.25s, eta=0:35:37
    2021-08-11 16:45:03 [INFO]	[TRAIN] Epoch=214/270, Step=56/74, loss=12.845989, lr=1.2e-05, time_each_step=0.22s, eta=0:35:36
    2021-08-11 16:45:04 [INFO]	[TRAIN] Epoch=214/270, Step=58/74, loss=15.938346, lr=1.2e-05, time_each_step=0.21s, eta=0:35:36
    2021-08-11 16:45:04 [INFO]	[TRAIN] Epoch=214/270, Step=60/74, loss=12.826698, lr=1.2e-05, time_each_step=0.2s, eta=0:35:35
    2021-08-11 16:45:04 [INFO]	[TRAIN] Epoch=214/270, Step=62/74, loss=14.082835, lr=1.2e-05, time_each_step=0.2s, eta=0:35:35
    2021-08-11 16:45:05 [INFO]	[TRAIN] Epoch=214/270, Step=64/74, loss=18.080416, lr=1.2e-05, time_each_step=0.19s, eta=0:35:34
    2021-08-11 16:45:05 [INFO]	[TRAIN] Epoch=214/270, Step=66/74, loss=22.394154, lr=1.2e-05, time_each_step=0.21s, eta=0:35:34
    2021-08-11 16:45:06 [INFO]	[TRAIN] Epoch=214/270, Step=68/74, loss=16.688463, lr=1.2e-05, time_each_step=0.21s, eta=0:35:34
    2021-08-11 16:45:06 [INFO]	[TRAIN] Epoch=214/270, Step=70/74, loss=12.000421, lr=1.2e-05, time_each_step=0.22s, eta=0:35:33
    2021-08-11 16:45:07 [INFO]	[TRAIN] Epoch=214/270, Step=72/74, loss=11.713861, lr=1.2e-05, time_each_step=0.22s, eta=0:35:33
    2021-08-11 16:45:07 [INFO]	[TRAIN] Epoch=214/270, Step=74/74, loss=12.404832, lr=1.2e-05, time_each_step=0.2s, eta=0:35:32
    2021-08-11 16:45:07 [INFO]	[TRAIN] Epoch 214 finished, loss=18.013683, lr=1.2e-05 .
    2021-08-11 16:45:26 [INFO]	[TRAIN] Epoch=215/270, Step=2/74, loss=8.742977, lr=1.2e-05, time_each_step=1.14s, eta=0:42:41
    2021-08-11 16:45:27 [INFO]	[TRAIN] Epoch=215/270, Step=4/74, loss=27.304802, lr=1.2e-05, time_each_step=1.17s, eta=0:42:41
    2021-08-11 16:45:28 [INFO]	[TRAIN] Epoch=215/270, Step=6/74, loss=15.394072, lr=1.2e-05, time_each_step=1.2s, eta=0:42:40
    2021-08-11 16:45:29 [INFO]	[TRAIN] Epoch=215/270, Step=8/74, loss=23.243355, lr=1.2e-05, time_each_step=1.22s, eta=0:42:40
    2021-08-11 16:45:30 [INFO]	[TRAIN] Epoch=215/270, Step=10/74, loss=28.356184, lr=1.2e-05, time_each_step=1.25s, eta=0:42:39
    2021-08-11 16:45:30 [INFO]	[TRAIN] Epoch=215/270, Step=12/74, loss=11.555666, lr=1.2e-05, time_each_step=1.26s, eta=0:42:37
    2021-08-11 16:45:32 [INFO]	[TRAIN] Epoch=215/270, Step=14/74, loss=14.375599, lr=1.2e-05, time_each_step=1.29s, eta=0:42:36
    2021-08-11 16:45:32 [INFO]	[TRAIN] Epoch=215/270, Step=16/74, loss=15.543676, lr=1.2e-05, time_each_step=1.3s, eta=0:42:34
    2021-08-11 16:45:33 [INFO]	[TRAIN] Epoch=215/270, Step=18/74, loss=19.458145, lr=1.2e-05, time_each_step=1.32s, eta=0:42:33
    2021-08-11 16:45:34 [INFO]	[TRAIN] Epoch=215/270, Step=20/74, loss=22.315491, lr=1.2e-05, time_each_step=1.34s, eta=0:42:31
    2021-08-11 16:45:35 [INFO]	[TRAIN] Epoch=215/270, Step=22/74, loss=11.556543, lr=1.2e-05, time_each_step=0.43s, eta=0:41:41
    2021-08-11 16:45:36 [INFO]	[TRAIN] Epoch=215/270, Step=24/74, loss=16.657625, lr=1.2e-05, time_each_step=0.42s, eta=0:41:40
    2021-08-11 16:45:37 [INFO]	[TRAIN] Epoch=215/270, Step=26/74, loss=10.7756, lr=1.2e-05, time_each_step=0.43s, eta=0:41:39
    2021-08-11 16:45:37 [INFO]	[TRAIN] Epoch=215/270, Step=28/74, loss=9.796882, lr=1.2e-05, time_each_step=0.41s, eta=0:41:38
    2021-08-11 16:45:38 [INFO]	[TRAIN] Epoch=215/270, Step=30/74, loss=13.214691, lr=1.2e-05, time_each_step=0.42s, eta=0:41:37
    2021-08-11 16:45:39 [INFO]	[TRAIN] Epoch=215/270, Step=32/74, loss=6.741063, lr=1.2e-05, time_each_step=0.42s, eta=0:41:36
    2021-08-11 16:45:40 [INFO]	[TRAIN] Epoch=215/270, Step=34/74, loss=14.929882, lr=1.2e-05, time_each_step=0.41s, eta=0:41:35
    2021-08-11 16:45:40 [INFO]	[TRAIN] Epoch=215/270, Step=36/74, loss=17.802006, lr=1.2e-05, time_each_step=0.41s, eta=0:41:34
    2021-08-11 16:45:41 [INFO]	[TRAIN] Epoch=215/270, Step=38/74, loss=32.419594, lr=1.2e-05, time_each_step=0.42s, eta=0:41:34
    2021-08-11 16:45:42 [INFO]	[TRAIN] Epoch=215/270, Step=40/74, loss=17.547508, lr=1.2e-05, time_each_step=0.41s, eta=0:41:33
    2021-08-11 16:45:42 [INFO]	[TRAIN] Epoch=215/270, Step=42/74, loss=9.132585, lr=1.2e-05, time_each_step=0.38s, eta=0:41:31
    2021-08-11 16:45:43 [INFO]	[TRAIN] Epoch=215/270, Step=44/74, loss=15.039034, lr=1.2e-05, time_each_step=0.37s, eta=0:41:30
    2021-08-11 16:45:43 [INFO]	[TRAIN] Epoch=215/270, Step=46/74, loss=11.211308, lr=1.2e-05, time_each_step=0.35s, eta=0:41:28
    2021-08-11 16:45:44 [INFO]	[TRAIN] Epoch=215/270, Step=48/74, loss=14.390839, lr=1.2e-05, time_each_step=0.34s, eta=0:41:28
    2021-08-11 16:45:44 [INFO]	[TRAIN] Epoch=215/270, Step=50/74, loss=20.944338, lr=1.2e-05, time_each_step=0.29s, eta=0:41:26
    2021-08-11 16:45:44 [INFO]	[TRAIN] Epoch=215/270, Step=52/74, loss=8.366522, lr=1.2e-05, time_each_step=0.28s, eta=0:41:25
    2021-08-11 16:45:45 [INFO]	[TRAIN] Epoch=215/270, Step=54/74, loss=13.864027, lr=1.2e-05, time_each_step=0.25s, eta=0:41:24
    2021-08-11 16:45:45 [INFO]	[TRAIN] Epoch=215/270, Step=56/74, loss=12.454994, lr=1.2e-05, time_each_step=0.23s, eta=0:41:23
    2021-08-11 16:45:45 [INFO]	[TRAIN] Epoch=215/270, Step=58/74, loss=22.927683, lr=1.2e-05, time_each_step=0.2s, eta=0:41:22
    2021-08-11 16:45:46 [INFO]	[TRAIN] Epoch=215/270, Step=60/74, loss=27.743906, lr=1.2e-05, time_each_step=0.19s, eta=0:41:21
    2021-08-11 16:45:46 [INFO]	[TRAIN] Epoch=215/270, Step=62/74, loss=4.831883, lr=1.2e-05, time_each_step=0.2s, eta=0:41:21
    2021-08-11 16:45:47 [INFO]	[TRAIN] Epoch=215/270, Step=64/74, loss=10.430577, lr=1.2e-05, time_each_step=0.19s, eta=0:41:21
    2021-08-11 16:45:47 [INFO]	[TRAIN] Epoch=215/270, Step=66/74, loss=17.051674, lr=1.2e-05, time_each_step=0.18s, eta=0:41:20
    2021-08-11 16:45:48 [INFO]	[TRAIN] Epoch=215/270, Step=68/74, loss=15.282059, lr=1.2e-05, time_each_step=0.2s, eta=0:41:20
    2021-08-11 16:45:48 [INFO]	[TRAIN] Epoch=215/270, Step=70/74, loss=12.666733, lr=1.2e-05, time_each_step=0.2s, eta=0:41:20
    2021-08-11 16:45:48 [INFO]	[TRAIN] Epoch=215/270, Step=72/74, loss=9.573277, lr=1.2e-05, time_each_step=0.2s, eta=0:41:19
    2021-08-11 16:45:49 [INFO]	[TRAIN] Epoch=215/270, Step=74/74, loss=10.449188, lr=1.2e-05, time_each_step=0.21s, eta=0:41:19
    2021-08-11 16:45:49 [INFO]	[TRAIN] Epoch 215 finished, loss=18.409063, lr=1.2e-05 .
    2021-08-11 16:46:03 [INFO]	[TRAIN] Epoch=216/270, Step=2/74, loss=38.855515, lr=1.2e-05, time_each_step=0.9s, eta=0:40:17
    2021-08-11 16:46:04 [INFO]	[TRAIN] Epoch=216/270, Step=4/74, loss=22.927998, lr=1.2e-05, time_each_step=0.92s, eta=0:40:16
    2021-08-11 16:46:05 [INFO]	[TRAIN] Epoch=216/270, Step=6/74, loss=13.900507, lr=1.2e-05, time_each_step=0.93s, eta=0:40:15
    2021-08-11 16:46:06 [INFO]	[TRAIN] Epoch=216/270, Step=8/74, loss=19.450819, lr=1.2e-05, time_each_step=0.96s, eta=0:40:16
    2021-08-11 16:46:07 [INFO]	[TRAIN] Epoch=216/270, Step=10/74, loss=21.663307, lr=1.2e-05, time_each_step=1.0s, eta=0:40:16
    2021-08-11 16:46:08 [INFO]	[TRAIN] Epoch=216/270, Step=12/74, loss=18.324274, lr=1.2e-05, time_each_step=1.04s, eta=0:40:16
    2021-08-11 16:46:09 [INFO]	[TRAIN] Epoch=216/270, Step=14/74, loss=25.00359, lr=1.2e-05, time_each_step=1.05s, eta=0:40:15
    2021-08-11 16:46:09 [INFO]	[TRAIN] Epoch=216/270, Step=16/74, loss=38.133942, lr=1.2e-05, time_each_step=1.07s, eta=0:40:14
    2021-08-11 16:46:10 [INFO]	[TRAIN] Epoch=216/270, Step=18/74, loss=10.835888, lr=1.2e-05, time_each_step=1.08s, eta=0:40:13
    2021-08-11 16:46:11 [INFO]	[TRAIN] Epoch=216/270, Step=20/74, loss=15.494614, lr=1.2e-05, time_each_step=1.1s, eta=0:40:12
    2021-08-11 16:46:12 [INFO]	[TRAIN] Epoch=216/270, Step=22/74, loss=15.816522, lr=1.2e-05, time_each_step=0.42s, eta=0:39:34
    2021-08-11 16:46:12 [INFO]	[TRAIN] Epoch=216/270, Step=24/74, loss=7.877674, lr=1.2e-05, time_each_step=0.42s, eta=0:39:33
    2021-08-11 16:46:13 [INFO]	[TRAIN] Epoch=216/270, Step=26/74, loss=13.843653, lr=1.2e-05, time_each_step=0.41s, eta=0:39:32
    2021-08-11 16:46:14 [INFO]	[TRAIN] Epoch=216/270, Step=28/74, loss=24.27862, lr=1.2e-05, time_each_step=0.39s, eta=0:39:30
    2021-08-11 16:46:14 [INFO]	[TRAIN] Epoch=216/270, Step=30/74, loss=20.691732, lr=1.2e-05, time_each_step=0.38s, eta=0:39:28
    2021-08-11 16:46:15 [INFO]	[TRAIN] Epoch=216/270, Step=32/74, loss=13.495251, lr=1.2e-05, time_each_step=0.36s, eta=0:39:27
    2021-08-11 16:46:16 [INFO]	[TRAIN] Epoch=216/270, Step=34/74, loss=9.100817, lr=1.2e-05, time_each_step=0.35s, eta=0:39:26
    2021-08-11 16:46:16 [INFO]	[TRAIN] Epoch=216/270, Step=36/74, loss=20.686829, lr=1.2e-05, time_each_step=0.35s, eta=0:39:25
    2021-08-11 16:46:17 [INFO]	[TRAIN] Epoch=216/270, Step=38/74, loss=8.649619, lr=1.2e-05, time_each_step=0.36s, eta=0:39:25
    2021-08-11 16:46:18 [INFO]	[TRAIN] Epoch=216/270, Step=40/74, loss=16.235264, lr=1.2e-05, time_each_step=0.36s, eta=0:39:24
    2021-08-11 16:46:19 [INFO]	[TRAIN] Epoch=216/270, Step=42/74, loss=22.570457, lr=1.2e-05, time_each_step=0.36s, eta=0:39:24
    2021-08-11 16:46:19 [INFO]	[TRAIN] Epoch=216/270, Step=44/74, loss=19.06497, lr=1.2e-05, time_each_step=0.35s, eta=0:39:22
    2021-08-11 16:46:20 [INFO]	[TRAIN] Epoch=216/270, Step=46/74, loss=24.486765, lr=1.2e-05, time_each_step=0.35s, eta=0:39:22
    2021-08-11 16:46:20 [INFO]	[TRAIN] Epoch=216/270, Step=48/74, loss=19.46998, lr=1.2e-05, time_each_step=0.34s, eta=0:39:21
    2021-08-11 16:46:21 [INFO]	[TRAIN] Epoch=216/270, Step=50/74, loss=19.960321, lr=1.2e-05, time_each_step=0.32s, eta=0:39:20
    2021-08-11 16:46:21 [INFO]	[TRAIN] Epoch=216/270, Step=52/74, loss=7.324166, lr=1.2e-05, time_each_step=0.3s, eta=0:39:18
    2021-08-11 16:46:21 [INFO]	[TRAIN] Epoch=216/270, Step=54/74, loss=35.136463, lr=1.2e-05, time_each_step=0.28s, eta=0:39:18
    2021-08-11 16:46:22 [INFO]	[TRAIN] Epoch=216/270, Step=56/74, loss=12.747271, lr=1.2e-05, time_each_step=0.26s, eta=0:39:17
    2021-08-11 16:46:22 [INFO]	[TRAIN] Epoch=216/270, Step=58/74, loss=13.935616, lr=1.2e-05, time_each_step=0.23s, eta=0:39:16
    2021-08-11 16:46:22 [INFO]	[TRAIN] Epoch=216/270, Step=60/74, loss=25.846432, lr=1.2e-05, time_each_step=0.21s, eta=0:39:15
    2021-08-11 16:46:23 [INFO]	[TRAIN] Epoch=216/270, Step=62/74, loss=21.374874, lr=1.2e-05, time_each_step=0.21s, eta=0:39:14
    2021-08-11 16:46:23 [INFO]	[TRAIN] Epoch=216/270, Step=64/74, loss=12.928996, lr=1.2e-05, time_each_step=0.21s, eta=0:39:14
    2021-08-11 16:46:24 [INFO]	[TRAIN] Epoch=216/270, Step=66/74, loss=11.735994, lr=1.2e-05, time_each_step=0.19s, eta=0:39:13
    2021-08-11 16:46:24 [INFO]	[TRAIN] Epoch=216/270, Step=68/74, loss=15.022127, lr=1.2e-05, time_each_step=0.19s, eta=0:39:13
    2021-08-11 16:46:25 [INFO]	[TRAIN] Epoch=216/270, Step=70/74, loss=19.916397, lr=1.2e-05, time_each_step=0.2s, eta=0:39:13
    2021-08-11 16:46:25 [INFO]	[TRAIN] Epoch=216/270, Step=72/74, loss=21.823439, lr=1.2e-05, time_each_step=0.2s, eta=0:39:12
    2021-08-11 16:46:25 [INFO]	[TRAIN] Epoch=216/270, Step=74/74, loss=20.068172, lr=1.2e-05, time_each_step=0.2s, eta=0:39:12
    2021-08-11 16:46:25 [INFO]	[TRAIN] Epoch 216 finished, loss=18.895765, lr=1.2e-05 .
    2021-08-11 16:46:44 [INFO]	[TRAIN] Epoch=217/270, Step=2/74, loss=23.162395, lr=1.2e-05, time_each_step=1.12s, eta=0:35:5
    2021-08-11 16:46:45 [INFO]	[TRAIN] Epoch=217/270, Step=4/74, loss=13.63248, lr=1.2e-05, time_each_step=1.16s, eta=0:35:5
    2021-08-11 16:46:46 [INFO]	[TRAIN] Epoch=217/270, Step=6/74, loss=30.461132, lr=1.2e-05, time_each_step=1.18s, eta=0:35:4
    2021-08-11 16:46:47 [INFO]	[TRAIN] Epoch=217/270, Step=8/74, loss=8.86902, lr=1.2e-05, time_each_step=1.19s, eta=0:35:2
    2021-08-11 16:46:48 [INFO]	[TRAIN] Epoch=217/270, Step=10/74, loss=13.469955, lr=1.2e-05, time_each_step=1.22s, eta=0:35:2
    2021-08-11 16:46:49 [INFO]	[TRAIN] Epoch=217/270, Step=12/74, loss=10.861221, lr=1.2e-05, time_each_step=1.26s, eta=0:35:2
    2021-08-11 16:46:50 [INFO]	[TRAIN] Epoch=217/270, Step=14/74, loss=28.200583, lr=1.2e-05, time_each_step=1.29s, eta=0:35:1
    2021-08-11 16:46:51 [INFO]	[TRAIN] Epoch=217/270, Step=16/74, loss=29.060352, lr=1.2e-05, time_each_step=1.31s, eta=0:35:0
    2021-08-11 16:46:52 [INFO]	[TRAIN] Epoch=217/270, Step=18/74, loss=16.998913, lr=1.2e-05, time_each_step=1.33s, eta=0:34:58
    2021-08-11 16:46:52 [INFO]	[TRAIN] Epoch=217/270, Step=20/74, loss=13.708225, lr=1.2e-05, time_each_step=1.33s, eta=0:34:56
    2021-08-11 16:46:53 [INFO]	[TRAIN] Epoch=217/270, Step=22/74, loss=16.91815, lr=1.2e-05, time_each_step=0.43s, eta=0:34:6
    2021-08-11 16:46:53 [INFO]	[TRAIN] Epoch=217/270, Step=24/74, loss=31.491083, lr=1.2e-05, time_each_step=0.42s, eta=0:34:5
    2021-08-11 16:46:54 [INFO]	[TRAIN] Epoch=217/270, Step=26/74, loss=19.790298, lr=1.2e-05, time_each_step=0.43s, eta=0:34:4
    2021-08-11 16:46:55 [INFO]	[TRAIN] Epoch=217/270, Step=28/74, loss=32.423542, lr=1.2e-05, time_each_step=0.41s, eta=0:34:3
    2021-08-11 16:46:56 [INFO]	[TRAIN] Epoch=217/270, Step=30/74, loss=13.318764, lr=1.2e-05, time_each_step=0.4s, eta=0:34:2
    2021-08-11 16:46:56 [INFO]	[TRAIN] Epoch=217/270, Step=32/74, loss=29.384729, lr=1.2e-05, time_each_step=0.38s, eta=0:34:0
    2021-08-11 16:46:57 [INFO]	[TRAIN] Epoch=217/270, Step=34/74, loss=22.436508, lr=1.2e-05, time_each_step=0.38s, eta=0:33:59
    2021-08-11 16:46:58 [INFO]	[TRAIN] Epoch=217/270, Step=36/74, loss=14.42316, lr=1.2e-05, time_each_step=0.35s, eta=0:33:58
    2021-08-11 16:46:58 [INFO]	[TRAIN] Epoch=217/270, Step=38/74, loss=12.560276, lr=1.2e-05, time_each_step=0.35s, eta=0:33:56
    2021-08-11 16:46:59 [INFO]	[TRAIN] Epoch=217/270, Step=40/74, loss=28.748623, lr=1.2e-05, time_each_step=0.34s, eta=0:33:56
    2021-08-11 16:47:00 [INFO]	[TRAIN] Epoch=217/270, Step=42/74, loss=26.177845, lr=1.2e-05, time_each_step=0.34s, eta=0:33:55
    2021-08-11 16:47:00 [INFO]	[TRAIN] Epoch=217/270, Step=44/74, loss=7.963518, lr=1.2e-05, time_each_step=0.34s, eta=0:33:54
    2021-08-11 16:47:01 [INFO]	[TRAIN] Epoch=217/270, Step=46/74, loss=17.320614, lr=1.2e-05, time_each_step=0.32s, eta=0:33:53
    2021-08-11 16:47:01 [INFO]	[TRAIN] Epoch=217/270, Step=48/74, loss=18.14529, lr=1.2e-05, time_each_step=0.32s, eta=0:33:52
    2021-08-11 16:47:02 [INFO]	[TRAIN] Epoch=217/270, Step=50/74, loss=17.963945, lr=1.2e-05, time_each_step=0.29s, eta=0:33:51
    2021-08-11 16:47:02 [INFO]	[TRAIN] Epoch=217/270, Step=52/74, loss=13.598485, lr=1.2e-05, time_each_step=0.28s, eta=0:33:50
    2021-08-11 16:47:02 [INFO]	[TRAIN] Epoch=217/270, Step=54/74, loss=15.173665, lr=1.2e-05, time_each_step=0.26s, eta=0:33:49
    2021-08-11 16:47:03 [INFO]	[TRAIN] Epoch=217/270, Step=56/74, loss=22.746357, lr=1.2e-05, time_each_step=0.24s, eta=0:33:48
    2021-08-11 16:47:03 [INFO]	[TRAIN] Epoch=217/270, Step=58/74, loss=19.189693, lr=1.2e-05, time_each_step=0.24s, eta=0:33:48
    2021-08-11 16:47:04 [INFO]	[TRAIN] Epoch=217/270, Step=60/74, loss=15.940199, lr=1.2e-05, time_each_step=0.24s, eta=0:33:47
    2021-08-11 16:47:04 [INFO]	[TRAIN] Epoch=217/270, Step=62/74, loss=15.066311, lr=1.2e-05, time_each_step=0.22s, eta=0:33:47
    2021-08-11 16:47:04 [INFO]	[TRAIN] Epoch=217/270, Step=64/74, loss=16.434803, lr=1.2e-05, time_each_step=0.21s, eta=0:33:46
    2021-08-11 16:47:05 [INFO]	[TRAIN] Epoch=217/270, Step=66/74, loss=25.089073, lr=1.2e-05, time_each_step=0.21s, eta=0:33:46
    2021-08-11 16:47:05 [INFO]	[TRAIN] Epoch=217/270, Step=68/74, loss=18.105923, lr=1.2e-05, time_each_step=0.2s, eta=0:33:45
    2021-08-11 16:47:06 [INFO]	[TRAIN] Epoch=217/270, Step=70/74, loss=20.108215, lr=1.2e-05, time_each_step=0.21s, eta=0:33:45
    2021-08-11 16:47:06 [INFO]	[TRAIN] Epoch=217/270, Step=72/74, loss=19.060957, lr=1.2e-05, time_each_step=0.21s, eta=0:33:44
    2021-08-11 16:47:06 [INFO]	[TRAIN] Epoch=217/270, Step=74/74, loss=15.882425, lr=1.2e-05, time_each_step=0.19s, eta=0:33:44
    2021-08-11 16:47:06 [INFO]	[TRAIN] Epoch 217 finished, loss=18.375982, lr=1.2e-05 .
    2021-08-11 16:47:12 [INFO]	[TRAIN] Epoch=218/270, Step=2/74, loss=22.006392, lr=1.2e-05, time_each_step=0.47s, eta=0:37:25
    2021-08-11 16:47:13 [INFO]	[TRAIN] Epoch=218/270, Step=4/74, loss=7.510611, lr=1.2e-05, time_each_step=0.49s, eta=0:37:26
    2021-08-11 16:47:14 [INFO]	[TRAIN] Epoch=218/270, Step=6/74, loss=16.833473, lr=1.2e-05, time_each_step=0.51s, eta=0:37:27
    2021-08-11 16:47:15 [INFO]	[TRAIN] Epoch=218/270, Step=8/74, loss=46.213169, lr=1.2e-05, time_each_step=0.54s, eta=0:37:27
    2021-08-11 16:47:16 [INFO]	[TRAIN] Epoch=218/270, Step=10/74, loss=21.254486, lr=1.2e-05, time_each_step=0.56s, eta=0:37:27
    2021-08-11 16:47:17 [INFO]	[TRAIN] Epoch=218/270, Step=12/74, loss=19.685207, lr=1.2e-05, time_each_step=0.58s, eta=0:37:27
    2021-08-11 16:47:17 [INFO]	[TRAIN] Epoch=218/270, Step=14/74, loss=46.027809, lr=1.2e-05, time_each_step=0.6s, eta=0:37:28
    2021-08-11 16:47:18 [INFO]	[TRAIN] Epoch=218/270, Step=16/74, loss=16.59911, lr=1.2e-05, time_each_step=0.62s, eta=0:37:27
    2021-08-11 16:47:19 [INFO]	[TRAIN] Epoch=218/270, Step=18/74, loss=20.635775, lr=1.2e-05, time_each_step=0.64s, eta=0:37:28
    2021-08-11 16:47:20 [INFO]	[TRAIN] Epoch=218/270, Step=20/74, loss=14.540745, lr=1.2e-05, time_each_step=0.67s, eta=0:37:28
    2021-08-11 16:47:21 [INFO]	[TRAIN] Epoch=218/270, Step=22/74, loss=16.796133, lr=1.2e-05, time_each_step=0.42s, eta=0:37:13
    2021-08-11 16:47:21 [INFO]	[TRAIN] Epoch=218/270, Step=24/74, loss=12.079405, lr=1.2e-05, time_each_step=0.41s, eta=0:37:12
    2021-08-11 16:47:22 [INFO]	[TRAIN] Epoch=218/270, Step=26/74, loss=13.808006, lr=1.2e-05, time_each_step=0.4s, eta=0:37:11
    2021-08-11 16:47:23 [INFO]	[TRAIN] Epoch=218/270, Step=28/74, loss=12.423537, lr=1.2e-05, time_each_step=0.4s, eta=0:37:10
    2021-08-11 16:47:23 [INFO]	[TRAIN] Epoch=218/270, Step=30/74, loss=19.241737, lr=1.2e-05, time_each_step=0.37s, eta=0:37:8
    2021-08-11 16:47:24 [INFO]	[TRAIN] Epoch=218/270, Step=32/74, loss=19.34363, lr=1.2e-05, time_each_step=0.36s, eta=0:37:7
    2021-08-11 16:47:25 [INFO]	[TRAIN] Epoch=218/270, Step=34/74, loss=16.594963, lr=1.2e-05, time_each_step=0.36s, eta=0:37:6
    2021-08-11 16:47:25 [INFO]	[TRAIN] Epoch=218/270, Step=36/74, loss=13.368069, lr=1.2e-05, time_each_step=0.36s, eta=0:37:5
    2021-08-11 16:47:26 [INFO]	[TRAIN] Epoch=218/270, Step=38/74, loss=12.431992, lr=1.2e-05, time_each_step=0.36s, eta=0:37:5
    2021-08-11 16:47:27 [INFO]	[TRAIN] Epoch=218/270, Step=40/74, loss=14.45079, lr=1.2e-05, time_each_step=0.36s, eta=0:37:4
    2021-08-11 16:47:28 [INFO]	[TRAIN] Epoch=218/270, Step=42/74, loss=29.375441, lr=1.2e-05, time_each_step=0.36s, eta=0:37:3
    2021-08-11 16:47:28 [INFO]	[TRAIN] Epoch=218/270, Step=44/74, loss=17.54678, lr=1.2e-05, time_each_step=0.36s, eta=0:37:2
    2021-08-11 16:47:29 [INFO]	[TRAIN] Epoch=218/270, Step=46/74, loss=11.384716, lr=1.2e-05, time_each_step=0.34s, eta=0:37:1
    2021-08-11 16:47:29 [INFO]	[TRAIN] Epoch=218/270, Step=48/74, loss=37.074471, lr=1.2e-05, time_each_step=0.33s, eta=0:37:0
    2021-08-11 16:47:30 [INFO]	[TRAIN] Epoch=218/270, Step=50/74, loss=30.573799, lr=1.2e-05, time_each_step=0.35s, eta=0:37:0
    2021-08-11 16:47:31 [INFO]	[TRAIN] Epoch=218/270, Step=52/74, loss=11.7295, lr=1.2e-05, time_each_step=0.35s, eta=0:36:59
    2021-08-11 16:47:31 [INFO]	[TRAIN] Epoch=218/270, Step=54/74, loss=11.122406, lr=1.2e-05, time_each_step=0.32s, eta=0:36:58
    2021-08-11 16:47:32 [INFO]	[TRAIN] Epoch=218/270, Step=56/74, loss=11.295933, lr=1.2e-05, time_each_step=0.32s, eta=0:36:57
    2021-08-11 16:47:32 [INFO]	[TRAIN] Epoch=218/270, Step=58/74, loss=13.554296, lr=1.2e-05, time_each_step=0.3s, eta=0:36:56
    2021-08-11 16:47:33 [INFO]	[TRAIN] Epoch=218/270, Step=60/74, loss=13.981423, lr=1.2e-05, time_each_step=0.29s, eta=0:36:56
    2021-08-11 16:47:33 [INFO]	[TRAIN] Epoch=218/270, Step=62/74, loss=17.070049, lr=1.2e-05, time_each_step=0.27s, eta=0:36:55
    2021-08-11 16:47:33 [INFO]	[TRAIN] Epoch=218/270, Step=64/74, loss=27.6075, lr=1.2e-05, time_each_step=0.25s, eta=0:36:54
    2021-08-11 16:47:34 [INFO]	[TRAIN] Epoch=218/270, Step=66/74, loss=44.828285, lr=1.2e-05, time_each_step=0.26s, eta=0:36:54
    2021-08-11 16:47:34 [INFO]	[TRAIN] Epoch=218/270, Step=68/74, loss=12.34809, lr=1.2e-05, time_each_step=0.24s, eta=0:36:53
    2021-08-11 16:47:35 [INFO]	[TRAIN] Epoch=218/270, Step=70/74, loss=16.523417, lr=1.2e-05, time_each_step=0.23s, eta=0:36:52
    2021-08-11 16:47:35 [INFO]	[TRAIN] Epoch=218/270, Step=72/74, loss=14.571172, lr=1.2e-05, time_each_step=0.22s, eta=0:36:52
    2021-08-11 16:47:35 [INFO]	[TRAIN] Epoch=218/270, Step=74/74, loss=10.119215, lr=1.2e-05, time_each_step=0.21s, eta=0:36:52
    2021-08-11 16:47:35 [INFO]	[TRAIN] Epoch 218 finished, loss=18.878756, lr=1.2e-05 .
    2021-08-11 16:47:44 [INFO]	[TRAIN] Epoch=219/270, Step=2/74, loss=20.807795, lr=1.2e-05, time_each_step=0.62s, eta=0:26:55
    2021-08-11 16:47:45 [INFO]	[TRAIN] Epoch=219/270, Step=4/74, loss=9.27593, lr=1.2e-05, time_each_step=0.64s, eta=0:26:55
    2021-08-11 16:47:45 [INFO]	[TRAIN] Epoch=219/270, Step=6/74, loss=16.721622, lr=1.2e-05, time_each_step=0.63s, eta=0:26:54
    2021-08-11 16:47:46 [INFO]	[TRAIN] Epoch=219/270, Step=8/74, loss=17.180872, lr=1.2e-05, time_each_step=0.65s, eta=0:26:53
    2021-08-11 16:47:47 [INFO]	[TRAIN] Epoch=219/270, Step=10/74, loss=31.007568, lr=1.2e-05, time_each_step=0.68s, eta=0:26:54
    2021-08-11 16:47:48 [INFO]	[TRAIN] Epoch=219/270, Step=12/74, loss=13.681253, lr=1.2e-05, time_each_step=0.7s, eta=0:26:53
    2021-08-11 16:47:49 [INFO]	[TRAIN] Epoch=219/270, Step=14/74, loss=21.583788, lr=1.2e-05, time_each_step=0.73s, eta=0:26:54
    2021-08-11 16:47:50 [INFO]	[TRAIN] Epoch=219/270, Step=16/74, loss=17.762241, lr=1.2e-05, time_each_step=0.76s, eta=0:26:54
    2021-08-11 16:47:50 [INFO]	[TRAIN] Epoch=219/270, Step=18/74, loss=18.138832, lr=1.2e-05, time_each_step=0.77s, eta=0:26:53
    2021-08-11 16:47:51 [INFO]	[TRAIN] Epoch=219/270, Step=20/74, loss=12.468763, lr=1.2e-05, time_each_step=0.77s, eta=0:26:52
    2021-08-11 16:47:51 [INFO]	[TRAIN] Epoch=219/270, Step=22/74, loss=21.388609, lr=1.2e-05, time_each_step=0.37s, eta=0:26:30
    2021-08-11 16:47:52 [INFO]	[TRAIN] Epoch=219/270, Step=24/74, loss=8.913402, lr=1.2e-05, time_each_step=0.35s, eta=0:26:28
    2021-08-11 16:47:52 [INFO]	[TRAIN] Epoch=219/270, Step=26/74, loss=26.643639, lr=1.2e-05, time_each_step=0.36s, eta=0:26:27
    2021-08-11 16:47:53 [INFO]	[TRAIN] Epoch=219/270, Step=28/74, loss=10.611567, lr=1.2e-05, time_each_step=0.35s, eta=0:26:27
    2021-08-11 16:47:54 [INFO]	[TRAIN] Epoch=219/270, Step=30/74, loss=17.408163, lr=1.2e-05, time_each_step=0.36s, eta=0:26:26
    2021-08-11 16:47:55 [INFO]	[TRAIN] Epoch=219/270, Step=32/74, loss=20.886614, lr=1.2e-05, time_each_step=0.36s, eta=0:26:25
    2021-08-11 16:47:56 [INFO]	[TRAIN] Epoch=219/270, Step=34/74, loss=6.666409, lr=1.2e-05, time_each_step=0.34s, eta=0:26:24
    2021-08-11 16:47:56 [INFO]	[TRAIN] Epoch=219/270, Step=36/74, loss=6.381313, lr=1.2e-05, time_each_step=0.33s, eta=0:26:23
    2021-08-11 16:47:57 [INFO]	[TRAIN] Epoch=219/270, Step=38/74, loss=21.038445, lr=1.2e-05, time_each_step=0.34s, eta=0:26:23
    2021-08-11 16:47:58 [INFO]	[TRAIN] Epoch=219/270, Step=40/74, loss=23.483738, lr=1.2e-05, time_each_step=0.35s, eta=0:26:22
    2021-08-11 16:47:59 [INFO]	[TRAIN] Epoch=219/270, Step=42/74, loss=38.646839, lr=1.2e-05, time_each_step=0.36s, eta=0:26:22
    2021-08-11 16:47:59 [INFO]	[TRAIN] Epoch=219/270, Step=44/74, loss=13.48942, lr=1.2e-05, time_each_step=0.37s, eta=0:26:22
    2021-08-11 16:48:00 [INFO]	[TRAIN] Epoch=219/270, Step=46/74, loss=16.676285, lr=1.2e-05, time_each_step=0.39s, eta=0:26:21
    2021-08-11 16:48:01 [INFO]	[TRAIN] Epoch=219/270, Step=48/74, loss=24.597958, lr=1.2e-05, time_each_step=0.38s, eta=0:26:20
    2021-08-11 16:48:01 [INFO]	[TRAIN] Epoch=219/270, Step=50/74, loss=19.806847, lr=1.2e-05, time_each_step=0.35s, eta=0:26:19
    2021-08-11 16:48:02 [INFO]	[TRAIN] Epoch=219/270, Step=52/74, loss=13.567019, lr=1.2e-05, time_each_step=0.34s, eta=0:26:18
    2021-08-11 16:48:02 [INFO]	[TRAIN] Epoch=219/270, Step=54/74, loss=23.279152, lr=1.2e-05, time_each_step=0.32s, eta=0:26:17
    2021-08-11 16:48:03 [INFO]	[TRAIN] Epoch=219/270, Step=56/74, loss=9.985188, lr=1.2e-05, time_each_step=0.32s, eta=0:26:16
    2021-08-11 16:48:03 [INFO]	[TRAIN] Epoch=219/270, Step=58/74, loss=16.534483, lr=1.2e-05, time_each_step=0.3s, eta=0:26:15
    2021-08-11 16:48:04 [INFO]	[TRAIN] Epoch=219/270, Step=60/74, loss=20.686884, lr=1.2e-05, time_each_step=0.28s, eta=0:26:14
    2021-08-11 16:48:04 [INFO]	[TRAIN] Epoch=219/270, Step=62/74, loss=9.994175, lr=1.2e-05, time_each_step=0.27s, eta=0:26:14
    2021-08-11 16:48:04 [INFO]	[TRAIN] Epoch=219/270, Step=64/74, loss=8.513011, lr=1.2e-05, time_each_step=0.26s, eta=0:26:13
    2021-08-11 16:48:05 [INFO]	[TRAIN] Epoch=219/270, Step=66/74, loss=22.170616, lr=1.2e-05, time_each_step=0.23s, eta=0:26:12
    2021-08-11 16:48:05 [INFO]	[TRAIN] Epoch=219/270, Step=68/74, loss=17.404449, lr=1.2e-05, time_each_step=0.23s, eta=0:26:12
    2021-08-11 16:48:06 [INFO]	[TRAIN] Epoch=219/270, Step=70/74, loss=12.559657, lr=1.2e-05, time_each_step=0.23s, eta=0:26:11
    2021-08-11 16:48:06 [INFO]	[TRAIN] Epoch=219/270, Step=72/74, loss=10.774733, lr=1.2e-05, time_each_step=0.22s, eta=0:26:11
    2021-08-11 16:48:06 [INFO]	[TRAIN] Epoch=219/270, Step=74/74, loss=13.081684, lr=1.2e-05, time_each_step=0.22s, eta=0:26:10
    2021-08-11 16:48:06 [INFO]	[TRAIN] Epoch 219 finished, loss=17.651533, lr=1.2e-05 .
    2021-08-11 16:48:15 [INFO]	[TRAIN] Epoch=220/270, Step=2/74, loss=26.240177, lr=1.2e-05, time_each_step=0.61s, eta=0:28:0
    2021-08-11 16:48:16 [INFO]	[TRAIN] Epoch=220/270, Step=4/74, loss=24.440294, lr=1.2e-05, time_each_step=0.63s, eta=0:28:0
    2021-08-11 16:48:17 [INFO]	[TRAIN] Epoch=220/270, Step=6/74, loss=5.858973, lr=1.2e-05, time_each_step=0.65s, eta=0:28:0
    2021-08-11 16:48:17 [INFO]	[TRAIN] Epoch=220/270, Step=8/74, loss=20.948164, lr=1.2e-05, time_each_step=0.66s, eta=0:28:0
    2021-08-11 16:48:18 [INFO]	[TRAIN] Epoch=220/270, Step=10/74, loss=13.567999, lr=1.2e-05, time_each_step=0.69s, eta=0:28:0
    2021-08-11 16:48:19 [INFO]	[TRAIN] Epoch=220/270, Step=12/74, loss=9.405213, lr=1.2e-05, time_each_step=0.72s, eta=0:28:1
    2021-08-11 16:48:20 [INFO]	[TRAIN] Epoch=220/270, Step=14/74, loss=50.365917, lr=1.2e-05, time_each_step=0.73s, eta=0:28:0
    2021-08-11 16:48:21 [INFO]	[TRAIN] Epoch=220/270, Step=16/74, loss=13.567141, lr=1.2e-05, time_each_step=0.74s, eta=0:27:59
    2021-08-11 16:48:21 [INFO]	[TRAIN] Epoch=220/270, Step=18/74, loss=21.016289, lr=1.2e-05, time_each_step=0.76s, eta=0:27:58
    2021-08-11 16:48:22 [INFO]	[TRAIN] Epoch=220/270, Step=20/74, loss=24.811798, lr=1.2e-05, time_each_step=0.78s, eta=0:27:58
    2021-08-11 16:48:23 [INFO]	[TRAIN] Epoch=220/270, Step=22/74, loss=27.315613, lr=1.2e-05, time_each_step=0.38s, eta=0:27:36
    2021-08-11 16:48:23 [INFO]	[TRAIN] Epoch=220/270, Step=24/74, loss=22.421471, lr=1.2e-05, time_each_step=0.38s, eta=0:27:35
    2021-08-11 16:48:24 [INFO]	[TRAIN] Epoch=220/270, Step=26/74, loss=10.915623, lr=1.2e-05, time_each_step=0.4s, eta=0:27:35
    2021-08-11 16:48:25 [INFO]	[TRAIN] Epoch=220/270, Step=28/74, loss=29.638971, lr=1.2e-05, time_each_step=0.4s, eta=0:27:35
    2021-08-11 16:48:26 [INFO]	[TRAIN] Epoch=220/270, Step=30/74, loss=15.306579, lr=1.2e-05, time_each_step=0.39s, eta=0:27:33
    2021-08-11 16:48:27 [INFO]	[TRAIN] Epoch=220/270, Step=32/74, loss=16.114647, lr=1.2e-05, time_each_step=0.38s, eta=0:27:32
    2021-08-11 16:48:28 [INFO]	[TRAIN] Epoch=220/270, Step=34/74, loss=17.136311, lr=1.2e-05, time_each_step=0.4s, eta=0:27:32
    2021-08-11 16:48:29 [INFO]	[TRAIN] Epoch=220/270, Step=36/74, loss=26.57428, lr=1.2e-05, time_each_step=0.4s, eta=0:27:31
    2021-08-11 16:48:29 [INFO]	[TRAIN] Epoch=220/270, Step=38/74, loss=13.609108, lr=1.2e-05, time_each_step=0.41s, eta=0:27:31
    2021-08-11 16:48:30 [INFO]	[TRAIN] Epoch=220/270, Step=40/74, loss=16.781765, lr=1.2e-05, time_each_step=0.41s, eta=0:27:30
    2021-08-11 16:48:31 [INFO]	[TRAIN] Epoch=220/270, Step=42/74, loss=18.556227, lr=1.2e-05, time_each_step=0.41s, eta=0:27:29
    2021-08-11 16:48:31 [INFO]	[TRAIN] Epoch=220/270, Step=44/74, loss=13.689873, lr=1.2e-05, time_each_step=0.39s, eta=0:27:28
    2021-08-11 16:48:32 [INFO]	[TRAIN] Epoch=220/270, Step=46/74, loss=25.751942, lr=1.2e-05, time_each_step=0.36s, eta=0:27:26
    2021-08-11 16:48:32 [INFO]	[TRAIN] Epoch=220/270, Step=48/74, loss=14.644397, lr=1.2e-05, time_each_step=0.33s, eta=0:27:24
    2021-08-11 16:48:32 [INFO]	[TRAIN] Epoch=220/270, Step=50/74, loss=8.791211, lr=1.2e-05, time_each_step=0.31s, eta=0:27:23
    2021-08-11 16:48:33 [INFO]	[TRAIN] Epoch=220/270, Step=52/74, loss=14.711018, lr=1.2e-05, time_each_step=0.29s, eta=0:27:22
    2021-08-11 16:48:33 [INFO]	[TRAIN] Epoch=220/270, Step=54/74, loss=40.98521, lr=1.2e-05, time_each_step=0.26s, eta=0:27:21
    2021-08-11 16:48:34 [INFO]	[TRAIN] Epoch=220/270, Step=56/74, loss=12.91325, lr=1.2e-05, time_each_step=0.26s, eta=0:27:21
    2021-08-11 16:48:34 [INFO]	[TRAIN] Epoch=220/270, Step=58/74, loss=27.200073, lr=1.2e-05, time_each_step=0.25s, eta=0:27:20
    2021-08-11 16:48:35 [INFO]	[TRAIN] Epoch=220/270, Step=60/74, loss=15.214983, lr=1.2e-05, time_each_step=0.23s, eta=0:27:19
    2021-08-11 16:48:35 [INFO]	[TRAIN] Epoch=220/270, Step=62/74, loss=26.718702, lr=1.2e-05, time_each_step=0.22s, eta=0:27:19
    2021-08-11 16:48:36 [INFO]	[TRAIN] Epoch=220/270, Step=64/74, loss=13.17728, lr=1.2e-05, time_each_step=0.22s, eta=0:27:18
    2021-08-11 16:48:36 [INFO]	[TRAIN] Epoch=220/270, Step=66/74, loss=12.211971, lr=1.2e-05, time_each_step=0.22s, eta=0:27:18
    2021-08-11 16:48:36 [INFO]	[TRAIN] Epoch=220/270, Step=68/74, loss=28.550457, lr=1.2e-05, time_each_step=0.22s, eta=0:27:17
    2021-08-11 16:48:37 [INFO]	[TRAIN] Epoch=220/270, Step=70/74, loss=20.381542, lr=1.2e-05, time_each_step=0.22s, eta=0:27:17
    2021-08-11 16:48:37 [INFO]	[TRAIN] Epoch=220/270, Step=72/74, loss=19.747974, lr=1.2e-05, time_each_step=0.23s, eta=0:27:16
    2021-08-11 16:48:38 [INFO]	[TRAIN] Epoch=220/270, Step=74/74, loss=16.628899, lr=1.2e-05, time_each_step=0.22s, eta=0:27:16
    2021-08-11 16:48:38 [INFO]	[TRAIN] Epoch 220 finished, loss=17.70649, lr=1.2e-05 .
    2021-08-11 16:48:38 [INFO]	Start to evaluating(total_samples=170, total_steps=22)...


    100%|██████████| 22/22 [00:17<00:00,  1.25it/s]


    2021-08-11 16:48:56 [INFO]	[EVAL] Finished, Epoch=220, bbox_map=79.775703 .
    2021-08-11 16:49:02 [INFO]	Model saved in output/yolov3_DarkNet53/best_model.
    2021-08-11 16:49:08 [INFO]	Model saved in output/yolov3_DarkNet53/epoch_220.
    2021-08-11 16:49:08 [INFO]	Current evaluated best model in eval_dataset is epoch_220, bbox_map=79.77570262579437
    2021-08-11 16:49:12 [INFO]	[TRAIN] Epoch=221/270, Step=2/74, loss=12.901583, lr=1.2e-05, time_each_step=0.43s, eta=0:27:40
    2021-08-11 16:49:13 [INFO]	[TRAIN] Epoch=221/270, Step=4/74, loss=35.128494, lr=1.2e-05, time_each_step=0.43s, eta=0:27:40
    2021-08-11 16:49:14 [INFO]	[TRAIN] Epoch=221/270, Step=6/74, loss=17.680473, lr=1.2e-05, time_each_step=0.46s, eta=0:27:41
    2021-08-11 16:49:14 [INFO]	[TRAIN] Epoch=221/270, Step=8/74, loss=18.277222, lr=1.2e-05, time_each_step=0.48s, eta=0:27:41
    2021-08-11 16:49:15 [INFO]	[TRAIN] Epoch=221/270, Step=10/74, loss=45.592731, lr=1.2e-05, time_each_step=0.5s, eta=0:27:41
    2021-08-11 16:49:16 [INFO]	[TRAIN] Epoch=221/270, Step=12/74, loss=16.024923, lr=1.2e-05, time_each_step=0.5s, eta=0:27:41
    2021-08-11 16:49:17 [INFO]	[TRAIN] Epoch=221/270, Step=14/74, loss=11.776155, lr=1.2e-05, time_each_step=0.52s, eta=0:27:41
    2021-08-11 16:49:18 [INFO]	[TRAIN] Epoch=221/270, Step=16/74, loss=13.385021, lr=1.2e-05, time_each_step=0.54s, eta=0:27:41
    2021-08-11 16:49:19 [INFO]	[TRAIN] Epoch=221/270, Step=18/74, loss=18.21924, lr=1.2e-05, time_each_step=0.57s, eta=0:27:42
    2021-08-11 16:49:20 [INFO]	[TRAIN] Epoch=221/270, Step=20/74, loss=11.246169, lr=1.2e-05, time_each_step=0.6s, eta=0:27:42
    2021-08-11 16:49:20 [INFO]	[TRAIN] Epoch=221/270, Step=22/74, loss=16.388065, lr=1.2e-05, time_each_step=0.41s, eta=0:27:31
    2021-08-11 16:49:21 [INFO]	[TRAIN] Epoch=221/270, Step=24/74, loss=20.445415, lr=1.2e-05, time_each_step=0.42s, eta=0:27:31
    2021-08-11 16:49:22 [INFO]	[TRAIN] Epoch=221/270, Step=26/74, loss=28.863976, lr=1.2e-05, time_each_step=0.42s, eta=0:27:30
    2021-08-11 16:49:23 [INFO]	[TRAIN] Epoch=221/270, Step=28/74, loss=14.926173, lr=1.2e-05, time_each_step=0.41s, eta=0:27:29
    2021-08-11 16:49:24 [INFO]	[TRAIN] Epoch=221/270, Step=30/74, loss=23.4361, lr=1.2e-05, time_each_step=0.42s, eta=0:27:28
    2021-08-11 16:49:25 [INFO]	[TRAIN] Epoch=221/270, Step=32/74, loss=15.759477, lr=1.2e-05, time_each_step=0.43s, eta=0:27:27
    2021-08-11 16:49:25 [INFO]	[TRAIN] Epoch=221/270, Step=34/74, loss=14.085126, lr=1.2e-05, time_each_step=0.43s, eta=0:27:27
    2021-08-11 16:49:26 [INFO]	[TRAIN] Epoch=221/270, Step=36/74, loss=21.670504, lr=1.2e-05, time_each_step=0.42s, eta=0:27:26
    2021-08-11 16:49:27 [INFO]	[TRAIN] Epoch=221/270, Step=38/74, loss=14.904561, lr=1.2e-05, time_each_step=0.41s, eta=0:27:24
    2021-08-11 16:49:27 [INFO]	[TRAIN] Epoch=221/270, Step=40/74, loss=11.682896, lr=1.2e-05, time_each_step=0.39s, eta=0:27:23
    2021-08-11 16:49:28 [INFO]	[TRAIN] Epoch=221/270, Step=42/74, loss=12.230433, lr=1.2e-05, time_each_step=0.39s, eta=0:27:22
    2021-08-11 16:49:29 [INFO]	[TRAIN] Epoch=221/270, Step=44/74, loss=17.196846, lr=1.2e-05, time_each_step=0.36s, eta=0:27:20
    2021-08-11 16:49:29 [INFO]	[TRAIN] Epoch=221/270, Step=46/74, loss=27.757282, lr=1.2e-05, time_each_step=0.35s, eta=0:27:19
    2021-08-11 16:49:29 [INFO]	[TRAIN] Epoch=221/270, Step=48/74, loss=10.052553, lr=1.2e-05, time_each_step=0.34s, eta=0:27:18
    2021-08-11 16:49:30 [INFO]	[TRAIN] Epoch=221/270, Step=50/74, loss=23.043007, lr=1.2e-05, time_each_step=0.29s, eta=0:27:17
    2021-08-11 16:49:30 [INFO]	[TRAIN] Epoch=221/270, Step=52/74, loss=21.777792, lr=1.2e-05, time_each_step=0.28s, eta=0:27:16
    2021-08-11 16:49:31 [INFO]	[TRAIN] Epoch=221/270, Step=54/74, loss=16.28937, lr=1.2e-05, time_each_step=0.27s, eta=0:27:15
    2021-08-11 16:49:31 [INFO]	[TRAIN] Epoch=221/270, Step=56/74, loss=10.220093, lr=1.2e-05, time_each_step=0.26s, eta=0:27:14
    2021-08-11 16:49:31 [INFO]	[TRAIN] Epoch=221/270, Step=58/74, loss=17.442411, lr=1.2e-05, time_each_step=0.23s, eta=0:27:13
    2021-08-11 16:49:32 [INFO]	[TRAIN] Epoch=221/270, Step=60/74, loss=8.513064, lr=1.2e-05, time_each_step=0.22s, eta=0:27:13
    2021-08-11 16:49:32 [INFO]	[TRAIN] Epoch=221/270, Step=62/74, loss=20.183498, lr=1.2e-05, time_each_step=0.2s, eta=0:27:12
    2021-08-11 16:49:33 [INFO]	[TRAIN] Epoch=221/270, Step=64/74, loss=17.545389, lr=1.2e-05, time_each_step=0.2s, eta=0:27:12
    2021-08-11 16:49:33 [INFO]	[TRAIN] Epoch=221/270, Step=66/74, loss=23.717251, lr=1.2e-05, time_each_step=0.2s, eta=0:27:11
    2021-08-11 16:49:34 [INFO]	[TRAIN] Epoch=221/270, Step=68/74, loss=15.109406, lr=1.2e-05, time_each_step=0.21s, eta=0:27:11
    2021-08-11 16:49:34 [INFO]	[TRAIN] Epoch=221/270, Step=70/74, loss=26.634619, lr=1.2e-05, time_each_step=0.21s, eta=0:27:10
    2021-08-11 16:49:34 [INFO]	[TRAIN] Epoch=221/270, Step=72/74, loss=24.730597, lr=1.2e-05, time_each_step=0.21s, eta=0:27:10
    2021-08-11 16:49:35 [INFO]	[TRAIN] Epoch=221/270, Step=74/74, loss=10.878078, lr=1.2e-05, time_each_step=0.2s, eta=0:27:10
    2021-08-11 16:49:35 [INFO]	[TRAIN] Epoch 221 finished, loss=19.517698, lr=1.2e-05 .
    2021-08-11 16:49:40 [INFO]	[TRAIN] Epoch=222/270, Step=2/74, loss=16.540642, lr=1.2e-05, time_each_step=0.42s, eta=0:23:40
    2021-08-11 16:49:41 [INFO]	[TRAIN] Epoch=222/270, Step=4/74, loss=24.506796, lr=1.2e-05, time_each_step=0.46s, eta=0:23:42
    2021-08-11 16:49:41 [INFO]	[TRAIN] Epoch=222/270, Step=6/74, loss=15.760235, lr=1.2e-05, time_each_step=0.47s, eta=0:23:42
    2021-08-11 16:49:42 [INFO]	[TRAIN] Epoch=222/270, Step=8/74, loss=16.361286, lr=1.2e-05, time_each_step=0.49s, eta=0:23:42
    2021-08-11 16:49:43 [INFO]	[TRAIN] Epoch=222/270, Step=10/74, loss=5.08089, lr=1.2e-05, time_each_step=0.51s, eta=0:23:42
    2021-08-11 16:49:44 [INFO]	[TRAIN] Epoch=222/270, Step=12/74, loss=19.842453, lr=1.2e-05, time_each_step=0.52s, eta=0:23:42
    2021-08-11 16:49:44 [INFO]	[TRAIN] Epoch=222/270, Step=14/74, loss=41.459557, lr=1.2e-05, time_each_step=0.52s, eta=0:23:41
    2021-08-11 16:49:45 [INFO]	[TRAIN] Epoch=222/270, Step=16/74, loss=20.389311, lr=1.2e-05, time_each_step=0.55s, eta=0:23:42
    2021-08-11 16:49:46 [INFO]	[TRAIN] Epoch=222/270, Step=18/74, loss=26.31966, lr=1.2e-05, time_each_step=0.57s, eta=0:23:42
    2021-08-11 16:49:46 [INFO]	[TRAIN] Epoch=222/270, Step=20/74, loss=23.009706, lr=1.2e-05, time_each_step=0.58s, eta=0:23:41
    2021-08-11 16:49:47 [INFO]	[TRAIN] Epoch=222/270, Step=22/74, loss=8.950674, lr=1.2e-05, time_each_step=0.37s, eta=0:23:29
    2021-08-11 16:49:48 [INFO]	[TRAIN] Epoch=222/270, Step=24/74, loss=14.471136, lr=1.2e-05, time_each_step=0.35s, eta=0:23:27
    2021-08-11 16:49:49 [INFO]	[TRAIN] Epoch=222/270, Step=26/74, loss=30.846901, lr=1.2e-05, time_each_step=0.36s, eta=0:23:27
    2021-08-11 16:49:49 [INFO]	[TRAIN] Epoch=222/270, Step=28/74, loss=8.82981, lr=1.2e-05, time_each_step=0.37s, eta=0:23:27
    2021-08-11 16:49:50 [INFO]	[TRAIN] Epoch=222/270, Step=30/74, loss=13.520654, lr=1.2e-05, time_each_step=0.38s, eta=0:23:26
    2021-08-11 16:49:51 [INFO]	[TRAIN] Epoch=222/270, Step=32/74, loss=19.372087, lr=1.2e-05, time_each_step=0.38s, eta=0:23:26
    2021-08-11 16:49:52 [INFO]	[TRAIN] Epoch=222/270, Step=34/74, loss=22.921967, lr=1.2e-05, time_each_step=0.38s, eta=0:23:25
    2021-08-11 16:49:53 [INFO]	[TRAIN] Epoch=222/270, Step=36/74, loss=12.782776, lr=1.2e-05, time_each_step=0.4s, eta=0:23:25
    2021-08-11 16:49:54 [INFO]	[TRAIN] Epoch=222/270, Step=38/74, loss=25.735487, lr=1.2e-05, time_each_step=0.39s, eta=0:23:24
    2021-08-11 16:49:54 [INFO]	[TRAIN] Epoch=222/270, Step=40/74, loss=13.519363, lr=1.2e-05, time_each_step=0.4s, eta=0:23:23
    2021-08-11 16:49:55 [INFO]	[TRAIN] Epoch=222/270, Step=42/74, loss=16.796116, lr=1.2e-05, time_each_step=0.4s, eta=0:23:23
    2021-08-11 16:49:56 [INFO]	[TRAIN] Epoch=222/270, Step=44/74, loss=26.482189, lr=1.2e-05, time_each_step=0.4s, eta=0:23:22
    2021-08-11 16:49:56 [INFO]	[TRAIN] Epoch=222/270, Step=46/74, loss=18.177818, lr=1.2e-05, time_each_step=0.37s, eta=0:23:20
    2021-08-11 16:49:56 [INFO]	[TRAIN] Epoch=222/270, Step=48/74, loss=17.379816, lr=1.2e-05, time_each_step=0.34s, eta=0:23:19
    2021-08-11 16:49:57 [INFO]	[TRAIN] Epoch=222/270, Step=50/74, loss=22.071012, lr=1.2e-05, time_each_step=0.32s, eta=0:23:17
    2021-08-11 16:49:57 [INFO]	[TRAIN] Epoch=222/270, Step=52/74, loss=21.21434, lr=1.2e-05, time_each_step=0.3s, eta=0:23:16
    2021-08-11 16:49:58 [INFO]	[TRAIN] Epoch=222/270, Step=54/74, loss=23.407719, lr=1.2e-05, time_each_step=0.28s, eta=0:23:15
    2021-08-11 16:49:58 [INFO]	[TRAIN] Epoch=222/270, Step=56/74, loss=15.918483, lr=1.2e-05, time_each_step=0.25s, eta=0:23:14
    2021-08-11 16:49:58 [INFO]	[TRAIN] Epoch=222/270, Step=58/74, loss=13.834743, lr=1.2e-05, time_each_step=0.23s, eta=0:23:13
    2021-08-11 16:49:59 [INFO]	[TRAIN] Epoch=222/270, Step=60/74, loss=7.088305, lr=1.2e-05, time_each_step=0.21s, eta=0:23:13
    2021-08-11 16:49:59 [INFO]	[TRAIN] Epoch=222/270, Step=62/74, loss=37.228058, lr=1.2e-05, time_each_step=0.2s, eta=0:23:12
    2021-08-11 16:49:59 [INFO]	[TRAIN] Epoch=222/270, Step=64/74, loss=22.326967, lr=1.2e-05, time_each_step=0.19s, eta=0:23:12
    2021-08-11 16:50:00 [INFO]	[TRAIN] Epoch=222/270, Step=66/74, loss=18.009033, lr=1.2e-05, time_each_step=0.19s, eta=0:23:11
    2021-08-11 16:50:01 [INFO]	[TRAIN] Epoch=222/270, Step=68/74, loss=17.33746, lr=1.2e-05, time_each_step=0.21s, eta=0:23:11
    2021-08-11 16:50:01 [INFO]	[TRAIN] Epoch=222/270, Step=70/74, loss=8.702988, lr=1.2e-05, time_each_step=0.22s, eta=0:23:11
    2021-08-11 16:50:01 [INFO]	[TRAIN] Epoch=222/270, Step=72/74, loss=23.223129, lr=1.2e-05, time_each_step=0.21s, eta=0:23:10
    2021-08-11 16:50:02 [INFO]	[TRAIN] Epoch=222/270, Step=74/74, loss=17.110924, lr=1.2e-05, time_each_step=0.2s, eta=0:23:10
    2021-08-11 16:50:02 [INFO]	[TRAIN] Epoch 222 finished, loss=18.753094, lr=1.2e-05 .
    2021-08-11 16:50:05 [INFO]	[TRAIN] Epoch=223/270, Step=2/74, loss=36.914951, lr=1.2e-05, time_each_step=0.38s, eta=0:22:58
    2021-08-11 16:50:07 [INFO]	[TRAIN] Epoch=223/270, Step=4/74, loss=27.366898, lr=1.2e-05, time_each_step=0.43s, eta=0:23:0
    2021-08-11 16:50:08 [INFO]	[TRAIN] Epoch=223/270, Step=6/74, loss=34.027512, lr=1.2e-05, time_each_step=0.45s, eta=0:23:0
    2021-08-11 16:50:08 [INFO]	[TRAIN] Epoch=223/270, Step=8/74, loss=10.302062, lr=1.2e-05, time_each_step=0.46s, eta=0:23:1
    2021-08-11 16:50:09 [INFO]	[TRAIN] Epoch=223/270, Step=10/74, loss=28.377586, lr=1.2e-05, time_each_step=0.48s, eta=0:23:1
    2021-08-11 16:50:10 [INFO]	[TRAIN] Epoch=223/270, Step=12/74, loss=12.784619, lr=1.2e-05, time_each_step=0.49s, eta=0:23:1
    2021-08-11 16:50:10 [INFO]	[TRAIN] Epoch=223/270, Step=14/74, loss=28.644562, lr=1.2e-05, time_each_step=0.5s, eta=0:23:0
    2021-08-11 16:50:11 [INFO]	[TRAIN] Epoch=223/270, Step=16/74, loss=17.560831, lr=1.2e-05, time_each_step=0.52s, eta=0:23:0
    2021-08-11 16:50:12 [INFO]	[TRAIN] Epoch=223/270, Step=18/74, loss=19.145788, lr=1.2e-05, time_each_step=0.55s, eta=0:23:1
    2021-08-11 16:50:13 [INFO]	[TRAIN] Epoch=223/270, Step=20/74, loss=13.367194, lr=1.2e-05, time_each_step=0.57s, eta=0:23:1
    2021-08-11 16:50:14 [INFO]	[TRAIN] Epoch=223/270, Step=22/74, loss=17.666258, lr=1.2e-05, time_each_step=0.42s, eta=0:22:52
    2021-08-11 16:50:15 [INFO]	[TRAIN] Epoch=223/270, Step=24/74, loss=24.431084, lr=1.2e-05, time_each_step=0.39s, eta=0:22:50
    2021-08-11 16:50:15 [INFO]	[TRAIN] Epoch=223/270, Step=26/74, loss=27.83186, lr=1.2e-05, time_each_step=0.39s, eta=0:22:48
    2021-08-11 16:50:16 [INFO]	[TRAIN] Epoch=223/270, Step=28/74, loss=14.187716, lr=1.2e-05, time_each_step=0.41s, eta=0:22:49
    2021-08-11 16:50:17 [INFO]	[TRAIN] Epoch=223/270, Step=30/74, loss=42.631927, lr=1.2e-05, time_each_step=0.42s, eta=0:22:48
    2021-08-11 16:50:18 [INFO]	[TRAIN] Epoch=223/270, Step=32/74, loss=19.225224, lr=1.2e-05, time_each_step=0.42s, eta=0:22:47
    2021-08-11 16:50:19 [INFO]	[TRAIN] Epoch=223/270, Step=34/74, loss=13.739012, lr=1.2e-05, time_each_step=0.41s, eta=0:22:46
    2021-08-11 16:50:19 [INFO]	[TRAIN] Epoch=223/270, Step=36/74, loss=13.509172, lr=1.2e-05, time_each_step=0.4s, eta=0:22:45
    2021-08-11 16:50:20 [INFO]	[TRAIN] Epoch=223/270, Step=38/74, loss=41.544197, lr=1.2e-05, time_each_step=0.39s, eta=0:22:44
    2021-08-11 16:50:21 [INFO]	[TRAIN] Epoch=223/270, Step=40/74, loss=23.098387, lr=1.2e-05, time_each_step=0.39s, eta=0:22:43
    2021-08-11 16:50:22 [INFO]	[TRAIN] Epoch=223/270, Step=42/74, loss=20.924109, lr=1.2e-05, time_each_step=0.39s, eta=0:22:42
    2021-08-11 16:50:22 [INFO]	[TRAIN] Epoch=223/270, Step=44/74, loss=10.478853, lr=1.2e-05, time_each_step=0.37s, eta=0:22:41
    2021-08-11 16:50:23 [INFO]	[TRAIN] Epoch=223/270, Step=46/74, loss=30.900467, lr=1.2e-05, time_each_step=0.37s, eta=0:22:40
    2021-08-11 16:50:23 [INFO]	[TRAIN] Epoch=223/270, Step=48/74, loss=22.608173, lr=1.2e-05, time_each_step=0.33s, eta=0:22:39
    2021-08-11 16:50:23 [INFO]	[TRAIN] Epoch=223/270, Step=50/74, loss=22.730757, lr=1.2e-05, time_each_step=0.3s, eta=0:22:37
    2021-08-11 16:50:24 [INFO]	[TRAIN] Epoch=223/270, Step=52/74, loss=21.705582, lr=1.2e-05, time_each_step=0.29s, eta=0:22:36
    2021-08-11 16:50:24 [INFO]	[TRAIN] Epoch=223/270, Step=54/74, loss=10.765329, lr=1.2e-05, time_each_step=0.28s, eta=0:22:36
    2021-08-11 16:50:25 [INFO]	[TRAIN] Epoch=223/270, Step=56/74, loss=24.321758, lr=1.2e-05, time_each_step=0.27s, eta=0:22:35
    2021-08-11 16:50:25 [INFO]	[TRAIN] Epoch=223/270, Step=58/74, loss=20.804455, lr=1.2e-05, time_each_step=0.25s, eta=0:22:34
    2021-08-11 16:50:26 [INFO]	[TRAIN] Epoch=223/270, Step=60/74, loss=9.410927, lr=1.2e-05, time_each_step=0.24s, eta=0:22:33
    2021-08-11 16:50:26 [INFO]	[TRAIN] Epoch=223/270, Step=62/74, loss=15.66717, lr=1.2e-05, time_each_step=0.22s, eta=0:22:33
    2021-08-11 16:50:26 [INFO]	[TRAIN] Epoch=223/270, Step=64/74, loss=14.666921, lr=1.2e-05, time_each_step=0.21s, eta=0:22:32
    2021-08-11 16:50:27 [INFO]	[TRAIN] Epoch=223/270, Step=66/74, loss=8.455933, lr=1.2e-05, time_each_step=0.2s, eta=0:22:32
    2021-08-11 16:50:27 [INFO]	[TRAIN] Epoch=223/270, Step=68/74, loss=13.034412, lr=1.2e-05, time_each_step=0.2s, eta=0:22:31
    2021-08-11 16:50:28 [INFO]	[TRAIN] Epoch=223/270, Step=70/74, loss=17.403864, lr=1.2e-05, time_each_step=0.22s, eta=0:22:31
    2021-08-11 16:50:28 [INFO]	[TRAIN] Epoch=223/270, Step=72/74, loss=18.228653, lr=1.2e-05, time_each_step=0.21s, eta=0:22:30
    2021-08-11 16:50:28 [INFO]	[TRAIN] Epoch=223/270, Step=74/74, loss=10.636234, lr=1.2e-05, time_each_step=0.21s, eta=0:22:30
    2021-08-11 16:50:28 [INFO]	[TRAIN] Epoch 223 finished, loss=18.906572, lr=1.2e-05 .
    2021-08-11 16:50:34 [INFO]	[TRAIN] Epoch=224/270, Step=2/74, loss=17.400362, lr=1.2e-05, time_each_step=0.44s, eta=0:22:35
    2021-08-11 16:50:35 [INFO]	[TRAIN] Epoch=224/270, Step=4/74, loss=12.832218, lr=1.2e-05, time_each_step=0.49s, eta=0:22:37
    2021-08-11 16:50:35 [INFO]	[TRAIN] Epoch=224/270, Step=6/74, loss=24.295334, lr=1.2e-05, time_each_step=0.49s, eta=0:22:37
    2021-08-11 16:50:36 [INFO]	[TRAIN] Epoch=224/270, Step=8/74, loss=12.656487, lr=1.2e-05, time_each_step=0.51s, eta=0:22:37
    2021-08-11 16:50:37 [INFO]	[TRAIN] Epoch=224/270, Step=10/74, loss=25.284456, lr=1.2e-05, time_each_step=0.53s, eta=0:22:37
    2021-08-11 16:50:38 [INFO]	[TRAIN] Epoch=224/270, Step=12/74, loss=15.261974, lr=1.2e-05, time_each_step=0.56s, eta=0:22:38
    2021-08-11 16:50:38 [INFO]	[TRAIN] Epoch=224/270, Step=14/74, loss=11.639347, lr=1.2e-05, time_each_step=0.56s, eta=0:22:37
    2021-08-11 16:50:39 [INFO]	[TRAIN] Epoch=224/270, Step=16/74, loss=17.390278, lr=1.2e-05, time_each_step=0.57s, eta=0:22:37
    2021-08-11 16:50:40 [INFO]	[TRAIN] Epoch=224/270, Step=18/74, loss=10.989554, lr=1.2e-05, time_each_step=0.59s, eta=0:22:36
    2021-08-11 16:50:41 [INFO]	[TRAIN] Epoch=224/270, Step=20/74, loss=12.851653, lr=1.2e-05, time_each_step=0.63s, eta=0:22:37
    2021-08-11 16:50:42 [INFO]	[TRAIN] Epoch=224/270, Step=22/74, loss=23.768898, lr=1.2e-05, time_each_step=0.41s, eta=0:22:24
    2021-08-11 16:50:42 [INFO]	[TRAIN] Epoch=224/270, Step=24/74, loss=10.923042, lr=1.2e-05, time_each_step=0.38s, eta=0:22:22
    2021-08-11 16:50:43 [INFO]	[TRAIN] Epoch=224/270, Step=26/74, loss=28.020878, lr=1.2e-05, time_each_step=0.38s, eta=0:22:22
    2021-08-11 16:50:44 [INFO]	[TRAIN] Epoch=224/270, Step=28/74, loss=14.991293, lr=1.2e-05, time_each_step=0.38s, eta=0:22:21
    2021-08-11 16:50:44 [INFO]	[TRAIN] Epoch=224/270, Step=30/74, loss=12.368369, lr=1.2e-05, time_each_step=0.37s, eta=0:22:20
    2021-08-11 16:50:45 [INFO]	[TRAIN] Epoch=224/270, Step=32/74, loss=12.003057, lr=1.2e-05, time_each_step=0.38s, eta=0:22:19
    2021-08-11 16:50:46 [INFO]	[TRAIN] Epoch=224/270, Step=34/74, loss=8.093502, lr=1.2e-05, time_each_step=0.38s, eta=0:22:18
    2021-08-11 16:50:47 [INFO]	[TRAIN] Epoch=224/270, Step=36/74, loss=9.879128, lr=1.2e-05, time_each_step=0.37s, eta=0:22:17
    2021-08-11 16:50:47 [INFO]	[TRAIN] Epoch=224/270, Step=38/74, loss=17.332268, lr=1.2e-05, time_each_step=0.36s, eta=0:22:16
    2021-08-11 16:50:48 [INFO]	[TRAIN] Epoch=224/270, Step=40/74, loss=12.006132, lr=1.2e-05, time_each_step=0.34s, eta=0:22:15
    2021-08-11 16:50:48 [INFO]	[TRAIN] Epoch=224/270, Step=42/74, loss=20.04464, lr=1.2e-05, time_each_step=0.34s, eta=0:22:14
    2021-08-11 16:50:49 [INFO]	[TRAIN] Epoch=224/270, Step=44/74, loss=25.438309, lr=1.2e-05, time_each_step=0.33s, eta=0:22:13
    2021-08-11 16:50:50 [INFO]	[TRAIN] Epoch=224/270, Step=46/74, loss=14.278654, lr=1.2e-05, time_each_step=0.32s, eta=0:22:12
    2021-08-11 16:50:50 [INFO]	[TRAIN] Epoch=224/270, Step=48/74, loss=20.890945, lr=1.2e-05, time_each_step=0.31s, eta=0:22:11
    2021-08-11 16:50:50 [INFO]	[TRAIN] Epoch=224/270, Step=50/74, loss=27.037884, lr=1.2e-05, time_each_step=0.3s, eta=0:22:10
    2021-08-11 16:50:51 [INFO]	[TRAIN] Epoch=224/270, Step=52/74, loss=8.594808, lr=1.2e-05, time_each_step=0.27s, eta=0:22:9
    2021-08-11 16:50:51 [INFO]	[TRAIN] Epoch=224/270, Step=54/74, loss=10.909224, lr=1.2e-05, time_each_step=0.26s, eta=0:22:8
    2021-08-11 16:50:51 [INFO]	[TRAIN] Epoch=224/270, Step=56/74, loss=19.36614, lr=1.2e-05, time_each_step=0.24s, eta=0:22:8
    2021-08-11 16:50:52 [INFO]	[TRAIN] Epoch=224/270, Step=58/74, loss=15.669732, lr=1.2e-05, time_each_step=0.23s, eta=0:22:7
    2021-08-11 16:50:52 [INFO]	[TRAIN] Epoch=224/270, Step=60/74, loss=18.131683, lr=1.2e-05, time_each_step=0.22s, eta=0:22:6
    2021-08-11 16:50:53 [INFO]	[TRAIN] Epoch=224/270, Step=62/74, loss=18.361528, lr=1.2e-05, time_each_step=0.2s, eta=0:22:6
    2021-08-11 16:50:53 [INFO]	[TRAIN] Epoch=224/270, Step=64/74, loss=12.639963, lr=1.2e-05, time_each_step=0.19s, eta=0:22:5
    2021-08-11 16:50:53 [INFO]	[TRAIN] Epoch=224/270, Step=66/74, loss=11.048095, lr=1.2e-05, time_each_step=0.18s, eta=0:22:5
    2021-08-11 16:50:54 [INFO]	[TRAIN] Epoch=224/270, Step=68/74, loss=18.05764, lr=1.2e-05, time_each_step=0.18s, eta=0:22:4
    2021-08-11 16:50:54 [INFO]	[TRAIN] Epoch=224/270, Step=70/74, loss=27.705536, lr=1.2e-05, time_each_step=0.18s, eta=0:22:4
    2021-08-11 16:50:54 [INFO]	[TRAIN] Epoch=224/270, Step=72/74, loss=19.672369, lr=1.2e-05, time_each_step=0.18s, eta=0:22:4
    2021-08-11 16:50:55 [INFO]	[TRAIN] Epoch=224/270, Step=74/74, loss=9.004534, lr=1.2e-05, time_each_step=0.18s, eta=0:22:3
    2021-08-11 16:50:55 [INFO]	[TRAIN] Epoch 224 finished, loss=17.324614, lr=1.2e-05 .
    2021-08-11 16:50:58 [INFO]	[TRAIN] Epoch=225/270, Step=2/74, loss=22.701216, lr=1.2e-05, time_each_step=0.33s, eta=0:21:31
    2021-08-11 16:50:59 [INFO]	[TRAIN] Epoch=225/270, Step=4/74, loss=16.574743, lr=1.2e-05, time_each_step=0.36s, eta=0:21:33
    2021-08-11 16:51:00 [INFO]	[TRAIN] Epoch=225/270, Step=6/74, loss=7.859447, lr=1.2e-05, time_each_step=0.37s, eta=0:21:33
    2021-08-11 16:51:00 [INFO]	[TRAIN] Epoch=225/270, Step=8/74, loss=20.405594, lr=1.2e-05, time_each_step=0.39s, eta=0:21:33
    2021-08-11 16:51:01 [INFO]	[TRAIN] Epoch=225/270, Step=10/74, loss=29.466129, lr=1.2e-05, time_each_step=0.42s, eta=0:21:35
    2021-08-11 16:51:02 [INFO]	[TRAIN] Epoch=225/270, Step=12/74, loss=18.640129, lr=1.2e-05, time_each_step=0.44s, eta=0:21:35
    2021-08-11 16:51:03 [INFO]	[TRAIN] Epoch=225/270, Step=14/74, loss=15.800745, lr=1.2e-05, time_each_step=0.47s, eta=0:21:36
    2021-08-11 16:51:04 [INFO]	[TRAIN] Epoch=225/270, Step=16/74, loss=15.037292, lr=1.2e-05, time_each_step=0.49s, eta=0:21:36
    2021-08-11 16:51:05 [INFO]	[TRAIN] Epoch=225/270, Step=18/74, loss=25.790203, lr=1.2e-05, time_each_step=0.52s, eta=0:21:37
    2021-08-11 16:51:05 [INFO]	[TRAIN] Epoch=225/270, Step=20/74, loss=22.283009, lr=1.2e-05, time_each_step=0.54s, eta=0:21:37
    2021-08-11 16:51:06 [INFO]	[TRAIN] Epoch=225/270, Step=22/74, loss=14.256881, lr=1.2e-05, time_each_step=0.4s, eta=0:21:29
    2021-08-11 16:51:07 [INFO]	[TRAIN] Epoch=225/270, Step=24/74, loss=14.293947, lr=1.2e-05, time_each_step=0.4s, eta=0:21:28
    2021-08-11 16:51:08 [INFO]	[TRAIN] Epoch=225/270, Step=26/74, loss=14.018841, lr=1.2e-05, time_each_step=0.41s, eta=0:21:27
    2021-08-11 16:51:09 [INFO]	[TRAIN] Epoch=225/270, Step=28/74, loss=15.659662, lr=1.2e-05, time_each_step=0.42s, eta=0:21:27
    2021-08-11 16:51:10 [INFO]	[TRAIN] Epoch=225/270, Step=30/74, loss=8.161355, lr=1.2e-05, time_each_step=0.42s, eta=0:21:26
    2021-08-11 16:51:11 [INFO]	[TRAIN] Epoch=225/270, Step=32/74, loss=28.086372, lr=1.2e-05, time_each_step=0.42s, eta=0:21:26
    2021-08-11 16:51:11 [INFO]	[TRAIN] Epoch=225/270, Step=34/74, loss=13.140149, lr=1.2e-05, time_each_step=0.42s, eta=0:21:25
    2021-08-11 16:51:12 [INFO]	[TRAIN] Epoch=225/270, Step=36/74, loss=13.923203, lr=1.2e-05, time_each_step=0.41s, eta=0:21:24
    2021-08-11 16:51:13 [INFO]	[TRAIN] Epoch=225/270, Step=38/74, loss=13.533383, lr=1.2e-05, time_each_step=0.41s, eta=0:21:23
    2021-08-11 16:51:14 [INFO]	[TRAIN] Epoch=225/270, Step=40/74, loss=15.361028, lr=1.2e-05, time_each_step=0.42s, eta=0:21:22
    2021-08-11 16:51:15 [INFO]	[TRAIN] Epoch=225/270, Step=42/74, loss=14.846086, lr=1.2e-05, time_each_step=0.43s, eta=0:21:21
    2021-08-11 16:51:15 [INFO]	[TRAIN] Epoch=225/270, Step=44/74, loss=27.769196, lr=1.2e-05, time_each_step=0.41s, eta=0:21:20
    2021-08-11 16:51:16 [INFO]	[TRAIN] Epoch=225/270, Step=46/74, loss=24.15485, lr=1.2e-05, time_each_step=0.4s, eta=0:21:19
    2021-08-11 16:51:16 [INFO]	[TRAIN] Epoch=225/270, Step=48/74, loss=33.902527, lr=1.2e-05, time_each_step=0.38s, eta=0:21:18
    2021-08-11 16:51:17 [INFO]	[TRAIN] Epoch=225/270, Step=50/74, loss=11.931836, lr=1.2e-05, time_each_step=0.35s, eta=0:21:16
    2021-08-11 16:51:17 [INFO]	[TRAIN] Epoch=225/270, Step=52/74, loss=18.966684, lr=1.2e-05, time_each_step=0.32s, eta=0:21:15
    2021-08-11 16:51:17 [INFO]	[TRAIN] Epoch=225/270, Step=54/74, loss=22.173351, lr=1.2e-05, time_each_step=0.31s, eta=0:21:14
    2021-08-11 16:51:18 [INFO]	[TRAIN] Epoch=225/270, Step=56/74, loss=15.484453, lr=1.2e-05, time_each_step=0.3s, eta=0:21:13
    2021-08-11 16:51:18 [INFO]	[TRAIN] Epoch=225/270, Step=58/74, loss=27.591972, lr=1.2e-05, time_each_step=0.27s, eta=0:21:12
    2021-08-11 16:51:19 [INFO]	[TRAIN] Epoch=225/270, Step=60/74, loss=16.011375, lr=1.2e-05, time_each_step=0.25s, eta=0:21:11
    2021-08-11 16:51:19 [INFO]	[TRAIN] Epoch=225/270, Step=62/74, loss=9.143081, lr=1.2e-05, time_each_step=0.24s, eta=0:21:11
    2021-08-11 16:51:20 [INFO]	[TRAIN] Epoch=225/270, Step=64/74, loss=13.274109, lr=1.2e-05, time_each_step=0.24s, eta=0:21:10
    2021-08-11 16:51:20 [INFO]	[TRAIN] Epoch=225/270, Step=66/74, loss=20.70092, lr=1.2e-05, time_each_step=0.23s, eta=0:21:10
    2021-08-11 16:51:21 [INFO]	[TRAIN] Epoch=225/270, Step=68/74, loss=10.248686, lr=1.2e-05, time_each_step=0.24s, eta=0:21:9
    2021-08-11 16:51:22 [INFO]	[TRAIN] Epoch=225/270, Step=70/74, loss=19.015144, lr=1.2e-05, time_each_step=0.24s, eta=0:21:9
    2021-08-11 16:51:22 [INFO]	[TRAIN] Epoch=225/270, Step=72/74, loss=23.995804, lr=1.2e-05, time_each_step=0.24s, eta=0:21:8
    2021-08-11 16:51:22 [INFO]	[TRAIN] Epoch=225/270, Step=74/74, loss=21.547583, lr=1.2e-05, time_each_step=0.24s, eta=0:21:8
    2021-08-11 16:51:22 [INFO]	[TRAIN] Epoch 225 finished, loss=18.936331, lr=1.2e-05 .
    2021-08-11 16:51:28 [INFO]	[TRAIN] Epoch=226/270, Step=2/74, loss=10.264343, lr=1.2e-05, time_each_step=0.48s, eta=0:22:22
    2021-08-11 16:51:29 [INFO]	[TRAIN] Epoch=226/270, Step=4/74, loss=22.004902, lr=1.2e-05, time_each_step=0.53s, eta=0:22:25
    2021-08-11 16:51:30 [INFO]	[TRAIN] Epoch=226/270, Step=6/74, loss=17.713814, lr=1.2e-05, time_each_step=0.53s, eta=0:22:24
    2021-08-11 16:51:30 [INFO]	[TRAIN] Epoch=226/270, Step=8/74, loss=9.187973, lr=1.2e-05, time_each_step=0.55s, eta=0:22:24
    2021-08-11 16:51:31 [INFO]	[TRAIN] Epoch=226/270, Step=10/74, loss=15.075253, lr=1.2e-05, time_each_step=0.56s, eta=0:22:23
    2021-08-11 16:51:32 [INFO]	[TRAIN] Epoch=226/270, Step=12/74, loss=43.134033, lr=1.2e-05, time_each_step=0.58s, eta=0:22:23
    2021-08-11 16:51:32 [INFO]	[TRAIN] Epoch=226/270, Step=14/74, loss=13.582828, lr=1.2e-05, time_each_step=0.57s, eta=0:22:21
    2021-08-11 16:51:33 [INFO]	[TRAIN] Epoch=226/270, Step=16/74, loss=16.585793, lr=1.2e-05, time_each_step=0.59s, eta=0:22:21
    2021-08-11 16:51:34 [INFO]	[TRAIN] Epoch=226/270, Step=18/74, loss=27.139648, lr=1.2e-05, time_each_step=0.6s, eta=0:22:21
    2021-08-11 16:51:35 [INFO]	[TRAIN] Epoch=226/270, Step=20/74, loss=15.755386, lr=1.2e-05, time_each_step=0.63s, eta=0:22:22
    2021-08-11 16:51:36 [INFO]	[TRAIN] Epoch=226/270, Step=22/74, loss=13.927664, lr=1.2e-05, time_each_step=0.41s, eta=0:22:9
    2021-08-11 16:51:37 [INFO]	[TRAIN] Epoch=226/270, Step=24/74, loss=49.758236, lr=1.2e-05, time_each_step=0.38s, eta=0:22:7
    2021-08-11 16:51:37 [INFO]	[TRAIN] Epoch=226/270, Step=26/74, loss=11.886544, lr=1.2e-05, time_each_step=0.39s, eta=0:22:6
    2021-08-11 16:51:38 [INFO]	[TRAIN] Epoch=226/270, Step=28/74, loss=22.251879, lr=1.2e-05, time_each_step=0.38s, eta=0:22:5
    2021-08-11 16:51:39 [INFO]	[TRAIN] Epoch=226/270, Step=30/74, loss=16.783878, lr=1.2e-05, time_each_step=0.39s, eta=0:22:4
    2021-08-11 16:51:40 [INFO]	[TRAIN] Epoch=226/270, Step=32/74, loss=12.515392, lr=1.2e-05, time_each_step=0.38s, eta=0:22:3
    2021-08-11 16:51:40 [INFO]	[TRAIN] Epoch=226/270, Step=34/74, loss=19.041847, lr=1.2e-05, time_each_step=0.39s, eta=0:22:3
    2021-08-11 16:51:41 [INFO]	[TRAIN] Epoch=226/270, Step=36/74, loss=16.654528, lr=1.2e-05, time_each_step=0.38s, eta=0:22:2
    2021-08-11 16:51:42 [INFO]	[TRAIN] Epoch=226/270, Step=38/74, loss=21.720858, lr=1.2e-05, time_each_step=0.38s, eta=0:22:1
    2021-08-11 16:51:42 [INFO]	[TRAIN] Epoch=226/270, Step=40/74, loss=22.576035, lr=1.2e-05, time_each_step=0.37s, eta=0:22:0
    2021-08-11 16:51:43 [INFO]	[TRAIN] Epoch=226/270, Step=42/74, loss=33.991547, lr=1.2e-05, time_each_step=0.35s, eta=0:21:59
    2021-08-11 16:51:43 [INFO]	[TRAIN] Epoch=226/270, Step=44/74, loss=21.575453, lr=1.2e-05, time_each_step=0.33s, eta=0:21:57
    2021-08-11 16:51:44 [INFO]	[TRAIN] Epoch=226/270, Step=46/74, loss=6.503709, lr=1.2e-05, time_each_step=0.31s, eta=0:21:56
    2021-08-11 16:51:44 [INFO]	[TRAIN] Epoch=226/270, Step=48/74, loss=22.111179, lr=1.2e-05, time_each_step=0.3s, eta=0:21:55
    2021-08-11 16:51:45 [INFO]	[TRAIN] Epoch=226/270, Step=50/74, loss=20.581169, lr=1.2e-05, time_each_step=0.29s, eta=0:21:54
    2021-08-11 16:51:45 [INFO]	[TRAIN] Epoch=226/270, Step=52/74, loss=16.801842, lr=1.2e-05, time_each_step=0.26s, eta=0:21:53
    2021-08-11 16:51:45 [INFO]	[TRAIN] Epoch=226/270, Step=54/74, loss=12.610758, lr=1.2e-05, time_each_step=0.26s, eta=0:21:52
    2021-08-11 16:51:46 [INFO]	[TRAIN] Epoch=226/270, Step=56/74, loss=23.515499, lr=1.2e-05, time_each_step=0.24s, eta=0:21:52
    2021-08-11 16:51:46 [INFO]	[TRAIN] Epoch=226/270, Step=58/74, loss=17.26656, lr=1.2e-05, time_each_step=0.23s, eta=0:21:51
    2021-08-11 16:51:47 [INFO]	[TRAIN] Epoch=226/270, Step=60/74, loss=22.009789, lr=1.2e-05, time_each_step=0.21s, eta=0:21:50
    2021-08-11 16:51:47 [INFO]	[TRAIN] Epoch=226/270, Step=62/74, loss=14.167645, lr=1.2e-05, time_each_step=0.2s, eta=0:21:50
    2021-08-11 16:51:47 [INFO]	[TRAIN] Epoch=226/270, Step=64/74, loss=22.598637, lr=1.2e-05, time_each_step=0.2s, eta=0:21:49
    2021-08-11 16:51:48 [INFO]	[TRAIN] Epoch=226/270, Step=66/74, loss=18.39142, lr=1.2e-05, time_each_step=0.2s, eta=0:21:49
    2021-08-11 16:51:48 [INFO]	[TRAIN] Epoch=226/270, Step=68/74, loss=20.728592, lr=1.2e-05, time_each_step=0.19s, eta=0:21:49
    2021-08-11 16:51:48 [INFO]	[TRAIN] Epoch=226/270, Step=70/74, loss=10.23965, lr=1.2e-05, time_each_step=0.18s, eta=0:21:48
    2021-08-11 16:51:49 [INFO]	[TRAIN] Epoch=226/270, Step=72/74, loss=15.850523, lr=1.2e-05, time_each_step=0.2s, eta=0:21:48
    2021-08-11 16:51:49 [INFO]	[TRAIN] Epoch=226/270, Step=74/74, loss=22.881638, lr=1.2e-05, time_each_step=0.2s, eta=0:21:47
    2021-08-11 16:51:49 [INFO]	[TRAIN] Epoch 226 finished, loss=19.156479, lr=1.2e-05 .
    2021-08-11 16:52:09 [INFO]	[TRAIN] Epoch=227/270, Step=2/74, loss=39.416161, lr=1.2e-05, time_each_step=1.18s, eta=0:22:21
    2021-08-11 16:52:10 [INFO]	[TRAIN] Epoch=227/270, Step=4/74, loss=10.541767, lr=1.2e-05, time_each_step=1.19s, eta=0:22:20
    2021-08-11 16:52:11 [INFO]	[TRAIN] Epoch=227/270, Step=6/74, loss=11.695128, lr=1.2e-05, time_each_step=1.23s, eta=0:22:20
    2021-08-11 16:52:12 [INFO]	[TRAIN] Epoch=227/270, Step=8/74, loss=16.431877, lr=1.2e-05, time_each_step=1.27s, eta=0:22:20
    2021-08-11 16:52:13 [INFO]	[TRAIN] Epoch=227/270, Step=10/74, loss=26.241016, lr=1.2e-05, time_each_step=1.29s, eta=0:22:18
    2021-08-11 16:52:14 [INFO]	[TRAIN] Epoch=227/270, Step=12/74, loss=13.931021, lr=1.2e-05, time_each_step=1.31s, eta=0:22:18
    2021-08-11 16:52:15 [INFO]	[TRAIN] Epoch=227/270, Step=14/74, loss=29.166046, lr=1.2e-05, time_each_step=1.36s, eta=0:22:18
    2021-08-11 16:52:16 [INFO]	[TRAIN] Epoch=227/270, Step=16/74, loss=13.099977, lr=1.2e-05, time_each_step=1.38s, eta=0:22:16
    2021-08-11 16:52:17 [INFO]	[TRAIN] Epoch=227/270, Step=18/74, loss=16.631184, lr=1.2e-05, time_each_step=1.39s, eta=0:22:14
    2021-08-11 16:52:18 [INFO]	[TRAIN] Epoch=227/270, Step=20/74, loss=19.53072, lr=1.2e-05, time_each_step=1.42s, eta=0:22:13
    2021-08-11 16:52:19 [INFO]	[TRAIN] Epoch=227/270, Step=22/74, loss=17.013287, lr=1.2e-05, time_each_step=0.46s, eta=0:21:20
    2021-08-11 16:52:20 [INFO]	[TRAIN] Epoch=227/270, Step=24/74, loss=37.307392, lr=1.2e-05, time_each_step=0.47s, eta=0:21:20
    2021-08-11 16:52:20 [INFO]	[TRAIN] Epoch=227/270, Step=26/74, loss=14.321753, lr=1.2e-05, time_each_step=0.47s, eta=0:21:19
    2021-08-11 16:52:21 [INFO]	[TRAIN] Epoch=227/270, Step=28/74, loss=26.909885, lr=1.2e-05, time_each_step=0.45s, eta=0:21:17
    2021-08-11 16:52:22 [INFO]	[TRAIN] Epoch=227/270, Step=30/74, loss=6.14201, lr=1.2e-05, time_each_step=0.44s, eta=0:21:16
    2021-08-11 16:52:23 [INFO]	[TRAIN] Epoch=227/270, Step=32/74, loss=21.022766, lr=1.2e-05, time_each_step=0.44s, eta=0:21:15
    2021-08-11 16:52:24 [INFO]	[TRAIN] Epoch=227/270, Step=34/74, loss=21.361643, lr=1.2e-05, time_each_step=0.43s, eta=0:21:13
    2021-08-11 16:52:24 [INFO]	[TRAIN] Epoch=227/270, Step=36/74, loss=13.911116, lr=1.2e-05, time_each_step=0.43s, eta=0:21:13
    2021-08-11 16:52:25 [INFO]	[TRAIN] Epoch=227/270, Step=38/74, loss=14.680681, lr=1.2e-05, time_each_step=0.42s, eta=0:21:11
    2021-08-11 16:52:26 [INFO]	[TRAIN] Epoch=227/270, Step=40/74, loss=27.977194, lr=1.2e-05, time_each_step=0.4s, eta=0:21:10
    2021-08-11 16:52:26 [INFO]	[TRAIN] Epoch=227/270, Step=42/74, loss=14.853185, lr=1.2e-05, time_each_step=0.38s, eta=0:21:9
    2021-08-11 16:52:27 [INFO]	[TRAIN] Epoch=227/270, Step=44/74, loss=19.302116, lr=1.2e-05, time_each_step=0.36s, eta=0:21:7
    2021-08-11 16:52:27 [INFO]	[TRAIN] Epoch=227/270, Step=46/74, loss=15.075768, lr=1.2e-05, time_each_step=0.33s, eta=0:21:5
    2021-08-11 16:52:28 [INFO]	[TRAIN] Epoch=227/270, Step=48/74, loss=27.38928, lr=1.2e-05, time_each_step=0.34s, eta=0:21:5
    2021-08-11 16:52:28 [INFO]	[TRAIN] Epoch=227/270, Step=50/74, loss=19.749687, lr=1.2e-05, time_each_step=0.32s, eta=0:21:4
    2021-08-11 16:52:29 [INFO]	[TRAIN] Epoch=227/270, Step=52/74, loss=23.84601, lr=1.2e-05, time_each_step=0.29s, eta=0:21:3
    2021-08-11 16:52:29 [INFO]	[TRAIN] Epoch=227/270, Step=54/74, loss=13.902782, lr=1.2e-05, time_each_step=0.27s, eta=0:21:2
    2021-08-11 16:52:29 [INFO]	[TRAIN] Epoch=227/270, Step=56/74, loss=25.986734, lr=1.2e-05, time_each_step=0.24s, eta=0:21:1
    2021-08-11 16:52:30 [INFO]	[TRAIN] Epoch=227/270, Step=58/74, loss=17.586426, lr=1.2e-05, time_each_step=0.23s, eta=0:21:0
    2021-08-11 16:52:30 [INFO]	[TRAIN] Epoch=227/270, Step=60/74, loss=20.987194, lr=1.2e-05, time_each_step=0.22s, eta=0:20:59
    2021-08-11 16:52:31 [INFO]	[TRAIN] Epoch=227/270, Step=62/74, loss=22.99728, lr=1.2e-05, time_each_step=0.22s, eta=0:20:59
    2021-08-11 16:52:31 [INFO]	[TRAIN] Epoch=227/270, Step=64/74, loss=19.480303, lr=1.2e-05, time_each_step=0.22s, eta=0:20:58
    2021-08-11 16:52:32 [INFO]	[TRAIN] Epoch=227/270, Step=66/74, loss=8.672686, lr=1.2e-05, time_each_step=0.23s, eta=0:20:58
    2021-08-11 16:52:32 [INFO]	[TRAIN] Epoch=227/270, Step=68/74, loss=14.824268, lr=1.2e-05, time_each_step=0.21s, eta=0:20:57
    2021-08-11 16:52:32 [INFO]	[TRAIN] Epoch=227/270, Step=70/74, loss=25.005447, lr=1.2e-05, time_each_step=0.22s, eta=0:20:57
    2021-08-11 16:52:33 [INFO]	[TRAIN] Epoch=227/270, Step=72/74, loss=15.204368, lr=1.2e-05, time_each_step=0.22s, eta=0:20:57
    2021-08-11 16:52:33 [INFO]	[TRAIN] Epoch=227/270, Step=74/74, loss=19.683655, lr=1.2e-05, time_each_step=0.21s, eta=0:20:56
    2021-08-11 16:52:33 [INFO]	[TRAIN] Epoch 227 finished, loss=18.691172, lr=1.2e-05 .
    2021-08-11 16:52:52 [INFO]	[TRAIN] Epoch=228/270, Step=2/74, loss=12.623032, lr=1.2e-05, time_each_step=1.11s, eta=0:33:31
    2021-08-11 16:52:52 [INFO]	[TRAIN] Epoch=228/270, Step=4/74, loss=23.143595, lr=1.2e-05, time_each_step=1.14s, eta=0:33:31
    2021-08-11 16:52:53 [INFO]	[TRAIN] Epoch=228/270, Step=6/74, loss=14.975276, lr=1.2e-05, time_each_step=1.13s, eta=0:33:28
    2021-08-11 16:52:55 [INFO]	[TRAIN] Epoch=228/270, Step=8/74, loss=17.919811, lr=1.2e-05, time_each_step=1.19s, eta=0:33:30
    2021-08-11 16:52:55 [INFO]	[TRAIN] Epoch=228/270, Step=10/74, loss=11.056935, lr=1.2e-05, time_each_step=1.2s, eta=0:33:28
    2021-08-11 16:52:56 [INFO]	[TRAIN] Epoch=228/270, Step=12/74, loss=13.156393, lr=1.2e-05, time_each_step=1.23s, eta=0:33:27
    2021-08-11 16:52:57 [INFO]	[TRAIN] Epoch=228/270, Step=14/74, loss=22.928368, lr=1.2e-05, time_each_step=1.24s, eta=0:33:26
    2021-08-11 16:52:57 [INFO]	[TRAIN] Epoch=228/270, Step=16/74, loss=13.661384, lr=1.2e-05, time_each_step=1.25s, eta=0:33:24
    2021-08-11 16:52:58 [INFO]	[TRAIN] Epoch=228/270, Step=18/74, loss=13.221783, lr=1.2e-05, time_each_step=1.26s, eta=0:33:22
    2021-08-11 16:52:59 [INFO]	[TRAIN] Epoch=228/270, Step=20/74, loss=8.093323, lr=1.2e-05, time_each_step=1.27s, eta=0:33:20
    2021-08-11 16:52:59 [INFO]	[TRAIN] Epoch=228/270, Step=22/74, loss=43.030006, lr=1.2e-05, time_each_step=0.39s, eta=0:32:31
    2021-08-11 16:53:00 [INFO]	[TRAIN] Epoch=228/270, Step=24/74, loss=24.44734, lr=1.2e-05, time_each_step=0.38s, eta=0:32:30
    2021-08-11 16:53:01 [INFO]	[TRAIN] Epoch=228/270, Step=26/74, loss=9.881692, lr=1.2e-05, time_each_step=0.4s, eta=0:32:30
    2021-08-11 16:53:02 [INFO]	[TRAIN] Epoch=228/270, Step=28/74, loss=23.762966, lr=1.2e-05, time_each_step=0.36s, eta=0:32:28
    2021-08-11 16:53:02 [INFO]	[TRAIN] Epoch=228/270, Step=30/74, loss=9.971821, lr=1.2e-05, time_each_step=0.36s, eta=0:32:27
    2021-08-11 16:53:03 [INFO]	[TRAIN] Epoch=228/270, Step=32/74, loss=15.874766, lr=1.2e-05, time_each_step=0.36s, eta=0:32:26
    2021-08-11 16:53:04 [INFO]	[TRAIN] Epoch=228/270, Step=34/74, loss=29.865515, lr=1.2e-05, time_each_step=0.36s, eta=0:32:25
    2021-08-11 16:53:05 [INFO]	[TRAIN] Epoch=228/270, Step=36/74, loss=15.515409, lr=1.2e-05, time_each_step=0.37s, eta=0:32:25
    2021-08-11 16:53:06 [INFO]	[TRAIN] Epoch=228/270, Step=38/74, loss=10.696683, lr=1.2e-05, time_each_step=0.38s, eta=0:32:25
    2021-08-11 16:53:06 [INFO]	[TRAIN] Epoch=228/270, Step=40/74, loss=48.464748, lr=1.2e-05, time_each_step=0.38s, eta=0:32:24
    2021-08-11 16:53:07 [INFO]	[TRAIN] Epoch=228/270, Step=42/74, loss=20.0089, lr=1.2e-05, time_each_step=0.38s, eta=0:32:24
    2021-08-11 16:53:08 [INFO]	[TRAIN] Epoch=228/270, Step=44/74, loss=12.676743, lr=1.2e-05, time_each_step=0.38s, eta=0:32:23
    2021-08-11 16:53:08 [INFO]	[TRAIN] Epoch=228/270, Step=46/74, loss=20.363779, lr=1.2e-05, time_each_step=0.38s, eta=0:32:22
    2021-08-11 16:53:09 [INFO]	[TRAIN] Epoch=228/270, Step=48/74, loss=17.995384, lr=1.2e-05, time_each_step=0.36s, eta=0:32:21
    2021-08-11 16:53:09 [INFO]	[TRAIN] Epoch=228/270, Step=50/74, loss=15.474464, lr=1.2e-05, time_each_step=0.35s, eta=0:32:20
    2021-08-11 16:53:10 [INFO]	[TRAIN] Epoch=228/270, Step=52/74, loss=12.955202, lr=1.2e-05, time_each_step=0.32s, eta=0:32:18
    2021-08-11 16:53:10 [INFO]	[TRAIN] Epoch=228/270, Step=54/74, loss=18.893835, lr=1.2e-05, time_each_step=0.32s, eta=0:32:18
    2021-08-11 16:53:11 [INFO]	[TRAIN] Epoch=228/270, Step=56/74, loss=22.350952, lr=1.2e-05, time_each_step=0.29s, eta=0:32:17
    2021-08-11 16:53:11 [INFO]	[TRAIN] Epoch=228/270, Step=58/74, loss=16.512312, lr=1.2e-05, time_each_step=0.27s, eta=0:32:16
    2021-08-11 16:53:11 [INFO]	[TRAIN] Epoch=228/270, Step=60/74, loss=7.429132, lr=1.2e-05, time_each_step=0.25s, eta=0:32:15
    2021-08-11 16:53:12 [INFO]	[TRAIN] Epoch=228/270, Step=62/74, loss=13.616375, lr=1.2e-05, time_each_step=0.23s, eta=0:32:14
    2021-08-11 16:53:12 [INFO]	[TRAIN] Epoch=228/270, Step=64/74, loss=18.587151, lr=1.2e-05, time_each_step=0.22s, eta=0:32:13
    2021-08-11 16:53:12 [INFO]	[TRAIN] Epoch=228/270, Step=66/74, loss=40.437164, lr=1.2e-05, time_each_step=0.2s, eta=0:32:13
    2021-08-11 16:53:13 [INFO]	[TRAIN] Epoch=228/270, Step=68/74, loss=11.176457, lr=1.2e-05, time_each_step=0.2s, eta=0:32:12
    2021-08-11 16:53:13 [INFO]	[TRAIN] Epoch=228/270, Step=70/74, loss=18.216022, lr=1.2e-05, time_each_step=0.19s, eta=0:32:12
    2021-08-11 16:53:14 [INFO]	[TRAIN] Epoch=228/270, Step=72/74, loss=11.464664, lr=1.2e-05, time_each_step=0.19s, eta=0:32:12
    2021-08-11 16:53:14 [INFO]	[TRAIN] Epoch=228/270, Step=74/74, loss=13.361542, lr=1.2e-05, time_each_step=0.18s, eta=0:32:11
    2021-08-11 16:53:14 [INFO]	[TRAIN] Epoch 228 finished, loss=18.460573, lr=1.2e-05 .
    2021-08-11 16:53:32 [INFO]	[TRAIN] Epoch=229/270, Step=2/74, loss=14.322053, lr=1.2e-05, time_each_step=1.07s, eta=0:30:35
    2021-08-11 16:53:33 [INFO]	[TRAIN] Epoch=229/270, Step=4/74, loss=36.45002, lr=1.2e-05, time_each_step=1.1s, eta=0:30:35
    2021-08-11 16:53:34 [INFO]	[TRAIN] Epoch=229/270, Step=6/74, loss=10.689413, lr=1.2e-05, time_each_step=1.13s, eta=0:30:35
    2021-08-11 16:53:35 [INFO]	[TRAIN] Epoch=229/270, Step=8/74, loss=12.990197, lr=1.2e-05, time_each_step=1.16s, eta=0:30:34
    2021-08-11 16:53:35 [INFO]	[TRAIN] Epoch=229/270, Step=10/74, loss=16.075466, lr=1.2e-05, time_each_step=1.15s, eta=0:30:32
    2021-08-11 16:53:36 [INFO]	[TRAIN] Epoch=229/270, Step=12/74, loss=53.036957, lr=1.2e-05, time_each_step=1.18s, eta=0:30:32
    2021-08-11 16:53:37 [INFO]	[TRAIN] Epoch=229/270, Step=14/74, loss=16.012466, lr=1.2e-05, time_each_step=1.2s, eta=0:30:30
    2021-08-11 16:53:38 [INFO]	[TRAIN] Epoch=229/270, Step=16/74, loss=13.643591, lr=1.2e-05, time_each_step=1.23s, eta=0:30:29
    2021-08-11 16:53:39 [INFO]	[TRAIN] Epoch=229/270, Step=18/74, loss=13.548014, lr=1.2e-05, time_each_step=1.25s, eta=0:30:28
    2021-08-11 16:53:39 [INFO]	[TRAIN] Epoch=229/270, Step=20/74, loss=23.220364, lr=1.2e-05, time_each_step=1.27s, eta=0:30:26
    2021-08-11 16:53:40 [INFO]	[TRAIN] Epoch=229/270, Step=22/74, loss=25.989721, lr=1.2e-05, time_each_step=0.4s, eta=0:29:39
    2021-08-11 16:53:41 [INFO]	[TRAIN] Epoch=229/270, Step=24/74, loss=18.776035, lr=1.2e-05, time_each_step=0.4s, eta=0:29:38
    2021-08-11 16:53:42 [INFO]	[TRAIN] Epoch=229/270, Step=26/74, loss=41.583843, lr=1.2e-05, time_each_step=0.41s, eta=0:29:38
    2021-08-11 16:53:43 [INFO]	[TRAIN] Epoch=229/270, Step=28/74, loss=9.144716, lr=1.2e-05, time_each_step=0.41s, eta=0:29:37
    2021-08-11 16:53:44 [INFO]	[TRAIN] Epoch=229/270, Step=30/74, loss=17.480467, lr=1.2e-05, time_each_step=0.43s, eta=0:29:37
    2021-08-11 16:53:44 [INFO]	[TRAIN] Epoch=229/270, Step=32/74, loss=15.337275, lr=1.2e-05, time_each_step=0.41s, eta=0:29:35
    2021-08-11 16:53:45 [INFO]	[TRAIN] Epoch=229/270, Step=34/74, loss=10.271612, lr=1.2e-05, time_each_step=0.41s, eta=0:29:34
    2021-08-11 16:53:46 [INFO]	[TRAIN] Epoch=229/270, Step=36/74, loss=16.350565, lr=1.2e-05, time_each_step=0.39s, eta=0:29:33
    2021-08-11 16:53:46 [INFO]	[TRAIN] Epoch=229/270, Step=38/74, loss=7.848026, lr=1.2e-05, time_each_step=0.38s, eta=0:29:32
    2021-08-11 16:53:47 [INFO]	[TRAIN] Epoch=229/270, Step=40/74, loss=15.014198, lr=1.2e-05, time_each_step=0.41s, eta=0:29:32
    2021-08-11 16:53:48 [INFO]	[TRAIN] Epoch=229/270, Step=42/74, loss=16.739927, lr=1.2e-05, time_each_step=0.4s, eta=0:29:31
    2021-08-11 16:53:49 [INFO]	[TRAIN] Epoch=229/270, Step=44/74, loss=27.063419, lr=1.2e-05, time_each_step=0.39s, eta=0:29:30
    2021-08-11 16:53:49 [INFO]	[TRAIN] Epoch=229/270, Step=46/74, loss=12.261404, lr=1.2e-05, time_each_step=0.36s, eta=0:29:28
    2021-08-11 16:53:50 [INFO]	[TRAIN] Epoch=229/270, Step=48/74, loss=18.390511, lr=1.2e-05, time_each_step=0.34s, eta=0:29:27
    2021-08-11 16:53:50 [INFO]	[TRAIN] Epoch=229/270, Step=50/74, loss=12.961023, lr=1.2e-05, time_each_step=0.31s, eta=0:29:26
    2021-08-11 16:53:50 [INFO]	[TRAIN] Epoch=229/270, Step=52/74, loss=29.875275, lr=1.2e-05, time_each_step=0.31s, eta=0:29:25
    2021-08-11 16:53:51 [INFO]	[TRAIN] Epoch=229/270, Step=54/74, loss=31.434233, lr=1.2e-05, time_each_step=0.3s, eta=0:29:24
    2021-08-11 16:53:51 [INFO]	[TRAIN] Epoch=229/270, Step=56/74, loss=11.338782, lr=1.2e-05, time_each_step=0.28s, eta=0:29:23
    2021-08-11 16:53:52 [INFO]	[TRAIN] Epoch=229/270, Step=58/74, loss=15.733812, lr=1.2e-05, time_each_step=0.26s, eta=0:29:22
    2021-08-11 16:53:52 [INFO]	[TRAIN] Epoch=229/270, Step=60/74, loss=9.252902, lr=1.2e-05, time_each_step=0.21s, eta=0:29:21
    2021-08-11 16:53:52 [INFO]	[TRAIN] Epoch=229/270, Step=62/74, loss=13.260901, lr=1.2e-05, time_each_step=0.2s, eta=0:29:21
    2021-08-11 16:53:53 [INFO]	[TRAIN] Epoch=229/270, Step=64/74, loss=21.521935, lr=1.2e-05, time_each_step=0.19s, eta=0:29:20
    2021-08-11 16:53:53 [INFO]	[TRAIN] Epoch=229/270, Step=66/74, loss=33.188896, lr=1.2e-05, time_each_step=0.19s, eta=0:29:20
    2021-08-11 16:53:53 [INFO]	[TRAIN] Epoch=229/270, Step=68/74, loss=9.057443, lr=1.2e-05, time_each_step=0.18s, eta=0:29:19
    2021-08-11 16:53:54 [INFO]	[TRAIN] Epoch=229/270, Step=70/74, loss=11.147206, lr=1.2e-05, time_each_step=0.19s, eta=0:29:19
    2021-08-11 16:53:54 [INFO]	[TRAIN] Epoch=229/270, Step=72/74, loss=22.581175, lr=1.2e-05, time_each_step=0.18s, eta=0:29:19
    2021-08-11 16:53:54 [INFO]	[TRAIN] Epoch=229/270, Step=74/74, loss=12.518712, lr=1.2e-05, time_each_step=0.18s, eta=0:29:18
    2021-08-11 16:53:54 [INFO]	[TRAIN] Epoch 229 finished, loss=18.63998, lr=1.2e-05 .
    2021-08-11 16:54:13 [INFO]	[TRAIN] Epoch=230/270, Step=2/74, loss=26.462992, lr=1.2e-05, time_each_step=1.12s, eta=0:29:47
    2021-08-11 16:54:14 [INFO]	[TRAIN] Epoch=230/270, Step=4/74, loss=18.878857, lr=1.2e-05, time_each_step=1.13s, eta=0:29:46
    2021-08-11 16:54:15 [INFO]	[TRAIN] Epoch=230/270, Step=6/74, loss=15.481678, lr=1.2e-05, time_each_step=1.16s, eta=0:29:46
    2021-08-11 16:54:16 [INFO]	[TRAIN] Epoch=230/270, Step=8/74, loss=24.019238, lr=1.2e-05, time_each_step=1.18s, eta=0:29:44
    2021-08-11 16:54:16 [INFO]	[TRAIN] Epoch=230/270, Step=10/74, loss=12.987392, lr=1.2e-05, time_each_step=1.19s, eta=0:29:43
    2021-08-11 16:54:17 [INFO]	[TRAIN] Epoch=230/270, Step=12/74, loss=7.414762, lr=1.2e-05, time_each_step=1.21s, eta=0:29:42
    2021-08-11 16:54:18 [INFO]	[TRAIN] Epoch=230/270, Step=14/74, loss=13.678523, lr=1.2e-05, time_each_step=1.25s, eta=0:29:41
    2021-08-11 16:54:19 [INFO]	[TRAIN] Epoch=230/270, Step=16/74, loss=22.357454, lr=1.2e-05, time_each_step=1.28s, eta=0:29:41
    2021-08-11 16:54:20 [INFO]	[TRAIN] Epoch=230/270, Step=18/74, loss=25.561256, lr=1.2e-05, time_each_step=1.28s, eta=0:29:38
    2021-08-11 16:54:21 [INFO]	[TRAIN] Epoch=230/270, Step=20/74, loss=18.088982, lr=1.2e-05, time_each_step=1.32s, eta=0:29:38
    2021-08-11 16:54:22 [INFO]	[TRAIN] Epoch=230/270, Step=22/74, loss=10.814209, lr=1.2e-05, time_each_step=0.41s, eta=0:28:48
    2021-08-11 16:54:23 [INFO]	[TRAIN] Epoch=230/270, Step=24/74, loss=15.916914, lr=1.2e-05, time_each_step=0.43s, eta=0:28:48
    2021-08-11 16:54:24 [INFO]	[TRAIN] Epoch=230/270, Step=26/74, loss=10.706078, lr=1.2e-05, time_each_step=0.43s, eta=0:28:47
    2021-08-11 16:54:25 [INFO]	[TRAIN] Epoch=230/270, Step=28/74, loss=20.646261, lr=1.2e-05, time_each_step=0.45s, eta=0:28:47
    2021-08-11 16:54:25 [INFO]	[TRAIN] Epoch=230/270, Step=30/74, loss=16.657631, lr=1.2e-05, time_each_step=0.44s, eta=0:28:46
    2021-08-11 16:54:26 [INFO]	[TRAIN] Epoch=230/270, Step=32/74, loss=13.662464, lr=1.2e-05, time_each_step=0.43s, eta=0:28:45
    2021-08-11 16:54:27 [INFO]	[TRAIN] Epoch=230/270, Step=34/74, loss=23.856247, lr=1.2e-05, time_each_step=0.41s, eta=0:28:43
    2021-08-11 16:54:28 [INFO]	[TRAIN] Epoch=230/270, Step=36/74, loss=37.832535, lr=1.2e-05, time_each_step=0.41s, eta=0:28:42
    2021-08-11 16:54:28 [INFO]	[TRAIN] Epoch=230/270, Step=38/74, loss=28.039242, lr=1.2e-05, time_each_step=0.44s, eta=0:28:42
    2021-08-11 16:54:29 [INFO]	[TRAIN] Epoch=230/270, Step=40/74, loss=20.525669, lr=1.2e-05, time_each_step=0.43s, eta=0:28:41
    2021-08-11 16:54:30 [INFO]	[TRAIN] Epoch=230/270, Step=42/74, loss=12.934464, lr=1.2e-05, time_each_step=0.41s, eta=0:28:40
    2021-08-11 16:54:30 [INFO]	[TRAIN] Epoch=230/270, Step=44/74, loss=8.878681, lr=1.2e-05, time_each_step=0.37s, eta=0:28:38
    2021-08-11 16:54:31 [INFO]	[TRAIN] Epoch=230/270, Step=46/74, loss=17.985691, lr=1.2e-05, time_each_step=0.35s, eta=0:28:36
    2021-08-11 16:54:31 [INFO]	[TRAIN] Epoch=230/270, Step=48/74, loss=16.589825, lr=1.2e-05, time_each_step=0.31s, eta=0:28:35
    2021-08-11 16:54:31 [INFO]	[TRAIN] Epoch=230/270, Step=50/74, loss=9.504726, lr=1.2e-05, time_each_step=0.31s, eta=0:28:34
    2021-08-11 16:54:32 [INFO]	[TRAIN] Epoch=230/270, Step=52/74, loss=13.359705, lr=1.2e-05, time_each_step=0.29s, eta=0:28:33
    2021-08-11 16:54:32 [INFO]	[TRAIN] Epoch=230/270, Step=54/74, loss=17.79944, lr=1.2e-05, time_each_step=0.29s, eta=0:28:32
    2021-08-11 16:54:33 [INFO]	[TRAIN] Epoch=230/270, Step=56/74, loss=13.805545, lr=1.2e-05, time_each_step=0.26s, eta=0:28:31
    2021-08-11 16:54:33 [INFO]	[TRAIN] Epoch=230/270, Step=58/74, loss=8.91696, lr=1.2e-05, time_each_step=0.23s, eta=0:28:30
    2021-08-11 16:54:33 [INFO]	[TRAIN] Epoch=230/270, Step=60/74, loss=15.499346, lr=1.2e-05, time_each_step=0.21s, eta=0:28:30
    2021-08-11 16:54:34 [INFO]	[TRAIN] Epoch=230/270, Step=62/74, loss=12.120895, lr=1.2e-05, time_each_step=0.2s, eta=0:28:29
    2021-08-11 16:54:34 [INFO]	[TRAIN] Epoch=230/270, Step=64/74, loss=13.779954, lr=1.2e-05, time_each_step=0.21s, eta=0:28:29
    2021-08-11 16:54:35 [INFO]	[TRAIN] Epoch=230/270, Step=66/74, loss=17.26186, lr=1.2e-05, time_each_step=0.22s, eta=0:28:28
    2021-08-11 16:54:36 [INFO]	[TRAIN] Epoch=230/270, Step=68/74, loss=9.658265, lr=1.2e-05, time_each_step=0.23s, eta=0:28:28
    2021-08-11 16:54:36 [INFO]	[TRAIN] Epoch=230/270, Step=70/74, loss=18.065702, lr=1.2e-05, time_each_step=0.22s, eta=0:28:28
    2021-08-11 16:54:36 [INFO]	[TRAIN] Epoch=230/270, Step=72/74, loss=17.74416, lr=1.2e-05, time_each_step=0.23s, eta=0:28:27
    2021-08-11 16:54:37 [INFO]	[TRAIN] Epoch=230/270, Step=74/74, loss=27.784727, lr=1.2e-05, time_each_step=0.22s, eta=0:28:27
    2021-08-11 16:54:37 [INFO]	[TRAIN] Epoch 230 finished, loss=17.271328, lr=1.2e-05 .
    2021-08-11 16:54:42 [INFO]	[TRAIN] Epoch=231/270, Step=2/74, loss=16.040705, lr=1.2e-05, time_each_step=0.45s, eta=0:29:33
    2021-08-11 16:54:42 [INFO]	[TRAIN] Epoch=231/270, Step=4/74, loss=19.664082, lr=1.2e-05, time_each_step=0.47s, eta=0:29:33
    2021-08-11 16:54:43 [INFO]	[TRAIN] Epoch=231/270, Step=6/74, loss=43.275909, lr=1.2e-05, time_each_step=0.5s, eta=0:29:34
    2021-08-11 16:54:44 [INFO]	[TRAIN] Epoch=231/270, Step=8/74, loss=15.879484, lr=1.2e-05, time_each_step=0.52s, eta=0:29:35
    2021-08-11 16:54:45 [INFO]	[TRAIN] Epoch=231/270, Step=10/74, loss=17.03397, lr=1.2e-05, time_each_step=0.54s, eta=0:29:35
    2021-08-11 16:54:46 [INFO]	[TRAIN] Epoch=231/270, Step=12/74, loss=20.48505, lr=1.2e-05, time_each_step=0.54s, eta=0:29:34
    2021-08-11 16:54:46 [INFO]	[TRAIN] Epoch=231/270, Step=14/74, loss=35.776649, lr=1.2e-05, time_each_step=0.54s, eta=0:29:33
    2021-08-11 16:54:47 [INFO]	[TRAIN] Epoch=231/270, Step=16/74, loss=12.032011, lr=1.2e-05, time_each_step=0.55s, eta=0:29:32
    2021-08-11 16:54:48 [INFO]	[TRAIN] Epoch=231/270, Step=18/74, loss=10.796225, lr=1.2e-05, time_each_step=0.58s, eta=0:29:33
    2021-08-11 16:54:49 [INFO]	[TRAIN] Epoch=231/270, Step=20/74, loss=15.124643, lr=1.2e-05, time_each_step=0.6s, eta=0:29:33
    2021-08-11 16:54:50 [INFO]	[TRAIN] Epoch=231/270, Step=22/74, loss=22.464909, lr=1.2e-05, time_each_step=0.4s, eta=0:29:21
    2021-08-11 16:54:51 [INFO]	[TRAIN] Epoch=231/270, Step=24/74, loss=22.321114, lr=1.2e-05, time_each_step=0.42s, eta=0:29:21
    2021-08-11 16:54:52 [INFO]	[TRAIN] Epoch=231/270, Step=26/74, loss=14.856009, lr=1.2e-05, time_each_step=0.41s, eta=0:29:20
    2021-08-11 16:54:52 [INFO]	[TRAIN] Epoch=231/270, Step=28/74, loss=9.544424, lr=1.2e-05, time_each_step=0.39s, eta=0:29:18
    2021-08-11 16:54:52 [INFO]	[TRAIN] Epoch=231/270, Step=30/74, loss=15.808888, lr=1.2e-05, time_each_step=0.37s, eta=0:29:17
    2021-08-11 16:54:53 [INFO]	[TRAIN] Epoch=231/270, Step=32/74, loss=15.677933, lr=1.2e-05, time_each_step=0.36s, eta=0:29:15
    2021-08-11 16:54:54 [INFO]	[TRAIN] Epoch=231/270, Step=34/74, loss=17.114399, lr=1.2e-05, time_each_step=0.37s, eta=0:29:15
    2021-08-11 16:54:54 [INFO]	[TRAIN] Epoch=231/270, Step=36/74, loss=16.243195, lr=1.2e-05, time_each_step=0.38s, eta=0:29:15
    2021-08-11 16:54:55 [INFO]	[TRAIN] Epoch=231/270, Step=38/74, loss=35.458439, lr=1.2e-05, time_each_step=0.38s, eta=0:29:14
    2021-08-11 16:54:56 [INFO]	[TRAIN] Epoch=231/270, Step=40/74, loss=14.065101, lr=1.2e-05, time_each_step=0.36s, eta=0:29:13
    2021-08-11 16:54:57 [INFO]	[TRAIN] Epoch=231/270, Step=42/74, loss=26.358315, lr=1.2e-05, time_each_step=0.35s, eta=0:29:12
    2021-08-11 16:54:57 [INFO]	[TRAIN] Epoch=231/270, Step=44/74, loss=22.228235, lr=1.2e-05, time_each_step=0.33s, eta=0:29:10
    2021-08-11 16:54:58 [INFO]	[TRAIN] Epoch=231/270, Step=46/74, loss=30.454334, lr=1.2e-05, time_each_step=0.32s, eta=0:29:9
    2021-08-11 16:54:59 [INFO]	[TRAIN] Epoch=231/270, Step=48/74, loss=8.154417, lr=1.2e-05, time_each_step=0.32s, eta=0:29:9
    2021-08-11 16:54:59 [INFO]	[TRAIN] Epoch=231/270, Step=50/74, loss=24.384163, lr=1.2e-05, time_each_step=0.33s, eta=0:29:8
    2021-08-11 16:55:00 [INFO]	[TRAIN] Epoch=231/270, Step=52/74, loss=15.721313, lr=1.2e-05, time_each_step=0.34s, eta=0:29:8
    2021-08-11 16:55:00 [INFO]	[TRAIN] Epoch=231/270, Step=54/74, loss=10.579548, lr=1.2e-05, time_each_step=0.32s, eta=0:29:7
    2021-08-11 16:55:01 [INFO]	[TRAIN] Epoch=231/270, Step=56/74, loss=6.002146, lr=1.2e-05, time_each_step=0.31s, eta=0:29:6
    2021-08-11 16:55:01 [INFO]	[TRAIN] Epoch=231/270, Step=58/74, loss=34.70768, lr=1.2e-05, time_each_step=0.29s, eta=0:29:5
    2021-08-11 16:55:01 [INFO]	[TRAIN] Epoch=231/270, Step=60/74, loss=18.71804, lr=1.2e-05, time_each_step=0.28s, eta=0:29:4
    2021-08-11 16:55:02 [INFO]	[TRAIN] Epoch=231/270, Step=62/74, loss=11.154879, lr=1.2e-05, time_each_step=0.25s, eta=0:29:3
    2021-08-11 16:55:02 [INFO]	[TRAIN] Epoch=231/270, Step=64/74, loss=9.101713, lr=1.2e-05, time_each_step=0.24s, eta=0:29:3
    2021-08-11 16:55:03 [INFO]	[TRAIN] Epoch=231/270, Step=66/74, loss=12.757042, lr=1.2e-05, time_each_step=0.24s, eta=0:29:2
    2021-08-11 16:55:03 [INFO]	[TRAIN] Epoch=231/270, Step=68/74, loss=17.605869, lr=1.2e-05, time_each_step=0.23s, eta=0:29:2
    2021-08-11 16:55:03 [INFO]	[TRAIN] Epoch=231/270, Step=70/74, loss=21.68425, lr=1.2e-05, time_each_step=0.22s, eta=0:29:1
    2021-08-11 16:55:04 [INFO]	[TRAIN] Epoch=231/270, Step=72/74, loss=15.126096, lr=1.2e-05, time_each_step=0.21s, eta=0:29:1
    2021-08-11 16:55:04 [INFO]	[TRAIN] Epoch=231/270, Step=74/74, loss=10.207505, lr=1.2e-05, time_each_step=0.21s, eta=0:29:0
    2021-08-11 16:55:04 [INFO]	[TRAIN] Epoch 231 finished, loss=17.91186, lr=1.2e-05 .
    2021-08-11 16:55:14 [INFO]	[TRAIN] Epoch=232/270, Step=2/74, loss=16.253139, lr=1.2e-05, time_each_step=0.69s, eta=0:19:49
    2021-08-11 16:55:15 [INFO]	[TRAIN] Epoch=232/270, Step=4/74, loss=13.647274, lr=1.2e-05, time_each_step=0.71s, eta=0:19:50
    2021-08-11 16:55:16 [INFO]	[TRAIN] Epoch=232/270, Step=6/74, loss=15.664081, lr=1.2e-05, time_each_step=0.74s, eta=0:19:50
    2021-08-11 16:55:17 [INFO]	[TRAIN] Epoch=232/270, Step=8/74, loss=28.40855, lr=1.2e-05, time_each_step=0.78s, eta=0:19:51
    2021-08-11 16:55:18 [INFO]	[TRAIN] Epoch=232/270, Step=10/74, loss=22.600538, lr=1.2e-05, time_each_step=0.79s, eta=0:19:50
    2021-08-11 16:55:19 [INFO]	[TRAIN] Epoch=232/270, Step=12/74, loss=7.519748, lr=1.2e-05, time_each_step=0.8s, eta=0:19:50
    2021-08-11 16:55:19 [INFO]	[TRAIN] Epoch=232/270, Step=14/74, loss=23.621761, lr=1.2e-05, time_each_step=0.81s, eta=0:19:48
    2021-08-11 16:55:20 [INFO]	[TRAIN] Epoch=232/270, Step=16/74, loss=34.276508, lr=1.2e-05, time_each_step=0.82s, eta=0:19:47
    2021-08-11 16:55:20 [INFO]	[TRAIN] Epoch=232/270, Step=18/74, loss=11.553493, lr=1.2e-05, time_each_step=0.81s, eta=0:19:45
    2021-08-11 16:55:21 [INFO]	[TRAIN] Epoch=232/270, Step=20/74, loss=17.828041, lr=1.2e-05, time_each_step=0.85s, eta=0:19:45
    2021-08-11 16:55:22 [INFO]	[TRAIN] Epoch=232/270, Step=22/74, loss=14.542041, lr=1.2e-05, time_each_step=0.39s, eta=0:19:20
    2021-08-11 16:55:23 [INFO]	[TRAIN] Epoch=232/270, Step=24/74, loss=20.461758, lr=1.2e-05, time_each_step=0.37s, eta=0:19:18
    2021-08-11 16:55:24 [INFO]	[TRAIN] Epoch=232/270, Step=26/74, loss=19.160461, lr=1.2e-05, time_each_step=0.38s, eta=0:19:18
    2021-08-11 16:55:25 [INFO]	[TRAIN] Epoch=232/270, Step=28/74, loss=14.407571, lr=1.2e-05, time_each_step=0.37s, eta=0:19:17
    2021-08-11 16:55:25 [INFO]	[TRAIN] Epoch=232/270, Step=30/74, loss=13.732161, lr=1.2e-05, time_each_step=0.36s, eta=0:19:16
    2021-08-11 16:55:26 [INFO]	[TRAIN] Epoch=232/270, Step=32/74, loss=22.275772, lr=1.2e-05, time_each_step=0.37s, eta=0:19:15
    2021-08-11 16:55:27 [INFO]	[TRAIN] Epoch=232/270, Step=34/74, loss=19.722218, lr=1.2e-05, time_each_step=0.39s, eta=0:19:15
    2021-08-11 16:55:28 [INFO]	[TRAIN] Epoch=232/270, Step=36/74, loss=14.074887, lr=1.2e-05, time_each_step=0.4s, eta=0:19:15
    2021-08-11 16:55:29 [INFO]	[TRAIN] Epoch=232/270, Step=38/74, loss=23.534763, lr=1.2e-05, time_each_step=0.42s, eta=0:19:15
    2021-08-11 16:55:29 [INFO]	[TRAIN] Epoch=232/270, Step=40/74, loss=11.908671, lr=1.2e-05, time_each_step=0.4s, eta=0:19:13
    2021-08-11 16:55:30 [INFO]	[TRAIN] Epoch=232/270, Step=42/74, loss=12.589185, lr=1.2e-05, time_each_step=0.39s, eta=0:19:12
    2021-08-11 16:55:30 [INFO]	[TRAIN] Epoch=232/270, Step=44/74, loss=15.618151, lr=1.2e-05, time_each_step=0.38s, eta=0:19:11
    2021-08-11 16:55:31 [INFO]	[TRAIN] Epoch=232/270, Step=46/74, loss=12.045186, lr=1.2e-05, time_each_step=0.35s, eta=0:19:10
    2021-08-11 16:55:32 [INFO]	[TRAIN] Epoch=232/270, Step=48/74, loss=12.354871, lr=1.2e-05, time_each_step=0.34s, eta=0:19:8
    2021-08-11 16:55:32 [INFO]	[TRAIN] Epoch=232/270, Step=50/74, loss=15.243501, lr=1.2e-05, time_each_step=0.33s, eta=0:19:8
    2021-08-11 16:55:32 [INFO]	[TRAIN] Epoch=232/270, Step=52/74, loss=45.354248, lr=1.2e-05, time_each_step=0.31s, eta=0:19:6
    2021-08-11 16:55:33 [INFO]	[TRAIN] Epoch=232/270, Step=54/74, loss=19.538992, lr=1.2e-05, time_each_step=0.29s, eta=0:19:5
    2021-08-11 16:55:33 [INFO]	[TRAIN] Epoch=232/270, Step=56/74, loss=44.518555, lr=1.2e-05, time_each_step=0.27s, eta=0:19:5
    2021-08-11 16:55:33 [INFO]	[TRAIN] Epoch=232/270, Step=58/74, loss=22.266165, lr=1.2e-05, time_each_step=0.24s, eta=0:19:3
    2021-08-11 16:55:34 [INFO]	[TRAIN] Epoch=232/270, Step=60/74, loss=10.90997, lr=1.2e-05, time_each_step=0.22s, eta=0:19:3
    2021-08-11 16:55:34 [INFO]	[TRAIN] Epoch=232/270, Step=62/74, loss=9.535625, lr=1.2e-05, time_each_step=0.21s, eta=0:19:2
    2021-08-11 16:55:35 [INFO]	[TRAIN] Epoch=232/270, Step=64/74, loss=16.372766, lr=1.2e-05, time_each_step=0.21s, eta=0:19:2
    2021-08-11 16:55:35 [INFO]	[TRAIN] Epoch=232/270, Step=66/74, loss=7.18703, lr=1.2e-05, time_each_step=0.19s, eta=0:19:1
    2021-08-11 16:55:35 [INFO]	[TRAIN] Epoch=232/270, Step=68/74, loss=22.25593, lr=1.2e-05, time_each_step=0.19s, eta=0:19:1
    2021-08-11 16:55:36 [INFO]	[TRAIN] Epoch=232/270, Step=70/74, loss=19.126715, lr=1.2e-05, time_each_step=0.19s, eta=0:19:0
    2021-08-11 16:55:36 [INFO]	[TRAIN] Epoch=232/270, Step=72/74, loss=8.599852, lr=1.2e-05, time_each_step=0.2s, eta=0:19:0
    2021-08-11 16:55:37 [INFO]	[TRAIN] Epoch=232/270, Step=74/74, loss=9.368492, lr=1.2e-05, time_each_step=0.19s, eta=0:19:0
    2021-08-11 16:55:37 [INFO]	[TRAIN] Epoch 232 finished, loss=17.739971, lr=1.2e-05 .
    2021-08-11 16:55:42 [INFO]	[TRAIN] Epoch=233/270, Step=2/74, loss=10.472622, lr=1.2e-05, time_each_step=0.45s, eta=0:21:58
    2021-08-11 16:55:43 [INFO]	[TRAIN] Epoch=233/270, Step=4/74, loss=13.37661, lr=1.2e-05, time_each_step=0.47s, eta=0:21:58
    2021-08-11 16:55:44 [INFO]	[TRAIN] Epoch=233/270, Step=6/74, loss=19.363901, lr=1.2e-05, time_each_step=0.49s, eta=0:21:58
    2021-08-11 16:55:45 [INFO]	[TRAIN] Epoch=233/270, Step=8/74, loss=21.407879, lr=1.2e-05, time_each_step=0.53s, eta=0:22:0
    2021-08-11 16:55:45 [INFO]	[TRAIN] Epoch=233/270, Step=10/74, loss=9.581757, lr=1.2e-05, time_each_step=0.54s, eta=0:21:59
    2021-08-11 16:55:46 [INFO]	[TRAIN] Epoch=233/270, Step=12/74, loss=9.013066, lr=1.2e-05, time_each_step=0.56s, eta=0:22:0
    2021-08-11 16:55:47 [INFO]	[TRAIN] Epoch=233/270, Step=14/74, loss=19.541183, lr=1.2e-05, time_each_step=0.58s, eta=0:22:0
    2021-08-11 16:55:48 [INFO]	[TRAIN] Epoch=233/270, Step=16/74, loss=24.195862, lr=1.2e-05, time_each_step=0.61s, eta=0:22:0
    2021-08-11 16:55:49 [INFO]	[TRAIN] Epoch=233/270, Step=18/74, loss=53.93298, lr=1.2e-05, time_each_step=0.62s, eta=0:22:0
    2021-08-11 16:55:50 [INFO]	[TRAIN] Epoch=233/270, Step=20/74, loss=21.145643, lr=1.2e-05, time_each_step=0.64s, eta=0:22:0
    2021-08-11 16:55:50 [INFO]	[TRAIN] Epoch=233/270, Step=22/74, loss=31.079456, lr=1.2e-05, time_each_step=0.4s, eta=0:21:46
    2021-08-11 16:55:51 [INFO]	[TRAIN] Epoch=233/270, Step=24/74, loss=17.058287, lr=1.2e-05, time_each_step=0.42s, eta=0:21:46
    2021-08-11 16:55:52 [INFO]	[TRAIN] Epoch=233/270, Step=26/74, loss=15.981009, lr=1.2e-05, time_each_step=0.41s, eta=0:21:45
    2021-08-11 16:55:53 [INFO]	[TRAIN] Epoch=233/270, Step=28/74, loss=11.453889, lr=1.2e-05, time_each_step=0.4s, eta=0:21:43
    2021-08-11 16:55:54 [INFO]	[TRAIN] Epoch=233/270, Step=30/74, loss=23.996275, lr=1.2e-05, time_each_step=0.41s, eta=0:21:43
    2021-08-11 16:55:55 [INFO]	[TRAIN] Epoch=233/270, Step=32/74, loss=12.113018, lr=1.2e-05, time_each_step=0.43s, eta=0:21:43
    2021-08-11 16:55:56 [INFO]	[TRAIN] Epoch=233/270, Step=34/74, loss=17.426825, lr=1.2e-05, time_each_step=0.42s, eta=0:21:42
    2021-08-11 16:55:56 [INFO]	[TRAIN] Epoch=233/270, Step=36/74, loss=17.336777, lr=1.2e-05, time_each_step=0.42s, eta=0:21:41
    2021-08-11 16:55:57 [INFO]	[TRAIN] Epoch=233/270, Step=38/74, loss=20.176159, lr=1.2e-05, time_each_step=0.41s, eta=0:21:40
    2021-08-11 16:55:58 [INFO]	[TRAIN] Epoch=233/270, Step=40/74, loss=15.72766, lr=1.2e-05, time_each_step=0.41s, eta=0:21:39
    2021-08-11 16:55:58 [INFO]	[TRAIN] Epoch=233/270, Step=42/74, loss=13.861635, lr=1.2e-05, time_each_step=0.39s, eta=0:21:37
    2021-08-11 16:55:58 [INFO]	[TRAIN] Epoch=233/270, Step=44/74, loss=29.807457, lr=1.2e-05, time_each_step=0.35s, eta=0:21:36
    2021-08-11 16:55:59 [INFO]	[TRAIN] Epoch=233/270, Step=46/74, loss=17.623159, lr=1.2e-05, time_each_step=0.33s, eta=0:21:34
    2021-08-11 16:55:59 [INFO]	[TRAIN] Epoch=233/270, Step=48/74, loss=11.409881, lr=1.2e-05, time_each_step=0.32s, eta=0:21:33
    2021-08-11 16:55:59 [INFO]	[TRAIN] Epoch=233/270, Step=50/74, loss=12.293667, lr=1.2e-05, time_each_step=0.29s, eta=0:21:32
    2021-08-11 16:56:00 [INFO]	[TRAIN] Epoch=233/270, Step=52/74, loss=7.529881, lr=1.2e-05, time_each_step=0.26s, eta=0:21:31
    2021-08-11 16:56:00 [INFO]	[TRAIN] Epoch=233/270, Step=54/74, loss=7.305065, lr=1.2e-05, time_each_step=0.23s, eta=0:21:30
    2021-08-11 16:56:00 [INFO]	[TRAIN] Epoch=233/270, Step=56/74, loss=16.000507, lr=1.2e-05, time_each_step=0.2s, eta=0:21:29
    2021-08-11 16:56:01 [INFO]	[TRAIN] Epoch=233/270, Step=58/74, loss=31.565777, lr=1.2e-05, time_each_step=0.19s, eta=0:21:28
    2021-08-11 16:56:01 [INFO]	[TRAIN] Epoch=233/270, Step=60/74, loss=45.341156, lr=1.2e-05, time_each_step=0.19s, eta=0:21:28
    2021-08-11 16:56:02 [INFO]	[TRAIN] Epoch=233/270, Step=62/74, loss=15.079685, lr=1.2e-05, time_each_step=0.18s, eta=0:21:27
    2021-08-11 16:56:02 [INFO]	[TRAIN] Epoch=233/270, Step=64/74, loss=10.614854, lr=1.2e-05, time_each_step=0.19s, eta=0:21:27
    2021-08-11 16:56:02 [INFO]	[TRAIN] Epoch=233/270, Step=66/74, loss=6.501537, lr=1.2e-05, time_each_step=0.19s, eta=0:21:26
    2021-08-11 16:56:03 [INFO]	[TRAIN] Epoch=233/270, Step=68/74, loss=10.552789, lr=1.2e-05, time_each_step=0.18s, eta=0:21:26
    2021-08-11 16:56:03 [INFO]	[TRAIN] Epoch=233/270, Step=70/74, loss=12.442563, lr=1.2e-05, time_each_step=0.19s, eta=0:21:26
    2021-08-11 16:56:04 [INFO]	[TRAIN] Epoch=233/270, Step=72/74, loss=16.383459, lr=1.2e-05, time_each_step=0.19s, eta=0:21:25
    2021-08-11 16:56:04 [INFO]	[TRAIN] Epoch=233/270, Step=74/74, loss=14.763552, lr=1.2e-05, time_each_step=0.2s, eta=0:21:25
    2021-08-11 16:56:04 [INFO]	[TRAIN] Epoch 233 finished, loss=18.496168, lr=1.2e-05 .
    2021-08-11 16:56:24 [INFO]	[TRAIN] Epoch=234/270, Step=2/74, loss=22.141254, lr=1.2e-05, time_each_step=1.17s, eta=0:19:19
    2021-08-11 16:56:25 [INFO]	[TRAIN] Epoch=234/270, Step=4/74, loss=28.382008, lr=1.2e-05, time_each_step=1.21s, eta=0:19:19
    2021-08-11 16:56:26 [INFO]	[TRAIN] Epoch=234/270, Step=6/74, loss=16.090317, lr=1.2e-05, time_each_step=1.24s, eta=0:19:18
    2021-08-11 16:56:27 [INFO]	[TRAIN] Epoch=234/270, Step=8/74, loss=13.482736, lr=1.2e-05, time_each_step=1.27s, eta=0:19:18
    2021-08-11 16:56:28 [INFO]	[TRAIN] Epoch=234/270, Step=10/74, loss=19.716759, lr=1.2e-05, time_each_step=1.29s, eta=0:19:17
    2021-08-11 16:56:28 [INFO]	[TRAIN] Epoch=234/270, Step=12/74, loss=16.905905, lr=1.2e-05, time_each_step=1.3s, eta=0:19:15
    2021-08-11 16:56:29 [INFO]	[TRAIN] Epoch=234/270, Step=14/74, loss=20.074024, lr=1.2e-05, time_each_step=1.33s, eta=0:19:14
    2021-08-11 16:56:30 [INFO]	[TRAIN] Epoch=234/270, Step=16/74, loss=23.152596, lr=1.2e-05, time_each_step=1.34s, eta=0:19:12
    2021-08-11 16:56:31 [INFO]	[TRAIN] Epoch=234/270, Step=18/74, loss=21.650145, lr=1.2e-05, time_each_step=1.37s, eta=0:19:11
    2021-08-11 16:56:32 [INFO]	[TRAIN] Epoch=234/270, Step=20/74, loss=8.029881, lr=1.2e-05, time_each_step=1.39s, eta=0:19:9
    2021-08-11 16:56:33 [INFO]	[TRAIN] Epoch=234/270, Step=22/74, loss=14.380716, lr=1.2e-05, time_each_step=0.44s, eta=0:18:17
    2021-08-11 16:56:33 [INFO]	[TRAIN] Epoch=234/270, Step=24/74, loss=14.303035, lr=1.2e-05, time_each_step=0.42s, eta=0:18:15
    2021-08-11 16:56:34 [INFO]	[TRAIN] Epoch=234/270, Step=26/74, loss=23.5951, lr=1.2e-05, time_each_step=0.38s, eta=0:18:13
    2021-08-11 16:56:34 [INFO]	[TRAIN] Epoch=234/270, Step=28/74, loss=12.741035, lr=1.2e-05, time_each_step=0.37s, eta=0:18:12
    2021-08-11 16:56:35 [INFO]	[TRAIN] Epoch=234/270, Step=30/74, loss=37.644745, lr=1.2e-05, time_each_step=0.36s, eta=0:18:10
    2021-08-11 16:56:36 [INFO]	[TRAIN] Epoch=234/270, Step=32/74, loss=20.585796, lr=1.2e-05, time_each_step=0.37s, eta=0:18:10
    2021-08-11 16:56:37 [INFO]	[TRAIN] Epoch=234/270, Step=34/74, loss=28.365807, lr=1.2e-05, time_each_step=0.39s, eta=0:18:10
    2021-08-11 16:56:38 [INFO]	[TRAIN] Epoch=234/270, Step=36/74, loss=28.314474, lr=1.2e-05, time_each_step=0.39s, eta=0:18:9
    2021-08-11 16:56:39 [INFO]	[TRAIN] Epoch=234/270, Step=38/74, loss=55.776371, lr=1.2e-05, time_each_step=0.38s, eta=0:18:8
    2021-08-11 16:56:39 [INFO]	[TRAIN] Epoch=234/270, Step=40/74, loss=15.641474, lr=1.2e-05, time_each_step=0.38s, eta=0:18:7
    2021-08-11 16:56:40 [INFO]	[TRAIN] Epoch=234/270, Step=42/74, loss=8.642563, lr=1.2e-05, time_each_step=0.36s, eta=0:18:6
    2021-08-11 16:56:40 [INFO]	[TRAIN] Epoch=234/270, Step=44/74, loss=16.197496, lr=1.2e-05, time_each_step=0.35s, eta=0:18:5
    2021-08-11 16:56:41 [INFO]	[TRAIN] Epoch=234/270, Step=46/74, loss=8.031217, lr=1.2e-05, time_each_step=0.36s, eta=0:18:4
    2021-08-11 16:56:41 [INFO]	[TRAIN] Epoch=234/270, Step=48/74, loss=13.886384, lr=1.2e-05, time_each_step=0.34s, eta=0:18:3
    2021-08-11 16:56:42 [INFO]	[TRAIN] Epoch=234/270, Step=50/74, loss=19.741539, lr=1.2e-05, time_each_step=0.32s, eta=0:18:2
    2021-08-11 16:56:42 [INFO]	[TRAIN] Epoch=234/270, Step=52/74, loss=13.169868, lr=1.2e-05, time_each_step=0.31s, eta=0:18:1
    2021-08-11 16:56:42 [INFO]	[TRAIN] Epoch=234/270, Step=54/74, loss=6.866007, lr=1.2e-05, time_each_step=0.27s, eta=0:18:0
    2021-08-11 16:56:43 [INFO]	[TRAIN] Epoch=234/270, Step=56/74, loss=28.96991, lr=1.2e-05, time_each_step=0.26s, eta=0:17:59
    2021-08-11 16:56:43 [INFO]	[TRAIN] Epoch=234/270, Step=58/74, loss=5.12965, lr=1.2e-05, time_each_step=0.24s, eta=0:17:58
    2021-08-11 16:56:44 [INFO]	[TRAIN] Epoch=234/270, Step=60/74, loss=12.335756, lr=1.2e-05, time_each_step=0.21s, eta=0:17:57
    2021-08-11 16:56:44 [INFO]	[TRAIN] Epoch=234/270, Step=62/74, loss=10.57721, lr=1.2e-05, time_each_step=0.2s, eta=0:17:57
    2021-08-11 16:56:44 [INFO]	[TRAIN] Epoch=234/270, Step=64/74, loss=27.949562, lr=1.2e-05, time_each_step=0.2s, eta=0:17:56
    2021-08-11 16:56:45 [INFO]	[TRAIN] Epoch=234/270, Step=66/74, loss=10.83457, lr=1.2e-05, time_each_step=0.2s, eta=0:17:56
    2021-08-11 16:56:45 [INFO]	[TRAIN] Epoch=234/270, Step=68/74, loss=27.91803, lr=1.2e-05, time_each_step=0.2s, eta=0:17:56
    2021-08-11 16:56:46 [INFO]	[TRAIN] Epoch=234/270, Step=70/74, loss=21.676796, lr=1.2e-05, time_each_step=0.2s, eta=0:17:55
    2021-08-11 16:56:46 [INFO]	[TRAIN] Epoch=234/270, Step=72/74, loss=17.294567, lr=1.2e-05, time_each_step=0.2s, eta=0:17:55
    2021-08-11 16:56:46 [INFO]	[TRAIN] Epoch=234/270, Step=74/74, loss=30.499439, lr=1.2e-05, time_each_step=0.2s, eta=0:17:54
    2021-08-11 16:56:46 [INFO]	[TRAIN] Epoch 234 finished, loss=18.798508, lr=1.2e-05 .
    2021-08-11 16:56:51 [INFO]	[TRAIN] Epoch=235/270, Step=2/74, loss=20.982983, lr=1.2e-05, time_each_step=0.43s, eta=0:26:42
    2021-08-11 16:56:52 [INFO]	[TRAIN] Epoch=235/270, Step=4/74, loss=20.473486, lr=1.2e-05, time_each_step=0.45s, eta=0:26:42
    2021-08-11 16:56:54 [INFO]	[TRAIN] Epoch=235/270, Step=6/74, loss=11.819785, lr=1.2e-05, time_each_step=0.49s, eta=0:26:44
    2021-08-11 16:56:55 [INFO]	[TRAIN] Epoch=235/270, Step=8/74, loss=21.018839, lr=1.2e-05, time_each_step=0.52s, eta=0:26:45
    2021-08-11 16:56:55 [INFO]	[TRAIN] Epoch=235/270, Step=10/74, loss=26.072556, lr=1.2e-05, time_each_step=0.54s, eta=0:26:45
    2021-08-11 16:56:56 [INFO]	[TRAIN] Epoch=235/270, Step=12/74, loss=14.430466, lr=1.2e-05, time_each_step=0.56s, eta=0:26:45
    2021-08-11 16:56:57 [INFO]	[TRAIN] Epoch=235/270, Step=14/74, loss=14.06953, lr=1.2e-05, time_each_step=0.58s, eta=0:26:45
    2021-08-11 16:56:58 [INFO]	[TRAIN] Epoch=235/270, Step=16/74, loss=22.559288, lr=1.2e-05, time_each_step=0.61s, eta=0:26:46
    2021-08-11 16:56:58 [INFO]	[TRAIN] Epoch=235/270, Step=18/74, loss=9.760132, lr=1.2e-05, time_each_step=0.63s, eta=0:26:46
    2021-08-11 16:56:59 [INFO]	[TRAIN] Epoch=235/270, Step=20/74, loss=21.545471, lr=1.2e-05, time_each_step=0.64s, eta=0:26:45
    2021-08-11 16:57:00 [INFO]	[TRAIN] Epoch=235/270, Step=22/74, loss=28.216114, lr=1.2e-05, time_each_step=0.42s, eta=0:26:32
    2021-08-11 16:57:00 [INFO]	[TRAIN] Epoch=235/270, Step=24/74, loss=15.10617, lr=1.2e-05, time_each_step=0.4s, eta=0:26:31
    2021-08-11 16:57:01 [INFO]	[TRAIN] Epoch=235/270, Step=26/74, loss=13.725085, lr=1.2e-05, time_each_step=0.39s, eta=0:26:29
    2021-08-11 16:57:02 [INFO]	[TRAIN] Epoch=235/270, Step=28/74, loss=22.407322, lr=1.2e-05, time_each_step=0.39s, eta=0:26:28
    2021-08-11 16:57:04 [INFO]	[TRAIN] Epoch=235/270, Step=30/74, loss=26.949459, lr=1.2e-05, time_each_step=0.41s, eta=0:26:29
    2021-08-11 16:57:04 [INFO]	[TRAIN] Epoch=235/270, Step=32/74, loss=24.886353, lr=1.2e-05, time_each_step=0.41s, eta=0:26:28
    2021-08-11 16:57:05 [INFO]	[TRAIN] Epoch=235/270, Step=34/74, loss=14.272245, lr=1.2e-05, time_each_step=0.42s, eta=0:26:27
    2021-08-11 16:57:06 [INFO]	[TRAIN] Epoch=235/270, Step=36/74, loss=21.591019, lr=1.2e-05, time_each_step=0.4s, eta=0:26:26
    2021-08-11 16:57:06 [INFO]	[TRAIN] Epoch=235/270, Step=38/74, loss=14.693469, lr=1.2e-05, time_each_step=0.38s, eta=0:26:24
    2021-08-11 16:57:07 [INFO]	[TRAIN] Epoch=235/270, Step=40/74, loss=29.65167, lr=1.2e-05, time_each_step=0.38s, eta=0:26:23
    2021-08-11 16:57:08 [INFO]	[TRAIN] Epoch=235/270, Step=42/74, loss=23.016388, lr=1.2e-05, time_each_step=0.39s, eta=0:26:23
    2021-08-11 16:57:08 [INFO]	[TRAIN] Epoch=235/270, Step=44/74, loss=7.874712, lr=1.2e-05, time_each_step=0.4s, eta=0:26:22
    2021-08-11 16:57:09 [INFO]	[TRAIN] Epoch=235/270, Step=46/74, loss=19.702066, lr=1.2e-05, time_each_step=0.38s, eta=0:26:21
    2021-08-11 16:57:09 [INFO]	[TRAIN] Epoch=235/270, Step=48/74, loss=24.235437, lr=1.2e-05, time_each_step=0.35s, eta=0:26:20
    2021-08-11 16:57:10 [INFO]	[TRAIN] Epoch=235/270, Step=50/74, loss=11.124243, lr=1.2e-05, time_each_step=0.31s, eta=0:26:18
    2021-08-11 16:57:10 [INFO]	[TRAIN] Epoch=235/270, Step=52/74, loss=17.654827, lr=1.2e-05, time_each_step=0.29s, eta=0:26:17
    2021-08-11 16:57:10 [INFO]	[TRAIN] Epoch=235/270, Step=54/74, loss=15.103708, lr=1.2e-05, time_each_step=0.26s, eta=0:26:16
    2021-08-11 16:57:11 [INFO]	[TRAIN] Epoch=235/270, Step=56/74, loss=17.309422, lr=1.2e-05, time_each_step=0.25s, eta=0:26:15
    2021-08-11 16:57:11 [INFO]	[TRAIN] Epoch=235/270, Step=58/74, loss=8.241359, lr=1.2e-05, time_each_step=0.26s, eta=0:26:15
    2021-08-11 16:57:12 [INFO]	[TRAIN] Epoch=235/270, Step=60/74, loss=20.97065, lr=1.2e-05, time_each_step=0.24s, eta=0:26:14
    2021-08-11 16:57:12 [INFO]	[TRAIN] Epoch=235/270, Step=62/74, loss=14.255836, lr=1.2e-05, time_each_step=0.23s, eta=0:26:13
    2021-08-11 16:57:13 [INFO]	[TRAIN] Epoch=235/270, Step=64/74, loss=7.823534, lr=1.2e-05, time_each_step=0.21s, eta=0:26:13
    2021-08-11 16:57:13 [INFO]	[TRAIN] Epoch=235/270, Step=66/74, loss=11.188637, lr=1.2e-05, time_each_step=0.22s, eta=0:26:12
    2021-08-11 16:57:14 [INFO]	[TRAIN] Epoch=235/270, Step=68/74, loss=18.654678, lr=1.2e-05, time_each_step=0.22s, eta=0:26:12
    2021-08-11 16:57:14 [INFO]	[TRAIN] Epoch=235/270, Step=70/74, loss=27.497524, lr=1.2e-05, time_each_step=0.23s, eta=0:26:11
    2021-08-11 16:57:15 [INFO]	[TRAIN] Epoch=235/270, Step=72/74, loss=49.488579, lr=1.2e-05, time_each_step=0.23s, eta=0:26:11
    2021-08-11 16:57:15 [INFO]	[TRAIN] Epoch=235/270, Step=74/74, loss=9.871933, lr=1.2e-05, time_each_step=0.24s, eta=0:26:11
    2021-08-11 16:57:15 [INFO]	[TRAIN] Epoch 235 finished, loss=18.010908, lr=1.2e-05 .
    2021-08-11 16:57:23 [INFO]	[TRAIN] Epoch=236/270, Step=2/74, loss=9.098766, lr=1.2e-05, time_each_step=0.6s, eta=0:18:36
    2021-08-11 16:57:24 [INFO]	[TRAIN] Epoch=236/270, Step=4/74, loss=21.321634, lr=1.2e-05, time_each_step=0.63s, eta=0:18:36
    2021-08-11 16:57:25 [INFO]	[TRAIN] Epoch=236/270, Step=6/74, loss=63.47266, lr=1.2e-05, time_each_step=0.65s, eta=0:18:36
    2021-08-11 16:57:26 [INFO]	[TRAIN] Epoch=236/270, Step=8/74, loss=7.702043, lr=1.2e-05, time_each_step=0.67s, eta=0:18:36
    2021-08-11 16:57:26 [INFO]	[TRAIN] Epoch=236/270, Step=10/74, loss=10.169343, lr=1.2e-05, time_each_step=0.68s, eta=0:18:35
    2021-08-11 16:57:27 [INFO]	[TRAIN] Epoch=236/270, Step=12/74, loss=13.90463, lr=1.2e-05, time_each_step=0.68s, eta=0:18:34
    2021-08-11 16:57:28 [INFO]	[TRAIN] Epoch=236/270, Step=14/74, loss=26.332712, lr=1.2e-05, time_each_step=0.69s, eta=0:18:34
    2021-08-11 16:57:29 [INFO]	[TRAIN] Epoch=236/270, Step=16/74, loss=20.912466, lr=1.2e-05, time_each_step=0.71s, eta=0:18:33
    2021-08-11 16:57:29 [INFO]	[TRAIN] Epoch=236/270, Step=18/74, loss=33.558506, lr=1.2e-05, time_each_step=0.72s, eta=0:18:32
    2021-08-11 16:57:30 [INFO]	[TRAIN] Epoch=236/270, Step=20/74, loss=32.073563, lr=1.2e-05, time_each_step=0.74s, eta=0:18:32
    2021-08-11 16:57:31 [INFO]	[TRAIN] Epoch=236/270, Step=22/74, loss=17.956362, lr=1.2e-05, time_each_step=0.39s, eta=0:18:12
    2021-08-11 16:57:31 [INFO]	[TRAIN] Epoch=236/270, Step=24/74, loss=21.190577, lr=1.2e-05, time_each_step=0.38s, eta=0:18:11
    2021-08-11 16:57:32 [INFO]	[TRAIN] Epoch=236/270, Step=26/74, loss=28.489363, lr=1.2e-05, time_each_step=0.37s, eta=0:18:10
    2021-08-11 16:57:33 [INFO]	[TRAIN] Epoch=236/270, Step=28/74, loss=18.561676, lr=1.2e-05, time_each_step=0.35s, eta=0:18:8
    2021-08-11 16:57:33 [INFO]	[TRAIN] Epoch=236/270, Step=30/74, loss=33.095848, lr=1.2e-05, time_each_step=0.36s, eta=0:18:8
    2021-08-11 16:57:34 [INFO]	[TRAIN] Epoch=236/270, Step=32/74, loss=25.900221, lr=1.2e-05, time_each_step=0.37s, eta=0:18:8
    2021-08-11 16:57:35 [INFO]	[TRAIN] Epoch=236/270, Step=34/74, loss=21.895996, lr=1.2e-05, time_each_step=0.37s, eta=0:18:7
    2021-08-11 16:57:36 [INFO]	[TRAIN] Epoch=236/270, Step=36/74, loss=8.267878, lr=1.2e-05, time_each_step=0.37s, eta=0:18:6
    2021-08-11 16:57:37 [INFO]	[TRAIN] Epoch=236/270, Step=38/74, loss=12.867302, lr=1.2e-05, time_each_step=0.39s, eta=0:18:6
    2021-08-11 16:57:38 [INFO]	[TRAIN] Epoch=236/270, Step=40/74, loss=14.530175, lr=1.2e-05, time_each_step=0.37s, eta=0:18:5
    2021-08-11 16:57:38 [INFO]	[TRAIN] Epoch=236/270, Step=42/74, loss=23.484997, lr=1.2e-05, time_each_step=0.36s, eta=0:18:4
    2021-08-11 16:57:39 [INFO]	[TRAIN] Epoch=236/270, Step=44/74, loss=27.411213, lr=1.2e-05, time_each_step=0.36s, eta=0:18:3
    2021-08-11 16:57:39 [INFO]	[TRAIN] Epoch=236/270, Step=46/74, loss=15.085065, lr=1.2e-05, time_each_step=0.35s, eta=0:18:2
    2021-08-11 16:57:40 [INFO]	[TRAIN] Epoch=236/270, Step=48/74, loss=10.446973, lr=1.2e-05, time_each_step=0.36s, eta=0:18:1
    2021-08-11 16:57:40 [INFO]	[TRAIN] Epoch=236/270, Step=50/74, loss=29.214558, lr=1.2e-05, time_each_step=0.33s, eta=0:18:0
    2021-08-11 16:57:40 [INFO]	[TRAIN] Epoch=236/270, Step=52/74, loss=20.689407, lr=1.2e-05, time_each_step=0.32s, eta=0:17:59
    2021-08-11 16:57:41 [INFO]	[TRAIN] Epoch=236/270, Step=54/74, loss=24.222401, lr=1.2e-05, time_each_step=0.3s, eta=0:17:58
    2021-08-11 16:57:42 [INFO]	[TRAIN] Epoch=236/270, Step=56/74, loss=17.232433, lr=1.2e-05, time_each_step=0.28s, eta=0:17:57
    2021-08-11 16:57:42 [INFO]	[TRAIN] Epoch=236/270, Step=58/74, loss=11.405386, lr=1.2e-05, time_each_step=0.25s, eta=0:17:56
    2021-08-11 16:57:43 [INFO]	[TRAIN] Epoch=236/270, Step=60/74, loss=11.374001, lr=1.2e-05, time_each_step=0.25s, eta=0:17:56
    2021-08-11 16:57:43 [INFO]	[TRAIN] Epoch=236/270, Step=62/74, loss=14.900549, lr=1.2e-05, time_each_step=0.25s, eta=0:17:55
    2021-08-11 16:57:43 [INFO]	[TRAIN] Epoch=236/270, Step=64/74, loss=23.37883, lr=1.2e-05, time_each_step=0.24s, eta=0:17:55
    2021-08-11 16:57:44 [INFO]	[TRAIN] Epoch=236/270, Step=66/74, loss=13.555744, lr=1.2e-05, time_each_step=0.23s, eta=0:17:54
    2021-08-11 16:57:44 [INFO]	[TRAIN] Epoch=236/270, Step=68/74, loss=11.951059, lr=1.2e-05, time_each_step=0.23s, eta=0:17:53
    2021-08-11 16:57:45 [INFO]	[TRAIN] Epoch=236/270, Step=70/74, loss=11.687618, lr=1.2e-05, time_each_step=0.23s, eta=0:17:53
    2021-08-11 16:57:45 [INFO]	[TRAIN] Epoch=236/270, Step=72/74, loss=19.40304, lr=1.2e-05, time_each_step=0.22s, eta=0:17:53
    2021-08-11 16:57:45 [INFO]	[TRAIN] Epoch=236/270, Step=74/74, loss=13.089571, lr=1.2e-05, time_each_step=0.21s, eta=0:17:52
    2021-08-11 16:57:45 [INFO]	[TRAIN] Epoch 236 finished, loss=19.208679, lr=1.2e-05 .
    2021-08-11 16:57:56 [INFO]	[TRAIN] Epoch=237/270, Step=2/74, loss=25.056274, lr=1.2e-05, time_each_step=0.71s, eta=0:18:50
    2021-08-11 16:57:56 [INFO]	[TRAIN] Epoch=237/270, Step=4/74, loss=34.917812, lr=1.2e-05, time_each_step=0.72s, eta=0:18:50
    2021-08-11 16:57:57 [INFO]	[TRAIN] Epoch=237/270, Step=6/74, loss=12.494419, lr=1.2e-05, time_each_step=0.74s, eta=0:18:50
    2021-08-11 16:57:58 [INFO]	[TRAIN] Epoch=237/270, Step=8/74, loss=22.258135, lr=1.2e-05, time_each_step=0.76s, eta=0:18:49
    2021-08-11 16:57:59 [INFO]	[TRAIN] Epoch=237/270, Step=10/74, loss=30.076237, lr=1.2e-05, time_each_step=0.79s, eta=0:18:50
    2021-08-11 16:58:00 [INFO]	[TRAIN] Epoch=237/270, Step=12/74, loss=11.013865, lr=1.2e-05, time_each_step=0.81s, eta=0:18:49
    2021-08-11 16:58:00 [INFO]	[TRAIN] Epoch=237/270, Step=14/74, loss=16.896778, lr=1.2e-05, time_each_step=0.81s, eta=0:18:48
    2021-08-11 16:58:01 [INFO]	[TRAIN] Epoch=237/270, Step=16/74, loss=12.486304, lr=1.2e-05, time_each_step=0.82s, eta=0:18:47
    2021-08-11 16:58:02 [INFO]	[TRAIN] Epoch=237/270, Step=18/74, loss=12.991793, lr=1.2e-05, time_each_step=0.85s, eta=0:18:47
    2021-08-11 16:58:03 [INFO]	[TRAIN] Epoch=237/270, Step=20/74, loss=11.635547, lr=1.2e-05, time_each_step=0.89s, eta=0:18:47
    2021-08-11 16:58:04 [INFO]	[TRAIN] Epoch=237/270, Step=22/74, loss=24.313066, lr=1.2e-05, time_each_step=0.4s, eta=0:18:20
    2021-08-11 16:58:04 [INFO]	[TRAIN] Epoch=237/270, Step=24/74, loss=10.622309, lr=1.2e-05, time_each_step=0.39s, eta=0:18:19
    2021-08-11 16:58:05 [INFO]	[TRAIN] Epoch=237/270, Step=26/74, loss=23.183794, lr=1.2e-05, time_each_step=0.37s, eta=0:18:17
    2021-08-11 16:58:06 [INFO]	[TRAIN] Epoch=237/270, Step=28/74, loss=19.792421, lr=1.2e-05, time_each_step=0.37s, eta=0:18:16
    2021-08-11 16:58:06 [INFO]	[TRAIN] Epoch=237/270, Step=30/74, loss=11.391111, lr=1.2e-05, time_each_step=0.36s, eta=0:18:15
    2021-08-11 16:58:07 [INFO]	[TRAIN] Epoch=237/270, Step=32/74, loss=19.771446, lr=1.2e-05, time_each_step=0.36s, eta=0:18:14
    2021-08-11 16:58:08 [INFO]	[TRAIN] Epoch=237/270, Step=34/74, loss=11.301933, lr=1.2e-05, time_each_step=0.38s, eta=0:18:14
    2021-08-11 16:58:09 [INFO]	[TRAIN] Epoch=237/270, Step=36/74, loss=13.438967, lr=1.2e-05, time_each_step=0.39s, eta=0:18:14
    2021-08-11 16:58:10 [INFO]	[TRAIN] Epoch=237/270, Step=38/74, loss=23.520676, lr=1.2e-05, time_each_step=0.39s, eta=0:18:13
    2021-08-11 16:58:11 [INFO]	[TRAIN] Epoch=237/270, Step=40/74, loss=21.162378, lr=1.2e-05, time_each_step=0.37s, eta=0:18:12
    2021-08-11 16:58:11 [INFO]	[TRAIN] Epoch=237/270, Step=42/74, loss=18.257824, lr=1.2e-05, time_each_step=0.38s, eta=0:18:11
    2021-08-11 16:58:12 [INFO]	[TRAIN] Epoch=237/270, Step=44/74, loss=16.674126, lr=1.2e-05, time_each_step=0.38s, eta=0:18:11
    2021-08-11 16:58:12 [INFO]	[TRAIN] Epoch=237/270, Step=46/74, loss=9.78237, lr=1.2e-05, time_each_step=0.38s, eta=0:18:10
    2021-08-11 16:58:13 [INFO]	[TRAIN] Epoch=237/270, Step=48/74, loss=18.183708, lr=1.2e-05, time_each_step=0.36s, eta=0:18:9
    2021-08-11 16:58:13 [INFO]	[TRAIN] Epoch=237/270, Step=50/74, loss=16.261261, lr=1.2e-05, time_each_step=0.35s, eta=0:18:7
    2021-08-11 16:58:14 [INFO]	[TRAIN] Epoch=237/270, Step=52/74, loss=18.412285, lr=1.2e-05, time_each_step=0.33s, eta=0:18:6
    2021-08-11 16:58:14 [INFO]	[TRAIN] Epoch=237/270, Step=54/74, loss=10.417347, lr=1.2e-05, time_each_step=0.31s, eta=0:18:5
    2021-08-11 16:58:14 [INFO]	[TRAIN] Epoch=237/270, Step=56/74, loss=12.472382, lr=1.2e-05, time_each_step=0.28s, eta=0:18:4
    2021-08-11 16:58:15 [INFO]	[TRAIN] Epoch=237/270, Step=58/74, loss=21.601397, lr=1.2e-05, time_each_step=0.25s, eta=0:18:3
    2021-08-11 16:58:15 [INFO]	[TRAIN] Epoch=237/270, Step=60/74, loss=27.023827, lr=1.2e-05, time_each_step=0.24s, eta=0:18:3
    2021-08-11 16:58:16 [INFO]	[TRAIN] Epoch=237/270, Step=62/74, loss=19.130199, lr=1.2e-05, time_each_step=0.22s, eta=0:18:2
    2021-08-11 16:58:16 [INFO]	[TRAIN] Epoch=237/270, Step=64/74, loss=45.281746, lr=1.2e-05, time_each_step=0.21s, eta=0:18:1
    2021-08-11 16:58:17 [INFO]	[TRAIN] Epoch=237/270, Step=66/74, loss=40.741409, lr=1.2e-05, time_each_step=0.21s, eta=0:18:1
    2021-08-11 16:58:17 [INFO]	[TRAIN] Epoch=237/270, Step=68/74, loss=24.927601, lr=1.2e-05, time_each_step=0.21s, eta=0:18:0
    2021-08-11 16:58:17 [INFO]	[TRAIN] Epoch=237/270, Step=70/74, loss=8.811799, lr=1.2e-05, time_each_step=0.21s, eta=0:18:0
    2021-08-11 16:58:18 [INFO]	[TRAIN] Epoch=237/270, Step=72/74, loss=12.282063, lr=1.2e-05, time_each_step=0.21s, eta=0:18:0
    2021-08-11 16:58:18 [INFO]	[TRAIN] Epoch=237/270, Step=74/74, loss=10.576115, lr=1.2e-05, time_each_step=0.21s, eta=0:17:59
    2021-08-11 16:58:18 [INFO]	[TRAIN] Epoch 237 finished, loss=18.714752, lr=1.2e-05 .
    2021-08-11 16:58:23 [INFO]	[TRAIN] Epoch=238/270, Step=2/74, loss=19.879738, lr=1.2e-05, time_each_step=0.43s, eta=0:19:33
    2021-08-11 16:58:24 [INFO]	[TRAIN] Epoch=238/270, Step=4/74, loss=18.255577, lr=1.2e-05, time_each_step=0.45s, eta=0:19:34
    2021-08-11 16:58:25 [INFO]	[TRAIN] Epoch=238/270, Step=6/74, loss=15.441153, lr=1.2e-05, time_each_step=0.48s, eta=0:19:35
    2021-08-11 16:58:26 [INFO]	[TRAIN] Epoch=238/270, Step=8/74, loss=11.93573, lr=1.2e-05, time_each_step=0.5s, eta=0:19:35
    2021-08-11 16:58:27 [INFO]	[TRAIN] Epoch=238/270, Step=10/74, loss=18.718056, lr=1.2e-05, time_each_step=0.53s, eta=0:19:36
    2021-08-11 16:58:27 [INFO]	[TRAIN] Epoch=238/270, Step=12/74, loss=8.7981, lr=1.2e-05, time_each_step=0.54s, eta=0:19:36
    2021-08-11 16:58:28 [INFO]	[TRAIN] Epoch=238/270, Step=14/74, loss=15.906961, lr=1.2e-05, time_each_step=0.57s, eta=0:19:37
    2021-08-11 16:58:29 [INFO]	[TRAIN] Epoch=238/270, Step=16/74, loss=19.203259, lr=1.2e-05, time_each_step=0.58s, eta=0:19:36
    2021-08-11 16:58:30 [INFO]	[TRAIN] Epoch=238/270, Step=18/74, loss=22.079292, lr=1.2e-05, time_each_step=0.6s, eta=0:19:36
    2021-08-11 16:58:31 [INFO]	[TRAIN] Epoch=238/270, Step=20/74, loss=13.403381, lr=1.2e-05, time_each_step=0.63s, eta=0:19:36
    2021-08-11 16:58:32 [INFO]	[TRAIN] Epoch=238/270, Step=22/74, loss=9.647318, lr=1.2e-05, time_each_step=0.44s, eta=0:19:26
    2021-08-11 16:58:33 [INFO]	[TRAIN] Epoch=238/270, Step=24/74, loss=24.05759, lr=1.2e-05, time_each_step=0.45s, eta=0:19:25
    2021-08-11 16:58:34 [INFO]	[TRAIN] Epoch=238/270, Step=26/74, loss=40.073154, lr=1.2e-05, time_each_step=0.43s, eta=0:19:23
    2021-08-11 16:58:34 [INFO]	[TRAIN] Epoch=238/270, Step=28/74, loss=10.496904, lr=1.2e-05, time_each_step=0.44s, eta=0:19:23
    2021-08-11 16:58:35 [INFO]	[TRAIN] Epoch=238/270, Step=30/74, loss=21.809954, lr=1.2e-05, time_each_step=0.44s, eta=0:19:22
    2021-08-11 16:58:36 [INFO]	[TRAIN] Epoch=238/270, Step=32/74, loss=17.191694, lr=1.2e-05, time_each_step=0.44s, eta=0:19:21
    2021-08-11 16:58:37 [INFO]	[TRAIN] Epoch=238/270, Step=34/74, loss=9.112284, lr=1.2e-05, time_each_step=0.42s, eta=0:19:19
    2021-08-11 16:58:37 [INFO]	[TRAIN] Epoch=238/270, Step=36/74, loss=13.613896, lr=1.2e-05, time_each_step=0.41s, eta=0:19:18
    2021-08-11 16:58:38 [INFO]	[TRAIN] Epoch=238/270, Step=38/74, loss=21.946339, lr=1.2e-05, time_each_step=0.4s, eta=0:19:17
    2021-08-11 16:58:39 [INFO]	[TRAIN] Epoch=238/270, Step=40/74, loss=18.461868, lr=1.2e-05, time_each_step=0.4s, eta=0:19:16
    2021-08-11 16:58:39 [INFO]	[TRAIN] Epoch=238/270, Step=42/74, loss=7.846549, lr=1.2e-05, time_each_step=0.38s, eta=0:19:14
    2021-08-11 16:58:40 [INFO]	[TRAIN] Epoch=238/270, Step=44/74, loss=12.594833, lr=1.2e-05, time_each_step=0.34s, eta=0:19:13
    2021-08-11 16:58:40 [INFO]	[TRAIN] Epoch=238/270, Step=46/74, loss=21.201981, lr=1.2e-05, time_each_step=0.33s, eta=0:19:12
    2021-08-11 16:58:41 [INFO]	[TRAIN] Epoch=238/270, Step=48/74, loss=29.003408, lr=1.2e-05, time_each_step=0.31s, eta=0:19:11
    2021-08-11 16:58:41 [INFO]	[TRAIN] Epoch=238/270, Step=50/74, loss=19.07567, lr=1.2e-05, time_each_step=0.28s, eta=0:19:9
    2021-08-11 16:58:42 [INFO]	[TRAIN] Epoch=238/270, Step=52/74, loss=13.080548, lr=1.2e-05, time_each_step=0.27s, eta=0:19:8
    2021-08-11 16:58:42 [INFO]	[TRAIN] Epoch=238/270, Step=54/74, loss=11.465803, lr=1.2e-05, time_each_step=0.26s, eta=0:19:8
    2021-08-11 16:58:42 [INFO]	[TRAIN] Epoch=238/270, Step=56/74, loss=26.930733, lr=1.2e-05, time_each_step=0.26s, eta=0:19:7
    2021-08-11 16:58:43 [INFO]	[TRAIN] Epoch=238/270, Step=58/74, loss=21.117977, lr=1.2e-05, time_each_step=0.26s, eta=0:19:7
    2021-08-11 16:58:43 [INFO]	[TRAIN] Epoch=238/270, Step=60/74, loss=11.930511, lr=1.2e-05, time_each_step=0.23s, eta=0:19:6
    2021-08-11 16:58:44 [INFO]	[TRAIN] Epoch=238/270, Step=62/74, loss=20.978832, lr=1.2e-05, time_each_step=0.22s, eta=0:19:5
    2021-08-11 16:58:45 [INFO]	[TRAIN] Epoch=238/270, Step=64/74, loss=7.597763, lr=1.2e-05, time_each_step=0.23s, eta=0:19:5
    2021-08-11 16:58:45 [INFO]	[TRAIN] Epoch=238/270, Step=66/74, loss=19.867626, lr=1.2e-05, time_each_step=0.24s, eta=0:19:4
    2021-08-11 16:58:46 [INFO]	[TRAIN] Epoch=238/270, Step=68/74, loss=23.481167, lr=1.2e-05, time_each_step=0.26s, eta=0:19:4
    2021-08-11 16:58:46 [INFO]	[TRAIN] Epoch=238/270, Step=70/74, loss=25.020447, lr=1.2e-05, time_each_step=0.26s, eta=0:19:4
    2021-08-11 16:58:47 [INFO]	[TRAIN] Epoch=238/270, Step=72/74, loss=9.711351, lr=1.2e-05, time_each_step=0.25s, eta=0:19:3
    2021-08-11 16:58:47 [INFO]	[TRAIN] Epoch=238/270, Step=74/74, loss=11.222701, lr=1.2e-05, time_each_step=0.25s, eta=0:19:2
    2021-08-11 16:58:47 [INFO]	[TRAIN] Epoch 238 finished, loss=18.246874, lr=1.2e-05 .
    2021-08-11 16:58:52 [INFO]	[TRAIN] Epoch=239/270, Step=2/74, loss=13.395836, lr=1.2e-05, time_each_step=0.49s, eta=0:16:58
    2021-08-11 16:58:53 [INFO]	[TRAIN] Epoch=239/270, Step=4/74, loss=30.43133, lr=1.2e-05, time_each_step=0.5s, eta=0:16:58
    2021-08-11 16:58:53 [INFO]	[TRAIN] Epoch=239/270, Step=6/74, loss=17.08345, lr=1.2e-05, time_each_step=0.51s, eta=0:16:58
    2021-08-11 16:58:54 [INFO]	[TRAIN] Epoch=239/270, Step=8/74, loss=12.817205, lr=1.2e-05, time_each_step=0.52s, eta=0:16:58
    2021-08-11 16:58:55 [INFO]	[TRAIN] Epoch=239/270, Step=10/74, loss=18.491655, lr=1.2e-05, time_each_step=0.54s, eta=0:16:57
    2021-08-11 16:58:56 [INFO]	[TRAIN] Epoch=239/270, Step=12/74, loss=11.951118, lr=1.2e-05, time_each_step=0.55s, eta=0:16:57
    2021-08-11 16:58:57 [INFO]	[TRAIN] Epoch=239/270, Step=14/74, loss=10.119401, lr=1.2e-05, time_each_step=0.55s, eta=0:16:56
    2021-08-11 16:58:58 [INFO]	[TRAIN] Epoch=239/270, Step=16/74, loss=20.881851, lr=1.2e-05, time_each_step=0.57s, eta=0:16:56
    2021-08-11 16:58:58 [INFO]	[TRAIN] Epoch=239/270, Step=18/74, loss=14.197053, lr=1.2e-05, time_each_step=0.58s, eta=0:16:56
    2021-08-11 16:58:59 [INFO]	[TRAIN] Epoch=239/270, Step=20/74, loss=16.318748, lr=1.2e-05, time_each_step=0.6s, eta=0:16:56
    2021-08-11 16:59:00 [INFO]	[TRAIN] Epoch=239/270, Step=22/74, loss=14.450546, lr=1.2e-05, time_each_step=0.38s, eta=0:16:43
    2021-08-11 16:59:00 [INFO]	[TRAIN] Epoch=239/270, Step=24/74, loss=30.716526, lr=1.2e-05, time_each_step=0.37s, eta=0:16:42
    2021-08-11 16:59:01 [INFO]	[TRAIN] Epoch=239/270, Step=26/74, loss=14.538327, lr=1.2e-05, time_each_step=0.38s, eta=0:16:41
    2021-08-11 16:59:02 [INFO]	[TRAIN] Epoch=239/270, Step=28/74, loss=27.308296, lr=1.2e-05, time_each_step=0.38s, eta=0:16:40
    2021-08-11 16:59:03 [INFO]	[TRAIN] Epoch=239/270, Step=30/74, loss=17.112295, lr=1.2e-05, time_each_step=0.37s, eta=0:16:40
    2021-08-11 16:59:03 [INFO]	[TRAIN] Epoch=239/270, Step=32/74, loss=13.620955, lr=1.2e-05, time_each_step=0.37s, eta=0:16:39
    2021-08-11 16:59:04 [INFO]	[TRAIN] Epoch=239/270, Step=34/74, loss=12.943404, lr=1.2e-05, time_each_step=0.38s, eta=0:16:38
    2021-08-11 16:59:05 [INFO]	[TRAIN] Epoch=239/270, Step=36/74, loss=13.036137, lr=1.2e-05, time_each_step=0.37s, eta=0:16:37
    2021-08-11 16:59:06 [INFO]	[TRAIN] Epoch=239/270, Step=38/74, loss=17.364674, lr=1.2e-05, time_each_step=0.38s, eta=0:16:37
    2021-08-11 16:59:07 [INFO]	[TRAIN] Epoch=239/270, Step=40/74, loss=24.387373, lr=1.2e-05, time_each_step=0.39s, eta=0:16:36
    2021-08-11 16:59:08 [INFO]	[TRAIN] Epoch=239/270, Step=42/74, loss=18.627323, lr=1.2e-05, time_each_step=0.39s, eta=0:16:35
    2021-08-11 16:59:08 [INFO]	[TRAIN] Epoch=239/270, Step=44/74, loss=18.933212, lr=1.2e-05, time_each_step=0.4s, eta=0:16:35
    2021-08-11 16:59:09 [INFO]	[TRAIN] Epoch=239/270, Step=46/74, loss=10.431252, lr=1.2e-05, time_each_step=0.38s, eta=0:16:34
    2021-08-11 16:59:09 [INFO]	[TRAIN] Epoch=239/270, Step=48/74, loss=9.722852, lr=1.2e-05, time_each_step=0.37s, eta=0:16:33
    2021-08-11 16:59:10 [INFO]	[TRAIN] Epoch=239/270, Step=50/74, loss=15.050695, lr=1.2e-05, time_each_step=0.34s, eta=0:16:31
    2021-08-11 16:59:10 [INFO]	[TRAIN] Epoch=239/270, Step=52/74, loss=19.528639, lr=1.2e-05, time_each_step=0.33s, eta=0:16:30
    2021-08-11 16:59:10 [INFO]	[TRAIN] Epoch=239/270, Step=54/74, loss=22.190855, lr=1.2e-05, time_each_step=0.31s, eta=0:16:29
    2021-08-11 16:59:11 [INFO]	[TRAIN] Epoch=239/270, Step=56/74, loss=21.652397, lr=1.2e-05, time_each_step=0.3s, eta=0:16:28
    2021-08-11 16:59:11 [INFO]	[TRAIN] Epoch=239/270, Step=58/74, loss=32.293682, lr=1.2e-05, time_each_step=0.28s, eta=0:16:28
    2021-08-11 16:59:12 [INFO]	[TRAIN] Epoch=239/270, Step=60/74, loss=15.010842, lr=1.2e-05, time_each_step=0.26s, eta=0:16:27
    2021-08-11 16:59:12 [INFO]	[TRAIN] Epoch=239/270, Step=62/74, loss=13.915657, lr=1.2e-05, time_each_step=0.23s, eta=0:16:26
    2021-08-11 16:59:13 [INFO]	[TRAIN] Epoch=239/270, Step=64/74, loss=8.268189, lr=1.2e-05, time_each_step=0.22s, eta=0:16:25
    2021-08-11 16:59:13 [INFO]	[TRAIN] Epoch=239/270, Step=66/74, loss=26.506546, lr=1.2e-05, time_each_step=0.22s, eta=0:16:25
    2021-08-11 16:59:13 [INFO]	[TRAIN] Epoch=239/270, Step=68/74, loss=20.89427, lr=1.2e-05, time_each_step=0.21s, eta=0:16:24
    2021-08-11 16:59:14 [INFO]	[TRAIN] Epoch=239/270, Step=70/74, loss=18.945723, lr=1.2e-05, time_each_step=0.22s, eta=0:16:24
    2021-08-11 16:59:14 [INFO]	[TRAIN] Epoch=239/270, Step=72/74, loss=11.009841, lr=1.2e-05, time_each_step=0.22s, eta=0:16:24
    2021-08-11 16:59:15 [INFO]	[TRAIN] Epoch=239/270, Step=74/74, loss=15.768436, lr=1.2e-05, time_each_step=0.23s, eta=0:16:23
    2021-08-11 16:59:15 [INFO]	[TRAIN] Epoch 239 finished, loss=18.930548, lr=1.2e-05 .
    2021-08-11 16:59:26 [INFO]	[TRAIN] Epoch=240/270, Step=2/74, loss=14.490562, lr=1.2e-05, time_each_step=0.76s, eta=0:16:24
    2021-08-11 16:59:27 [INFO]	[TRAIN] Epoch=240/270, Step=4/74, loss=21.477831, lr=1.2e-05, time_each_step=0.78s, eta=0:16:24
    2021-08-11 16:59:28 [INFO]	[TRAIN] Epoch=240/270, Step=6/74, loss=24.356236, lr=1.2e-05, time_each_step=0.78s, eta=0:16:22
    2021-08-11 16:59:29 [INFO]	[TRAIN] Epoch=240/270, Step=8/74, loss=22.809361, lr=1.2e-05, time_each_step=0.81s, eta=0:16:23
    2021-08-11 16:59:29 [INFO]	[TRAIN] Epoch=240/270, Step=10/74, loss=25.917053, lr=1.2e-05, time_each_step=0.82s, eta=0:16:22
    2021-08-11 16:59:30 [INFO]	[TRAIN] Epoch=240/270, Step=12/74, loss=20.384102, lr=1.2e-05, time_each_step=0.85s, eta=0:16:22
    2021-08-11 16:59:31 [INFO]	[TRAIN] Epoch=240/270, Step=14/74, loss=18.225855, lr=1.2e-05, time_each_step=0.88s, eta=0:16:22
    2021-08-11 16:59:32 [INFO]	[TRAIN] Epoch=240/270, Step=16/74, loss=10.263442, lr=1.2e-05, time_each_step=0.89s, eta=0:16:21
    2021-08-11 16:59:33 [INFO]	[TRAIN] Epoch=240/270, Step=18/74, loss=15.002516, lr=1.2e-05, time_each_step=0.91s, eta=0:16:20
    2021-08-11 16:59:34 [INFO]	[TRAIN] Epoch=240/270, Step=20/74, loss=14.701139, lr=1.2e-05, time_each_step=0.93s, eta=0:16:19
    2021-08-11 16:59:34 [INFO]	[TRAIN] Epoch=240/270, Step=22/74, loss=21.186275, lr=1.2e-05, time_each_step=0.41s, eta=0:15:50
    2021-08-11 16:59:35 [INFO]	[TRAIN] Epoch=240/270, Step=24/74, loss=9.525335, lr=1.2e-05, time_each_step=0.4s, eta=0:15:49
    2021-08-11 16:59:36 [INFO]	[TRAIN] Epoch=240/270, Step=26/74, loss=15.871971, lr=1.2e-05, time_each_step=0.4s, eta=0:15:49
    2021-08-11 16:59:36 [INFO]	[TRAIN] Epoch=240/270, Step=28/74, loss=11.702564, lr=1.2e-05, time_each_step=0.39s, eta=0:15:47
    2021-08-11 16:59:37 [INFO]	[TRAIN] Epoch=240/270, Step=30/74, loss=20.599182, lr=1.2e-05, time_each_step=0.4s, eta=0:15:47
    2021-08-11 16:59:38 [INFO]	[TRAIN] Epoch=240/270, Step=32/74, loss=11.757607, lr=1.2e-05, time_each_step=0.38s, eta=0:15:45
    2021-08-11 16:59:39 [INFO]	[TRAIN] Epoch=240/270, Step=34/74, loss=30.49309, lr=1.2e-05, time_each_step=0.38s, eta=0:15:44
    2021-08-11 16:59:40 [INFO]	[TRAIN] Epoch=240/270, Step=36/74, loss=8.092444, lr=1.2e-05, time_each_step=0.4s, eta=0:15:44
    2021-08-11 16:59:40 [INFO]	[TRAIN] Epoch=240/270, Step=38/74, loss=13.734905, lr=1.2e-05, time_each_step=0.38s, eta=0:15:43
    2021-08-11 16:59:41 [INFO]	[TRAIN] Epoch=240/270, Step=40/74, loss=9.02582, lr=1.2e-05, time_each_step=0.38s, eta=0:15:42
    2021-08-11 16:59:42 [INFO]	[TRAIN] Epoch=240/270, Step=42/74, loss=14.667058, lr=1.2e-05, time_each_step=0.39s, eta=0:15:42
    2021-08-11 16:59:42 [INFO]	[TRAIN] Epoch=240/270, Step=44/74, loss=23.73757, lr=1.2e-05, time_each_step=0.37s, eta=0:15:40
    2021-08-11 16:59:43 [INFO]	[TRAIN] Epoch=240/270, Step=46/74, loss=23.628708, lr=1.2e-05, time_each_step=0.36s, eta=0:15:39
    2021-08-11 16:59:43 [INFO]	[TRAIN] Epoch=240/270, Step=48/74, loss=17.358765, lr=1.2e-05, time_each_step=0.35s, eta=0:15:38
    2021-08-11 16:59:44 [INFO]	[TRAIN] Epoch=240/270, Step=50/74, loss=5.166744, lr=1.2e-05, time_each_step=0.34s, eta=0:15:37
    2021-08-11 16:59:44 [INFO]	[TRAIN] Epoch=240/270, Step=52/74, loss=40.419807, lr=1.2e-05, time_each_step=0.33s, eta=0:15:36
    2021-08-11 16:59:45 [INFO]	[TRAIN] Epoch=240/270, Step=54/74, loss=12.543241, lr=1.2e-05, time_each_step=0.31s, eta=0:15:35
    2021-08-11 16:59:45 [INFO]	[TRAIN] Epoch=240/270, Step=56/74, loss=10.453357, lr=1.2e-05, time_each_step=0.29s, eta=0:15:34
    2021-08-11 16:59:46 [INFO]	[TRAIN] Epoch=240/270, Step=58/74, loss=11.344215, lr=1.2e-05, time_each_step=0.27s, eta=0:15:33
    2021-08-11 16:59:46 [INFO]	[TRAIN] Epoch=240/270, Step=60/74, loss=19.273491, lr=1.2e-05, time_each_step=0.25s, eta=0:15:33
    2021-08-11 16:59:47 [INFO]	[TRAIN] Epoch=240/270, Step=62/74, loss=25.038614, lr=1.2e-05, time_each_step=0.23s, eta=0:15:32
    2021-08-11 16:59:47 [INFO]	[TRAIN] Epoch=240/270, Step=64/74, loss=34.897259, lr=1.2e-05, time_each_step=0.23s, eta=0:15:31
    2021-08-11 16:59:48 [INFO]	[TRAIN] Epoch=240/270, Step=66/74, loss=20.631874, lr=1.2e-05, time_each_step=0.23s, eta=0:15:31
    2021-08-11 16:59:48 [INFO]	[TRAIN] Epoch=240/270, Step=68/74, loss=13.651733, lr=1.2e-05, time_each_step=0.23s, eta=0:15:31
    2021-08-11 16:59:48 [INFO]	[TRAIN] Epoch=240/270, Step=70/74, loss=10.303677, lr=1.2e-05, time_each_step=0.22s, eta=0:15:30
    2021-08-11 16:59:49 [INFO]	[TRAIN] Epoch=240/270, Step=72/74, loss=23.521605, lr=1.2e-05, time_each_step=0.21s, eta=0:15:30
    2021-08-11 16:59:49 [INFO]	[TRAIN] Epoch=240/270, Step=74/74, loss=29.066713, lr=1.2e-05, time_each_step=0.22s, eta=0:15:29
    2021-08-11 16:59:49 [INFO]	[TRAIN] Epoch 240 finished, loss=18.199718, lr=1.2e-05 .
    2021-08-11 16:59:49 [INFO]	Start to evaluating(total_samples=170, total_steps=22)...


    100%|██████████| 22/22 [00:09<00:00,  2.41it/s]


    2021-08-11 16:59:58 [INFO]	[EVAL] Finished, Epoch=240, bbox_map=80.86807 .
    2021-08-11 17:00:05 [INFO]	Model saved in output/yolov3_DarkNet53/best_model.
    2021-08-11 17:00:11 [INFO]	Model saved in output/yolov3_DarkNet53/epoch_240.
    2021-08-11 17:00:11 [INFO]	Current evaluated best model in eval_dataset is epoch_240, bbox_map=80.86807042180423
    2021-08-11 17:00:22 [INFO]	[TRAIN] Epoch=241/270, Step=2/74, loss=11.535477, lr=1e-06, time_each_step=0.76s, eta=0:18:5
    2021-08-11 17:00:23 [INFO]	[TRAIN] Epoch=241/270, Step=4/74, loss=22.066263, lr=1e-06, time_each_step=0.77s, eta=0:18:5
    2021-08-11 17:00:23 [INFO]	[TRAIN] Epoch=241/270, Step=6/74, loss=10.241523, lr=1e-06, time_each_step=0.78s, eta=0:18:4
    2021-08-11 17:00:24 [INFO]	[TRAIN] Epoch=241/270, Step=8/74, loss=10.65864, lr=1e-06, time_each_step=0.8s, eta=0:18:4
    2021-08-11 17:00:25 [INFO]	[TRAIN] Epoch=241/270, Step=10/74, loss=9.797065, lr=1e-06, time_each_step=0.83s, eta=0:18:4
    2021-08-11 17:00:26 [INFO]	[TRAIN] Epoch=241/270, Step=12/74, loss=32.086838, lr=1e-06, time_each_step=0.84s, eta=0:18:3
    2021-08-11 17:00:27 [INFO]	[TRAIN] Epoch=241/270, Step=14/74, loss=15.751364, lr=1e-06, time_each_step=0.86s, eta=0:18:2
    2021-08-11 17:00:27 [INFO]	[TRAIN] Epoch=241/270, Step=16/74, loss=25.515068, lr=1e-06, time_each_step=0.88s, eta=0:18:2
    2021-08-11 17:00:28 [INFO]	[TRAIN] Epoch=241/270, Step=18/74, loss=9.923073, lr=1e-06, time_each_step=0.9s, eta=0:18:1
    2021-08-11 17:00:29 [INFO]	[TRAIN] Epoch=241/270, Step=20/74, loss=12.027571, lr=1e-06, time_each_step=0.92s, eta=0:18:0
    2021-08-11 17:00:30 [INFO]	[TRAIN] Epoch=241/270, Step=22/74, loss=13.928857, lr=1e-06, time_each_step=0.38s, eta=0:17:31
    2021-08-11 17:00:30 [INFO]	[TRAIN] Epoch=241/270, Step=24/74, loss=11.653267, lr=1e-06, time_each_step=0.38s, eta=0:17:30
    2021-08-11 17:00:31 [INFO]	[TRAIN] Epoch=241/270, Step=26/74, loss=29.142967, lr=1e-06, time_each_step=0.39s, eta=0:17:30
    2021-08-11 17:00:32 [INFO]	[TRAIN] Epoch=241/270, Step=28/74, loss=23.639099, lr=1e-06, time_each_step=0.38s, eta=0:17:29
    2021-08-11 17:00:32 [INFO]	[TRAIN] Epoch=241/270, Step=30/74, loss=17.126972, lr=1e-06, time_each_step=0.37s, eta=0:17:27
    2021-08-11 17:00:33 [INFO]	[TRAIN] Epoch=241/270, Step=32/74, loss=15.370466, lr=1e-06, time_each_step=0.36s, eta=0:17:26
    2021-08-11 17:00:33 [INFO]	[TRAIN] Epoch=241/270, Step=34/74, loss=15.95752, lr=1e-06, time_each_step=0.33s, eta=0:17:24
    2021-08-11 17:00:34 [INFO]	[TRAIN] Epoch=241/270, Step=36/74, loss=23.153084, lr=1e-06, time_each_step=0.33s, eta=0:17:23
    2021-08-11 17:00:35 [INFO]	[TRAIN] Epoch=241/270, Step=38/74, loss=20.367516, lr=1e-06, time_each_step=0.32s, eta=0:17:23
    2021-08-11 17:00:35 [INFO]	[TRAIN] Epoch=241/270, Step=40/74, loss=22.118156, lr=1e-06, time_each_step=0.32s, eta=0:17:22
    2021-08-11 17:00:36 [INFO]	[TRAIN] Epoch=241/270, Step=42/74, loss=12.972654, lr=1e-06, time_each_step=0.33s, eta=0:17:22
    2021-08-11 17:00:37 [INFO]	[TRAIN] Epoch=241/270, Step=44/74, loss=10.593813, lr=1e-06, time_each_step=0.31s, eta=0:17:20
    2021-08-11 17:00:37 [INFO]	[TRAIN] Epoch=241/270, Step=46/74, loss=16.225231, lr=1e-06, time_each_step=0.31s, eta=0:17:20
    2021-08-11 17:00:38 [INFO]	[TRAIN] Epoch=241/270, Step=48/74, loss=20.389725, lr=1e-06, time_each_step=0.3s, eta=0:17:19
    2021-08-11 17:00:38 [INFO]	[TRAIN] Epoch=241/270, Step=50/74, loss=10.035303, lr=1e-06, time_each_step=0.27s, eta=0:17:18
    2021-08-11 17:00:38 [INFO]	[TRAIN] Epoch=241/270, Step=52/74, loss=35.046227, lr=1e-06, time_each_step=0.27s, eta=0:17:17
    2021-08-11 17:00:39 [INFO]	[TRAIN] Epoch=241/270, Step=54/74, loss=20.357445, lr=1e-06, time_each_step=0.28s, eta=0:17:17
    2021-08-11 17:00:39 [INFO]	[TRAIN] Epoch=241/270, Step=56/74, loss=35.89119, lr=1e-06, time_each_step=0.26s, eta=0:17:16
    2021-08-11 17:00:39 [INFO]	[TRAIN] Epoch=241/270, Step=58/74, loss=25.268059, lr=1e-06, time_each_step=0.24s, eta=0:17:15
    2021-08-11 17:00:40 [INFO]	[TRAIN] Epoch=241/270, Step=60/74, loss=20.196384, lr=1e-06, time_each_step=0.23s, eta=0:17:14
    2021-08-11 17:00:40 [INFO]	[TRAIN] Epoch=241/270, Step=62/74, loss=13.768053, lr=1e-06, time_each_step=0.2s, eta=0:17:13
    2021-08-11 17:00:41 [INFO]	[TRAIN] Epoch=241/270, Step=64/74, loss=12.730719, lr=1e-06, time_each_step=0.2s, eta=0:17:13
    2021-08-11 17:00:41 [INFO]	[TRAIN] Epoch=241/270, Step=66/74, loss=7.307656, lr=1e-06, time_each_step=0.18s, eta=0:17:12
    2021-08-11 17:00:41 [INFO]	[TRAIN] Epoch=241/270, Step=68/74, loss=18.246716, lr=1e-06, time_each_step=0.19s, eta=0:17:12
    2021-08-11 17:00:42 [INFO]	[TRAIN] Epoch=241/270, Step=70/74, loss=15.309887, lr=1e-06, time_each_step=0.19s, eta=0:17:12
    2021-08-11 17:00:42 [INFO]	[TRAIN] Epoch=241/270, Step=72/74, loss=9.595078, lr=1e-06, time_each_step=0.19s, eta=0:17:11
    2021-08-11 17:00:42 [INFO]	[TRAIN] Epoch=241/270, Step=74/74, loss=9.851895, lr=1e-06, time_each_step=0.19s, eta=0:17:11
    2021-08-11 17:00:42 [INFO]	[TRAIN] Epoch 241 finished, loss=18.611897, lr=1e-06 .
    2021-08-11 17:00:51 [INFO]	[TRAIN] Epoch=242/270, Step=2/74, loss=25.443901, lr=1e-06, time_each_step=0.58s, eta=0:16:15
    2021-08-11 17:00:52 [INFO]	[TRAIN] Epoch=242/270, Step=4/74, loss=26.835497, lr=1e-06, time_each_step=0.61s, eta=0:16:16
    2021-08-11 17:00:52 [INFO]	[TRAIN] Epoch=242/270, Step=6/74, loss=22.926611, lr=1e-06, time_each_step=0.62s, eta=0:16:16
    2021-08-11 17:00:53 [INFO]	[TRAIN] Epoch=242/270, Step=8/74, loss=16.246428, lr=1e-06, time_each_step=0.65s, eta=0:16:17
    2021-08-11 17:00:54 [INFO]	[TRAIN] Epoch=242/270, Step=10/74, loss=22.006714, lr=1e-06, time_each_step=0.66s, eta=0:16:16
    2021-08-11 17:00:55 [INFO]	[TRAIN] Epoch=242/270, Step=12/74, loss=18.331486, lr=1e-06, time_each_step=0.68s, eta=0:16:16
    2021-08-11 17:00:56 [INFO]	[TRAIN] Epoch=242/270, Step=14/74, loss=28.835039, lr=1e-06, time_each_step=0.71s, eta=0:16:17
    2021-08-11 17:00:56 [INFO]	[TRAIN] Epoch=242/270, Step=16/74, loss=15.01655, lr=1e-06, time_each_step=0.73s, eta=0:16:16
    2021-08-11 17:00:57 [INFO]	[TRAIN] Epoch=242/270, Step=18/74, loss=20.244135, lr=1e-06, time_each_step=0.75s, eta=0:16:16
    2021-08-11 17:00:58 [INFO]	[TRAIN] Epoch=242/270, Step=20/74, loss=22.124863, lr=1e-06, time_each_step=0.78s, eta=0:16:16
    2021-08-11 17:00:59 [INFO]	[TRAIN] Epoch=242/270, Step=22/74, loss=14.397309, lr=1e-06, time_each_step=0.41s, eta=0:15:55
    2021-08-11 17:01:00 [INFO]	[TRAIN] Epoch=242/270, Step=24/74, loss=42.576569, lr=1e-06, time_each_step=0.41s, eta=0:15:54
    2021-08-11 17:01:01 [INFO]	[TRAIN] Epoch=242/270, Step=26/74, loss=8.606262, lr=1e-06, time_each_step=0.42s, eta=0:15:54
    2021-08-11 17:01:02 [INFO]	[TRAIN] Epoch=242/270, Step=28/74, loss=15.514957, lr=1e-06, time_each_step=0.43s, eta=0:15:53
    2021-08-11 17:01:02 [INFO]	[TRAIN] Epoch=242/270, Step=30/74, loss=13.647926, lr=1e-06, time_each_step=0.43s, eta=0:15:53
    2021-08-11 17:01:03 [INFO]	[TRAIN] Epoch=242/270, Step=32/74, loss=27.680475, lr=1e-06, time_each_step=0.41s, eta=0:15:51
    2021-08-11 17:01:03 [INFO]	[TRAIN] Epoch=242/270, Step=34/74, loss=27.24301, lr=1e-06, time_each_step=0.39s, eta=0:15:49
    2021-08-11 17:01:04 [INFO]	[TRAIN] Epoch=242/270, Step=36/74, loss=17.184544, lr=1e-06, time_each_step=0.39s, eta=0:15:48
    2021-08-11 17:01:05 [INFO]	[TRAIN] Epoch=242/270, Step=38/74, loss=30.55978, lr=1e-06, time_each_step=0.38s, eta=0:15:47
    2021-08-11 17:01:05 [INFO]	[TRAIN] Epoch=242/270, Step=40/74, loss=14.60528, lr=1e-06, time_each_step=0.36s, eta=0:15:46
    2021-08-11 17:01:06 [INFO]	[TRAIN] Epoch=242/270, Step=42/74, loss=8.957699, lr=1e-06, time_each_step=0.35s, eta=0:15:45
    2021-08-11 17:01:06 [INFO]	[TRAIN] Epoch=242/270, Step=44/74, loss=26.71204, lr=1e-06, time_each_step=0.33s, eta=0:15:44
    2021-08-11 17:01:07 [INFO]	[TRAIN] Epoch=242/270, Step=46/74, loss=13.597443, lr=1e-06, time_each_step=0.3s, eta=0:15:42
    2021-08-11 17:01:07 [INFO]	[TRAIN] Epoch=242/270, Step=48/74, loss=14.175631, lr=1e-06, time_each_step=0.27s, eta=0:15:41
    2021-08-11 17:01:08 [INFO]	[TRAIN] Epoch=242/270, Step=50/74, loss=23.738445, lr=1e-06, time_each_step=0.28s, eta=0:15:40
    2021-08-11 17:01:08 [INFO]	[TRAIN] Epoch=242/270, Step=52/74, loss=7.52597, lr=1e-06, time_each_step=0.27s, eta=0:15:40
    2021-08-11 17:01:09 [INFO]	[TRAIN] Epoch=242/270, Step=54/74, loss=10.418258, lr=1e-06, time_each_step=0.27s, eta=0:15:39
    2021-08-11 17:01:09 [INFO]	[TRAIN] Epoch=242/270, Step=56/74, loss=6.598096, lr=1e-06, time_each_step=0.26s, eta=0:15:38
    2021-08-11 17:01:10 [INFO]	[TRAIN] Epoch=242/270, Step=58/74, loss=9.459455, lr=1e-06, time_each_step=0.25s, eta=0:15:38
    2021-08-11 17:01:10 [INFO]	[TRAIN] Epoch=242/270, Step=60/74, loss=19.868238, lr=1e-06, time_each_step=0.26s, eta=0:15:37
    2021-08-11 17:01:11 [INFO]	[TRAIN] Epoch=242/270, Step=62/74, loss=19.487602, lr=1e-06, time_each_step=0.24s, eta=0:15:37
    2021-08-11 17:01:11 [INFO]	[TRAIN] Epoch=242/270, Step=64/74, loss=18.338032, lr=1e-06, time_each_step=0.24s, eta=0:15:36
    2021-08-11 17:01:11 [INFO]	[TRAIN] Epoch=242/270, Step=66/74, loss=9.63187, lr=1e-06, time_each_step=0.23s, eta=0:15:36
    2021-08-11 17:01:12 [INFO]	[TRAIN] Epoch=242/270, Step=68/74, loss=16.749269, lr=1e-06, time_each_step=0.24s, eta=0:15:35
    2021-08-11 17:01:13 [INFO]	[TRAIN] Epoch=242/270, Step=70/74, loss=12.876534, lr=1e-06, time_each_step=0.23s, eta=0:15:35
    2021-08-11 17:01:13 [INFO]	[TRAIN] Epoch=242/270, Step=72/74, loss=20.403763, lr=1e-06, time_each_step=0.24s, eta=0:15:34
    2021-08-11 17:01:13 [INFO]	[TRAIN] Epoch=242/270, Step=74/74, loss=9.201097, lr=1e-06, time_each_step=0.22s, eta=0:15:34
    2021-08-11 17:01:13 [INFO]	[TRAIN] Epoch 242 finished, loss=17.695589, lr=1e-06 .
    2021-08-11 17:01:32 [INFO]	[TRAIN] Epoch=243/270, Step=2/74, loss=10.153955, lr=1e-06, time_each_step=1.13s, eta=0:15:57
    2021-08-11 17:01:33 [INFO]	[TRAIN] Epoch=243/270, Step=4/74, loss=16.063742, lr=1e-06, time_each_step=1.17s, eta=0:15:57
    2021-08-11 17:01:34 [INFO]	[TRAIN] Epoch=243/270, Step=6/74, loss=14.327685, lr=1e-06, time_each_step=1.16s, eta=0:15:54
    2021-08-11 17:01:35 [INFO]	[TRAIN] Epoch=243/270, Step=8/74, loss=27.911728, lr=1e-06, time_each_step=1.19s, eta=0:15:54
    2021-08-11 17:01:35 [INFO]	[TRAIN] Epoch=243/270, Step=10/74, loss=15.463316, lr=1e-06, time_each_step=1.21s, eta=0:15:53
    2021-08-11 17:01:36 [INFO]	[TRAIN] Epoch=243/270, Step=12/74, loss=17.488811, lr=1e-06, time_each_step=1.23s, eta=0:15:52
    2021-08-11 17:01:37 [INFO]	[TRAIN] Epoch=243/270, Step=14/74, loss=19.846691, lr=1e-06, time_each_step=1.23s, eta=0:15:49
    2021-08-11 17:01:37 [INFO]	[TRAIN] Epoch=243/270, Step=16/74, loss=11.583936, lr=1e-06, time_each_step=1.23s, eta=0:15:47
    2021-08-11 17:01:38 [INFO]	[TRAIN] Epoch=243/270, Step=18/74, loss=11.85739, lr=1e-06, time_each_step=1.27s, eta=0:15:46
    2021-08-11 17:01:39 [INFO]	[TRAIN] Epoch=243/270, Step=20/74, loss=21.497314, lr=1e-06, time_each_step=1.3s, eta=0:15:45
    2021-08-11 17:01:40 [INFO]	[TRAIN] Epoch=243/270, Step=22/74, loss=14.634483, lr=1e-06, time_each_step=0.42s, eta=0:14:57
    2021-08-11 17:01:41 [INFO]	[TRAIN] Epoch=243/270, Step=24/74, loss=20.098389, lr=1e-06, time_each_step=0.39s, eta=0:14:55
    2021-08-11 17:01:42 [INFO]	[TRAIN] Epoch=243/270, Step=26/74, loss=19.099812, lr=1e-06, time_each_step=0.4s, eta=0:14:55
    2021-08-11 17:01:42 [INFO]	[TRAIN] Epoch=243/270, Step=28/74, loss=10.896868, lr=1e-06, time_each_step=0.39s, eta=0:14:53
    2021-08-11 17:01:43 [INFO]	[TRAIN] Epoch=243/270, Step=30/74, loss=16.068869, lr=1e-06, time_each_step=0.37s, eta=0:14:52
    2021-08-11 17:01:44 [INFO]	[TRAIN] Epoch=243/270, Step=32/74, loss=19.538366, lr=1e-06, time_each_step=0.39s, eta=0:14:52
    2021-08-11 17:01:45 [INFO]	[TRAIN] Epoch=243/270, Step=34/74, loss=19.376928, lr=1e-06, time_each_step=0.4s, eta=0:14:51
    2021-08-11 17:01:45 [INFO]	[TRAIN] Epoch=243/270, Step=36/74, loss=16.096798, lr=1e-06, time_each_step=0.4s, eta=0:14:51
    2021-08-11 17:01:46 [INFO]	[TRAIN] Epoch=243/270, Step=38/74, loss=17.913303, lr=1e-06, time_each_step=0.38s, eta=0:14:49
    2021-08-11 17:01:46 [INFO]	[TRAIN] Epoch=243/270, Step=40/74, loss=29.981161, lr=1e-06, time_each_step=0.36s, eta=0:14:47
    2021-08-11 17:01:47 [INFO]	[TRAIN] Epoch=243/270, Step=42/74, loss=18.251661, lr=1e-06, time_each_step=0.34s, eta=0:14:46
    2021-08-11 17:01:47 [INFO]	[TRAIN] Epoch=243/270, Step=44/74, loss=23.520075, lr=1e-06, time_each_step=0.33s, eta=0:14:45
    2021-08-11 17:01:48 [INFO]	[TRAIN] Epoch=243/270, Step=46/74, loss=18.912148, lr=1e-06, time_each_step=0.31s, eta=0:14:44
    2021-08-11 17:01:48 [INFO]	[TRAIN] Epoch=243/270, Step=48/74, loss=18.128706, lr=1e-06, time_each_step=0.31s, eta=0:14:43
    2021-08-11 17:01:49 [INFO]	[TRAIN] Epoch=243/270, Step=50/74, loss=6.984323, lr=1e-06, time_each_step=0.3s, eta=0:14:43
    2021-08-11 17:01:49 [INFO]	[TRAIN] Epoch=243/270, Step=52/74, loss=5.372986, lr=1e-06, time_each_step=0.27s, eta=0:14:41
    2021-08-11 17:01:50 [INFO]	[TRAIN] Epoch=243/270, Step=54/74, loss=28.881767, lr=1e-06, time_each_step=0.25s, eta=0:14:40
    2021-08-11 17:01:50 [INFO]	[TRAIN] Epoch=243/270, Step=56/74, loss=15.599659, lr=1e-06, time_each_step=0.24s, eta=0:14:40
    2021-08-11 17:01:50 [INFO]	[TRAIN] Epoch=243/270, Step=58/74, loss=18.899357, lr=1e-06, time_each_step=0.23s, eta=0:14:39
    2021-08-11 17:01:51 [INFO]	[TRAIN] Epoch=243/270, Step=60/74, loss=18.936497, lr=1e-06, time_each_step=0.22s, eta=0:14:38
    2021-08-11 17:01:51 [INFO]	[TRAIN] Epoch=243/270, Step=62/74, loss=15.872746, lr=1e-06, time_each_step=0.23s, eta=0:14:38
    2021-08-11 17:01:52 [INFO]	[TRAIN] Epoch=243/270, Step=64/74, loss=16.031605, lr=1e-06, time_each_step=0.23s, eta=0:14:38
    2021-08-11 17:01:52 [INFO]	[TRAIN] Epoch=243/270, Step=66/74, loss=16.653622, lr=1e-06, time_each_step=0.22s, eta=0:14:37
    2021-08-11 17:01:52 [INFO]	[TRAIN] Epoch=243/270, Step=68/74, loss=26.191395, lr=1e-06, time_each_step=0.2s, eta=0:14:37
    2021-08-11 17:01:53 [INFO]	[TRAIN] Epoch=243/270, Step=70/74, loss=10.964648, lr=1e-06, time_each_step=0.2s, eta=0:14:36
    2021-08-11 17:01:53 [INFO]	[TRAIN] Epoch=243/270, Step=72/74, loss=24.123714, lr=1e-06, time_each_step=0.2s, eta=0:14:36
    2021-08-11 17:01:54 [INFO]	[TRAIN] Epoch=243/270, Step=74/74, loss=12.084595, lr=1e-06, time_each_step=0.2s, eta=0:14:35
    2021-08-11 17:01:54 [INFO]	[TRAIN] Epoch 243 finished, loss=16.600851, lr=1e-06 .
    2021-08-11 17:02:09 [INFO]	[TRAIN] Epoch=244/270, Step=2/74, loss=22.62752, lr=1e-06, time_each_step=0.93s, eta=0:19:21
    2021-08-11 17:02:10 [INFO]	[TRAIN] Epoch=244/270, Step=4/74, loss=16.297169, lr=1e-06, time_each_step=0.97s, eta=0:19:21
    2021-08-11 17:02:11 [INFO]	[TRAIN] Epoch=244/270, Step=6/74, loss=19.846275, lr=1e-06, time_each_step=0.99s, eta=0:19:21
    2021-08-11 17:02:11 [INFO]	[TRAIN] Epoch=244/270, Step=8/74, loss=12.623703, lr=1e-06, time_each_step=1.0s, eta=0:19:20
    2021-08-11 17:02:12 [INFO]	[TRAIN] Epoch=244/270, Step=10/74, loss=11.775082, lr=1e-06, time_each_step=1.03s, eta=0:19:20
    2021-08-11 17:02:13 [INFO]	[TRAIN] Epoch=244/270, Step=12/74, loss=17.718668, lr=1e-06, time_each_step=1.05s, eta=0:19:19
    2021-08-11 17:02:14 [INFO]	[TRAIN] Epoch=244/270, Step=14/74, loss=10.377023, lr=1e-06, time_each_step=1.07s, eta=0:19:18
    2021-08-11 17:02:14 [INFO]	[TRAIN] Epoch=244/270, Step=16/74, loss=13.48332, lr=1e-06, time_each_step=1.08s, eta=0:19:16
    2021-08-11 17:02:15 [INFO]	[TRAIN] Epoch=244/270, Step=18/74, loss=9.705256, lr=1e-06, time_each_step=1.09s, eta=0:19:15
    2021-08-11 17:02:16 [INFO]	[TRAIN] Epoch=244/270, Step=20/74, loss=47.327702, lr=1e-06, time_each_step=1.1s, eta=0:19:13
    2021-08-11 17:02:17 [INFO]	[TRAIN] Epoch=244/270, Step=22/74, loss=33.491909, lr=1e-06, time_each_step=0.4s, eta=0:18:35
    2021-08-11 17:02:17 [INFO]	[TRAIN] Epoch=244/270, Step=24/74, loss=13.715075, lr=1e-06, time_each_step=0.37s, eta=0:18:33
    2021-08-11 17:02:18 [INFO]	[TRAIN] Epoch=244/270, Step=26/74, loss=10.140522, lr=1e-06, time_each_step=0.37s, eta=0:18:32
    2021-08-11 17:02:19 [INFO]	[TRAIN] Epoch=244/270, Step=28/74, loss=25.414326, lr=1e-06, time_each_step=0.37s, eta=0:18:31
    2021-08-11 17:02:20 [INFO]	[TRAIN] Epoch=244/270, Step=30/74, loss=18.8759, lr=1e-06, time_each_step=0.38s, eta=0:18:31
    2021-08-11 17:02:21 [INFO]	[TRAIN] Epoch=244/270, Step=32/74, loss=10.972717, lr=1e-06, time_each_step=0.41s, eta=0:18:31
    2021-08-11 17:02:22 [INFO]	[TRAIN] Epoch=244/270, Step=34/74, loss=25.244059, lr=1e-06, time_each_step=0.42s, eta=0:18:31
    2021-08-11 17:02:23 [INFO]	[TRAIN] Epoch=244/270, Step=36/74, loss=21.113123, lr=1e-06, time_each_step=0.42s, eta=0:18:30
    2021-08-11 17:02:24 [INFO]	[TRAIN] Epoch=244/270, Step=38/74, loss=27.965067, lr=1e-06, time_each_step=0.42s, eta=0:18:29
    2021-08-11 17:02:24 [INFO]	[TRAIN] Epoch=244/270, Step=40/74, loss=23.901588, lr=1e-06, time_each_step=0.42s, eta=0:18:28
    2021-08-11 17:02:25 [INFO]	[TRAIN] Epoch=244/270, Step=42/74, loss=19.449499, lr=1e-06, time_each_step=0.41s, eta=0:18:27
    2021-08-11 17:02:25 [INFO]	[TRAIN] Epoch=244/270, Step=44/74, loss=12.930136, lr=1e-06, time_each_step=0.4s, eta=0:18:26
    2021-08-11 17:02:26 [INFO]	[TRAIN] Epoch=244/270, Step=46/74, loss=9.658491, lr=1e-06, time_each_step=0.37s, eta=0:18:24
    2021-08-11 17:02:26 [INFO]	[TRAIN] Epoch=244/270, Step=48/74, loss=10.182258, lr=1e-06, time_each_step=0.35s, eta=0:18:23
    2021-08-11 17:02:26 [INFO]	[TRAIN] Epoch=244/270, Step=50/74, loss=6.233103, lr=1e-06, time_each_step=0.3s, eta=0:18:21
    2021-08-11 17:02:27 [INFO]	[TRAIN] Epoch=244/270, Step=52/74, loss=9.539843, lr=1e-06, time_each_step=0.27s, eta=0:18:20
    2021-08-11 17:02:27 [INFO]	[TRAIN] Epoch=244/270, Step=54/74, loss=11.820639, lr=1e-06, time_each_step=0.24s, eta=0:18:19
    2021-08-11 17:02:27 [INFO]	[TRAIN] Epoch=244/270, Step=56/74, loss=14.67368, lr=1e-06, time_each_step=0.23s, eta=0:18:18
    2021-08-11 17:02:28 [INFO]	[TRAIN] Epoch=244/270, Step=58/74, loss=18.018665, lr=1e-06, time_each_step=0.21s, eta=0:18:17
    2021-08-11 17:02:28 [INFO]	[TRAIN] Epoch=244/270, Step=60/74, loss=16.473162, lr=1e-06, time_each_step=0.19s, eta=0:18:17
    2021-08-11 17:02:29 [INFO]	[TRAIN] Epoch=244/270, Step=62/74, loss=10.598766, lr=1e-06, time_each_step=0.19s, eta=0:18:16
    2021-08-11 17:02:29 [INFO]	[TRAIN] Epoch=244/270, Step=64/74, loss=16.775881, lr=1e-06, time_each_step=0.19s, eta=0:18:16
    2021-08-11 17:02:29 [INFO]	[TRAIN] Epoch=244/270, Step=66/74, loss=10.70035, lr=1e-06, time_each_step=0.19s, eta=0:18:15
    2021-08-11 17:02:30 [INFO]	[TRAIN] Epoch=244/270, Step=68/74, loss=7.947384, lr=1e-06, time_each_step=0.19s, eta=0:18:15
    2021-08-11 17:02:30 [INFO]	[TRAIN] Epoch=244/270, Step=70/74, loss=10.531668, lr=1e-06, time_each_step=0.2s, eta=0:18:15
    2021-08-11 17:02:31 [INFO]	[TRAIN] Epoch=244/270, Step=72/74, loss=21.171227, lr=1e-06, time_each_step=0.2s, eta=0:18:14
    2021-08-11 17:02:31 [INFO]	[TRAIN] Epoch=244/270, Step=74/74, loss=19.849859, lr=1e-06, time_each_step=0.21s, eta=0:18:14
    2021-08-11 17:02:31 [INFO]	[TRAIN] Epoch 244 finished, loss=16.665752, lr=1e-06 .
    2021-08-11 17:02:46 [INFO]	[TRAIN] Epoch=245/270, Step=2/74, loss=20.582527, lr=1e-06, time_each_step=0.94s, eta=0:17:25
    2021-08-11 17:02:47 [INFO]	[TRAIN] Epoch=245/270, Step=4/74, loss=18.475304, lr=1e-06, time_each_step=0.96s, eta=0:17:24
    2021-08-11 17:02:48 [INFO]	[TRAIN] Epoch=245/270, Step=6/74, loss=9.881778, lr=1e-06, time_each_step=0.97s, eta=0:17:23
    2021-08-11 17:02:48 [INFO]	[TRAIN] Epoch=245/270, Step=8/74, loss=12.846863, lr=1e-06, time_each_step=0.98s, eta=0:17:21
    2021-08-11 17:02:49 [INFO]	[TRAIN] Epoch=245/270, Step=10/74, loss=15.334642, lr=1e-06, time_each_step=0.99s, eta=0:17:20
    2021-08-11 17:02:49 [INFO]	[TRAIN] Epoch=245/270, Step=12/74, loss=10.786816, lr=1e-06, time_each_step=1.0s, eta=0:17:19
    2021-08-11 17:02:50 [INFO]	[TRAIN] Epoch=245/270, Step=14/74, loss=25.519316, lr=1e-06, time_each_step=1.03s, eta=0:17:19
    2021-08-11 17:02:51 [INFO]	[TRAIN] Epoch=245/270, Step=16/74, loss=8.366788, lr=1e-06, time_each_step=1.04s, eta=0:17:17
    2021-08-11 17:02:52 [INFO]	[TRAIN] Epoch=245/270, Step=18/74, loss=19.789598, lr=1e-06, time_each_step=1.05s, eta=0:17:16
    2021-08-11 17:02:53 [INFO]	[TRAIN] Epoch=245/270, Step=20/74, loss=20.289238, lr=1e-06, time_each_step=1.08s, eta=0:17:15
    2021-08-11 17:02:54 [INFO]	[TRAIN] Epoch=245/270, Step=22/74, loss=29.858475, lr=1e-06, time_each_step=0.37s, eta=0:16:36
    2021-08-11 17:02:55 [INFO]	[TRAIN] Epoch=245/270, Step=24/74, loss=11.404804, lr=1e-06, time_each_step=0.38s, eta=0:16:36
    2021-08-11 17:02:56 [INFO]	[TRAIN] Epoch=245/270, Step=26/74, loss=20.892941, lr=1e-06, time_each_step=0.4s, eta=0:16:36
    2021-08-11 17:02:56 [INFO]	[TRAIN] Epoch=245/270, Step=28/74, loss=30.701942, lr=1e-06, time_each_step=0.4s, eta=0:16:35
    2021-08-11 17:02:57 [INFO]	[TRAIN] Epoch=245/270, Step=30/74, loss=11.379564, lr=1e-06, time_each_step=0.41s, eta=0:16:35
    2021-08-11 17:02:58 [INFO]	[TRAIN] Epoch=245/270, Step=32/74, loss=25.891459, lr=1e-06, time_each_step=0.41s, eta=0:16:34
    2021-08-11 17:02:59 [INFO]	[TRAIN] Epoch=245/270, Step=34/74, loss=4.612766, lr=1e-06, time_each_step=0.41s, eta=0:16:33
    2021-08-11 17:03:00 [INFO]	[TRAIN] Epoch=245/270, Step=36/74, loss=15.273854, lr=1e-06, time_each_step=0.43s, eta=0:16:33
    2021-08-11 17:03:00 [INFO]	[TRAIN] Epoch=245/270, Step=38/74, loss=10.646331, lr=1e-06, time_each_step=0.42s, eta=0:16:32
    2021-08-11 17:03:01 [INFO]	[TRAIN] Epoch=245/270, Step=40/74, loss=38.575882, lr=1e-06, time_each_step=0.4s, eta=0:16:30
    2021-08-11 17:03:01 [INFO]	[TRAIN] Epoch=245/270, Step=42/74, loss=14.439584, lr=1e-06, time_each_step=0.4s, eta=0:16:30
    2021-08-11 17:03:02 [INFO]	[TRAIN] Epoch=245/270, Step=44/74, loss=8.043037, lr=1e-06, time_each_step=0.37s, eta=0:16:28
    2021-08-11 17:03:03 [INFO]	[TRAIN] Epoch=245/270, Step=46/74, loss=29.971743, lr=1e-06, time_each_step=0.35s, eta=0:16:27
    2021-08-11 17:03:03 [INFO]	[TRAIN] Epoch=245/270, Step=48/74, loss=15.628805, lr=1e-06, time_each_step=0.34s, eta=0:16:26
    2021-08-11 17:03:04 [INFO]	[TRAIN] Epoch=245/270, Step=50/74, loss=16.627636, lr=1e-06, time_each_step=0.33s, eta=0:16:25
    2021-08-11 17:03:04 [INFO]	[TRAIN] Epoch=245/270, Step=52/74, loss=16.311714, lr=1e-06, time_each_step=0.31s, eta=0:16:24
    2021-08-11 17:03:04 [INFO]	[TRAIN] Epoch=245/270, Step=54/74, loss=13.975349, lr=1e-06, time_each_step=0.28s, eta=0:16:22
    2021-08-11 17:03:04 [INFO]	[TRAIN] Epoch=245/270, Step=56/74, loss=18.237743, lr=1e-06, time_each_step=0.24s, eta=0:16:21
    2021-08-11 17:03:05 [INFO]	[TRAIN] Epoch=245/270, Step=58/74, loss=23.726017, lr=1e-06, time_each_step=0.25s, eta=0:16:21
    2021-08-11 17:03:06 [INFO]	[TRAIN] Epoch=245/270, Step=60/74, loss=14.392422, lr=1e-06, time_each_step=0.24s, eta=0:16:20
    2021-08-11 17:03:06 [INFO]	[TRAIN] Epoch=245/270, Step=62/74, loss=12.793919, lr=1e-06, time_each_step=0.22s, eta=0:16:19
    2021-08-11 17:03:06 [INFO]	[TRAIN] Epoch=245/270, Step=64/74, loss=5.548244, lr=1e-06, time_each_step=0.21s, eta=0:16:19
    2021-08-11 17:03:07 [INFO]	[TRAIN] Epoch=245/270, Step=66/74, loss=24.530855, lr=1e-06, time_each_step=0.2s, eta=0:16:18
    2021-08-11 17:03:07 [INFO]	[TRAIN] Epoch=245/270, Step=68/74, loss=14.559402, lr=1e-06, time_each_step=0.19s, eta=0:16:18
    2021-08-11 17:03:07 [INFO]	[TRAIN] Epoch=245/270, Step=70/74, loss=13.373835, lr=1e-06, time_each_step=0.19s, eta=0:16:18
    2021-08-11 17:03:08 [INFO]	[TRAIN] Epoch=245/270, Step=72/74, loss=21.069689, lr=1e-06, time_each_step=0.19s, eta=0:16:17
    2021-08-11 17:03:08 [INFO]	[TRAIN] Epoch=245/270, Step=74/74, loss=24.33477, lr=1e-06, time_each_step=0.2s, eta=0:16:17
    2021-08-11 17:03:08 [INFO]	[TRAIN] Epoch 245 finished, loss=18.394566, lr=1e-06 .
    2021-08-11 17:03:13 [INFO]	[TRAIN] Epoch=246/270, Step=2/74, loss=19.393652, lr=1e-06, time_each_step=0.43s, eta=0:16:1
    2021-08-11 17:03:14 [INFO]	[TRAIN] Epoch=246/270, Step=4/74, loss=17.060955, lr=1e-06, time_each_step=0.46s, eta=0:16:2
    2021-08-11 17:03:15 [INFO]	[TRAIN] Epoch=246/270, Step=6/74, loss=14.266935, lr=1e-06, time_each_step=0.48s, eta=0:16:2
    2021-08-11 17:03:16 [INFO]	[TRAIN] Epoch=246/270, Step=8/74, loss=12.787843, lr=1e-06, time_each_step=0.5s, eta=0:16:3
    2021-08-11 17:03:17 [INFO]	[TRAIN] Epoch=246/270, Step=10/74, loss=42.261795, lr=1e-06, time_each_step=0.53s, eta=0:16:4
    2021-08-11 17:03:18 [INFO]	[TRAIN] Epoch=246/270, Step=12/74, loss=30.739559, lr=1e-06, time_each_step=0.55s, eta=0:16:4
    2021-08-11 17:03:18 [INFO]	[TRAIN] Epoch=246/270, Step=14/74, loss=9.936224, lr=1e-06, time_each_step=0.57s, eta=0:16:4
    2021-08-11 17:03:19 [INFO]	[TRAIN] Epoch=246/270, Step=16/74, loss=22.601925, lr=1e-06, time_each_step=0.59s, eta=0:16:4
    2021-08-11 17:03:20 [INFO]	[TRAIN] Epoch=246/270, Step=18/74, loss=25.323807, lr=1e-06, time_each_step=0.61s, eta=0:16:4
    2021-08-11 17:03:21 [INFO]	[TRAIN] Epoch=246/270, Step=20/74, loss=8.065105, lr=1e-06, time_each_step=0.63s, eta=0:16:4
    2021-08-11 17:03:21 [INFO]	[TRAIN] Epoch=246/270, Step=22/74, loss=26.491903, lr=1e-06, time_each_step=0.42s, eta=0:15:51
    2021-08-11 17:03:23 [INFO]	[TRAIN] Epoch=246/270, Step=24/74, loss=16.922173, lr=1e-06, time_each_step=0.42s, eta=0:15:51
    2021-08-11 17:03:23 [INFO]	[TRAIN] Epoch=246/270, Step=26/74, loss=24.175718, lr=1e-06, time_each_step=0.41s, eta=0:15:50
    2021-08-11 17:03:25 [INFO]	[TRAIN] Epoch=246/270, Step=28/74, loss=11.72251, lr=1e-06, time_each_step=0.43s, eta=0:15:50
    2021-08-11 17:03:26 [INFO]	[TRAIN] Epoch=246/270, Step=30/74, loss=25.574686, lr=1e-06, time_each_step=0.45s, eta=0:15:50
    2021-08-11 17:03:27 [INFO]	[TRAIN] Epoch=246/270, Step=32/74, loss=26.4874, lr=1e-06, time_each_step=0.45s, eta=0:15:49
    2021-08-11 17:03:27 [INFO]	[TRAIN] Epoch=246/270, Step=34/74, loss=17.340433, lr=1e-06, time_each_step=0.45s, eta=0:15:48
    2021-08-11 17:03:28 [INFO]	[TRAIN] Epoch=246/270, Step=36/74, loss=18.085037, lr=1e-06, time_each_step=0.45s, eta=0:15:47
    2021-08-11 17:03:29 [INFO]	[TRAIN] Epoch=246/270, Step=38/74, loss=13.256207, lr=1e-06, time_each_step=0.44s, eta=0:15:46
    2021-08-11 17:03:29 [INFO]	[TRAIN] Epoch=246/270, Step=40/74, loss=15.143068, lr=1e-06, time_each_step=0.42s, eta=0:15:44
    2021-08-11 17:03:30 [INFO]	[TRAIN] Epoch=246/270, Step=42/74, loss=22.240026, lr=1e-06, time_each_step=0.42s, eta=0:15:43
    2021-08-11 17:03:30 [INFO]	[TRAIN] Epoch=246/270, Step=44/74, loss=15.419198, lr=1e-06, time_each_step=0.39s, eta=0:15:42
    2021-08-11 17:03:31 [INFO]	[TRAIN] Epoch=246/270, Step=46/74, loss=38.923153, lr=1e-06, time_each_step=0.37s, eta=0:15:40
    2021-08-11 17:03:31 [INFO]	[TRAIN] Epoch=246/270, Step=48/74, loss=11.110708, lr=1e-06, time_each_step=0.33s, eta=0:15:38
    2021-08-11 17:03:32 [INFO]	[TRAIN] Epoch=246/270, Step=50/74, loss=22.107491, lr=1e-06, time_each_step=0.29s, eta=0:15:37
    2021-08-11 17:03:32 [INFO]	[TRAIN] Epoch=246/270, Step=52/74, loss=11.173767, lr=1e-06, time_each_step=0.28s, eta=0:15:36
    2021-08-11 17:03:33 [INFO]	[TRAIN] Epoch=246/270, Step=54/74, loss=19.635609, lr=1e-06, time_each_step=0.27s, eta=0:15:35
    2021-08-11 17:03:33 [INFO]	[TRAIN] Epoch=246/270, Step=56/74, loss=20.106424, lr=1e-06, time_each_step=0.25s, eta=0:15:34
    2021-08-11 17:03:34 [INFO]	[TRAIN] Epoch=246/270, Step=58/74, loss=14.48654, lr=1e-06, time_each_step=0.25s, eta=0:15:34
    2021-08-11 17:03:34 [INFO]	[TRAIN] Epoch=246/270, Step=60/74, loss=8.959755, lr=1e-06, time_each_step=0.24s, eta=0:15:33
    2021-08-11 17:03:34 [INFO]	[TRAIN] Epoch=246/270, Step=62/74, loss=19.211189, lr=1e-06, time_each_step=0.23s, eta=0:15:33
    2021-08-11 17:03:35 [INFO]	[TRAIN] Epoch=246/270, Step=64/74, loss=14.931278, lr=1e-06, time_each_step=0.23s, eta=0:15:32
    2021-08-11 17:03:35 [INFO]	[TRAIN] Epoch=246/270, Step=66/74, loss=9.376048, lr=1e-06, time_each_step=0.22s, eta=0:15:32
    2021-08-11 17:03:36 [INFO]	[TRAIN] Epoch=246/270, Step=68/74, loss=15.133278, lr=1e-06, time_each_step=0.22s, eta=0:15:31
    2021-08-11 17:03:36 [INFO]	[TRAIN] Epoch=246/270, Step=70/74, loss=16.938923, lr=1e-06, time_each_step=0.23s, eta=0:15:31
    2021-08-11 17:03:37 [INFO]	[TRAIN] Epoch=246/270, Step=72/74, loss=25.377907, lr=1e-06, time_each_step=0.21s, eta=0:15:30
    2021-08-11 17:03:37 [INFO]	[TRAIN] Epoch=246/270, Step=74/74, loss=12.197422, lr=1e-06, time_each_step=0.22s, eta=0:15:30
    2021-08-11 17:03:37 [INFO]	[TRAIN] Epoch 246 finished, loss=19.347858, lr=1e-06 .
    2021-08-11 17:03:44 [INFO]	[TRAIN] Epoch=247/270, Step=2/74, loss=12.913391, lr=1e-06, time_each_step=0.56s, eta=0:12:26
    2021-08-11 17:03:45 [INFO]	[TRAIN] Epoch=247/270, Step=4/74, loss=14.018398, lr=1e-06, time_each_step=0.57s, eta=0:12:25
    2021-08-11 17:03:46 [INFO]	[TRAIN] Epoch=247/270, Step=6/74, loss=32.884518, lr=1e-06, time_each_step=0.62s, eta=0:12:27
    2021-08-11 17:03:47 [INFO]	[TRAIN] Epoch=247/270, Step=8/74, loss=19.519625, lr=1e-06, time_each_step=0.63s, eta=0:12:27
    2021-08-11 17:03:48 [INFO]	[TRAIN] Epoch=247/270, Step=10/74, loss=5.880631, lr=1e-06, time_each_step=0.65s, eta=0:12:27
    2021-08-11 17:03:48 [INFO]	[TRAIN] Epoch=247/270, Step=12/74, loss=16.012562, lr=1e-06, time_each_step=0.66s, eta=0:12:26
    2021-08-11 17:03:49 [INFO]	[TRAIN] Epoch=247/270, Step=14/74, loss=32.123165, lr=1e-06, time_each_step=0.67s, eta=0:12:26
    2021-08-11 17:03:50 [INFO]	[TRAIN] Epoch=247/270, Step=16/74, loss=31.895119, lr=1e-06, time_each_step=0.67s, eta=0:12:24
    2021-08-11 17:03:50 [INFO]	[TRAIN] Epoch=247/270, Step=18/74, loss=9.137186, lr=1e-06, time_each_step=0.69s, eta=0:12:24
    2021-08-11 17:03:51 [INFO]	[TRAIN] Epoch=247/270, Step=20/74, loss=17.749104, lr=1e-06, time_each_step=0.7s, eta=0:12:23
    2021-08-11 17:03:51 [INFO]	[TRAIN] Epoch=247/270, Step=22/74, loss=11.591364, lr=1e-06, time_each_step=0.36s, eta=0:12:4
    2021-08-11 17:03:52 [INFO]	[TRAIN] Epoch=247/270, Step=24/74, loss=22.318741, lr=1e-06, time_each_step=0.36s, eta=0:12:3
    2021-08-11 17:03:53 [INFO]	[TRAIN] Epoch=247/270, Step=26/74, loss=18.872833, lr=1e-06, time_each_step=0.33s, eta=0:12:1
    2021-08-11 17:03:54 [INFO]	[TRAIN] Epoch=247/270, Step=28/74, loss=18.62401, lr=1e-06, time_each_step=0.34s, eta=0:12:1
    2021-08-11 17:03:55 [INFO]	[TRAIN] Epoch=247/270, Step=30/74, loss=14.246393, lr=1e-06, time_each_step=0.35s, eta=0:12:1
    2021-08-11 17:03:56 [INFO]	[TRAIN] Epoch=247/270, Step=32/74, loss=13.215023, lr=1e-06, time_each_step=0.36s, eta=0:12:0
    2021-08-11 17:03:56 [INFO]	[TRAIN] Epoch=247/270, Step=34/74, loss=11.608496, lr=1e-06, time_each_step=0.37s, eta=0:12:0
    2021-08-11 17:03:57 [INFO]	[TRAIN] Epoch=247/270, Step=36/74, loss=25.646847, lr=1e-06, time_each_step=0.38s, eta=0:12:0
    2021-08-11 17:03:58 [INFO]	[TRAIN] Epoch=247/270, Step=38/74, loss=23.524014, lr=1e-06, time_each_step=0.38s, eta=0:11:59
    2021-08-11 17:03:59 [INFO]	[TRAIN] Epoch=247/270, Step=40/74, loss=13.953949, lr=1e-06, time_each_step=0.39s, eta=0:11:58
    2021-08-11 17:03:59 [INFO]	[TRAIN] Epoch=247/270, Step=42/74, loss=12.731934, lr=1e-06, time_each_step=0.4s, eta=0:11:58
    2021-08-11 17:04:00 [INFO]	[TRAIN] Epoch=247/270, Step=44/74, loss=15.227879, lr=1e-06, time_each_step=0.4s, eta=0:11:57
    2021-08-11 17:04:01 [INFO]	[TRAIN] Epoch=247/270, Step=46/74, loss=32.726402, lr=1e-06, time_each_step=0.39s, eta=0:11:56
    2021-08-11 17:04:01 [INFO]	[TRAIN] Epoch=247/270, Step=48/74, loss=19.981142, lr=1e-06, time_each_step=0.36s, eta=0:11:55
    2021-08-11 17:04:02 [INFO]	[TRAIN] Epoch=247/270, Step=50/74, loss=25.544643, lr=1e-06, time_each_step=0.35s, eta=0:11:54
    2021-08-11 17:04:02 [INFO]	[TRAIN] Epoch=247/270, Step=52/74, loss=24.263756, lr=1e-06, time_each_step=0.33s, eta=0:11:53
    2021-08-11 17:04:02 [INFO]	[TRAIN] Epoch=247/270, Step=54/74, loss=11.421052, lr=1e-06, time_each_step=0.3s, eta=0:11:51
    2021-08-11 17:04:03 [INFO]	[TRAIN] Epoch=247/270, Step=56/74, loss=14.292194, lr=1e-06, time_each_step=0.28s, eta=0:11:50
    2021-08-11 17:04:03 [INFO]	[TRAIN] Epoch=247/270, Step=58/74, loss=13.034138, lr=1e-06, time_each_step=0.26s, eta=0:11:49
    2021-08-11 17:04:04 [INFO]	[TRAIN] Epoch=247/270, Step=60/74, loss=11.557163, lr=1e-06, time_each_step=0.25s, eta=0:11:49
    2021-08-11 17:04:04 [INFO]	[TRAIN] Epoch=247/270, Step=62/74, loss=13.584066, lr=1e-06, time_each_step=0.22s, eta=0:11:48
    2021-08-11 17:04:04 [INFO]	[TRAIN] Epoch=247/270, Step=64/74, loss=10.153553, lr=1e-06, time_each_step=0.21s, eta=0:11:47
    2021-08-11 17:04:05 [INFO]	[TRAIN] Epoch=247/270, Step=66/74, loss=10.553637, lr=1e-06, time_each_step=0.21s, eta=0:11:47
    2021-08-11 17:04:05 [INFO]	[TRAIN] Epoch=247/270, Step=68/74, loss=20.224911, lr=1e-06, time_each_step=0.2s, eta=0:11:47
    2021-08-11 17:04:06 [INFO]	[TRAIN] Epoch=247/270, Step=70/74, loss=11.190817, lr=1e-06, time_each_step=0.2s, eta=0:11:46
    2021-08-11 17:04:06 [INFO]	[TRAIN] Epoch=247/270, Step=72/74, loss=25.696638, lr=1e-06, time_each_step=0.2s, eta=0:11:46
    2021-08-11 17:04:06 [INFO]	[TRAIN] Epoch=247/270, Step=74/74, loss=12.510579, lr=1e-06, time_each_step=0.2s, eta=0:11:45
    2021-08-11 17:04:06 [INFO]	[TRAIN] Epoch 247 finished, loss=16.792812, lr=1e-06 .
    2021-08-11 17:04:17 [INFO]	[TRAIN] Epoch=248/270, Step=2/74, loss=8.736426, lr=1e-06, time_each_step=0.7s, eta=0:12:20
    2021-08-11 17:04:18 [INFO]	[TRAIN] Epoch=248/270, Step=4/74, loss=9.961096, lr=1e-06, time_each_step=0.72s, eta=0:12:21
    2021-08-11 17:04:18 [INFO]	[TRAIN] Epoch=248/270, Step=6/74, loss=25.710936, lr=1e-06, time_each_step=0.73s, eta=0:12:20
    2021-08-11 17:04:19 [INFO]	[TRAIN] Epoch=248/270, Step=8/74, loss=13.405343, lr=1e-06, time_each_step=0.77s, eta=0:12:21
    2021-08-11 17:04:20 [INFO]	[TRAIN] Epoch=248/270, Step=10/74, loss=25.740032, lr=1e-06, time_each_step=0.78s, eta=0:12:20
    2021-08-11 17:04:21 [INFO]	[TRAIN] Epoch=248/270, Step=12/74, loss=9.775371, lr=1e-06, time_each_step=0.81s, eta=0:12:20
    2021-08-11 17:04:22 [INFO]	[TRAIN] Epoch=248/270, Step=14/74, loss=13.549339, lr=1e-06, time_each_step=0.83s, eta=0:12:20
    2021-08-11 17:04:23 [INFO]	[TRAIN] Epoch=248/270, Step=16/74, loss=13.476641, lr=1e-06, time_each_step=0.84s, eta=0:12:19
    2021-08-11 17:04:23 [INFO]	[TRAIN] Epoch=248/270, Step=18/74, loss=15.981735, lr=1e-06, time_each_step=0.87s, eta=0:12:19
    2021-08-11 17:04:24 [INFO]	[TRAIN] Epoch=248/270, Step=20/74, loss=18.544794, lr=1e-06, time_each_step=0.9s, eta=0:12:19
    2021-08-11 17:04:25 [INFO]	[TRAIN] Epoch=248/270, Step=22/74, loss=30.716187, lr=1e-06, time_each_step=0.41s, eta=0:11:51
    2021-08-11 17:04:26 [INFO]	[TRAIN] Epoch=248/270, Step=24/74, loss=10.552999, lr=1e-06, time_each_step=0.41s, eta=0:11:51
    2021-08-11 17:04:27 [INFO]	[TRAIN] Epoch=248/270, Step=26/74, loss=19.623566, lr=1e-06, time_each_step=0.41s, eta=0:11:50
    2021-08-11 17:04:27 [INFO]	[TRAIN] Epoch=248/270, Step=28/74, loss=13.163982, lr=1e-06, time_each_step=0.41s, eta=0:11:49
    2021-08-11 17:04:28 [INFO]	[TRAIN] Epoch=248/270, Step=30/74, loss=18.666567, lr=1e-06, time_each_step=0.41s, eta=0:11:48
    2021-08-11 17:04:29 [INFO]	[TRAIN] Epoch=248/270, Step=32/74, loss=29.29714, lr=1e-06, time_each_step=0.4s, eta=0:11:47
    2021-08-11 17:04:30 [INFO]	[TRAIN] Epoch=248/270, Step=34/74, loss=12.706154, lr=1e-06, time_each_step=0.39s, eta=0:11:46
    2021-08-11 17:04:31 [INFO]	[TRAIN] Epoch=248/270, Step=36/74, loss=16.128239, lr=1e-06, time_each_step=0.4s, eta=0:11:45
    2021-08-11 17:04:31 [INFO]	[TRAIN] Epoch=248/270, Step=38/74, loss=13.938298, lr=1e-06, time_each_step=0.39s, eta=0:11:44
    2021-08-11 17:04:32 [INFO]	[TRAIN] Epoch=248/270, Step=40/74, loss=19.05117, lr=1e-06, time_each_step=0.37s, eta=0:11:43
    2021-08-11 17:04:32 [INFO]	[TRAIN] Epoch=248/270, Step=42/74, loss=14.040598, lr=1e-06, time_each_step=0.37s, eta=0:11:42
    2021-08-11 17:04:33 [INFO]	[TRAIN] Epoch=248/270, Step=44/74, loss=14.960966, lr=1e-06, time_each_step=0.34s, eta=0:11:40
    2021-08-11 17:04:33 [INFO]	[TRAIN] Epoch=248/270, Step=46/74, loss=20.18202, lr=1e-06, time_each_step=0.33s, eta=0:11:39
    2021-08-11 17:04:34 [INFO]	[TRAIN] Epoch=248/270, Step=48/74, loss=18.053272, lr=1e-06, time_each_step=0.33s, eta=0:11:39
    2021-08-11 17:04:34 [INFO]	[TRAIN] Epoch=248/270, Step=50/74, loss=11.525761, lr=1e-06, time_each_step=0.31s, eta=0:11:38
    2021-08-11 17:04:35 [INFO]	[TRAIN] Epoch=248/270, Step=52/74, loss=12.967419, lr=1e-06, time_each_step=0.29s, eta=0:11:37
    2021-08-11 17:04:35 [INFO]	[TRAIN] Epoch=248/270, Step=54/74, loss=14.91556, lr=1e-06, time_each_step=0.27s, eta=0:11:36
    2021-08-11 17:04:35 [INFO]	[TRAIN] Epoch=248/270, Step=56/74, loss=16.221928, lr=1e-06, time_each_step=0.24s, eta=0:11:35
    2021-08-11 17:04:36 [INFO]	[TRAIN] Epoch=248/270, Step=58/74, loss=7.170778, lr=1e-06, time_each_step=0.24s, eta=0:11:34
    2021-08-11 17:04:36 [INFO]	[TRAIN] Epoch=248/270, Step=60/74, loss=15.985259, lr=1e-06, time_each_step=0.24s, eta=0:11:33
    2021-08-11 17:04:37 [INFO]	[TRAIN] Epoch=248/270, Step=62/74, loss=16.648193, lr=1e-06, time_each_step=0.23s, eta=0:11:33
    2021-08-11 17:04:37 [INFO]	[TRAIN] Epoch=248/270, Step=64/74, loss=8.970004, lr=1e-06, time_each_step=0.23s, eta=0:11:32
    2021-08-11 17:04:38 [INFO]	[TRAIN] Epoch=248/270, Step=66/74, loss=20.966614, lr=1e-06, time_each_step=0.23s, eta=0:11:32
    2021-08-11 17:04:38 [INFO]	[TRAIN] Epoch=248/270, Step=68/74, loss=21.010601, lr=1e-06, time_each_step=0.21s, eta=0:11:31
    2021-08-11 17:04:39 [INFO]	[TRAIN] Epoch=248/270, Step=70/74, loss=13.075502, lr=1e-06, time_each_step=0.21s, eta=0:11:31
    2021-08-11 17:04:39 [INFO]	[TRAIN] Epoch=248/270, Step=72/74, loss=15.305817, lr=1e-06, time_each_step=0.2s, eta=0:11:31
    2021-08-11 17:04:39 [INFO]	[TRAIN] Epoch=248/270, Step=74/74, loss=22.950216, lr=1e-06, time_each_step=0.21s, eta=0:11:30
    2021-08-11 17:04:39 [INFO]	[TRAIN] Epoch 248 finished, loss=17.182688, lr=1e-06 .
    2021-08-11 17:04:49 [INFO]	[TRAIN] Epoch=249/270, Step=2/74, loss=25.436716, lr=1e-06, time_each_step=0.7s, eta=0:13:3
    2021-08-11 17:04:50 [INFO]	[TRAIN] Epoch=249/270, Step=4/74, loss=17.046783, lr=1e-06, time_each_step=0.71s, eta=0:13:3
    2021-08-11 17:04:51 [INFO]	[TRAIN] Epoch=249/270, Step=6/74, loss=31.224266, lr=1e-06, time_each_step=0.74s, eta=0:13:4
    2021-08-11 17:04:52 [INFO]	[TRAIN] Epoch=249/270, Step=8/74, loss=24.080883, lr=1e-06, time_each_step=0.76s, eta=0:13:3
    2021-08-11 17:04:53 [INFO]	[TRAIN] Epoch=249/270, Step=10/74, loss=9.49597, lr=1e-06, time_each_step=0.79s, eta=0:13:3
    2021-08-11 17:04:54 [INFO]	[TRAIN] Epoch=249/270, Step=12/74, loss=10.636784, lr=1e-06, time_each_step=0.8s, eta=0:13:3
    2021-08-11 17:04:55 [INFO]	[TRAIN] Epoch=249/270, Step=14/74, loss=13.838444, lr=1e-06, time_each_step=0.82s, eta=0:13:2
    2021-08-11 17:04:55 [INFO]	[TRAIN] Epoch=249/270, Step=16/74, loss=10.775339, lr=1e-06, time_each_step=0.84s, eta=0:13:2
    2021-08-11 17:04:56 [INFO]	[TRAIN] Epoch=249/270, Step=18/74, loss=13.077501, lr=1e-06, time_each_step=0.86s, eta=0:13:1
    2021-08-11 17:04:57 [INFO]	[TRAIN] Epoch=249/270, Step=20/74, loss=21.50943, lr=1e-06, time_each_step=0.87s, eta=0:13:0
    2021-08-11 17:04:58 [INFO]	[TRAIN] Epoch=249/270, Step=22/74, loss=25.228195, lr=1e-06, time_each_step=0.42s, eta=0:12:35
    2021-08-11 17:04:59 [INFO]	[TRAIN] Epoch=249/270, Step=24/74, loss=16.940939, lr=1e-06, time_each_step=0.42s, eta=0:12:34
    2021-08-11 17:05:00 [INFO]	[TRAIN] Epoch=249/270, Step=26/74, loss=24.150478, lr=1e-06, time_each_step=0.42s, eta=0:12:33
    2021-08-11 17:05:01 [INFO]	[TRAIN] Epoch=249/270, Step=28/74, loss=11.381535, lr=1e-06, time_each_step=0.43s, eta=0:12:33
    2021-08-11 17:05:01 [INFO]	[TRAIN] Epoch=249/270, Step=30/74, loss=13.878761, lr=1e-06, time_each_step=0.42s, eta=0:12:32
    2021-08-11 17:05:03 [INFO]	[TRAIN] Epoch=249/270, Step=32/74, loss=14.867743, lr=1e-06, time_each_step=0.44s, eta=0:12:31
    2021-08-11 17:05:03 [INFO]	[TRAIN] Epoch=249/270, Step=34/74, loss=15.837925, lr=1e-06, time_each_step=0.44s, eta=0:12:31
    2021-08-11 17:05:04 [INFO]	[TRAIN] Epoch=249/270, Step=36/74, loss=28.823784, lr=1e-06, time_each_step=0.43s, eta=0:12:29
    2021-08-11 17:05:05 [INFO]	[TRAIN] Epoch=249/270, Step=38/74, loss=11.424738, lr=1e-06, time_each_step=0.43s, eta=0:12:29
    2021-08-11 17:05:05 [INFO]	[TRAIN] Epoch=249/270, Step=40/74, loss=32.522514, lr=1e-06, time_each_step=0.41s, eta=0:12:27
    2021-08-11 17:05:05 [INFO]	[TRAIN] Epoch=249/270, Step=42/74, loss=18.307156, lr=1e-06, time_each_step=0.37s, eta=0:12:25
    2021-08-11 17:05:06 [INFO]	[TRAIN] Epoch=249/270, Step=44/74, loss=5.80807, lr=1e-06, time_each_step=0.35s, eta=0:12:23
    2021-08-11 17:05:06 [INFO]	[TRAIN] Epoch=249/270, Step=46/74, loss=12.731306, lr=1e-06, time_each_step=0.31s, eta=0:12:22
    2021-08-11 17:05:07 [INFO]	[TRAIN] Epoch=249/270, Step=48/74, loss=34.959572, lr=1e-06, time_each_step=0.29s, eta=0:12:21
    2021-08-11 17:05:07 [INFO]	[TRAIN] Epoch=249/270, Step=50/74, loss=14.66158, lr=1e-06, time_each_step=0.28s, eta=0:12:20
    2021-08-11 17:05:07 [INFO]	[TRAIN] Epoch=249/270, Step=52/74, loss=12.393915, lr=1e-06, time_each_step=0.24s, eta=0:12:18
    2021-08-11 17:05:08 [INFO]	[TRAIN] Epoch=249/270, Step=54/74, loss=41.253216, lr=1e-06, time_each_step=0.22s, eta=0:12:17
    2021-08-11 17:05:08 [INFO]	[TRAIN] Epoch=249/270, Step=56/74, loss=29.682508, lr=1e-06, time_each_step=0.2s, eta=0:12:17
    2021-08-11 17:05:08 [INFO]	[TRAIN] Epoch=249/270, Step=58/74, loss=15.177559, lr=1e-06, time_each_step=0.19s, eta=0:12:16
    2021-08-11 17:05:09 [INFO]	[TRAIN] Epoch=249/270, Step=60/74, loss=18.619528, lr=1e-06, time_each_step=0.2s, eta=0:12:16
    2021-08-11 17:05:09 [INFO]	[TRAIN] Epoch=249/270, Step=62/74, loss=25.346653, lr=1e-06, time_each_step=0.2s, eta=0:12:15
    2021-08-11 17:05:10 [INFO]	[TRAIN] Epoch=249/270, Step=64/74, loss=31.4828, lr=1e-06, time_each_step=0.2s, eta=0:12:15
    2021-08-11 17:05:10 [INFO]	[TRAIN] Epoch=249/270, Step=66/74, loss=22.694853, lr=1e-06, time_each_step=0.2s, eta=0:12:15
    2021-08-11 17:05:10 [INFO]	[TRAIN] Epoch=249/270, Step=68/74, loss=13.566778, lr=1e-06, time_each_step=0.18s, eta=0:12:14
    2021-08-11 17:05:11 [INFO]	[TRAIN] Epoch=249/270, Step=70/74, loss=20.012783, lr=1e-06, time_each_step=0.18s, eta=0:12:14
    2021-08-11 17:05:11 [INFO]	[TRAIN] Epoch=249/270, Step=72/74, loss=14.670004, lr=1e-06, time_each_step=0.18s, eta=0:12:13
    2021-08-11 17:05:11 [INFO]	[TRAIN] Epoch=249/270, Step=74/74, loss=20.88302, lr=1e-06, time_each_step=0.19s, eta=0:12:13
    2021-08-11 17:05:11 [INFO]	[TRAIN] Epoch 249 finished, loss=18.764696, lr=1e-06 .
    2021-08-11 17:05:30 [INFO]	[TRAIN] Epoch=250/270, Step=2/74, loss=15.516754, lr=1e-06, time_each_step=1.09s, eta=0:12:43
    2021-08-11 17:05:31 [INFO]	[TRAIN] Epoch=250/270, Step=4/74, loss=19.887632, lr=1e-06, time_each_step=1.11s, eta=0:12:43
    2021-08-11 17:05:31 [INFO]	[TRAIN] Epoch=250/270, Step=6/74, loss=9.344942, lr=1e-06, time_each_step=1.13s, eta=0:12:42
    2021-08-11 17:05:33 [INFO]	[TRAIN] Epoch=250/270, Step=8/74, loss=23.658495, lr=1e-06, time_each_step=1.17s, eta=0:12:42
    2021-08-11 17:05:34 [INFO]	[TRAIN] Epoch=250/270, Step=10/74, loss=17.988005, lr=1e-06, time_each_step=1.19s, eta=0:12:41
    2021-08-11 17:05:34 [INFO]	[TRAIN] Epoch=250/270, Step=12/74, loss=30.074842, lr=1e-06, time_each_step=1.21s, eta=0:12:40
    2021-08-11 17:05:35 [INFO]	[TRAIN] Epoch=250/270, Step=14/74, loss=23.070602, lr=1e-06, time_each_step=1.23s, eta=0:12:39
    2021-08-11 17:05:36 [INFO]	[TRAIN] Epoch=250/270, Step=16/74, loss=46.848694, lr=1e-06, time_each_step=1.27s, eta=0:12:38
    2021-08-11 17:05:37 [INFO]	[TRAIN] Epoch=250/270, Step=18/74, loss=19.295521, lr=1e-06, time_each_step=1.28s, eta=0:12:36
    2021-08-11 17:05:38 [INFO]	[TRAIN] Epoch=250/270, Step=20/74, loss=22.909472, lr=1e-06, time_each_step=1.31s, eta=0:12:36
    2021-08-11 17:05:39 [INFO]	[TRAIN] Epoch=250/270, Step=22/74, loss=15.12096, lr=1e-06, time_each_step=0.44s, eta=0:11:47
    2021-08-11 17:05:39 [INFO]	[TRAIN] Epoch=250/270, Step=24/74, loss=8.799507, lr=1e-06, time_each_step=0.43s, eta=0:11:46
    2021-08-11 17:05:40 [INFO]	[TRAIN] Epoch=250/270, Step=26/74, loss=12.608936, lr=1e-06, time_each_step=0.41s, eta=0:11:45
    2021-08-11 17:05:41 [INFO]	[TRAIN] Epoch=250/270, Step=28/74, loss=10.000281, lr=1e-06, time_each_step=0.4s, eta=0:11:43
    2021-08-11 17:05:41 [INFO]	[TRAIN] Epoch=250/270, Step=30/74, loss=17.578199, lr=1e-06, time_each_step=0.38s, eta=0:11:41
    2021-08-11 17:05:42 [INFO]	[TRAIN] Epoch=250/270, Step=32/74, loss=25.678068, lr=1e-06, time_each_step=0.37s, eta=0:11:40
    2021-08-11 17:05:42 [INFO]	[TRAIN] Epoch=250/270, Step=34/74, loss=12.8106, lr=1e-06, time_each_step=0.37s, eta=0:11:40
    2021-08-11 17:05:43 [INFO]	[TRAIN] Epoch=250/270, Step=36/74, loss=22.451517, lr=1e-06, time_each_step=0.36s, eta=0:11:39
    2021-08-11 17:05:44 [INFO]	[TRAIN] Epoch=250/270, Step=38/74, loss=18.3699, lr=1e-06, time_each_step=0.4s, eta=0:11:39
    2021-08-11 17:05:45 [INFO]	[TRAIN] Epoch=250/270, Step=40/74, loss=24.964939, lr=1e-06, time_each_step=0.38s, eta=0:11:38
    2021-08-11 17:05:46 [INFO]	[TRAIN] Epoch=250/270, Step=42/74, loss=9.969803, lr=1e-06, time_each_step=0.36s, eta=0:11:36
    2021-08-11 17:05:46 [INFO]	[TRAIN] Epoch=250/270, Step=44/74, loss=8.613879, lr=1e-06, time_each_step=0.35s, eta=0:11:35
    2021-08-11 17:05:47 [INFO]	[TRAIN] Epoch=250/270, Step=46/74, loss=15.357561, lr=1e-06, time_each_step=0.34s, eta=0:11:34
    2021-08-11 17:05:47 [INFO]	[TRAIN] Epoch=250/270, Step=48/74, loss=31.376991, lr=1e-06, time_each_step=0.32s, eta=0:11:33
    2021-08-11 17:05:47 [INFO]	[TRAIN] Epoch=250/270, Step=50/74, loss=12.665736, lr=1e-06, time_each_step=0.31s, eta=0:11:32
    2021-08-11 17:05:48 [INFO]	[TRAIN] Epoch=250/270, Step=52/74, loss=10.935116, lr=1e-06, time_each_step=0.3s, eta=0:11:31
    2021-08-11 17:05:48 [INFO]	[TRAIN] Epoch=250/270, Step=54/74, loss=10.292356, lr=1e-06, time_each_step=0.29s, eta=0:11:31
    2021-08-11 17:05:49 [INFO]	[TRAIN] Epoch=250/270, Step=56/74, loss=33.737671, lr=1e-06, time_each_step=0.27s, eta=0:11:30
    2021-08-11 17:05:49 [INFO]	[TRAIN] Epoch=250/270, Step=58/74, loss=21.505514, lr=1e-06, time_each_step=0.23s, eta=0:11:29
    2021-08-11 17:05:50 [INFO]	[TRAIN] Epoch=250/270, Step=60/74, loss=14.869049, lr=1e-06, time_each_step=0.21s, eta=0:11:28
    2021-08-11 17:05:50 [INFO]	[TRAIN] Epoch=250/270, Step=62/74, loss=26.966469, lr=1e-06, time_each_step=0.22s, eta=0:11:27
    2021-08-11 17:05:51 [INFO]	[TRAIN] Epoch=250/270, Step=64/74, loss=15.983927, lr=1e-06, time_each_step=0.22s, eta=0:11:27
    2021-08-11 17:05:51 [INFO]	[TRAIN] Epoch=250/270, Step=66/74, loss=14.394541, lr=1e-06, time_each_step=0.22s, eta=0:11:27
    2021-08-11 17:05:52 [INFO]	[TRAIN] Epoch=250/270, Step=68/74, loss=16.632973, lr=1e-06, time_each_step=0.24s, eta=0:11:26
    2021-08-11 17:05:52 [INFO]	[TRAIN] Epoch=250/270, Step=70/74, loss=20.295414, lr=1e-06, time_each_step=0.25s, eta=0:11:26
    2021-08-11 17:05:53 [INFO]	[TRAIN] Epoch=250/270, Step=72/74, loss=11.05209, lr=1e-06, time_each_step=0.24s, eta=0:11:25
    2021-08-11 17:05:53 [INFO]	[TRAIN] Epoch=250/270, Step=74/74, loss=29.38755, lr=1e-06, time_each_step=0.23s, eta=0:11:25
    2021-08-11 17:05:53 [INFO]	[TRAIN] Epoch 250 finished, loss=17.838915, lr=1e-06 .
    2021-08-11 17:06:04 [INFO]	[TRAIN] Epoch=251/270, Step=2/74, loss=4.741169, lr=1e-06, time_each_step=0.77s, eta=0:14:44
    2021-08-11 17:06:05 [INFO]	[TRAIN] Epoch=251/270, Step=4/74, loss=22.173359, lr=1e-06, time_each_step=0.79s, eta=0:14:44
    2021-08-11 17:06:05 [INFO]	[TRAIN] Epoch=251/270, Step=6/74, loss=13.326117, lr=1e-06, time_each_step=0.79s, eta=0:14:43
    2021-08-11 17:06:06 [INFO]	[TRAIN] Epoch=251/270, Step=8/74, loss=7.74002, lr=1e-06, time_each_step=0.8s, eta=0:14:41
    2021-08-11 17:06:07 [INFO]	[TRAIN] Epoch=251/270, Step=10/74, loss=12.082022, lr=1e-06, time_each_step=0.8s, eta=0:14:40
    2021-08-11 17:06:07 [INFO]	[TRAIN] Epoch=251/270, Step=12/74, loss=12.613152, lr=1e-06, time_each_step=0.81s, eta=0:14:39
    2021-08-11 17:06:08 [INFO]	[TRAIN] Epoch=251/270, Step=14/74, loss=16.180534, lr=1e-06, time_each_step=0.81s, eta=0:14:38
    2021-08-11 17:06:09 [INFO]	[TRAIN] Epoch=251/270, Step=16/74, loss=30.975925, lr=1e-06, time_each_step=0.83s, eta=0:14:37
    2021-08-11 17:06:10 [INFO]	[TRAIN] Epoch=251/270, Step=18/74, loss=34.29493, lr=1e-06, time_each_step=0.85s, eta=0:14:37
    2021-08-11 17:06:10 [INFO]	[TRAIN] Epoch=251/270, Step=20/74, loss=12.715586, lr=1e-06, time_each_step=0.88s, eta=0:14:36
    2021-08-11 17:06:11 [INFO]	[TRAIN] Epoch=251/270, Step=22/74, loss=14.746756, lr=1e-06, time_each_step=0.34s, eta=0:14:7
    2021-08-11 17:06:11 [INFO]	[TRAIN] Epoch=251/270, Step=24/74, loss=23.25938, lr=1e-06, time_each_step=0.32s, eta=0:14:5
    2021-08-11 17:06:12 [INFO]	[TRAIN] Epoch=251/270, Step=26/74, loss=10.453472, lr=1e-06, time_each_step=0.33s, eta=0:14:5
    2021-08-11 17:06:13 [INFO]	[TRAIN] Epoch=251/270, Step=28/74, loss=18.576229, lr=1e-06, time_each_step=0.34s, eta=0:14:4
    2021-08-11 17:06:14 [INFO]	[TRAIN] Epoch=251/270, Step=30/74, loss=22.765055, lr=1e-06, time_each_step=0.35s, eta=0:14:4
    2021-08-11 17:06:14 [INFO]	[TRAIN] Epoch=251/270, Step=32/74, loss=18.480577, lr=1e-06, time_each_step=0.35s, eta=0:14:4
    2021-08-11 17:06:15 [INFO]	[TRAIN] Epoch=251/270, Step=34/74, loss=18.4716, lr=1e-06, time_each_step=0.36s, eta=0:14:3
    2021-08-11 17:06:16 [INFO]	[TRAIN] Epoch=251/270, Step=36/74, loss=11.224945, lr=1e-06, time_each_step=0.34s, eta=0:14:2
    2021-08-11 17:06:16 [INFO]	[TRAIN] Epoch=251/270, Step=38/74, loss=10.929225, lr=1e-06, time_each_step=0.33s, eta=0:14:1
    2021-08-11 17:06:17 [INFO]	[TRAIN] Epoch=251/270, Step=40/74, loss=9.544022, lr=1e-06, time_each_step=0.34s, eta=0:14:0
    2021-08-11 17:06:18 [INFO]	[TRAIN] Epoch=251/270, Step=42/74, loss=20.258823, lr=1e-06, time_each_step=0.35s, eta=0:14:0
    2021-08-11 17:06:18 [INFO]	[TRAIN] Epoch=251/270, Step=44/74, loss=10.456823, lr=1e-06, time_each_step=0.34s, eta=0:13:59
    2021-08-11 17:06:19 [INFO]	[TRAIN] Epoch=251/270, Step=46/74, loss=9.838835, lr=1e-06, time_each_step=0.34s, eta=0:13:58
    2021-08-11 17:06:19 [INFO]	[TRAIN] Epoch=251/270, Step=48/74, loss=13.095489, lr=1e-06, time_each_step=0.34s, eta=0:13:58
    2021-08-11 17:06:20 [INFO]	[TRAIN] Epoch=251/270, Step=50/74, loss=16.203066, lr=1e-06, time_each_step=0.32s, eta=0:13:57
    2021-08-11 17:06:20 [INFO]	[TRAIN] Epoch=251/270, Step=52/74, loss=24.252714, lr=1e-06, time_each_step=0.31s, eta=0:13:56
    2021-08-11 17:06:21 [INFO]	[TRAIN] Epoch=251/270, Step=54/74, loss=17.37846, lr=1e-06, time_each_step=0.28s, eta=0:13:54
    2021-08-11 17:06:21 [INFO]	[TRAIN] Epoch=251/270, Step=56/74, loss=15.41783, lr=1e-06, time_each_step=0.28s, eta=0:13:54
    2021-08-11 17:06:21 [INFO]	[TRAIN] Epoch=251/270, Step=58/74, loss=13.122435, lr=1e-06, time_each_step=0.26s, eta=0:13:53
    2021-08-11 17:06:22 [INFO]	[TRAIN] Epoch=251/270, Step=60/74, loss=8.408763, lr=1e-06, time_each_step=0.24s, eta=0:13:52
    2021-08-11 17:06:22 [INFO]	[TRAIN] Epoch=251/270, Step=62/74, loss=9.171288, lr=1e-06, time_each_step=0.22s, eta=0:13:52
    2021-08-11 17:06:23 [INFO]	[TRAIN] Epoch=251/270, Step=64/74, loss=10.28375, lr=1e-06, time_each_step=0.23s, eta=0:13:51
    2021-08-11 17:06:23 [INFO]	[TRAIN] Epoch=251/270, Step=66/74, loss=15.498437, lr=1e-06, time_each_step=0.23s, eta=0:13:51
    2021-08-11 17:06:24 [INFO]	[TRAIN] Epoch=251/270, Step=68/74, loss=6.838031, lr=1e-06, time_each_step=0.22s, eta=0:13:50
    2021-08-11 17:06:24 [INFO]	[TRAIN] Epoch=251/270, Step=70/74, loss=20.7367, lr=1e-06, time_each_step=0.22s, eta=0:13:50
    2021-08-11 17:06:25 [INFO]	[TRAIN] Epoch=251/270, Step=72/74, loss=23.468521, lr=1e-06, time_each_step=0.21s, eta=0:13:49
    2021-08-11 17:06:25 [INFO]	[TRAIN] Epoch=251/270, Step=74/74, loss=8.699705, lr=1e-06, time_each_step=0.21s, eta=0:13:49
    2021-08-11 17:06:25 [INFO]	[TRAIN] Epoch 251 finished, loss=15.026512, lr=1e-06 .
    2021-08-11 17:06:31 [INFO]	[TRAIN] Epoch=252/270, Step=2/74, loss=8.049917, lr=1e-06, time_each_step=0.49s, eta=0:10:57
    2021-08-11 17:06:32 [INFO]	[TRAIN] Epoch=252/270, Step=4/74, loss=7.05515, lr=1e-06, time_each_step=0.51s, eta=0:10:57
    2021-08-11 17:06:32 [INFO]	[TRAIN] Epoch=252/270, Step=6/74, loss=13.621916, lr=1e-06, time_each_step=0.52s, eta=0:10:57
    2021-08-11 17:06:33 [INFO]	[TRAIN] Epoch=252/270, Step=8/74, loss=11.774642, lr=1e-06, time_each_step=0.53s, eta=0:10:57
    2021-08-11 17:06:34 [INFO]	[TRAIN] Epoch=252/270, Step=10/74, loss=6.422836, lr=1e-06, time_each_step=0.55s, eta=0:10:56
    2021-08-11 17:06:35 [INFO]	[TRAIN] Epoch=252/270, Step=12/74, loss=19.863209, lr=1e-06, time_each_step=0.56s, eta=0:10:56
    2021-08-11 17:06:35 [INFO]	[TRAIN] Epoch=252/270, Step=14/74, loss=10.44985, lr=1e-06, time_each_step=0.57s, eta=0:10:56
    2021-08-11 17:06:36 [INFO]	[TRAIN] Epoch=252/270, Step=16/74, loss=22.949745, lr=1e-06, time_each_step=0.58s, eta=0:10:55
    2021-08-11 17:06:37 [INFO]	[TRAIN] Epoch=252/270, Step=18/74, loss=13.506962, lr=1e-06, time_each_step=0.62s, eta=0:10:56
    2021-08-11 17:06:38 [INFO]	[TRAIN] Epoch=252/270, Step=20/74, loss=20.946768, lr=1e-06, time_each_step=0.64s, eta=0:10:56
    2021-08-11 17:06:39 [INFO]	[TRAIN] Epoch=252/270, Step=22/74, loss=16.341473, lr=1e-06, time_each_step=0.38s, eta=0:10:42
    2021-08-11 17:06:39 [INFO]	[TRAIN] Epoch=252/270, Step=24/74, loss=14.526646, lr=1e-06, time_each_step=0.38s, eta=0:10:40
    2021-08-11 17:06:40 [INFO]	[TRAIN] Epoch=252/270, Step=26/74, loss=11.918167, lr=1e-06, time_each_step=0.39s, eta=0:10:40
    2021-08-11 17:06:41 [INFO]	[TRAIN] Epoch=252/270, Step=28/74, loss=16.27808, lr=1e-06, time_each_step=0.39s, eta=0:10:40
    2021-08-11 17:06:42 [INFO]	[TRAIN] Epoch=252/270, Step=30/74, loss=13.193041, lr=1e-06, time_each_step=0.39s, eta=0:10:39
    2021-08-11 17:06:42 [INFO]	[TRAIN] Epoch=252/270, Step=32/74, loss=21.716352, lr=1e-06, time_each_step=0.39s, eta=0:10:38
    2021-08-11 17:06:43 [INFO]	[TRAIN] Epoch=252/270, Step=34/74, loss=11.369489, lr=1e-06, time_each_step=0.38s, eta=0:10:37
    2021-08-11 17:06:44 [INFO]	[TRAIN] Epoch=252/270, Step=36/74, loss=17.558861, lr=1e-06, time_each_step=0.38s, eta=0:10:36
    2021-08-11 17:06:45 [INFO]	[TRAIN] Epoch=252/270, Step=38/74, loss=27.151606, lr=1e-06, time_each_step=0.38s, eta=0:10:35
    2021-08-11 17:06:45 [INFO]	[TRAIN] Epoch=252/270, Step=40/74, loss=14.126827, lr=1e-06, time_each_step=0.37s, eta=0:10:34
    2021-08-11 17:06:46 [INFO]	[TRAIN] Epoch=252/270, Step=42/74, loss=13.569977, lr=1e-06, time_each_step=0.37s, eta=0:10:33
    2021-08-11 17:06:47 [INFO]	[TRAIN] Epoch=252/270, Step=44/74, loss=23.140781, lr=1e-06, time_each_step=0.38s, eta=0:10:33
    2021-08-11 17:06:47 [INFO]	[TRAIN] Epoch=252/270, Step=46/74, loss=23.969357, lr=1e-06, time_each_step=0.35s, eta=0:10:31
    2021-08-11 17:06:48 [INFO]	[TRAIN] Epoch=252/270, Step=48/74, loss=33.330593, lr=1e-06, time_each_step=0.34s, eta=0:10:30
    2021-08-11 17:06:48 [INFO]	[TRAIN] Epoch=252/270, Step=50/74, loss=6.469801, lr=1e-06, time_each_step=0.31s, eta=0:10:29
    2021-08-11 17:06:48 [INFO]	[TRAIN] Epoch=252/270, Step=52/74, loss=4.990591, lr=1e-06, time_each_step=0.29s, eta=0:10:28
    2021-08-11 17:06:48 [INFO]	[TRAIN] Epoch=252/270, Step=54/74, loss=17.021963, lr=1e-06, time_each_step=0.27s, eta=0:10:27
    2021-08-11 17:06:49 [INFO]	[TRAIN] Epoch=252/270, Step=56/74, loss=7.480293, lr=1e-06, time_each_step=0.26s, eta=0:10:26
    2021-08-11 17:06:49 [INFO]	[TRAIN] Epoch=252/270, Step=58/74, loss=15.4279, lr=1e-06, time_each_step=0.23s, eta=0:10:25
    2021-08-11 17:06:49 [INFO]	[TRAIN] Epoch=252/270, Step=60/74, loss=26.985542, lr=1e-06, time_each_step=0.22s, eta=0:10:25
    2021-08-11 17:06:50 [INFO]	[TRAIN] Epoch=252/270, Step=62/74, loss=13.370476, lr=1e-06, time_each_step=0.2s, eta=0:10:24
    2021-08-11 17:06:51 [INFO]	[TRAIN] Epoch=252/270, Step=64/74, loss=21.202286, lr=1e-06, time_each_step=0.19s, eta=0:10:23
    2021-08-11 17:06:51 [INFO]	[TRAIN] Epoch=252/270, Step=66/74, loss=13.621408, lr=1e-06, time_each_step=0.19s, eta=0:10:23
    2021-08-11 17:06:51 [INFO]	[TRAIN] Epoch=252/270, Step=68/74, loss=12.514141, lr=1e-06, time_each_step=0.2s, eta=0:10:23
    2021-08-11 17:06:52 [INFO]	[TRAIN] Epoch=252/270, Step=70/74, loss=8.249167, lr=1e-06, time_each_step=0.2s, eta=0:10:22
    2021-08-11 17:06:52 [INFO]	[TRAIN] Epoch=252/270, Step=72/74, loss=13.601095, lr=1e-06, time_each_step=0.2s, eta=0:10:22
    2021-08-11 17:06:52 [INFO]	[TRAIN] Epoch=252/270, Step=74/74, loss=11.848031, lr=1e-06, time_each_step=0.2s, eta=0:10:22
    2021-08-11 17:06:52 [INFO]	[TRAIN] Epoch 252 finished, loss=15.180588, lr=1e-06 .
    2021-08-11 17:06:57 [INFO]	[TRAIN] Epoch=253/270, Step=2/74, loss=16.974449, lr=1e-06, time_each_step=0.44s, eta=0:9:1
    2021-08-11 17:06:59 [INFO]	[TRAIN] Epoch=253/270, Step=4/74, loss=7.649978, lr=1e-06, time_each_step=0.48s, eta=0:9:3
    2021-08-11 17:06:59 [INFO]	[TRAIN] Epoch=253/270, Step=6/74, loss=15.863036, lr=1e-06, time_each_step=0.49s, eta=0:9:3
    2021-08-11 17:07:00 [INFO]	[TRAIN] Epoch=253/270, Step=8/74, loss=15.084548, lr=1e-06, time_each_step=0.5s, eta=0:9:2
    2021-08-11 17:07:01 [INFO]	[TRAIN] Epoch=253/270, Step=10/74, loss=25.246933, lr=1e-06, time_each_step=0.5s, eta=0:9:1
    2021-08-11 17:07:01 [INFO]	[TRAIN] Epoch=253/270, Step=12/74, loss=10.369659, lr=1e-06, time_each_step=0.52s, eta=0:9:1
    2021-08-11 17:07:02 [INFO]	[TRAIN] Epoch=253/270, Step=14/74, loss=32.42268, lr=1e-06, time_each_step=0.52s, eta=0:9:1
    2021-08-11 17:07:03 [INFO]	[TRAIN] Epoch=253/270, Step=16/74, loss=17.482101, lr=1e-06, time_each_step=0.54s, eta=0:9:1
    2021-08-11 17:07:04 [INFO]	[TRAIN] Epoch=253/270, Step=18/74, loss=9.112768, lr=1e-06, time_each_step=0.57s, eta=0:9:1
    2021-08-11 17:07:04 [INFO]	[TRAIN] Epoch=253/270, Step=20/74, loss=8.459485, lr=1e-06, time_each_step=0.6s, eta=0:9:1
    2021-08-11 17:07:05 [INFO]	[TRAIN] Epoch=253/270, Step=22/74, loss=45.781078, lr=1e-06, time_each_step=0.38s, eta=0:8:49
    2021-08-11 17:07:06 [INFO]	[TRAIN] Epoch=253/270, Step=24/74, loss=12.999506, lr=1e-06, time_each_step=0.36s, eta=0:8:47
    2021-08-11 17:07:06 [INFO]	[TRAIN] Epoch=253/270, Step=26/74, loss=20.683805, lr=1e-06, time_each_step=0.36s, eta=0:8:46
    2021-08-11 17:07:07 [INFO]	[TRAIN] Epoch=253/270, Step=28/74, loss=8.678984, lr=1e-06, time_each_step=0.35s, eta=0:8:45
    2021-08-11 17:07:08 [INFO]	[TRAIN] Epoch=253/270, Step=30/74, loss=9.819713, lr=1e-06, time_each_step=0.37s, eta=0:8:45
    2021-08-11 17:07:09 [INFO]	[TRAIN] Epoch=253/270, Step=32/74, loss=14.386811, lr=1e-06, time_each_step=0.36s, eta=0:8:45
    2021-08-11 17:07:09 [INFO]	[TRAIN] Epoch=253/270, Step=34/74, loss=6.590312, lr=1e-06, time_each_step=0.36s, eta=0:8:44
    2021-08-11 17:07:10 [INFO]	[TRAIN] Epoch=253/270, Step=36/74, loss=15.087677, lr=1e-06, time_each_step=0.38s, eta=0:8:44
    2021-08-11 17:07:11 [INFO]	[TRAIN] Epoch=253/270, Step=38/74, loss=3.70972, lr=1e-06, time_each_step=0.36s, eta=0:8:42
    2021-08-11 17:07:12 [INFO]	[TRAIN] Epoch=253/270, Step=40/74, loss=9.634487, lr=1e-06, time_each_step=0.36s, eta=0:8:42
    2021-08-11 17:07:12 [INFO]	[TRAIN] Epoch=253/270, Step=42/74, loss=12.321681, lr=1e-06, time_each_step=0.35s, eta=0:8:40
    2021-08-11 17:07:13 [INFO]	[TRAIN] Epoch=253/270, Step=44/74, loss=8.964947, lr=1e-06, time_each_step=0.35s, eta=0:8:40
    2021-08-11 17:07:13 [INFO]	[TRAIN] Epoch=253/270, Step=46/74, loss=25.015938, lr=1e-06, time_each_step=0.34s, eta=0:8:39
    2021-08-11 17:07:14 [INFO]	[TRAIN] Epoch=253/270, Step=48/74, loss=16.118599, lr=1e-06, time_each_step=0.32s, eta=0:8:38
    2021-08-11 17:07:14 [INFO]	[TRAIN] Epoch=253/270, Step=50/74, loss=12.392443, lr=1e-06, time_each_step=0.3s, eta=0:8:36
    2021-08-11 17:07:14 [INFO]	[TRAIN] Epoch=253/270, Step=52/74, loss=34.23666, lr=1e-06, time_each_step=0.28s, eta=0:8:35
    2021-08-11 17:07:15 [INFO]	[TRAIN] Epoch=253/270, Step=54/74, loss=26.653383, lr=1e-06, time_each_step=0.27s, eta=0:8:35
    2021-08-11 17:07:15 [INFO]	[TRAIN] Epoch=253/270, Step=56/74, loss=3.703007, lr=1e-06, time_each_step=0.24s, eta=0:8:33
    2021-08-11 17:07:16 [INFO]	[TRAIN] Epoch=253/270, Step=58/74, loss=10.503441, lr=1e-06, time_each_step=0.24s, eta=0:8:33
    2021-08-11 17:07:16 [INFO]	[TRAIN] Epoch=253/270, Step=60/74, loss=13.034462, lr=1e-06, time_each_step=0.22s, eta=0:8:32
    2021-08-11 17:07:17 [INFO]	[TRAIN] Epoch=253/270, Step=62/74, loss=13.425259, lr=1e-06, time_each_step=0.22s, eta=0:8:32
    2021-08-11 17:07:17 [INFO]	[TRAIN] Epoch=253/270, Step=64/74, loss=23.535744, lr=1e-06, time_each_step=0.21s, eta=0:8:31
    2021-08-11 17:07:18 [INFO]	[TRAIN] Epoch=253/270, Step=66/74, loss=43.734787, lr=1e-06, time_each_step=0.22s, eta=0:8:31
    2021-08-11 17:07:18 [INFO]	[TRAIN] Epoch=253/270, Step=68/74, loss=13.021381, lr=1e-06, time_each_step=0.22s, eta=0:8:31
    2021-08-11 17:07:19 [INFO]	[TRAIN] Epoch=253/270, Step=70/74, loss=20.921488, lr=1e-06, time_each_step=0.23s, eta=0:8:30
    2021-08-11 17:07:19 [INFO]	[TRAIN] Epoch=253/270, Step=72/74, loss=13.947205, lr=1e-06, time_each_step=0.24s, eta=0:8:30
    2021-08-11 17:07:19 [INFO]	[TRAIN] Epoch=253/270, Step=74/74, loss=21.145916, lr=1e-06, time_each_step=0.23s, eta=0:8:29
    2021-08-11 17:07:19 [INFO]	[TRAIN] Epoch 253 finished, loss=15.131572, lr=1e-06 .
    2021-08-11 17:07:34 [INFO]	[TRAIN] Epoch=254/270, Step=2/74, loss=10.400893, lr=1e-06, time_each_step=0.97s, eta=0:9:3
    2021-08-11 17:07:35 [INFO]	[TRAIN] Epoch=254/270, Step=4/74, loss=14.972385, lr=1e-06, time_each_step=0.96s, eta=0:9:1
    2021-08-11 17:07:36 [INFO]	[TRAIN] Epoch=254/270, Step=6/74, loss=41.523643, lr=1e-06, time_each_step=0.99s, eta=0:9:0
    2021-08-11 17:07:37 [INFO]	[TRAIN] Epoch=254/270, Step=8/74, loss=19.848717, lr=1e-06, time_each_step=1.0s, eta=0:8:59
    2021-08-11 17:07:37 [INFO]	[TRAIN] Epoch=254/270, Step=10/74, loss=12.470103, lr=1e-06, time_each_step=1.01s, eta=0:8:58
    2021-08-11 17:07:38 [INFO]	[TRAIN] Epoch=254/270, Step=12/74, loss=13.920292, lr=1e-06, time_each_step=1.02s, eta=0:8:57
    2021-08-11 17:07:39 [INFO]	[TRAIN] Epoch=254/270, Step=14/74, loss=19.248417, lr=1e-06, time_each_step=1.04s, eta=0:8:56
    2021-08-11 17:07:40 [INFO]	[TRAIN] Epoch=254/270, Step=16/74, loss=8.234797, lr=1e-06, time_each_step=1.05s, eta=0:8:54
    2021-08-11 17:07:40 [INFO]	[TRAIN] Epoch=254/270, Step=18/74, loss=15.694025, lr=1e-06, time_each_step=1.07s, eta=0:8:53
    2021-08-11 17:07:41 [INFO]	[TRAIN] Epoch=254/270, Step=20/74, loss=4.840869, lr=1e-06, time_each_step=1.1s, eta=0:8:53
    2021-08-11 17:07:42 [INFO]	[TRAIN] Epoch=254/270, Step=22/74, loss=19.16832, lr=1e-06, time_each_step=0.4s, eta=0:8:14
    2021-08-11 17:07:43 [INFO]	[TRAIN] Epoch=254/270, Step=24/74, loss=4.260191, lr=1e-06, time_each_step=0.4s, eta=0:8:13
    2021-08-11 17:07:44 [INFO]	[TRAIN] Epoch=254/270, Step=26/74, loss=14.78524, lr=1e-06, time_each_step=0.39s, eta=0:8:12
    2021-08-11 17:07:45 [INFO]	[TRAIN] Epoch=254/270, Step=28/74, loss=11.216967, lr=1e-06, time_each_step=0.4s, eta=0:8:12
    2021-08-11 17:07:45 [INFO]	[TRAIN] Epoch=254/270, Step=30/74, loss=21.982391, lr=1e-06, time_each_step=0.4s, eta=0:8:11
    2021-08-11 17:07:46 [INFO]	[TRAIN] Epoch=254/270, Step=32/74, loss=17.064186, lr=1e-06, time_each_step=0.41s, eta=0:8:10
    2021-08-11 17:07:47 [INFO]	[TRAIN] Epoch=254/270, Step=34/74, loss=9.552307, lr=1e-06, time_each_step=0.4s, eta=0:8:9
    2021-08-11 17:07:47 [INFO]	[TRAIN] Epoch=254/270, Step=36/74, loss=19.760361, lr=1e-06, time_each_step=0.39s, eta=0:8:8
    2021-08-11 17:07:48 [INFO]	[TRAIN] Epoch=254/270, Step=38/74, loss=14.490678, lr=1e-06, time_each_step=0.4s, eta=0:8:7
    2021-08-11 17:07:49 [INFO]	[TRAIN] Epoch=254/270, Step=40/74, loss=13.241762, lr=1e-06, time_each_step=0.38s, eta=0:8:6
    2021-08-11 17:07:50 [INFO]	[TRAIN] Epoch=254/270, Step=42/74, loss=15.956383, lr=1e-06, time_each_step=0.37s, eta=0:8:5
    2021-08-11 17:07:50 [INFO]	[TRAIN] Epoch=254/270, Step=44/74, loss=5.09209, lr=1e-06, time_each_step=0.36s, eta=0:8:4
    2021-08-11 17:07:51 [INFO]	[TRAIN] Epoch=254/270, Step=46/74, loss=8.915663, lr=1e-06, time_each_step=0.34s, eta=0:8:3
    2021-08-11 17:07:51 [INFO]	[TRAIN] Epoch=254/270, Step=48/74, loss=10.039442, lr=1e-06, time_each_step=0.33s, eta=0:8:2
    2021-08-11 17:07:52 [INFO]	[TRAIN] Epoch=254/270, Step=50/74, loss=24.89356, lr=1e-06, time_each_step=0.32s, eta=0:8:1
    2021-08-11 17:07:52 [INFO]	[TRAIN] Epoch=254/270, Step=52/74, loss=15.13843, lr=1e-06, time_each_step=0.29s, eta=0:8:0
    2021-08-11 17:07:53 [INFO]	[TRAIN] Epoch=254/270, Step=54/74, loss=13.596148, lr=1e-06, time_each_step=0.29s, eta=0:7:59
    2021-08-11 17:07:53 [INFO]	[TRAIN] Epoch=254/270, Step=56/74, loss=20.804777, lr=1e-06, time_each_step=0.28s, eta=0:7:58
    2021-08-11 17:07:53 [INFO]	[TRAIN] Epoch=254/270, Step=58/74, loss=11.022715, lr=1e-06, time_each_step=0.25s, eta=0:7:57
    2021-08-11 17:07:54 [INFO]	[TRAIN] Epoch=254/270, Step=60/74, loss=8.268953, lr=1e-06, time_each_step=0.22s, eta=0:7:56
    2021-08-11 17:07:54 [INFO]	[TRAIN] Epoch=254/270, Step=62/74, loss=20.086895, lr=1e-06, time_each_step=0.22s, eta=0:7:56
    2021-08-11 17:07:54 [INFO]	[TRAIN] Epoch=254/270, Step=64/74, loss=14.430285, lr=1e-06, time_each_step=0.22s, eta=0:7:55
    2021-08-11 17:07:55 [INFO]	[TRAIN] Epoch=254/270, Step=66/74, loss=24.921217, lr=1e-06, time_each_step=0.21s, eta=0:7:55
    2021-08-11 17:07:55 [INFO]	[TRAIN] Epoch=254/270, Step=68/74, loss=13.339614, lr=1e-06, time_each_step=0.2s, eta=0:7:54
    2021-08-11 17:07:56 [INFO]	[TRAIN] Epoch=254/270, Step=70/74, loss=21.226299, lr=1e-06, time_each_step=0.2s, eta=0:7:54
    2021-08-11 17:07:56 [INFO]	[TRAIN] Epoch=254/270, Step=72/74, loss=12.975788, lr=1e-06, time_each_step=0.2s, eta=0:7:54
    2021-08-11 17:07:56 [INFO]	[TRAIN] Epoch=254/270, Step=74/74, loss=19.632549, lr=1e-06, time_each_step=0.19s, eta=0:7:53
    2021-08-11 17:07:56 [INFO]	[TRAIN] Epoch 254 finished, loss=15.191967, lr=1e-06 .
    2021-08-11 17:08:02 [INFO]	[TRAIN] Epoch=255/270, Step=2/74, loss=9.525954, lr=1e-06, time_each_step=0.45s, eta=0:10:31
    2021-08-11 17:08:03 [INFO]	[TRAIN] Epoch=255/270, Step=4/74, loss=10.848519, lr=1e-06, time_each_step=0.47s, eta=0:10:31
    2021-08-11 17:08:03 [INFO]	[TRAIN] Epoch=255/270, Step=6/74, loss=28.09985, lr=1e-06, time_each_step=0.48s, eta=0:10:31
    2021-08-11 17:08:04 [INFO]	[TRAIN] Epoch=255/270, Step=8/74, loss=16.513971, lr=1e-06, time_each_step=0.52s, eta=0:10:32
    2021-08-11 17:08:05 [INFO]	[TRAIN] Epoch=255/270, Step=10/74, loss=8.939127, lr=1e-06, time_each_step=0.53s, eta=0:10:32
    2021-08-11 17:08:06 [INFO]	[TRAIN] Epoch=255/270, Step=12/74, loss=16.081264, lr=1e-06, time_each_step=0.56s, eta=0:10:33
    2021-08-11 17:08:07 [INFO]	[TRAIN] Epoch=255/270, Step=14/74, loss=9.263111, lr=1e-06, time_each_step=0.58s, eta=0:10:33
    2021-08-11 17:08:07 [INFO]	[TRAIN] Epoch=255/270, Step=16/74, loss=14.463104, lr=1e-06, time_each_step=0.6s, eta=0:10:33
    2021-08-11 17:08:08 [INFO]	[TRAIN] Epoch=255/270, Step=18/74, loss=13.942567, lr=1e-06, time_each_step=0.6s, eta=0:10:32
    2021-08-11 17:08:09 [INFO]	[TRAIN] Epoch=255/270, Step=20/74, loss=14.363647, lr=1e-06, time_each_step=0.61s, eta=0:10:31
    2021-08-11 17:08:09 [INFO]	[TRAIN] Epoch=255/270, Step=22/74, loss=13.017046, lr=1e-06, time_each_step=0.37s, eta=0:10:18
    2021-08-11 17:08:10 [INFO]	[TRAIN] Epoch=255/270, Step=24/74, loss=42.820972, lr=1e-06, time_each_step=0.38s, eta=0:10:17
    2021-08-11 17:08:11 [INFO]	[TRAIN] Epoch=255/270, Step=26/74, loss=21.345657, lr=1e-06, time_each_step=0.38s, eta=0:10:17
    2021-08-11 17:08:11 [INFO]	[TRAIN] Epoch=255/270, Step=28/74, loss=19.301645, lr=1e-06, time_each_step=0.35s, eta=0:10:14
    2021-08-11 17:08:12 [INFO]	[TRAIN] Epoch=255/270, Step=30/74, loss=7.513134, lr=1e-06, time_each_step=0.35s, eta=0:10:14
    2021-08-11 17:08:13 [INFO]	[TRAIN] Epoch=255/270, Step=32/74, loss=6.885709, lr=1e-06, time_each_step=0.36s, eta=0:10:13
    2021-08-11 17:08:14 [INFO]	[TRAIN] Epoch=255/270, Step=34/74, loss=12.083869, lr=1e-06, time_each_step=0.35s, eta=0:10:12
    2021-08-11 17:08:14 [INFO]	[TRAIN] Epoch=255/270, Step=36/74, loss=12.997612, lr=1e-06, time_each_step=0.35s, eta=0:10:12
    2021-08-11 17:08:15 [INFO]	[TRAIN] Epoch=255/270, Step=38/74, loss=8.393893, lr=1e-06, time_each_step=0.36s, eta=0:10:11
    2021-08-11 17:08:16 [INFO]	[TRAIN] Epoch=255/270, Step=40/74, loss=26.273588, lr=1e-06, time_each_step=0.35s, eta=0:10:10
    2021-08-11 17:08:16 [INFO]	[TRAIN] Epoch=255/270, Step=42/74, loss=7.948487, lr=1e-06, time_each_step=0.35s, eta=0:10:9
    2021-08-11 17:08:17 [INFO]	[TRAIN] Epoch=255/270, Step=44/74, loss=29.490536, lr=1e-06, time_each_step=0.35s, eta=0:10:9
    2021-08-11 17:08:18 [INFO]	[TRAIN] Epoch=255/270, Step=46/74, loss=9.406487, lr=1e-06, time_each_step=0.35s, eta=0:10:8
    2021-08-11 17:08:18 [INFO]	[TRAIN] Epoch=255/270, Step=48/74, loss=12.391485, lr=1e-06, time_each_step=0.36s, eta=0:10:8
    2021-08-11 17:08:19 [INFO]	[TRAIN] Epoch=255/270, Step=50/74, loss=25.257416, lr=1e-06, time_each_step=0.34s, eta=0:10:7
    2021-08-11 17:08:19 [INFO]	[TRAIN] Epoch=255/270, Step=52/74, loss=8.744259, lr=1e-06, time_each_step=0.31s, eta=0:10:5
    2021-08-11 17:08:20 [INFO]	[TRAIN] Epoch=255/270, Step=54/74, loss=23.952559, lr=1e-06, time_each_step=0.29s, eta=0:10:4
    2021-08-11 17:08:20 [INFO]	[TRAIN] Epoch=255/270, Step=56/74, loss=12.512297, lr=1e-06, time_each_step=0.28s, eta=0:10:3
    2021-08-11 17:08:20 [INFO]	[TRAIN] Epoch=255/270, Step=58/74, loss=27.02335, lr=1e-06, time_each_step=0.27s, eta=0:10:3
    2021-08-11 17:08:21 [INFO]	[TRAIN] Epoch=255/270, Step=60/74, loss=23.37269, lr=1e-06, time_each_step=0.26s, eta=0:10:2
    2021-08-11 17:08:21 [INFO]	[TRAIN] Epoch=255/270, Step=62/74, loss=15.444545, lr=1e-06, time_each_step=0.24s, eta=0:10:1
    2021-08-11 17:08:22 [INFO]	[TRAIN] Epoch=255/270, Step=64/74, loss=11.298046, lr=1e-06, time_each_step=0.22s, eta=0:10:1
    2021-08-11 17:08:22 [INFO]	[TRAIN] Epoch=255/270, Step=66/74, loss=3.692283, lr=1e-06, time_each_step=0.2s, eta=0:10:0
    2021-08-11 17:08:22 [INFO]	[TRAIN] Epoch=255/270, Step=68/74, loss=26.164692, lr=1e-06, time_each_step=0.19s, eta=0:9:59
    2021-08-11 17:08:23 [INFO]	[TRAIN] Epoch=255/270, Step=70/74, loss=39.643959, lr=1e-06, time_each_step=0.19s, eta=0:9:59
    2021-08-11 17:08:23 [INFO]	[TRAIN] Epoch=255/270, Step=72/74, loss=18.491186, lr=1e-06, time_each_step=0.19s, eta=0:9:59
    2021-08-11 17:08:23 [INFO]	[TRAIN] Epoch=255/270, Step=74/74, loss=10.251589, lr=1e-06, time_each_step=0.19s, eta=0:9:58
    2021-08-11 17:08:23 [INFO]	[TRAIN] Epoch 255 finished, loss=15.344928, lr=1e-06 .
    2021-08-11 17:08:28 [INFO]	[TRAIN] Epoch=256/270, Step=2/74, loss=14.96295, lr=1e-06, time_each_step=0.41s, eta=0:7:30
    2021-08-11 17:08:29 [INFO]	[TRAIN] Epoch=256/270, Step=4/74, loss=15.381952, lr=1e-06, time_each_step=0.43s, eta=0:7:31
    2021-08-11 17:08:30 [INFO]	[TRAIN] Epoch=256/270, Step=6/74, loss=52.786079, lr=1e-06, time_each_step=0.45s, eta=0:7:31
    2021-08-11 17:08:30 [INFO]	[TRAIN] Epoch=256/270, Step=8/74, loss=7.084476, lr=1e-06, time_each_step=0.46s, eta=0:7:31
    2021-08-11 17:08:31 [INFO]	[TRAIN] Epoch=256/270, Step=10/74, loss=19.697966, lr=1e-06, time_each_step=0.47s, eta=0:7:31
    2021-08-11 17:08:32 [INFO]	[TRAIN] Epoch=256/270, Step=12/74, loss=7.618664, lr=1e-06, time_each_step=0.49s, eta=0:7:31
    2021-08-11 17:08:32 [INFO]	[TRAIN] Epoch=256/270, Step=14/74, loss=16.340002, lr=1e-06, time_each_step=0.5s, eta=0:7:30
    2021-08-11 17:08:33 [INFO]	[TRAIN] Epoch=256/270, Step=16/74, loss=10.225991, lr=1e-06, time_each_step=0.52s, eta=0:7:31
    2021-08-11 17:08:34 [INFO]	[TRAIN] Epoch=256/270, Step=18/74, loss=29.922745, lr=1e-06, time_each_step=0.53s, eta=0:7:31
    2021-08-11 17:08:34 [INFO]	[TRAIN] Epoch=256/270, Step=20/74, loss=5.779226, lr=1e-06, time_each_step=0.53s, eta=0:7:29
    2021-08-11 17:08:35 [INFO]	[TRAIN] Epoch=256/270, Step=22/74, loss=13.220135, lr=1e-06, time_each_step=0.34s, eta=0:7:18
    2021-08-11 17:08:36 [INFO]	[TRAIN] Epoch=256/270, Step=24/74, loss=22.283033, lr=1e-06, time_each_step=0.35s, eta=0:7:18
    2021-08-11 17:08:37 [INFO]	[TRAIN] Epoch=256/270, Step=26/74, loss=18.386635, lr=1e-06, time_each_step=0.35s, eta=0:7:18
    2021-08-11 17:08:38 [INFO]	[TRAIN] Epoch=256/270, Step=28/74, loss=37.158157, lr=1e-06, time_each_step=0.35s, eta=0:7:17
    2021-08-11 17:08:38 [INFO]	[TRAIN] Epoch=256/270, Step=30/74, loss=20.004808, lr=1e-06, time_each_step=0.37s, eta=0:7:17
    2021-08-11 17:08:39 [INFO]	[TRAIN] Epoch=256/270, Step=32/74, loss=22.986973, lr=1e-06, time_each_step=0.38s, eta=0:7:17
    2021-08-11 17:08:40 [INFO]	[TRAIN] Epoch=256/270, Step=34/74, loss=17.164377, lr=1e-06, time_each_step=0.39s, eta=0:7:16
    2021-08-11 17:08:41 [INFO]	[TRAIN] Epoch=256/270, Step=36/74, loss=18.926855, lr=1e-06, time_each_step=0.39s, eta=0:7:15
    2021-08-11 17:08:42 [INFO]	[TRAIN] Epoch=256/270, Step=38/74, loss=13.270466, lr=1e-06, time_each_step=0.4s, eta=0:7:15
    2021-08-11 17:08:43 [INFO]	[TRAIN] Epoch=256/270, Step=40/74, loss=14.346155, lr=1e-06, time_each_step=0.42s, eta=0:7:15
    2021-08-11 17:08:43 [INFO]	[TRAIN] Epoch=256/270, Step=42/74, loss=53.21376, lr=1e-06, time_each_step=0.39s, eta=0:7:13
    2021-08-11 17:08:43 [INFO]	[TRAIN] Epoch=256/270, Step=44/74, loss=9.754535, lr=1e-06, time_each_step=0.37s, eta=0:7:12
    2021-08-11 17:08:44 [INFO]	[TRAIN] Epoch=256/270, Step=46/74, loss=9.969367, lr=1e-06, time_each_step=0.35s, eta=0:7:10
    2021-08-11 17:08:44 [INFO]	[TRAIN] Epoch=256/270, Step=48/74, loss=10.475223, lr=1e-06, time_each_step=0.33s, eta=0:7:9
    2021-08-11 17:08:45 [INFO]	[TRAIN] Epoch=256/270, Step=50/74, loss=8.170597, lr=1e-06, time_each_step=0.32s, eta=0:7:8
    2021-08-11 17:08:45 [INFO]	[TRAIN] Epoch=256/270, Step=52/74, loss=8.076791, lr=1e-06, time_each_step=0.29s, eta=0:7:7
    2021-08-11 17:08:45 [INFO]	[TRAIN] Epoch=256/270, Step=54/74, loss=13.37895, lr=1e-06, time_each_step=0.27s, eta=0:7:6
    2021-08-11 17:08:46 [INFO]	[TRAIN] Epoch=256/270, Step=56/74, loss=16.544079, lr=1e-06, time_each_step=0.25s, eta=0:7:5
    2021-08-11 17:08:46 [INFO]	[TRAIN] Epoch=256/270, Step=58/74, loss=11.032365, lr=1e-06, time_each_step=0.22s, eta=0:7:4
    2021-08-11 17:08:46 [INFO]	[TRAIN] Epoch=256/270, Step=60/74, loss=8.142362, lr=1e-06, time_each_step=0.19s, eta=0:7:3
    2021-08-11 17:08:47 [INFO]	[TRAIN] Epoch=256/270, Step=62/74, loss=13.21563, lr=1e-06, time_each_step=0.2s, eta=0:7:3
    2021-08-11 17:08:47 [INFO]	[TRAIN] Epoch=256/270, Step=64/74, loss=16.926414, lr=1e-06, time_each_step=0.19s, eta=0:7:3
    2021-08-11 17:08:48 [INFO]	[TRAIN] Epoch=256/270, Step=66/74, loss=9.144679, lr=1e-06, time_each_step=0.19s, eta=0:7:2
    2021-08-11 17:08:48 [INFO]	[TRAIN] Epoch=256/270, Step=68/74, loss=7.260593, lr=1e-06, time_each_step=0.19s, eta=0:7:2
    2021-08-11 17:08:48 [INFO]	[TRAIN] Epoch=256/270, Step=70/74, loss=11.009262, lr=1e-06, time_each_step=0.17s, eta=0:7:1
    2021-08-11 17:08:49 [INFO]	[TRAIN] Epoch=256/270, Step=72/74, loss=11.125177, lr=1e-06, time_each_step=0.18s, eta=0:7:1
    2021-08-11 17:08:49 [INFO]	[TRAIN] Epoch=256/270, Step=74/74, loss=8.686429, lr=1e-06, time_each_step=0.18s, eta=0:7:1
    2021-08-11 17:08:49 [INFO]	[TRAIN] Epoch 256 finished, loss=15.403913, lr=1e-06 .
    2021-08-11 17:09:06 [INFO]	[TRAIN] Epoch=257/270, Step=2/74, loss=10.958818, lr=1e-06, time_each_step=1.01s, eta=0:7:27
    2021-08-11 17:09:07 [INFO]	[TRAIN] Epoch=257/270, Step=4/74, loss=17.663116, lr=1e-06, time_each_step=1.03s, eta=0:7:27
    2021-08-11 17:09:08 [INFO]	[TRAIN] Epoch=257/270, Step=6/74, loss=22.308752, lr=1e-06, time_each_step=1.06s, eta=0:7:27
    2021-08-11 17:09:09 [INFO]	[TRAIN] Epoch=257/270, Step=8/74, loss=18.199976, lr=1e-06, time_each_step=1.09s, eta=0:7:27
    2021-08-11 17:09:10 [INFO]	[TRAIN] Epoch=257/270, Step=10/74, loss=28.384846, lr=1e-06, time_each_step=1.12s, eta=0:7:26
    2021-08-11 17:09:10 [INFO]	[TRAIN] Epoch=257/270, Step=12/74, loss=3.201189, lr=1e-06, time_each_step=1.14s, eta=0:7:25
    2021-08-11 17:09:11 [INFO]	[TRAIN] Epoch=257/270, Step=14/74, loss=22.08392, lr=1e-06, time_each_step=1.17s, eta=0:7:25
    2021-08-11 17:09:12 [INFO]	[TRAIN] Epoch=257/270, Step=16/74, loss=13.365091, lr=1e-06, time_each_step=1.2s, eta=0:7:24
    2021-08-11 17:09:13 [INFO]	[TRAIN] Epoch=257/270, Step=18/74, loss=16.705379, lr=1e-06, time_each_step=1.2s, eta=0:7:22
    2021-08-11 17:09:14 [INFO]	[TRAIN] Epoch=257/270, Step=20/74, loss=12.665179, lr=1e-06, time_each_step=1.23s, eta=0:7:21
    2021-08-11 17:09:14 [INFO]	[TRAIN] Epoch=257/270, Step=22/74, loss=9.750372, lr=1e-06, time_each_step=0.42s, eta=0:6:36
    2021-08-11 17:09:15 [INFO]	[TRAIN] Epoch=257/270, Step=24/74, loss=9.202817, lr=1e-06, time_each_step=0.42s, eta=0:6:35
    2021-08-11 17:09:16 [INFO]	[TRAIN] Epoch=257/270, Step=26/74, loss=15.518665, lr=1e-06, time_each_step=0.41s, eta=0:6:34
    2021-08-11 17:09:17 [INFO]	[TRAIN] Epoch=257/270, Step=28/74, loss=9.48228, lr=1e-06, time_each_step=0.4s, eta=0:6:33
    2021-08-11 17:09:17 [INFO]	[TRAIN] Epoch=257/270, Step=30/74, loss=13.022295, lr=1e-06, time_each_step=0.38s, eta=0:6:31
    2021-08-11 17:09:18 [INFO]	[TRAIN] Epoch=257/270, Step=32/74, loss=10.351373, lr=1e-06, time_each_step=0.36s, eta=0:6:30
    2021-08-11 17:09:19 [INFO]	[TRAIN] Epoch=257/270, Step=34/74, loss=18.349911, lr=1e-06, time_each_step=0.38s, eta=0:6:30
    2021-08-11 17:09:20 [INFO]	[TRAIN] Epoch=257/270, Step=36/74, loss=9.179929, lr=1e-06, time_each_step=0.4s, eta=0:6:29
    2021-08-11 17:09:21 [INFO]	[TRAIN] Epoch=257/270, Step=38/74, loss=18.681194, lr=1e-06, time_each_step=0.41s, eta=0:6:29
    2021-08-11 17:09:21 [INFO]	[TRAIN] Epoch=257/270, Step=40/74, loss=18.199963, lr=1e-06, time_each_step=0.4s, eta=0:6:28
    2021-08-11 17:09:22 [INFO]	[TRAIN] Epoch=257/270, Step=42/74, loss=20.528391, lr=1e-06, time_each_step=0.38s, eta=0:6:26
    2021-08-11 17:09:22 [INFO]	[TRAIN] Epoch=257/270, Step=44/74, loss=40.682034, lr=1e-06, time_each_step=0.36s, eta=0:6:25
    2021-08-11 17:09:23 [INFO]	[TRAIN] Epoch=257/270, Step=46/74, loss=15.1455, lr=1e-06, time_each_step=0.34s, eta=0:6:24
    2021-08-11 17:09:23 [INFO]	[TRAIN] Epoch=257/270, Step=48/74, loss=9.266222, lr=1e-06, time_each_step=0.31s, eta=0:6:22
    2021-08-11 17:09:23 [INFO]	[TRAIN] Epoch=257/270, Step=50/74, loss=11.47665, lr=1e-06, time_each_step=0.3s, eta=0:6:21
    2021-08-11 17:09:24 [INFO]	[TRAIN] Epoch=257/270, Step=52/74, loss=10.002847, lr=1e-06, time_each_step=0.3s, eta=0:6:21
    2021-08-11 17:09:24 [INFO]	[TRAIN] Epoch=257/270, Step=54/74, loss=37.336594, lr=1e-06, time_each_step=0.27s, eta=0:6:20
    2021-08-11 17:09:25 [INFO]	[TRAIN] Epoch=257/270, Step=56/74, loss=9.459026, lr=1e-06, time_each_step=0.22s, eta=0:6:18
    2021-08-11 17:09:25 [INFO]	[TRAIN] Epoch=257/270, Step=58/74, loss=19.136656, lr=1e-06, time_each_step=0.21s, eta=0:6:18
    2021-08-11 17:09:25 [INFO]	[TRAIN] Epoch=257/270, Step=60/74, loss=4.593424, lr=1e-06, time_each_step=0.2s, eta=0:6:17
    2021-08-11 17:09:26 [INFO]	[TRAIN] Epoch=257/270, Step=62/74, loss=16.361736, lr=1e-06, time_each_step=0.2s, eta=0:6:17
    2021-08-11 17:09:26 [INFO]	[TRAIN] Epoch=257/270, Step=64/74, loss=25.10318, lr=1e-06, time_each_step=0.19s, eta=0:6:16
    2021-08-11 17:09:27 [INFO]	[TRAIN] Epoch=257/270, Step=66/74, loss=6.669871, lr=1e-06, time_each_step=0.2s, eta=0:6:16
    2021-08-11 17:09:27 [INFO]	[TRAIN] Epoch=257/270, Step=68/74, loss=9.997705, lr=1e-06, time_each_step=0.21s, eta=0:6:16
    2021-08-11 17:09:28 [INFO]	[TRAIN] Epoch=257/270, Step=70/74, loss=18.872543, lr=1e-06, time_each_step=0.22s, eta=0:6:15
    2021-08-11 17:09:28 [INFO]	[TRAIN] Epoch=257/270, Step=72/74, loss=10.680059, lr=1e-06, time_each_step=0.22s, eta=0:6:15
    2021-08-11 17:09:29 [INFO]	[TRAIN] Epoch=257/270, Step=74/74, loss=26.482895, lr=1e-06, time_each_step=0.22s, eta=0:6:14
    2021-08-11 17:09:29 [INFO]	[TRAIN] Epoch 257 finished, loss=14.764569, lr=1e-06 .
    2021-08-11 17:09:35 [INFO]	[TRAIN] Epoch=258/270, Step=2/74, loss=13.125465, lr=1e-06, time_each_step=0.53s, eta=0:9:18
    2021-08-11 17:09:36 [INFO]	[TRAIN] Epoch=258/270, Step=4/74, loss=17.539017, lr=1e-06, time_each_step=0.56s, eta=0:9:19
    2021-08-11 17:09:37 [INFO]	[TRAIN] Epoch=258/270, Step=6/74, loss=25.938293, lr=1e-06, time_each_step=0.57s, eta=0:9:19
    2021-08-11 17:09:38 [INFO]	[TRAIN] Epoch=258/270, Step=8/74, loss=34.852863, lr=1e-06, time_each_step=0.59s, eta=0:9:19
    2021-08-11 17:09:38 [INFO]	[TRAIN] Epoch=258/270, Step=10/74, loss=19.773535, lr=1e-06, time_each_step=0.6s, eta=0:9:18
    2021-08-11 17:09:39 [INFO]	[TRAIN] Epoch=258/270, Step=12/74, loss=13.024531, lr=1e-06, time_each_step=0.61s, eta=0:9:17
    2021-08-11 17:09:39 [INFO]	[TRAIN] Epoch=258/270, Step=14/74, loss=21.930157, lr=1e-06, time_each_step=0.61s, eta=0:9:16
    2021-08-11 17:09:40 [INFO]	[TRAIN] Epoch=258/270, Step=16/74, loss=22.925898, lr=1e-06, time_each_step=0.61s, eta=0:9:15
    2021-08-11 17:09:41 [INFO]	[TRAIN] Epoch=258/270, Step=18/74, loss=17.541412, lr=1e-06, time_each_step=0.63s, eta=0:9:15
    2021-08-11 17:09:41 [INFO]	[TRAIN] Epoch=258/270, Step=20/74, loss=16.398024, lr=1e-06, time_each_step=0.63s, eta=0:9:14
    2021-08-11 17:09:42 [INFO]	[TRAIN] Epoch=258/270, Step=22/74, loss=8.606546, lr=1e-06, time_each_step=0.35s, eta=0:8:58
    2021-08-11 17:09:43 [INFO]	[TRAIN] Epoch=258/270, Step=24/74, loss=15.902449, lr=1e-06, time_each_step=0.34s, eta=0:8:57
    2021-08-11 17:09:44 [INFO]	[TRAIN] Epoch=258/270, Step=26/74, loss=16.000172, lr=1e-06, time_each_step=0.36s, eta=0:8:57
    2021-08-11 17:09:45 [INFO]	[TRAIN] Epoch=258/270, Step=28/74, loss=27.501753, lr=1e-06, time_each_step=0.35s, eta=0:8:56
    2021-08-11 17:09:45 [INFO]	[TRAIN] Epoch=258/270, Step=30/74, loss=18.356163, lr=1e-06, time_each_step=0.36s, eta=0:8:55
    2021-08-11 17:09:46 [INFO]	[TRAIN] Epoch=258/270, Step=32/74, loss=33.042316, lr=1e-06, time_each_step=0.37s, eta=0:8:55
    2021-08-11 17:09:47 [INFO]	[TRAIN] Epoch=258/270, Step=34/74, loss=12.490485, lr=1e-06, time_each_step=0.4s, eta=0:8:56
    2021-08-11 17:09:48 [INFO]	[TRAIN] Epoch=258/270, Step=36/74, loss=7.376755, lr=1e-06, time_each_step=0.4s, eta=0:8:55
    2021-08-11 17:09:48 [INFO]	[TRAIN] Epoch=258/270, Step=38/74, loss=11.249283, lr=1e-06, time_each_step=0.38s, eta=0:8:53
    2021-08-11 17:09:49 [INFO]	[TRAIN] Epoch=258/270, Step=40/74, loss=8.688505, lr=1e-06, time_each_step=0.39s, eta=0:8:53
    2021-08-11 17:09:50 [INFO]	[TRAIN] Epoch=258/270, Step=42/74, loss=21.288055, lr=1e-06, time_each_step=0.37s, eta=0:8:51
    2021-08-11 17:09:50 [INFO]	[TRAIN] Epoch=258/270, Step=44/74, loss=23.832272, lr=1e-06, time_each_step=0.35s, eta=0:8:50
    2021-08-11 17:09:51 [INFO]	[TRAIN] Epoch=258/270, Step=46/74, loss=8.642611, lr=1e-06, time_each_step=0.33s, eta=0:8:49
    2021-08-11 17:09:51 [INFO]	[TRAIN] Epoch=258/270, Step=48/74, loss=4.616112, lr=1e-06, time_each_step=0.33s, eta=0:8:48
    2021-08-11 17:09:52 [INFO]	[TRAIN] Epoch=258/270, Step=50/74, loss=8.447266, lr=1e-06, time_each_step=0.31s, eta=0:8:47
    2021-08-11 17:09:52 [INFO]	[TRAIN] Epoch=258/270, Step=52/74, loss=24.642147, lr=1e-06, time_each_step=0.29s, eta=0:8:46
    2021-08-11 17:09:52 [INFO]	[TRAIN] Epoch=258/270, Step=54/74, loss=11.737359, lr=1e-06, time_each_step=0.26s, eta=0:8:45
    2021-08-11 17:09:53 [INFO]	[TRAIN] Epoch=258/270, Step=56/74, loss=8.356226, lr=1e-06, time_each_step=0.26s, eta=0:8:44
    2021-08-11 17:09:53 [INFO]	[TRAIN] Epoch=258/270, Step=58/74, loss=8.320799, lr=1e-06, time_each_step=0.25s, eta=0:8:44
    2021-08-11 17:09:54 [INFO]	[TRAIN] Epoch=258/270, Step=60/74, loss=9.376822, lr=1e-06, time_each_step=0.24s, eta=0:8:43
    2021-08-11 17:09:54 [INFO]	[TRAIN] Epoch=258/270, Step=62/74, loss=13.789095, lr=1e-06, time_each_step=0.23s, eta=0:8:42
    2021-08-11 17:09:55 [INFO]	[TRAIN] Epoch=258/270, Step=64/74, loss=8.588686, lr=1e-06, time_each_step=0.23s, eta=0:8:42
    2021-08-11 17:09:55 [INFO]	[TRAIN] Epoch=258/270, Step=66/74, loss=6.100574, lr=1e-06, time_each_step=0.24s, eta=0:8:42
    2021-08-11 17:09:56 [INFO]	[TRAIN] Epoch=258/270, Step=68/74, loss=11.943432, lr=1e-06, time_each_step=0.24s, eta=0:8:41
    2021-08-11 17:09:56 [INFO]	[TRAIN] Epoch=258/270, Step=70/74, loss=15.322968, lr=1e-06, time_each_step=0.24s, eta=0:8:41
    2021-08-11 17:09:57 [INFO]	[TRAIN] Epoch=258/270, Step=72/74, loss=9.274514, lr=1e-06, time_each_step=0.24s, eta=0:8:40
    2021-08-11 17:09:57 [INFO]	[TRAIN] Epoch=258/270, Step=74/74, loss=9.84803, lr=1e-06, time_each_step=0.24s, eta=0:8:40
    2021-08-11 17:09:57 [INFO]	[TRAIN] Epoch 258 finished, loss=14.758622, lr=1e-06 .
    2021-08-11 17:10:17 [INFO]	[TRAIN] Epoch=259/270, Step=2/74, loss=13.768973, lr=1e-06, time_each_step=1.19s, eta=0:7:22
    2021-08-11 17:10:18 [INFO]	[TRAIN] Epoch=259/270, Step=4/74, loss=8.820073, lr=1e-06, time_each_step=1.22s, eta=0:7:22
    2021-08-11 17:10:19 [INFO]	[TRAIN] Epoch=259/270, Step=6/74, loss=26.452139, lr=1e-06, time_each_step=1.24s, eta=0:7:20
    2021-08-11 17:10:20 [INFO]	[TRAIN] Epoch=259/270, Step=8/74, loss=11.031857, lr=1e-06, time_each_step=1.27s, eta=0:7:20
    2021-08-11 17:10:20 [INFO]	[TRAIN] Epoch=259/270, Step=10/74, loss=20.092245, lr=1e-06, time_each_step=1.27s, eta=0:7:17
    2021-08-11 17:10:21 [INFO]	[TRAIN] Epoch=259/270, Step=12/74, loss=24.850004, lr=1e-06, time_each_step=1.27s, eta=0:7:15
    2021-08-11 17:10:22 [INFO]	[TRAIN] Epoch=259/270, Step=14/74, loss=13.250856, lr=1e-06, time_each_step=1.28s, eta=0:7:13
    2021-08-11 17:10:22 [INFO]	[TRAIN] Epoch=259/270, Step=16/74, loss=23.949038, lr=1e-06, time_each_step=1.3s, eta=0:7:11
    2021-08-11 17:10:23 [INFO]	[TRAIN] Epoch=259/270, Step=18/74, loss=49.899109, lr=1e-06, time_each_step=1.32s, eta=0:7:10
    2021-08-11 17:10:24 [INFO]	[TRAIN] Epoch=259/270, Step=20/74, loss=16.754211, lr=1e-06, time_each_step=1.33s, eta=0:7:8
    2021-08-11 17:10:24 [INFO]	[TRAIN] Epoch=259/270, Step=22/74, loss=5.938996, lr=1e-06, time_each_step=0.37s, eta=0:6:15
    2021-08-11 17:10:25 [INFO]	[TRAIN] Epoch=259/270, Step=24/74, loss=12.857103, lr=1e-06, time_each_step=0.35s, eta=0:6:14
    2021-08-11 17:10:26 [INFO]	[TRAIN] Epoch=259/270, Step=26/74, loss=12.965918, lr=1e-06, time_each_step=0.34s, eta=0:6:12
    2021-08-11 17:10:26 [INFO]	[TRAIN] Epoch=259/270, Step=28/74, loss=7.091433, lr=1e-06, time_each_step=0.34s, eta=0:6:11
    2021-08-11 17:10:27 [INFO]	[TRAIN] Epoch=259/270, Step=30/74, loss=18.626848, lr=1e-06, time_each_step=0.36s, eta=0:6:11
    2021-08-11 17:10:28 [INFO]	[TRAIN] Epoch=259/270, Step=32/74, loss=9.257769, lr=1e-06, time_each_step=0.35s, eta=0:6:11
    2021-08-11 17:10:29 [INFO]	[TRAIN] Epoch=259/270, Step=34/74, loss=16.441935, lr=1e-06, time_each_step=0.35s, eta=0:6:10
    2021-08-11 17:10:29 [INFO]	[TRAIN] Epoch=259/270, Step=36/74, loss=14.27683, lr=1e-06, time_each_step=0.34s, eta=0:6:9
    2021-08-11 17:10:30 [INFO]	[TRAIN] Epoch=259/270, Step=38/74, loss=17.031891, lr=1e-06, time_each_step=0.34s, eta=0:6:8
    2021-08-11 17:10:31 [INFO]	[TRAIN] Epoch=259/270, Step=40/74, loss=10.892885, lr=1e-06, time_each_step=0.36s, eta=0:6:8
    2021-08-11 17:10:32 [INFO]	[TRAIN] Epoch=259/270, Step=42/74, loss=17.606377, lr=1e-06, time_each_step=0.37s, eta=0:6:8
    2021-08-11 17:10:32 [INFO]	[TRAIN] Epoch=259/270, Step=44/74, loss=25.346224, lr=1e-06, time_each_step=0.37s, eta=0:6:7
    2021-08-11 17:10:33 [INFO]	[TRAIN] Epoch=259/270, Step=46/74, loss=15.492008, lr=1e-06, time_each_step=0.36s, eta=0:6:6
    2021-08-11 17:10:33 [INFO]	[TRAIN] Epoch=259/270, Step=48/74, loss=22.679863, lr=1e-06, time_each_step=0.34s, eta=0:6:5
    2021-08-11 17:10:34 [INFO]	[TRAIN] Epoch=259/270, Step=50/74, loss=5.782081, lr=1e-06, time_each_step=0.31s, eta=0:6:3
    2021-08-11 17:10:34 [INFO]	[TRAIN] Epoch=259/270, Step=52/74, loss=10.509785, lr=1e-06, time_each_step=0.3s, eta=0:6:2
    2021-08-11 17:10:34 [INFO]	[TRAIN] Epoch=259/270, Step=54/74, loss=5.854953, lr=1e-06, time_each_step=0.3s, eta=0:6:2
    2021-08-11 17:10:35 [INFO]	[TRAIN] Epoch=259/270, Step=56/74, loss=16.586637, lr=1e-06, time_each_step=0.28s, eta=0:6:1
    2021-08-11 17:10:35 [INFO]	[TRAIN] Epoch=259/270, Step=58/74, loss=9.801418, lr=1e-06, time_each_step=0.26s, eta=0:6:0
    2021-08-11 17:10:35 [INFO]	[TRAIN] Epoch=259/270, Step=60/74, loss=11.371695, lr=1e-06, time_each_step=0.22s, eta=0:5:59
    2021-08-11 17:10:36 [INFO]	[TRAIN] Epoch=259/270, Step=62/74, loss=27.149532, lr=1e-06, time_each_step=0.19s, eta=0:5:58
    2021-08-11 17:10:36 [INFO]	[TRAIN] Epoch=259/270, Step=64/74, loss=9.145175, lr=1e-06, time_each_step=0.18s, eta=0:5:58
    2021-08-11 17:10:36 [INFO]	[TRAIN] Epoch=259/270, Step=66/74, loss=27.987051, lr=1e-06, time_each_step=0.18s, eta=0:5:57
    2021-08-11 17:10:37 [INFO]	[TRAIN] Epoch=259/270, Step=68/74, loss=8.28043, lr=1e-06, time_each_step=0.18s, eta=0:5:57
    2021-08-11 17:10:37 [INFO]	[TRAIN] Epoch=259/270, Step=70/74, loss=11.16331, lr=1e-06, time_each_step=0.18s, eta=0:5:57
    2021-08-11 17:10:38 [INFO]	[TRAIN] Epoch=259/270, Step=72/74, loss=8.148554, lr=1e-06, time_each_step=0.18s, eta=0:5:56
    2021-08-11 17:10:38 [INFO]	[TRAIN] Epoch=259/270, Step=74/74, loss=7.573781, lr=1e-06, time_each_step=0.18s, eta=0:5:56
    2021-08-11 17:10:38 [INFO]	[TRAIN] Epoch 259 finished, loss=14.815032, lr=1e-06 .
    2021-08-11 17:10:44 [INFO]	[TRAIN] Epoch=260/270, Step=2/74, loss=13.698999, lr=1e-06, time_each_step=0.43s, eta=0:8:2
    2021-08-11 17:10:45 [INFO]	[TRAIN] Epoch=260/270, Step=4/74, loss=11.498531, lr=1e-06, time_each_step=0.48s, eta=0:8:4
    2021-08-11 17:10:46 [INFO]	[TRAIN] Epoch=260/270, Step=6/74, loss=21.263178, lr=1e-06, time_each_step=0.5s, eta=0:8:5
    2021-08-11 17:10:46 [INFO]	[TRAIN] Epoch=260/270, Step=8/74, loss=9.445423, lr=1e-06, time_each_step=0.54s, eta=0:8:6
    2021-08-11 17:10:47 [INFO]	[TRAIN] Epoch=260/270, Step=10/74, loss=17.949398, lr=1e-06, time_each_step=0.54s, eta=0:8:6
    2021-08-11 17:10:48 [INFO]	[TRAIN] Epoch=260/270, Step=12/74, loss=12.111229, lr=1e-06, time_each_step=0.56s, eta=0:8:6
    2021-08-11 17:10:48 [INFO]	[TRAIN] Epoch=260/270, Step=14/74, loss=21.134165, lr=1e-06, time_each_step=0.58s, eta=0:8:6
    2021-08-11 17:10:49 [INFO]	[TRAIN] Epoch=260/270, Step=16/74, loss=15.988692, lr=1e-06, time_each_step=0.59s, eta=0:8:5
    2021-08-11 17:10:50 [INFO]	[TRAIN] Epoch=260/270, Step=18/74, loss=13.758341, lr=1e-06, time_each_step=0.61s, eta=0:8:5
    2021-08-11 17:10:51 [INFO]	[TRAIN] Epoch=260/270, Step=20/74, loss=11.321309, lr=1e-06, time_each_step=0.62s, eta=0:8:4
    2021-08-11 17:10:51 [INFO]	[TRAIN] Epoch=260/270, Step=22/74, loss=13.458244, lr=1e-06, time_each_step=0.38s, eta=0:7:51
    2021-08-11 17:10:52 [INFO]	[TRAIN] Epoch=260/270, Step=24/74, loss=15.720349, lr=1e-06, time_each_step=0.35s, eta=0:7:49
    2021-08-11 17:10:52 [INFO]	[TRAIN] Epoch=260/270, Step=26/74, loss=21.631405, lr=1e-06, time_each_step=0.34s, eta=0:7:47
    2021-08-11 17:10:53 [INFO]	[TRAIN] Epoch=260/270, Step=28/74, loss=8.654344, lr=1e-06, time_each_step=0.33s, eta=0:7:46
    2021-08-11 17:10:54 [INFO]	[TRAIN] Epoch=260/270, Step=30/74, loss=21.142399, lr=1e-06, time_each_step=0.35s, eta=0:7:46
    2021-08-11 17:10:55 [INFO]	[TRAIN] Epoch=260/270, Step=32/74, loss=40.811459, lr=1e-06, time_each_step=0.37s, eta=0:7:47
    2021-08-11 17:10:56 [INFO]	[TRAIN] Epoch=260/270, Step=34/74, loss=26.678383, lr=1e-06, time_each_step=0.39s, eta=0:7:47
    2021-08-11 17:10:57 [INFO]	[TRAIN] Epoch=260/270, Step=36/74, loss=6.304432, lr=1e-06, time_each_step=0.4s, eta=0:7:46
    2021-08-11 17:10:58 [INFO]	[TRAIN] Epoch=260/270, Step=38/74, loss=17.743992, lr=1e-06, time_each_step=0.4s, eta=0:7:45
    2021-08-11 17:10:58 [INFO]	[TRAIN] Epoch=260/270, Step=40/74, loss=11.314267, lr=1e-06, time_each_step=0.38s, eta=0:7:44
    2021-08-11 17:10:59 [INFO]	[TRAIN] Epoch=260/270, Step=42/74, loss=11.242833, lr=1e-06, time_each_step=0.37s, eta=0:7:43
    2021-08-11 17:10:59 [INFO]	[TRAIN] Epoch=260/270, Step=44/74, loss=7.898633, lr=1e-06, time_each_step=0.38s, eta=0:7:42
    2021-08-11 17:11:00 [INFO]	[TRAIN] Epoch=260/270, Step=46/74, loss=11.274952, lr=1e-06, time_each_step=0.37s, eta=0:7:41
    2021-08-11 17:11:00 [INFO]	[TRAIN] Epoch=260/270, Step=48/74, loss=10.246872, lr=1e-06, time_each_step=0.37s, eta=0:7:41
    2021-08-11 17:11:01 [INFO]	[TRAIN] Epoch=260/270, Step=50/74, loss=10.612854, lr=1e-06, time_each_step=0.35s, eta=0:7:39
    2021-08-11 17:11:01 [INFO]	[TRAIN] Epoch=260/270, Step=52/74, loss=40.596527, lr=1e-06, time_each_step=0.3s, eta=0:7:38
    2021-08-11 17:11:02 [INFO]	[TRAIN] Epoch=260/270, Step=54/74, loss=6.055552, lr=1e-06, time_each_step=0.27s, eta=0:7:36
    2021-08-11 17:11:02 [INFO]	[TRAIN] Epoch=260/270, Step=56/74, loss=19.467571, lr=1e-06, time_each_step=0.24s, eta=0:7:35
    2021-08-11 17:11:02 [INFO]	[TRAIN] Epoch=260/270, Step=58/74, loss=10.388046, lr=1e-06, time_each_step=0.23s, eta=0:7:35
    2021-08-11 17:11:03 [INFO]	[TRAIN] Epoch=260/270, Step=60/74, loss=11.446394, lr=1e-06, time_each_step=0.23s, eta=0:7:34
    2021-08-11 17:11:03 [INFO]	[TRAIN] Epoch=260/270, Step=62/74, loss=15.900101, lr=1e-06, time_each_step=0.23s, eta=0:7:34
    2021-08-11 17:11:04 [INFO]	[TRAIN] Epoch=260/270, Step=64/74, loss=13.294798, lr=1e-06, time_each_step=0.22s, eta=0:7:33
    2021-08-11 17:11:04 [INFO]	[TRAIN] Epoch=260/270, Step=66/74, loss=8.115381, lr=1e-06, time_each_step=0.21s, eta=0:7:33
    2021-08-11 17:11:04 [INFO]	[TRAIN] Epoch=260/270, Step=68/74, loss=8.288696, lr=1e-06, time_each_step=0.21s, eta=0:7:32
    2021-08-11 17:11:05 [INFO]	[TRAIN] Epoch=260/270, Step=70/74, loss=16.527277, lr=1e-06, time_each_step=0.19s, eta=0:7:32
    2021-08-11 17:11:05 [INFO]	[TRAIN] Epoch=260/270, Step=72/74, loss=12.09454, lr=1e-06, time_each_step=0.2s, eta=0:7:31
    2021-08-11 17:11:06 [INFO]	[TRAIN] Epoch=260/270, Step=74/74, loss=16.535328, lr=1e-06, time_each_step=0.2s, eta=0:7:31
    2021-08-11 17:11:06 [INFO]	[TRAIN] Epoch 260 finished, loss=14.527512, lr=1e-06 .
    2021-08-11 17:11:06 [INFO]	Start to evaluating(total_samples=170, total_steps=22)...


    100%|██████████| 22/22 [00:13<00:00,  1.62it/s]


    2021-08-11 17:11:19 [INFO]	[EVAL] Finished, Epoch=260, bbox_map=80.396761 .
    2021-08-11 17:11:25 [INFO]	Model saved in output/yolov3_DarkNet53/epoch_260.
    2021-08-11 17:11:25 [INFO]	Current evaluated best model in eval_dataset is epoch_240, bbox_map=80.86807042180423
    2021-08-11 17:11:30 [INFO]	[TRAIN] Epoch=261/270, Step=2/74, loss=17.006172, lr=1e-06, time_each_step=0.41s, eta=0:4:56
    2021-08-11 17:11:31 [INFO]	[TRAIN] Epoch=261/270, Step=4/74, loss=10.051881, lr=1e-06, time_each_step=0.41s, eta=0:4:55
    2021-08-11 17:11:31 [INFO]	[TRAIN] Epoch=261/270, Step=6/74, loss=30.074577, lr=1e-06, time_each_step=0.43s, eta=0:4:56
    2021-08-11 17:11:32 [INFO]	[TRAIN] Epoch=261/270, Step=8/74, loss=20.997822, lr=1e-06, time_each_step=0.46s, eta=0:4:57
    2021-08-11 17:11:33 [INFO]	[TRAIN] Epoch=261/270, Step=10/74, loss=11.063059, lr=1e-06, time_each_step=0.47s, eta=0:4:57
    2021-08-11 17:11:34 [INFO]	[TRAIN] Epoch=261/270, Step=12/74, loss=9.20413, lr=1e-06, time_each_step=0.48s, eta=0:4:57
    2021-08-11 17:11:34 [INFO]	[TRAIN] Epoch=261/270, Step=14/74, loss=20.907181, lr=1e-06, time_each_step=0.49s, eta=0:4:56
    2021-08-11 17:11:35 [INFO]	[TRAIN] Epoch=261/270, Step=16/74, loss=12.567531, lr=1e-06, time_each_step=0.52s, eta=0:4:57
    2021-08-11 17:11:36 [INFO]	[TRAIN] Epoch=261/270, Step=18/74, loss=14.393622, lr=1e-06, time_each_step=0.53s, eta=0:4:56
    2021-08-11 17:11:36 [INFO]	[TRAIN] Epoch=261/270, Step=20/74, loss=19.664099, lr=1e-06, time_each_step=0.52s, eta=0:4:55
    2021-08-11 17:11:37 [INFO]	[TRAIN] Epoch=261/270, Step=22/74, loss=24.716476, lr=1e-06, time_each_step=0.34s, eta=0:4:44
    2021-08-11 17:11:37 [INFO]	[TRAIN] Epoch=261/270, Step=24/74, loss=8.640923, lr=1e-06, time_each_step=0.34s, eta=0:4:44
    2021-08-11 17:11:38 [INFO]	[TRAIN] Epoch=261/270, Step=26/74, loss=9.6955, lr=1e-06, time_each_step=0.32s, eta=0:4:42
    2021-08-11 17:11:38 [INFO]	[TRAIN] Epoch=261/270, Step=28/74, loss=8.823997, lr=1e-06, time_each_step=0.31s, eta=0:4:41
    2021-08-11 17:11:39 [INFO]	[TRAIN] Epoch=261/270, Step=30/74, loss=10.38156, lr=1e-06, time_each_step=0.32s, eta=0:4:41
    2021-08-11 17:11:40 [INFO]	[TRAIN] Epoch=261/270, Step=32/74, loss=27.773401, lr=1e-06, time_each_step=0.33s, eta=0:4:41
    2021-08-11 17:11:41 [INFO]	[TRAIN] Epoch=261/270, Step=34/74, loss=13.722922, lr=1e-06, time_each_step=0.35s, eta=0:4:41
    2021-08-11 17:11:42 [INFO]	[TRAIN] Epoch=261/270, Step=36/74, loss=12.648627, lr=1e-06, time_each_step=0.34s, eta=0:4:39
    2021-08-11 17:11:42 [INFO]	[TRAIN] Epoch=261/270, Step=38/74, loss=14.007172, lr=1e-06, time_each_step=0.34s, eta=0:4:39
    2021-08-11 17:11:43 [INFO]	[TRAIN] Epoch=261/270, Step=40/74, loss=15.413913, lr=1e-06, time_each_step=0.35s, eta=0:4:39
    2021-08-11 17:11:44 [INFO]	[TRAIN] Epoch=261/270, Step=42/74, loss=7.767055, lr=1e-06, time_each_step=0.35s, eta=0:4:38
    2021-08-11 17:11:44 [INFO]	[TRAIN] Epoch=261/270, Step=44/74, loss=17.870152, lr=1e-06, time_each_step=0.35s, eta=0:4:37
    2021-08-11 17:11:45 [INFO]	[TRAIN] Epoch=261/270, Step=46/74, loss=12.124239, lr=1e-06, time_each_step=0.35s, eta=0:4:36
    2021-08-11 17:11:45 [INFO]	[TRAIN] Epoch=261/270, Step=48/74, loss=6.673959, lr=1e-06, time_each_step=0.34s, eta=0:4:35
    2021-08-11 17:11:46 [INFO]	[TRAIN] Epoch=261/270, Step=50/74, loss=10.435584, lr=1e-06, time_each_step=0.32s, eta=0:4:34
    2021-08-11 17:11:46 [INFO]	[TRAIN] Epoch=261/270, Step=52/74, loss=9.248923, lr=1e-06, time_each_step=0.29s, eta=0:4:33
    2021-08-11 17:11:47 [INFO]	[TRAIN] Epoch=261/270, Step=54/74, loss=19.613966, lr=1e-06, time_each_step=0.27s, eta=0:4:32
    2021-08-11 17:11:47 [INFO]	[TRAIN] Epoch=261/270, Step=56/74, loss=30.476727, lr=1e-06, time_each_step=0.26s, eta=0:4:31
    2021-08-11 17:11:47 [INFO]	[TRAIN] Epoch=261/270, Step=58/74, loss=18.036705, lr=1e-06, time_each_step=0.25s, eta=0:4:31
    2021-08-11 17:11:48 [INFO]	[TRAIN] Epoch=261/270, Step=60/74, loss=49.415802, lr=1e-06, time_each_step=0.24s, eta=0:4:30
    2021-08-11 17:11:48 [INFO]	[TRAIN] Epoch=261/270, Step=62/74, loss=9.9712, lr=1e-06, time_each_step=0.22s, eta=0:4:29
    2021-08-11 17:11:48 [INFO]	[TRAIN] Epoch=261/270, Step=64/74, loss=6.265686, lr=1e-06, time_each_step=0.2s, eta=0:4:29
    2021-08-11 17:11:49 [INFO]	[TRAIN] Epoch=261/270, Step=66/74, loss=6.644839, lr=1e-06, time_each_step=0.2s, eta=0:4:28
    2021-08-11 17:11:49 [INFO]	[TRAIN] Epoch=261/270, Step=68/74, loss=11.147438, lr=1e-06, time_each_step=0.19s, eta=0:4:28
    2021-08-11 17:11:50 [INFO]	[TRAIN] Epoch=261/270, Step=70/74, loss=14.344288, lr=1e-06, time_each_step=0.2s, eta=0:4:27
    2021-08-11 17:11:50 [INFO]	[TRAIN] Epoch=261/270, Step=72/74, loss=7.660811, lr=1e-06, time_each_step=0.2s, eta=0:4:27
    2021-08-11 17:11:50 [INFO]	[TRAIN] Epoch=261/270, Step=74/74, loss=15.478492, lr=1e-06, time_each_step=0.19s, eta=0:4:27
    2021-08-11 17:11:50 [INFO]	[TRAIN] Epoch 261 finished, loss=14.941282, lr=1e-06 .
    2021-08-11 17:12:01 [INFO]	[TRAIN] Epoch=262/270, Step=2/74, loss=21.569279, lr=1e-06, time_each_step=0.69s, eta=0:4:28
    2021-08-11 17:12:02 [INFO]	[TRAIN] Epoch=262/270, Step=4/74, loss=10.565322, lr=1e-06, time_each_step=0.72s, eta=0:4:28
    2021-08-11 17:12:03 [INFO]	[TRAIN] Epoch=262/270, Step=6/74, loss=11.803533, lr=1e-06, time_each_step=0.75s, eta=0:4:29
    2021-08-11 17:12:03 [INFO]	[TRAIN] Epoch=262/270, Step=8/74, loss=7.556475, lr=1e-06, time_each_step=0.75s, eta=0:4:27
    2021-08-11 17:12:04 [INFO]	[TRAIN] Epoch=262/270, Step=10/74, loss=8.30157, lr=1e-06, time_each_step=0.77s, eta=0:4:27
    2021-08-11 17:12:05 [INFO]	[TRAIN] Epoch=262/270, Step=12/74, loss=17.801443, lr=1e-06, time_each_step=0.8s, eta=0:4:28
    2021-08-11 17:12:06 [INFO]	[TRAIN] Epoch=262/270, Step=14/74, loss=4.645299, lr=1e-06, time_each_step=0.82s, eta=0:4:27
    2021-08-11 17:12:06 [INFO]	[TRAIN] Epoch=262/270, Step=16/74, loss=14.647806, lr=1e-06, time_each_step=0.83s, eta=0:4:26
    2021-08-11 17:12:07 [INFO]	[TRAIN] Epoch=262/270, Step=18/74, loss=5.256688, lr=1e-06, time_each_step=0.86s, eta=0:4:26
    2021-08-11 17:12:08 [INFO]	[TRAIN] Epoch=262/270, Step=20/74, loss=14.399006, lr=1e-06, time_each_step=0.87s, eta=0:4:25
    2021-08-11 17:12:08 [INFO]	[TRAIN] Epoch=262/270, Step=22/74, loss=10.680769, lr=1e-06, time_each_step=0.37s, eta=0:3:57
    2021-08-11 17:12:09 [INFO]	[TRAIN] Epoch=262/270, Step=24/74, loss=24.655081, lr=1e-06, time_each_step=0.35s, eta=0:3:56
    2021-08-11 17:12:10 [INFO]	[TRAIN] Epoch=262/270, Step=26/74, loss=11.754101, lr=1e-06, time_each_step=0.35s, eta=0:3:55
    2021-08-11 17:12:11 [INFO]	[TRAIN] Epoch=262/270, Step=28/74, loss=13.017207, lr=1e-06, time_each_step=0.37s, eta=0:3:55
    2021-08-11 17:12:11 [INFO]	[TRAIN] Epoch=262/270, Step=30/74, loss=18.859344, lr=1e-06, time_each_step=0.37s, eta=0:3:55
    2021-08-11 17:12:12 [INFO]	[TRAIN] Epoch=262/270, Step=32/74, loss=8.379134, lr=1e-06, time_each_step=0.35s, eta=0:3:53
    2021-08-11 17:12:13 [INFO]	[TRAIN] Epoch=262/270, Step=34/74, loss=16.587429, lr=1e-06, time_each_step=0.36s, eta=0:3:53
    2021-08-11 17:12:13 [INFO]	[TRAIN] Epoch=262/270, Step=36/74, loss=9.672565, lr=1e-06, time_each_step=0.36s, eta=0:3:52
    2021-08-11 17:12:14 [INFO]	[TRAIN] Epoch=262/270, Step=38/74, loss=7.747082, lr=1e-06, time_each_step=0.34s, eta=0:3:50
    2021-08-11 17:12:14 [INFO]	[TRAIN] Epoch=262/270, Step=40/74, loss=4.569695, lr=1e-06, time_each_step=0.34s, eta=0:3:50
    2021-08-11 17:12:15 [INFO]	[TRAIN] Epoch=262/270, Step=42/74, loss=16.848177, lr=1e-06, time_each_step=0.35s, eta=0:3:49
    2021-08-11 17:12:16 [INFO]	[TRAIN] Epoch=262/270, Step=44/74, loss=12.897886, lr=1e-06, time_each_step=0.34s, eta=0:3:48
    2021-08-11 17:12:16 [INFO]	[TRAIN] Epoch=262/270, Step=46/74, loss=14.182905, lr=1e-06, time_each_step=0.32s, eta=0:3:47
    2021-08-11 17:12:17 [INFO]	[TRAIN] Epoch=262/270, Step=48/74, loss=12.666061, lr=1e-06, time_each_step=0.3s, eta=0:3:46
    2021-08-11 17:12:17 [INFO]	[TRAIN] Epoch=262/270, Step=50/74, loss=3.129289, lr=1e-06, time_each_step=0.29s, eta=0:3:45
    2021-08-11 17:12:18 [INFO]	[TRAIN] Epoch=262/270, Step=52/74, loss=16.787191, lr=1e-06, time_each_step=0.28s, eta=0:3:44
    2021-08-11 17:12:18 [INFO]	[TRAIN] Epoch=262/270, Step=54/74, loss=28.447594, lr=1e-06, time_each_step=0.26s, eta=0:3:43
    2021-08-11 17:12:19 [INFO]	[TRAIN] Epoch=262/270, Step=56/74, loss=12.933433, lr=1e-06, time_each_step=0.26s, eta=0:3:43
    2021-08-11 17:12:19 [INFO]	[TRAIN] Epoch=262/270, Step=58/74, loss=8.028517, lr=1e-06, time_each_step=0.27s, eta=0:3:42
    2021-08-11 17:12:19 [INFO]	[TRAIN] Epoch=262/270, Step=60/74, loss=15.873732, lr=1e-06, time_each_step=0.25s, eta=0:3:42
    2021-08-11 17:12:20 [INFO]	[TRAIN] Epoch=262/270, Step=62/74, loss=11.259022, lr=1e-06, time_each_step=0.24s, eta=0:3:41
    2021-08-11 17:12:20 [INFO]	[TRAIN] Epoch=262/270, Step=64/74, loss=13.496889, lr=1e-06, time_each_step=0.23s, eta=0:3:40
    2021-08-11 17:12:21 [INFO]	[TRAIN] Epoch=262/270, Step=66/74, loss=9.591476, lr=1e-06, time_each_step=0.22s, eta=0:3:40
    2021-08-11 17:12:21 [INFO]	[TRAIN] Epoch=262/270, Step=68/74, loss=7.595275, lr=1e-06, time_each_step=0.21s, eta=0:3:39
    2021-08-11 17:12:21 [INFO]	[TRAIN] Epoch=262/270, Step=70/74, loss=6.97455, lr=1e-06, time_each_step=0.2s, eta=0:3:39
    2021-08-11 17:12:21 [INFO]	[TRAIN] Epoch=262/270, Step=72/74, loss=51.355453, lr=1e-06, time_each_step=0.2s, eta=0:3:39
    2021-08-11 17:12:22 [INFO]	[TRAIN] Epoch=262/270, Step=74/74, loss=8.331343, lr=1e-06, time_each_step=0.2s, eta=0:3:38
    2021-08-11 17:12:22 [INFO]	[TRAIN] Epoch 262 finished, loss=14.276849, lr=1e-06 .
    2021-08-11 17:12:28 [INFO]	[TRAIN] Epoch=263/270, Step=2/74, loss=12.851072, lr=1e-06, time_each_step=0.49s, eta=0:4:36
    2021-08-11 17:12:29 [INFO]	[TRAIN] Epoch=263/270, Step=4/74, loss=29.731972, lr=1e-06, time_each_step=0.5s, eta=0:4:35
    2021-08-11 17:12:30 [INFO]	[TRAIN] Epoch=263/270, Step=6/74, loss=29.824102, lr=1e-06, time_each_step=0.5s, eta=0:4:35
    2021-08-11 17:12:30 [INFO]	[TRAIN] Epoch=263/270, Step=8/74, loss=12.578581, lr=1e-06, time_each_step=0.52s, eta=0:4:35
    2021-08-11 17:12:31 [INFO]	[TRAIN] Epoch=263/270, Step=10/74, loss=4.502079, lr=1e-06, time_each_step=0.54s, eta=0:4:35
    2021-08-11 17:12:32 [INFO]	[TRAIN] Epoch=263/270, Step=12/74, loss=32.910797, lr=1e-06, time_each_step=0.56s, eta=0:4:35
    2021-08-11 17:12:32 [INFO]	[TRAIN] Epoch=263/270, Step=14/74, loss=10.432022, lr=1e-06, time_each_step=0.57s, eta=0:4:35
    2021-08-11 17:12:33 [INFO]	[TRAIN] Epoch=263/270, Step=16/74, loss=12.76318, lr=1e-06, time_each_step=0.61s, eta=0:4:36
    2021-08-11 17:12:34 [INFO]	[TRAIN] Epoch=263/270, Step=18/74, loss=20.02832, lr=1e-06, time_each_step=0.64s, eta=0:4:36
    2021-08-11 17:12:35 [INFO]	[TRAIN] Epoch=263/270, Step=20/74, loss=11.929358, lr=1e-06, time_each_step=0.65s, eta=0:4:36
    2021-08-11 17:12:36 [INFO]	[TRAIN] Epoch=263/270, Step=22/74, loss=13.922813, lr=1e-06, time_each_step=0.36s, eta=0:4:19
    2021-08-11 17:12:36 [INFO]	[TRAIN] Epoch=263/270, Step=24/74, loss=8.557539, lr=1e-06, time_each_step=0.36s, eta=0:4:18
    2021-08-11 17:12:37 [INFO]	[TRAIN] Epoch=263/270, Step=26/74, loss=9.858669, lr=1e-06, time_each_step=0.37s, eta=0:4:18
    2021-08-11 17:12:38 [INFO]	[TRAIN] Epoch=263/270, Step=28/74, loss=9.750536, lr=1e-06, time_each_step=0.37s, eta=0:4:18
    2021-08-11 17:12:38 [INFO]	[TRAIN] Epoch=263/270, Step=30/74, loss=9.727373, lr=1e-06, time_each_step=0.37s, eta=0:4:17
    2021-08-11 17:12:39 [INFO]	[TRAIN] Epoch=263/270, Step=32/74, loss=8.347857, lr=1e-06, time_each_step=0.37s, eta=0:4:16
    2021-08-11 17:12:40 [INFO]	[TRAIN] Epoch=263/270, Step=34/74, loss=12.581409, lr=1e-06, time_each_step=0.37s, eta=0:4:16
    2021-08-11 17:12:41 [INFO]	[TRAIN] Epoch=263/270, Step=36/74, loss=6.772708, lr=1e-06, time_each_step=0.36s, eta=0:4:14
    2021-08-11 17:12:41 [INFO]	[TRAIN] Epoch=263/270, Step=38/74, loss=14.550104, lr=1e-06, time_each_step=0.35s, eta=0:4:13
    2021-08-11 17:12:42 [INFO]	[TRAIN] Epoch=263/270, Step=40/74, loss=8.945641, lr=1e-06, time_each_step=0.36s, eta=0:4:13
    2021-08-11 17:12:43 [INFO]	[TRAIN] Epoch=263/270, Step=42/74, loss=12.18516, lr=1e-06, time_each_step=0.35s, eta=0:4:12
    2021-08-11 17:12:43 [INFO]	[TRAIN] Epoch=263/270, Step=44/74, loss=11.48828, lr=1e-06, time_each_step=0.35s, eta=0:4:11
    2021-08-11 17:12:43 [INFO]	[TRAIN] Epoch=263/270, Step=46/74, loss=23.742268, lr=1e-06, time_each_step=0.33s, eta=0:4:10
    2021-08-11 17:12:44 [INFO]	[TRAIN] Epoch=263/270, Step=48/74, loss=15.225953, lr=1e-06, time_each_step=0.32s, eta=0:4:9
    2021-08-11 17:12:44 [INFO]	[TRAIN] Epoch=263/270, Step=50/74, loss=6.281875, lr=1e-06, time_each_step=0.31s, eta=0:4:8
    2021-08-11 17:12:45 [INFO]	[TRAIN] Epoch=263/270, Step=52/74, loss=16.465477, lr=1e-06, time_each_step=0.28s, eta=0:4:7
    2021-08-11 17:12:45 [INFO]	[TRAIN] Epoch=263/270, Step=54/74, loss=10.282433, lr=1e-06, time_each_step=0.27s, eta=0:4:6
    2021-08-11 17:12:46 [INFO]	[TRAIN] Epoch=263/270, Step=56/74, loss=12.01482, lr=1e-06, time_each_step=0.26s, eta=0:4:5
    2021-08-11 17:12:46 [INFO]	[TRAIN] Epoch=263/270, Step=58/74, loss=23.84207, lr=1e-06, time_each_step=0.24s, eta=0:4:4
    2021-08-11 17:12:46 [INFO]	[TRAIN] Epoch=263/270, Step=60/74, loss=13.000978, lr=1e-06, time_each_step=0.21s, eta=0:4:4
    2021-08-11 17:12:47 [INFO]	[TRAIN] Epoch=263/270, Step=62/74, loss=19.929043, lr=1e-06, time_each_step=0.2s, eta=0:4:3
    2021-08-11 17:12:47 [INFO]	[TRAIN] Epoch=263/270, Step=64/74, loss=20.280788, lr=1e-06, time_each_step=0.2s, eta=0:4:3
    2021-08-11 17:12:48 [INFO]	[TRAIN] Epoch=263/270, Step=66/74, loss=16.624842, lr=1e-06, time_each_step=0.21s, eta=0:4:2
    2021-08-11 17:12:48 [INFO]	[TRAIN] Epoch=263/270, Step=68/74, loss=9.61423, lr=1e-06, time_each_step=0.19s, eta=0:4:2
    2021-08-11 17:12:48 [INFO]	[TRAIN] Epoch=263/270, Step=70/74, loss=19.731926, lr=1e-06, time_each_step=0.19s, eta=0:4:1
    2021-08-11 17:12:49 [INFO]	[TRAIN] Epoch=263/270, Step=72/74, loss=13.981488, lr=1e-06, time_each_step=0.2s, eta=0:4:1
    2021-08-11 17:12:49 [INFO]	[TRAIN] Epoch=263/270, Step=74/74, loss=22.167665, lr=1e-06, time_each_step=0.2s, eta=0:4:1
    2021-08-11 17:12:49 [INFO]	[TRAIN] Epoch 263 finished, loss=14.66241, lr=1e-06 .
    2021-08-11 17:12:55 [INFO]	[TRAIN] Epoch=264/270, Step=2/74, loss=4.877478, lr=1e-06, time_each_step=0.46s, eta=0:3:37
    2021-08-11 17:12:56 [INFO]	[TRAIN] Epoch=264/270, Step=4/74, loss=9.223963, lr=1e-06, time_each_step=0.48s, eta=0:3:37
    2021-08-11 17:12:56 [INFO]	[TRAIN] Epoch=264/270, Step=6/74, loss=8.534245, lr=1e-06, time_each_step=0.49s, eta=0:3:37
    2021-08-11 17:12:57 [INFO]	[TRAIN] Epoch=264/270, Step=8/74, loss=18.208883, lr=1e-06, time_each_step=0.51s, eta=0:3:37
    2021-08-11 17:12:58 [INFO]	[TRAIN] Epoch=264/270, Step=10/74, loss=4.248755, lr=1e-06, time_each_step=0.52s, eta=0:3:37
    2021-08-11 17:12:59 [INFO]	[TRAIN] Epoch=264/270, Step=12/74, loss=18.843798, lr=1e-06, time_each_step=0.55s, eta=0:3:38
    2021-08-11 17:13:00 [INFO]	[TRAIN] Epoch=264/270, Step=14/74, loss=25.634148, lr=1e-06, time_each_step=0.59s, eta=0:3:39
    2021-08-11 17:13:00 [INFO]	[TRAIN] Epoch=264/270, Step=16/74, loss=24.454725, lr=1e-06, time_each_step=0.6s, eta=0:3:38
    2021-08-11 17:13:01 [INFO]	[TRAIN] Epoch=264/270, Step=18/74, loss=16.070749, lr=1e-06, time_each_step=0.6s, eta=0:3:37
    2021-08-11 17:13:01 [INFO]	[TRAIN] Epoch=264/270, Step=20/74, loss=22.290388, lr=1e-06, time_each_step=0.6s, eta=0:3:36
    2021-08-11 17:13:02 [INFO]	[TRAIN] Epoch=264/270, Step=22/74, loss=44.108288, lr=1e-06, time_each_step=0.34s, eta=0:3:21
    2021-08-11 17:13:03 [INFO]	[TRAIN] Epoch=264/270, Step=24/74, loss=14.704079, lr=1e-06, time_each_step=0.35s, eta=0:3:21
    2021-08-11 17:13:03 [INFO]	[TRAIN] Epoch=264/270, Step=26/74, loss=12.053633, lr=1e-06, time_each_step=0.37s, eta=0:3:21
    2021-08-11 17:13:04 [INFO]	[TRAIN] Epoch=264/270, Step=28/74, loss=10.259748, lr=1e-06, time_each_step=0.37s, eta=0:3:20
    2021-08-11 17:13:05 [INFO]	[TRAIN] Epoch=264/270, Step=30/74, loss=10.110012, lr=1e-06, time_each_step=0.37s, eta=0:3:20
    2021-08-11 17:13:06 [INFO]	[TRAIN] Epoch=264/270, Step=32/74, loss=8.231581, lr=1e-06, time_each_step=0.36s, eta=0:3:19
    2021-08-11 17:13:06 [INFO]	[TRAIN] Epoch=264/270, Step=34/74, loss=7.739195, lr=1e-06, time_each_step=0.34s, eta=0:3:17
    2021-08-11 17:13:07 [INFO]	[TRAIN] Epoch=264/270, Step=36/74, loss=16.308685, lr=1e-06, time_each_step=0.34s, eta=0:3:17
    2021-08-11 17:13:08 [INFO]	[TRAIN] Epoch=264/270, Step=38/74, loss=16.202679, lr=1e-06, time_each_step=0.34s, eta=0:3:16
    2021-08-11 17:13:08 [INFO]	[TRAIN] Epoch=264/270, Step=40/74, loss=7.14956, lr=1e-06, time_each_step=0.35s, eta=0:3:16
    2021-08-11 17:13:09 [INFO]	[TRAIN] Epoch=264/270, Step=42/74, loss=24.537769, lr=1e-06, time_each_step=0.35s, eta=0:3:15
    2021-08-11 17:13:10 [INFO]	[TRAIN] Epoch=264/270, Step=44/74, loss=10.751055, lr=1e-06, time_each_step=0.34s, eta=0:3:14
    2021-08-11 17:13:10 [INFO]	[TRAIN] Epoch=264/270, Step=46/74, loss=15.039386, lr=1e-06, time_each_step=0.34s, eta=0:3:13
    2021-08-11 17:13:11 [INFO]	[TRAIN] Epoch=264/270, Step=48/74, loss=13.754917, lr=1e-06, time_each_step=0.33s, eta=0:3:12
    2021-08-11 17:13:11 [INFO]	[TRAIN] Epoch=264/270, Step=50/74, loss=19.560081, lr=1e-06, time_each_step=0.3s, eta=0:3:11
    2021-08-11 17:13:11 [INFO]	[TRAIN] Epoch=264/270, Step=52/74, loss=18.127144, lr=1e-06, time_each_step=0.29s, eta=0:3:10
    2021-08-11 17:13:12 [INFO]	[TRAIN] Epoch=264/270, Step=54/74, loss=28.507919, lr=1e-06, time_each_step=0.28s, eta=0:3:9
    2021-08-11 17:13:12 [INFO]	[TRAIN] Epoch=264/270, Step=56/74, loss=18.262617, lr=1e-06, time_each_step=0.28s, eta=0:3:9
    2021-08-11 17:13:13 [INFO]	[TRAIN] Epoch=264/270, Step=58/74, loss=9.825419, lr=1e-06, time_each_step=0.26s, eta=0:3:8
    2021-08-11 17:13:13 [INFO]	[TRAIN] Epoch=264/270, Step=60/74, loss=17.822636, lr=1e-06, time_each_step=0.25s, eta=0:3:7
    2021-08-11 17:13:14 [INFO]	[TRAIN] Epoch=264/270, Step=62/74, loss=11.463007, lr=1e-06, time_each_step=0.24s, eta=0:3:7
    2021-08-11 17:13:14 [INFO]	[TRAIN] Epoch=264/270, Step=64/74, loss=10.649443, lr=1e-06, time_each_step=0.22s, eta=0:3:6
    2021-08-11 17:13:14 [INFO]	[TRAIN] Epoch=264/270, Step=66/74, loss=18.088892, lr=1e-06, time_each_step=0.21s, eta=0:3:5
    2021-08-11 17:13:15 [INFO]	[TRAIN] Epoch=264/270, Step=68/74, loss=9.178354, lr=1e-06, time_each_step=0.2s, eta=0:3:5
    2021-08-11 17:13:15 [INFO]	[TRAIN] Epoch=264/270, Step=70/74, loss=22.051174, lr=1e-06, time_each_step=0.21s, eta=0:3:4
    2021-08-11 17:13:15 [INFO]	[TRAIN] Epoch=264/270, Step=72/74, loss=7.927711, lr=1e-06, time_each_step=0.2s, eta=0:3:4
    2021-08-11 17:13:16 [INFO]	[TRAIN] Epoch=264/270, Step=74/74, loss=24.366537, lr=1e-06, time_each_step=0.19s, eta=0:3:4
    2021-08-11 17:13:16 [INFO]	[TRAIN] Epoch 264 finished, loss=14.912363, lr=1e-06 .
    2021-08-11 17:13:21 [INFO]	[TRAIN] Epoch=265/270, Step=2/74, loss=8.1581, lr=1e-06, time_each_step=0.43s, eta=0:3:3
    2021-08-11 17:13:22 [INFO]	[TRAIN] Epoch=265/270, Step=4/74, loss=28.927212, lr=1e-06, time_each_step=0.44s, eta=0:3:3
    2021-08-11 17:13:23 [INFO]	[TRAIN] Epoch=265/270, Step=6/74, loss=12.227845, lr=1e-06, time_each_step=0.47s, eta=0:3:4
    2021-08-11 17:13:23 [INFO]	[TRAIN] Epoch=265/270, Step=8/74, loss=33.446045, lr=1e-06, time_each_step=0.49s, eta=0:3:4
    2021-08-11 17:13:25 [INFO]	[TRAIN] Epoch=265/270, Step=10/74, loss=21.80851, lr=1e-06, time_each_step=0.53s, eta=0:3:6
    2021-08-11 17:13:25 [INFO]	[TRAIN] Epoch=265/270, Step=12/74, loss=9.241692, lr=1e-06, time_each_step=0.53s, eta=0:3:5
    2021-08-11 17:13:26 [INFO]	[TRAIN] Epoch=265/270, Step=14/74, loss=26.792168, lr=1e-06, time_each_step=0.55s, eta=0:3:5
    2021-08-11 17:13:27 [INFO]	[TRAIN] Epoch=265/270, Step=16/74, loss=5.750618, lr=1e-06, time_each_step=0.57s, eta=0:3:5
    2021-08-11 17:13:28 [INFO]	[TRAIN] Epoch=265/270, Step=18/74, loss=13.119104, lr=1e-06, time_each_step=0.6s, eta=0:3:5
    2021-08-11 17:13:29 [INFO]	[TRAIN] Epoch=265/270, Step=20/74, loss=7.470059, lr=1e-06, time_each_step=0.64s, eta=0:3:6
    2021-08-11 17:13:29 [INFO]	[TRAIN] Epoch=265/270, Step=22/74, loss=15.617425, lr=1e-06, time_each_step=0.41s, eta=0:2:53
    2021-08-11 17:13:30 [INFO]	[TRAIN] Epoch=265/270, Step=24/74, loss=12.150475, lr=1e-06, time_each_step=0.41s, eta=0:2:52
    2021-08-11 17:13:31 [INFO]	[TRAIN] Epoch=265/270, Step=26/74, loss=18.458033, lr=1e-06, time_each_step=0.4s, eta=0:2:51
    2021-08-11 17:13:32 [INFO]	[TRAIN] Epoch=265/270, Step=28/74, loss=11.663721, lr=1e-06, time_each_step=0.41s, eta=0:2:50
    2021-08-11 17:13:32 [INFO]	[TRAIN] Epoch=265/270, Step=30/74, loss=11.698418, lr=1e-06, time_each_step=0.39s, eta=0:2:49
    2021-08-11 17:13:33 [INFO]	[TRAIN] Epoch=265/270, Step=32/74, loss=10.279189, lr=1e-06, time_each_step=0.42s, eta=0:2:49
    2021-08-11 17:13:35 [INFO]	[TRAIN] Epoch=265/270, Step=34/74, loss=24.368803, lr=1e-06, time_each_step=0.43s, eta=0:2:49
    2021-08-11 17:13:35 [INFO]	[TRAIN] Epoch=265/270, Step=36/74, loss=17.076321, lr=1e-06, time_each_step=0.43s, eta=0:2:48
    2021-08-11 17:13:36 [INFO]	[TRAIN] Epoch=265/270, Step=38/74, loss=12.014997, lr=1e-06, time_each_step=0.41s, eta=0:2:47
    2021-08-11 17:13:37 [INFO]	[TRAIN] Epoch=265/270, Step=40/74, loss=6.559946, lr=1e-06, time_each_step=0.4s, eta=0:2:45
    2021-08-11 17:13:37 [INFO]	[TRAIN] Epoch=265/270, Step=42/74, loss=13.593258, lr=1e-06, time_each_step=0.38s, eta=0:2:44
    2021-08-11 17:13:37 [INFO]	[TRAIN] Epoch=265/270, Step=44/74, loss=17.076271, lr=1e-06, time_each_step=0.36s, eta=0:2:42
    2021-08-11 17:13:38 [INFO]	[TRAIN] Epoch=265/270, Step=46/74, loss=27.877846, lr=1e-06, time_each_step=0.34s, eta=0:2:41
    2021-08-11 17:13:38 [INFO]	[TRAIN] Epoch=265/270, Step=48/74, loss=18.58753, lr=1e-06, time_each_step=0.32s, eta=0:2:40
    2021-08-11 17:13:39 [INFO]	[TRAIN] Epoch=265/270, Step=50/74, loss=7.02708, lr=1e-06, time_each_step=0.31s, eta=0:2:39
    2021-08-11 17:13:39 [INFO]	[TRAIN] Epoch=265/270, Step=52/74, loss=5.127255, lr=1e-06, time_each_step=0.28s, eta=0:2:38
    2021-08-11 17:13:39 [INFO]	[TRAIN] Epoch=265/270, Step=54/74, loss=13.815021, lr=1e-06, time_each_step=0.24s, eta=0:2:36
    2021-08-11 17:13:40 [INFO]	[TRAIN] Epoch=265/270, Step=56/74, loss=11.095653, lr=1e-06, time_each_step=0.22s, eta=0:2:36
    2021-08-11 17:13:40 [INFO]	[TRAIN] Epoch=265/270, Step=58/74, loss=8.690472, lr=1e-06, time_each_step=0.22s, eta=0:2:35
    2021-08-11 17:13:41 [INFO]	[TRAIN] Epoch=265/270, Step=60/74, loss=13.920027, lr=1e-06, time_each_step=0.2s, eta=0:2:34
    2021-08-11 17:13:41 [INFO]	[TRAIN] Epoch=265/270, Step=62/74, loss=14.348707, lr=1e-06, time_each_step=0.19s, eta=0:2:34
    2021-08-11 17:13:41 [INFO]	[TRAIN] Epoch=265/270, Step=64/74, loss=17.08016, lr=1e-06, time_each_step=0.2s, eta=0:2:34
    2021-08-11 17:13:42 [INFO]	[TRAIN] Epoch=265/270, Step=66/74, loss=17.518044, lr=1e-06, time_each_step=0.21s, eta=0:2:33
    2021-08-11 17:13:42 [INFO]	[TRAIN] Epoch=265/270, Step=68/74, loss=10.141204, lr=1e-06, time_each_step=0.21s, eta=0:2:33
    2021-08-11 17:13:43 [INFO]	[TRAIN] Epoch=265/270, Step=70/74, loss=15.978114, lr=1e-06, time_each_step=0.2s, eta=0:2:32
    2021-08-11 17:13:43 [INFO]	[TRAIN] Epoch=265/270, Step=72/74, loss=51.542538, lr=1e-06, time_each_step=0.2s, eta=0:2:32
    2021-08-11 17:13:44 [INFO]	[TRAIN] Epoch=265/270, Step=74/74, loss=7.21471, lr=1e-06, time_each_step=0.21s, eta=0:2:32
    2021-08-11 17:13:44 [INFO]	[TRAIN] Epoch 265 finished, loss=15.008563, lr=1e-06 .
    2021-08-11 17:13:58 [INFO]	[TRAIN] Epoch=266/270, Step=2/74, loss=8.903593, lr=1e-06, time_each_step=0.91s, eta=0:3:16
    2021-08-11 17:13:59 [INFO]	[TRAIN] Epoch=266/270, Step=4/74, loss=19.549173, lr=1e-06, time_each_step=0.93s, eta=0:3:15
    2021-08-11 17:13:59 [INFO]	[TRAIN] Epoch=266/270, Step=6/74, loss=7.480794, lr=1e-06, time_each_step=0.94s, eta=0:3:14
    2021-08-11 17:14:00 [INFO]	[TRAIN] Epoch=266/270, Step=8/74, loss=25.259087, lr=1e-06, time_each_step=0.96s, eta=0:3:14
    2021-08-11 17:14:01 [INFO]	[TRAIN] Epoch=266/270, Step=10/74, loss=8.477297, lr=1e-06, time_each_step=0.97s, eta=0:3:13
    2021-08-11 17:14:01 [INFO]	[TRAIN] Epoch=266/270, Step=12/74, loss=11.349889, lr=1e-06, time_each_step=0.98s, eta=0:3:11
    2021-08-11 17:14:03 [INFO]	[TRAIN] Epoch=266/270, Step=14/74, loss=5.193202, lr=1e-06, time_each_step=1.02s, eta=0:3:12
    2021-08-11 17:14:03 [INFO]	[TRAIN] Epoch=266/270, Step=16/74, loss=9.049095, lr=1e-06, time_each_step=1.04s, eta=0:3:11
    2021-08-11 17:14:04 [INFO]	[TRAIN] Epoch=266/270, Step=18/74, loss=30.621714, lr=1e-06, time_each_step=1.05s, eta=0:3:9
    2021-08-11 17:14:05 [INFO]	[TRAIN] Epoch=266/270, Step=20/74, loss=13.359322, lr=1e-06, time_each_step=1.08s, eta=0:3:9
    2021-08-11 17:14:06 [INFO]	[TRAIN] Epoch=266/270, Step=22/74, loss=10.971096, lr=1e-06, time_each_step=0.41s, eta=0:2:32
    2021-08-11 17:14:07 [INFO]	[TRAIN] Epoch=266/270, Step=24/74, loss=21.682896, lr=1e-06, time_each_step=0.4s, eta=0:2:31
    2021-08-11 17:14:08 [INFO]	[TRAIN] Epoch=266/270, Step=26/74, loss=45.345612, lr=1e-06, time_each_step=0.41s, eta=0:2:30
    2021-08-11 17:14:08 [INFO]	[TRAIN] Epoch=266/270, Step=28/74, loss=12.047031, lr=1e-06, time_each_step=0.41s, eta=0:2:30
    2021-08-11 17:14:09 [INFO]	[TRAIN] Epoch=266/270, Step=30/74, loss=47.149517, lr=1e-06, time_each_step=0.42s, eta=0:2:29
    2021-08-11 17:14:10 [INFO]	[TRAIN] Epoch=266/270, Step=32/74, loss=11.887918, lr=1e-06, time_each_step=0.43s, eta=0:2:28
    2021-08-11 17:14:11 [INFO]	[TRAIN] Epoch=266/270, Step=34/74, loss=18.315575, lr=1e-06, time_each_step=0.4s, eta=0:2:26
    2021-08-11 17:14:11 [INFO]	[TRAIN] Epoch=266/270, Step=36/74, loss=11.163218, lr=1e-06, time_each_step=0.39s, eta=0:2:25
    2021-08-11 17:14:12 [INFO]	[TRAIN] Epoch=266/270, Step=38/74, loss=5.241792, lr=1e-06, time_each_step=0.38s, eta=0:2:24
    2021-08-11 17:14:12 [INFO]	[TRAIN] Epoch=266/270, Step=40/74, loss=14.544541, lr=1e-06, time_each_step=0.36s, eta=0:2:23
    2021-08-11 17:14:13 [INFO]	[TRAIN] Epoch=266/270, Step=42/74, loss=8.243265, lr=1e-06, time_each_step=0.35s, eta=0:2:22
    2021-08-11 17:14:14 [INFO]	[TRAIN] Epoch=266/270, Step=44/74, loss=24.946148, lr=1e-06, time_each_step=0.34s, eta=0:2:21
    2021-08-11 17:14:14 [INFO]	[TRAIN] Epoch=266/270, Step=46/74, loss=19.801098, lr=1e-06, time_each_step=0.33s, eta=0:2:20
    2021-08-11 17:14:15 [INFO]	[TRAIN] Epoch=266/270, Step=48/74, loss=10.556282, lr=1e-06, time_each_step=0.31s, eta=0:2:18
    2021-08-11 17:14:15 [INFO]	[TRAIN] Epoch=266/270, Step=50/74, loss=25.659245, lr=1e-06, time_each_step=0.29s, eta=0:2:17
    2021-08-11 17:14:15 [INFO]	[TRAIN] Epoch=266/270, Step=52/74, loss=20.593723, lr=1e-06, time_each_step=0.27s, eta=0:2:16
    2021-08-11 17:14:16 [INFO]	[TRAIN] Epoch=266/270, Step=54/74, loss=8.755241, lr=1e-06, time_each_step=0.26s, eta=0:2:16
    2021-08-11 17:14:16 [INFO]	[TRAIN] Epoch=266/270, Step=56/74, loss=11.625953, lr=1e-06, time_each_step=0.26s, eta=0:2:15
    2021-08-11 17:14:17 [INFO]	[TRAIN] Epoch=266/270, Step=58/74, loss=8.112548, lr=1e-06, time_each_step=0.25s, eta=0:2:14
    2021-08-11 17:14:17 [INFO]	[TRAIN] Epoch=266/270, Step=60/74, loss=8.854031, lr=1e-06, time_each_step=0.23s, eta=0:2:14
    2021-08-11 17:14:17 [INFO]	[TRAIN] Epoch=266/270, Step=62/74, loss=8.002543, lr=1e-06, time_each_step=0.21s, eta=0:2:13
    2021-08-11 17:14:18 [INFO]	[TRAIN] Epoch=266/270, Step=64/74, loss=7.278928, lr=1e-06, time_each_step=0.22s, eta=0:2:13
    2021-08-11 17:14:18 [INFO]	[TRAIN] Epoch=266/270, Step=66/74, loss=52.70789, lr=1e-06, time_each_step=0.2s, eta=0:2:12
    2021-08-11 17:14:18 [INFO]	[TRAIN] Epoch=266/270, Step=68/74, loss=16.305218, lr=1e-06, time_each_step=0.19s, eta=0:2:12
    2021-08-11 17:14:19 [INFO]	[TRAIN] Epoch=266/270, Step=70/74, loss=12.068129, lr=1e-06, time_each_step=0.2s, eta=0:2:11
    2021-08-11 17:14:19 [INFO]	[TRAIN] Epoch=266/270, Step=72/74, loss=8.015064, lr=1e-06, time_each_step=0.19s, eta=0:2:11
    2021-08-11 17:14:20 [INFO]	[TRAIN] Epoch=266/270, Step=74/74, loss=8.126495, lr=1e-06, time_each_step=0.19s, eta=0:2:10
    2021-08-11 17:14:20 [INFO]	[TRAIN] Epoch 266 finished, loss=14.894748, lr=1e-06 .
    2021-08-11 17:14:25 [INFO]	[TRAIN] Epoch=267/270, Step=2/74, loss=17.994005, lr=1e-06, time_each_step=0.43s, eta=0:2:39
    2021-08-11 17:14:26 [INFO]	[TRAIN] Epoch=267/270, Step=4/74, loss=7.17708, lr=1e-06, time_each_step=0.48s, eta=0:2:42
    2021-08-11 17:14:27 [INFO]	[TRAIN] Epoch=267/270, Step=6/74, loss=9.839626, lr=1e-06, time_each_step=0.51s, eta=0:2:43
    2021-08-11 17:14:28 [INFO]	[TRAIN] Epoch=267/270, Step=8/74, loss=20.203054, lr=1e-06, time_each_step=0.54s, eta=0:2:44
    2021-08-11 17:14:29 [INFO]	[TRAIN] Epoch=267/270, Step=10/74, loss=28.001677, lr=1e-06, time_each_step=0.55s, eta=0:2:43
    2021-08-11 17:14:30 [INFO]	[TRAIN] Epoch=267/270, Step=12/74, loss=9.417952, lr=1e-06, time_each_step=0.57s, eta=0:2:44
    2021-08-11 17:14:31 [INFO]	[TRAIN] Epoch=267/270, Step=14/74, loss=7.31035, lr=1e-06, time_each_step=0.6s, eta=0:2:44
    2021-08-11 17:14:31 [INFO]	[TRAIN] Epoch=267/270, Step=16/74, loss=10.258673, lr=1e-06, time_each_step=0.62s, eta=0:2:44
    2021-08-11 17:14:32 [INFO]	[TRAIN] Epoch=267/270, Step=18/74, loss=8.106693, lr=1e-06, time_each_step=0.62s, eta=0:2:43
    2021-08-11 17:14:32 [INFO]	[TRAIN] Epoch=267/270, Step=20/74, loss=18.853271, lr=1e-06, time_each_step=0.63s, eta=0:2:42
    2021-08-11 17:14:33 [INFO]	[TRAIN] Epoch=267/270, Step=22/74, loss=8.566168, lr=1e-06, time_each_step=0.4s, eta=0:2:29
    2021-08-11 17:14:34 [INFO]	[TRAIN] Epoch=267/270, Step=24/74, loss=21.616142, lr=1e-06, time_each_step=0.37s, eta=0:2:27
    2021-08-11 17:14:35 [INFO]	[TRAIN] Epoch=267/270, Step=26/74, loss=4.962493, lr=1e-06, time_each_step=0.37s, eta=0:2:26
    2021-08-11 17:14:35 [INFO]	[TRAIN] Epoch=267/270, Step=28/74, loss=23.643415, lr=1e-06, time_each_step=0.36s, eta=0:2:25
    2021-08-11 17:14:36 [INFO]	[TRAIN] Epoch=267/270, Step=30/74, loss=62.091881, lr=1e-06, time_each_step=0.35s, eta=0:2:23
    2021-08-11 17:14:36 [INFO]	[TRAIN] Epoch=267/270, Step=32/74, loss=9.728424, lr=1e-06, time_each_step=0.34s, eta=0:2:22
    2021-08-11 17:14:37 [INFO]	[TRAIN] Epoch=267/270, Step=34/74, loss=15.639208, lr=1e-06, time_each_step=0.34s, eta=0:2:22
    2021-08-11 17:14:38 [INFO]	[TRAIN] Epoch=267/270, Step=36/74, loss=6.404258, lr=1e-06, time_each_step=0.35s, eta=0:2:21
    2021-08-11 17:14:39 [INFO]	[TRAIN] Epoch=267/270, Step=38/74, loss=7.973228, lr=1e-06, time_each_step=0.37s, eta=0:2:22
    2021-08-11 17:14:40 [INFO]	[TRAIN] Epoch=267/270, Step=40/74, loss=16.687273, lr=1e-06, time_each_step=0.37s, eta=0:2:20
    2021-08-11 17:14:41 [INFO]	[TRAIN] Epoch=267/270, Step=42/74, loss=6.94892, lr=1e-06, time_each_step=0.38s, eta=0:2:20
    2021-08-11 17:14:41 [INFO]	[TRAIN] Epoch=267/270, Step=44/74, loss=11.794623, lr=1e-06, time_each_step=0.38s, eta=0:2:19
    2021-08-11 17:14:42 [INFO]	[TRAIN] Epoch=267/270, Step=46/74, loss=28.343838, lr=1e-06, time_each_step=0.35s, eta=0:2:18
    2021-08-11 17:14:42 [INFO]	[TRAIN] Epoch=267/270, Step=48/74, loss=18.468971, lr=1e-06, time_each_step=0.34s, eta=0:2:17
    2021-08-11 17:14:43 [INFO]	[TRAIN] Epoch=267/270, Step=50/74, loss=13.988197, lr=1e-06, time_each_step=0.34s, eta=0:2:16
    2021-08-11 17:14:43 [INFO]	[TRAIN] Epoch=267/270, Step=52/74, loss=14.37686, lr=1e-06, time_each_step=0.33s, eta=0:2:15
    2021-08-11 17:14:43 [INFO]	[TRAIN] Epoch=267/270, Step=54/74, loss=10.479725, lr=1e-06, time_each_step=0.3s, eta=0:2:14
    2021-08-11 17:14:44 [INFO]	[TRAIN] Epoch=267/270, Step=56/74, loss=16.837383, lr=1e-06, time_each_step=0.28s, eta=0:2:13
    2021-08-11 17:14:44 [INFO]	[TRAIN] Epoch=267/270, Step=58/74, loss=7.680025, lr=1e-06, time_each_step=0.25s, eta=0:2:12
    2021-08-11 17:14:44 [INFO]	[TRAIN] Epoch=267/270, Step=60/74, loss=6.812506, lr=1e-06, time_each_step=0.24s, eta=0:2:11
    2021-08-11 17:14:45 [INFO]	[TRAIN] Epoch=267/270, Step=62/74, loss=14.724276, lr=1e-06, time_each_step=0.21s, eta=0:2:11
    2021-08-11 17:14:45 [INFO]	[TRAIN] Epoch=267/270, Step=64/74, loss=20.570221, lr=1e-06, time_each_step=0.19s, eta=0:2:10
    2021-08-11 17:14:46 [INFO]	[TRAIN] Epoch=267/270, Step=66/74, loss=10.349286, lr=1e-06, time_each_step=0.2s, eta=0:2:10
    2021-08-11 17:14:46 [INFO]	[TRAIN] Epoch=267/270, Step=68/74, loss=5.400686, lr=1e-06, time_each_step=0.19s, eta=0:2:9
    2021-08-11 17:14:46 [INFO]	[TRAIN] Epoch=267/270, Step=70/74, loss=23.81424, lr=1e-06, time_each_step=0.18s, eta=0:2:9
    2021-08-11 17:14:47 [INFO]	[TRAIN] Epoch=267/270, Step=72/74, loss=28.280018, lr=1e-06, time_each_step=0.17s, eta=0:2:8
    2021-08-11 17:14:47 [INFO]	[TRAIN] Epoch=267/270, Step=74/74, loss=9.94648, lr=1e-06, time_each_step=0.19s, eta=0:2:8
    2021-08-11 17:14:47 [INFO]	[TRAIN] Epoch 267 finished, loss=15.910323, lr=1e-06 .
    2021-08-11 17:15:02 [INFO]	[TRAIN] Epoch=268/270, Step=2/74, loss=25.136723, lr=1e-06, time_each_step=0.9s, eta=0:2:19
    2021-08-11 17:15:02 [INFO]	[TRAIN] Epoch=268/270, Step=4/74, loss=31.766642, lr=1e-06, time_each_step=0.9s, eta=0:2:18
    2021-08-11 17:15:03 [INFO]	[TRAIN] Epoch=268/270, Step=6/74, loss=8.570219, lr=1e-06, time_each_step=0.93s, eta=0:2:17
    2021-08-11 17:15:04 [INFO]	[TRAIN] Epoch=268/270, Step=8/74, loss=9.193214, lr=1e-06, time_each_step=0.95s, eta=0:2:17
    2021-08-11 17:15:04 [INFO]	[TRAIN] Epoch=268/270, Step=10/74, loss=37.096573, lr=1e-06, time_each_step=0.97s, eta=0:2:16
    2021-08-11 17:15:05 [INFO]	[TRAIN] Epoch=268/270, Step=12/74, loss=29.592339, lr=1e-06, time_each_step=0.98s, eta=0:2:15
    2021-08-11 17:15:06 [INFO]	[TRAIN] Epoch=268/270, Step=14/74, loss=8.634319, lr=1e-06, time_each_step=1.01s, eta=0:2:15
    2021-08-11 17:15:07 [INFO]	[TRAIN] Epoch=268/270, Step=16/74, loss=12.715829, lr=1e-06, time_each_step=1.03s, eta=0:2:14
    2021-08-11 17:15:07 [INFO]	[TRAIN] Epoch=268/270, Step=18/74, loss=17.122631, lr=1e-06, time_each_step=1.04s, eta=0:2:13
    2021-08-11 17:15:08 [INFO]	[TRAIN] Epoch=268/270, Step=20/74, loss=13.453587, lr=1e-06, time_each_step=1.03s, eta=0:2:10
    2021-08-11 17:15:09 [INFO]	[TRAIN] Epoch=268/270, Step=22/74, loss=10.610682, lr=1e-06, time_each_step=0.34s, eta=0:1:32
    2021-08-11 17:15:10 [INFO]	[TRAIN] Epoch=268/270, Step=24/74, loss=11.372307, lr=1e-06, time_each_step=0.36s, eta=0:1:32
    2021-08-11 17:15:10 [INFO]	[TRAIN] Epoch=268/270, Step=26/74, loss=15.978659, lr=1e-06, time_each_step=0.37s, eta=0:1:32
    2021-08-11 17:15:11 [INFO]	[TRAIN] Epoch=268/270, Step=28/74, loss=8.244082, lr=1e-06, time_each_step=0.36s, eta=0:1:31
    2021-08-11 17:15:12 [INFO]	[TRAIN] Epoch=268/270, Step=30/74, loss=8.888625, lr=1e-06, time_each_step=0.37s, eta=0:1:30
    2021-08-11 17:15:13 [INFO]	[TRAIN] Epoch=268/270, Step=32/74, loss=20.199593, lr=1e-06, time_each_step=0.36s, eta=0:1:30
    2021-08-11 17:15:13 [INFO]	[TRAIN] Epoch=268/270, Step=34/74, loss=18.431459, lr=1e-06, time_each_step=0.36s, eta=0:1:29
    2021-08-11 17:15:14 [INFO]	[TRAIN] Epoch=268/270, Step=36/74, loss=18.751652, lr=1e-06, time_each_step=0.35s, eta=0:1:28
    2021-08-11 17:15:15 [INFO]	[TRAIN] Epoch=268/270, Step=38/74, loss=10.698001, lr=1e-06, time_each_step=0.37s, eta=0:1:28
    2021-08-11 17:15:16 [INFO]	[TRAIN] Epoch=268/270, Step=40/74, loss=26.78175, lr=1e-06, time_each_step=0.38s, eta=0:1:27
    2021-08-11 17:15:16 [INFO]	[TRAIN] Epoch=268/270, Step=42/74, loss=9.536353, lr=1e-06, time_each_step=0.37s, eta=0:1:26
    2021-08-11 17:15:17 [INFO]	[TRAIN] Epoch=268/270, Step=44/74, loss=22.126873, lr=1e-06, time_each_step=0.36s, eta=0:1:25
    2021-08-11 17:15:17 [INFO]	[TRAIN] Epoch=268/270, Step=46/74, loss=6.65977, lr=1e-06, time_each_step=0.33s, eta=0:1:24
    2021-08-11 17:15:17 [INFO]	[TRAIN] Epoch=268/270, Step=48/74, loss=6.190161, lr=1e-06, time_each_step=0.32s, eta=0:1:23
    2021-08-11 17:15:18 [INFO]	[TRAIN] Epoch=268/270, Step=50/74, loss=17.10891, lr=1e-06, time_each_step=0.3s, eta=0:1:22
    2021-08-11 17:15:18 [INFO]	[TRAIN] Epoch=268/270, Step=52/74, loss=11.831927, lr=1e-06, time_each_step=0.28s, eta=0:1:20
    2021-08-11 17:15:18 [INFO]	[TRAIN] Epoch=268/270, Step=54/74, loss=17.972275, lr=1e-06, time_each_step=0.26s, eta=0:1:20
    2021-08-11 17:15:19 [INFO]	[TRAIN] Epoch=268/270, Step=56/74, loss=9.06725, lr=1e-06, time_each_step=0.23s, eta=0:1:19
    2021-08-11 17:15:19 [INFO]	[TRAIN] Epoch=268/270, Step=58/74, loss=15.010383, lr=1e-06, time_each_step=0.22s, eta=0:1:18
    2021-08-11 17:15:20 [INFO]	[TRAIN] Epoch=268/270, Step=60/74, loss=25.383997, lr=1e-06, time_each_step=0.2s, eta=0:1:17
    2021-08-11 17:15:20 [INFO]	[TRAIN] Epoch=268/270, Step=62/74, loss=17.399338, lr=1e-06, time_each_step=0.21s, eta=0:1:17
    2021-08-11 17:15:21 [INFO]	[TRAIN] Epoch=268/270, Step=64/74, loss=4.979928, lr=1e-06, time_each_step=0.2s, eta=0:1:16
    2021-08-11 17:15:21 [INFO]	[TRAIN] Epoch=268/270, Step=66/74, loss=17.200165, lr=1e-06, time_each_step=0.2s, eta=0:1:16
    2021-08-11 17:15:21 [INFO]	[TRAIN] Epoch=268/270, Step=68/74, loss=13.511909, lr=1e-06, time_each_step=0.2s, eta=0:1:16
    2021-08-11 17:15:22 [INFO]	[TRAIN] Epoch=268/270, Step=70/74, loss=11.523186, lr=1e-06, time_each_step=0.2s, eta=0:1:15
    2021-08-11 17:15:22 [INFO]	[TRAIN] Epoch=268/270, Step=72/74, loss=12.103643, lr=1e-06, time_each_step=0.21s, eta=0:1:15
    2021-08-11 17:15:23 [INFO]	[TRAIN] Epoch=268/270, Step=74/74, loss=29.691963, lr=1e-06, time_each_step=0.21s, eta=0:1:14
    2021-08-11 17:15:23 [INFO]	[TRAIN] Epoch 268 finished, loss=15.156486, lr=1e-06 .
    2021-08-11 17:15:29 [INFO]	[TRAIN] Epoch=269/270, Step=2/74, loss=8.620803, lr=1e-06, time_each_step=0.5s, eta=0:1:31
    2021-08-11 17:15:30 [INFO]	[TRAIN] Epoch=269/270, Step=4/74, loss=10.423522, lr=1e-06, time_each_step=0.51s, eta=0:1:31
    2021-08-11 17:15:30 [INFO]	[TRAIN] Epoch=269/270, Step=6/74, loss=16.06321, lr=1e-06, time_each_step=0.54s, eta=0:1:31
    2021-08-11 17:15:31 [INFO]	[TRAIN] Epoch=269/270, Step=8/74, loss=7.452971, lr=1e-06, time_each_step=0.55s, eta=0:1:31
    2021-08-11 17:15:32 [INFO]	[TRAIN] Epoch=269/270, Step=10/74, loss=11.125712, lr=1e-06, time_each_step=0.56s, eta=0:1:31
    2021-08-11 17:15:33 [INFO]	[TRAIN] Epoch=269/270, Step=12/74, loss=18.695545, lr=1e-06, time_each_step=0.58s, eta=0:1:31
    2021-08-11 17:15:33 [INFO]	[TRAIN] Epoch=269/270, Step=14/74, loss=9.123857, lr=1e-06, time_each_step=0.6s, eta=0:1:31
    2021-08-11 17:15:34 [INFO]	[TRAIN] Epoch=269/270, Step=16/74, loss=41.664055, lr=1e-06, time_each_step=0.6s, eta=0:1:30
    2021-08-11 17:15:35 [INFO]	[TRAIN] Epoch=269/270, Step=18/74, loss=9.793462, lr=1e-06, time_each_step=0.61s, eta=0:1:29
    2021-08-11 17:15:35 [INFO]	[TRAIN] Epoch=269/270, Step=20/74, loss=7.277332, lr=1e-06, time_each_step=0.61s, eta=0:1:28
    2021-08-11 17:15:36 [INFO]	[TRAIN] Epoch=269/270, Step=22/74, loss=8.93724, lr=1e-06, time_each_step=0.34s, eta=0:1:13
    2021-08-11 17:15:36 [INFO]	[TRAIN] Epoch=269/270, Step=24/74, loss=51.087254, lr=1e-06, time_each_step=0.34s, eta=0:1:12
    2021-08-11 17:15:37 [INFO]	[TRAIN] Epoch=269/270, Step=26/74, loss=12.97851, lr=1e-06, time_each_step=0.35s, eta=0:1:11
    2021-08-11 17:15:38 [INFO]	[TRAIN] Epoch=269/270, Step=28/74, loss=27.033352, lr=1e-06, time_each_step=0.33s, eta=0:1:10
    2021-08-11 17:15:38 [INFO]	[TRAIN] Epoch=269/270, Step=30/74, loss=3.709891, lr=1e-06, time_each_step=0.32s, eta=0:1:9
    2021-08-11 17:15:39 [INFO]	[TRAIN] Epoch=269/270, Step=32/74, loss=13.079575, lr=1e-06, time_each_step=0.32s, eta=0:1:8
    2021-08-11 17:15:40 [INFO]	[TRAIN] Epoch=269/270, Step=34/74, loss=7.409634, lr=1e-06, time_each_step=0.31s, eta=0:1:7
    2021-08-11 17:15:40 [INFO]	[TRAIN] Epoch=269/270, Step=36/74, loss=19.644018, lr=1e-06, time_each_step=0.32s, eta=0:1:7
    2021-08-11 17:15:41 [INFO]	[TRAIN] Epoch=269/270, Step=38/74, loss=16.248272, lr=1e-06, time_each_step=0.33s, eta=0:1:7
    2021-08-11 17:15:42 [INFO]	[TRAIN] Epoch=269/270, Step=40/74, loss=10.125792, lr=1e-06, time_each_step=0.36s, eta=0:1:7
    2021-08-11 17:15:43 [INFO]	[TRAIN] Epoch=269/270, Step=42/74, loss=10.961687, lr=1e-06, time_each_step=0.37s, eta=0:1:6
    2021-08-11 17:15:44 [INFO]	[TRAIN] Epoch=269/270, Step=44/74, loss=6.371451, lr=1e-06, time_each_step=0.37s, eta=0:1:6
    2021-08-11 17:15:44 [INFO]	[TRAIN] Epoch=269/270, Step=46/74, loss=18.136482, lr=1e-06, time_each_step=0.35s, eta=0:1:5
    2021-08-11 17:15:45 [INFO]	[TRAIN] Epoch=269/270, Step=48/74, loss=12.800804, lr=1e-06, time_each_step=0.35s, eta=0:1:4
    2021-08-11 17:15:45 [INFO]	[TRAIN] Epoch=269/270, Step=50/74, loss=8.124687, lr=1e-06, time_each_step=0.35s, eta=0:1:3
    2021-08-11 17:15:46 [INFO]	[TRAIN] Epoch=269/270, Step=52/74, loss=16.952156, lr=1e-06, time_each_step=0.33s, eta=0:1:2
    2021-08-11 17:15:46 [INFO]	[TRAIN] Epoch=269/270, Step=54/74, loss=12.525278, lr=1e-06, time_each_step=0.33s, eta=0:1:1
    2021-08-11 17:15:47 [INFO]	[TRAIN] Epoch=269/270, Step=56/74, loss=19.233416, lr=1e-06, time_each_step=0.32s, eta=0:1:0
    2021-08-11 17:15:47 [INFO]	[TRAIN] Epoch=269/270, Step=58/74, loss=19.142153, lr=1e-06, time_each_step=0.3s, eta=0:1:0
    2021-08-11 17:15:47 [INFO]	[TRAIN] Epoch=269/270, Step=60/74, loss=7.444398, lr=1e-06, time_each_step=0.27s, eta=0:0:59
    2021-08-11 17:15:48 [INFO]	[TRAIN] Epoch=269/270, Step=62/74, loss=8.266433, lr=1e-06, time_each_step=0.25s, eta=0:0:58
    2021-08-11 17:15:48 [INFO]	[TRAIN] Epoch=269/270, Step=64/74, loss=7.998219, lr=1e-06, time_each_step=0.23s, eta=0:0:57
    2021-08-11 17:15:49 [INFO]	[TRAIN] Epoch=269/270, Step=66/74, loss=11.646782, lr=1e-06, time_each_step=0.22s, eta=0:0:57
    2021-08-11 17:15:49 [INFO]	[TRAIN] Epoch=269/270, Step=68/74, loss=9.86458, lr=1e-06, time_each_step=0.22s, eta=0:0:56
    2021-08-11 17:15:50 [INFO]	[TRAIN] Epoch=269/270, Step=70/74, loss=11.985245, lr=1e-06, time_each_step=0.21s, eta=0:0:56
    2021-08-11 17:15:50 [INFO]	[TRAIN] Epoch=269/270, Step=72/74, loss=16.405117, lr=1e-06, time_each_step=0.21s, eta=0:0:55
    2021-08-11 17:15:50 [INFO]	[TRAIN] Epoch=269/270, Step=74/74, loss=14.434589, lr=1e-06, time_each_step=0.21s, eta=0:0:55
    2021-08-11 17:15:50 [INFO]	[TRAIN] Epoch 269 finished, loss=14.541382, lr=1e-06 .
    2021-08-11 17:15:55 [INFO]	[TRAIN] Epoch=270/270, Step=2/74, loss=9.637806, lr=1e-06, time_each_step=0.43s, eta=0:0:50
    2021-08-11 17:15:56 [INFO]	[TRAIN] Epoch=270/270, Step=4/74, loss=5.8643, lr=1e-06, time_each_step=0.44s, eta=0:0:50
    2021-08-11 17:15:57 [INFO]	[TRAIN] Epoch=270/270, Step=6/74, loss=38.6721, lr=1e-06, time_each_step=0.48s, eta=0:0:52
    2021-08-11 17:15:58 [INFO]	[TRAIN] Epoch=270/270, Step=8/74, loss=15.004152, lr=1e-06, time_each_step=0.49s, eta=0:0:52
    2021-08-11 17:15:58 [INFO]	[TRAIN] Epoch=270/270, Step=10/74, loss=18.059345, lr=1e-06, time_each_step=0.51s, eta=0:0:52
    2021-08-11 17:15:59 [INFO]	[TRAIN] Epoch=270/270, Step=12/74, loss=7.693106, lr=1e-06, time_each_step=0.52s, eta=0:0:52
    2021-08-11 17:16:00 [INFO]	[TRAIN] Epoch=270/270, Step=14/74, loss=8.713644, lr=1e-06, time_each_step=0.53s, eta=0:0:51
    2021-08-11 17:16:00 [INFO]	[TRAIN] Epoch=270/270, Step=16/74, loss=14.208525, lr=1e-06, time_each_step=0.53s, eta=0:0:50
    2021-08-11 17:16:01 [INFO]	[TRAIN] Epoch=270/270, Step=18/74, loss=10.726345, lr=1e-06, time_each_step=0.57s, eta=0:0:51
    2021-08-11 17:16:02 [INFO]	[TRAIN] Epoch=270/270, Step=20/74, loss=34.498352, lr=1e-06, time_each_step=0.59s, eta=0:0:51
    2021-08-11 17:16:03 [INFO]	[TRAIN] Epoch=270/270, Step=22/74, loss=7.191622, lr=1e-06, time_each_step=0.38s, eta=0:0:39
    2021-08-11 17:16:03 [INFO]	[TRAIN] Epoch=270/270, Step=24/74, loss=28.084156, lr=1e-06, time_each_step=0.38s, eta=0:0:38
    2021-08-11 17:16:04 [INFO]	[TRAIN] Epoch=270/270, Step=26/74, loss=6.184209, lr=1e-06, time_each_step=0.36s, eta=0:0:37
    2021-08-11 17:16:05 [INFO]	[TRAIN] Epoch=270/270, Step=28/74, loss=8.632387, lr=1e-06, time_each_step=0.36s, eta=0:0:36
    2021-08-11 17:16:06 [INFO]	[TRAIN] Epoch=270/270, Step=30/74, loss=24.151634, lr=1e-06, time_each_step=0.35s, eta=0:0:35
    2021-08-11 17:16:06 [INFO]	[TRAIN] Epoch=270/270, Step=32/74, loss=7.779696, lr=1e-06, time_each_step=0.34s, eta=0:0:34
    2021-08-11 17:16:07 [INFO]	[TRAIN] Epoch=270/270, Step=34/74, loss=8.334702, lr=1e-06, time_each_step=0.34s, eta=0:0:33
    2021-08-11 17:16:08 [INFO]	[TRAIN] Epoch=270/270, Step=36/74, loss=6.964208, lr=1e-06, time_each_step=0.37s, eta=0:0:33
    2021-08-11 17:16:09 [INFO]	[TRAIN] Epoch=270/270, Step=38/74, loss=22.250017, lr=1e-06, time_each_step=0.36s, eta=0:0:32
    2021-08-11 17:16:09 [INFO]	[TRAIN] Epoch=270/270, Step=40/74, loss=23.341454, lr=1e-06, time_each_step=0.35s, eta=0:0:31
    2021-08-11 17:16:10 [INFO]	[TRAIN] Epoch=270/270, Step=42/74, loss=4.484751, lr=1e-06, time_each_step=0.35s, eta=0:0:31
    2021-08-11 17:16:10 [INFO]	[TRAIN] Epoch=270/270, Step=44/74, loss=11.939711, lr=1e-06, time_each_step=0.35s, eta=0:0:30
    2021-08-11 17:16:11 [INFO]	[TRAIN] Epoch=270/270, Step=46/74, loss=31.420664, lr=1e-06, time_each_step=0.33s, eta=0:0:29
    2021-08-11 17:16:11 [INFO]	[TRAIN] Epoch=270/270, Step=48/74, loss=14.95013, lr=1e-06, time_each_step=0.32s, eta=0:0:28
    2021-08-11 17:16:12 [INFO]	[TRAIN] Epoch=270/270, Step=50/74, loss=10.776379, lr=1e-06, time_each_step=0.31s, eta=0:0:27
    2021-08-11 17:16:12 [INFO]	[TRAIN] Epoch=270/270, Step=52/74, loss=25.036549, lr=1e-06, time_each_step=0.31s, eta=0:0:26
    2021-08-11 17:16:13 [INFO]	[TRAIN] Epoch=270/270, Step=54/74, loss=13.867204, lr=1e-06, time_each_step=0.31s, eta=0:0:26
    2021-08-11 17:16:13 [INFO]	[TRAIN] Epoch=270/270, Step=56/74, loss=8.444964, lr=1e-06, time_each_step=0.29s, eta=0:0:25
    2021-08-11 17:16:14 [INFO]	[TRAIN] Epoch=270/270, Step=58/74, loss=22.975393, lr=1e-06, time_each_step=0.25s, eta=0:0:23
    2021-08-11 17:16:14 [INFO]	[TRAIN] Epoch=270/270, Step=60/74, loss=14.196902, lr=1e-06, time_each_step=0.24s, eta=0:0:23
    2021-08-11 17:16:14 [INFO]	[TRAIN] Epoch=270/270, Step=62/74, loss=23.591305, lr=1e-06, time_each_step=0.22s, eta=0:0:22
    2021-08-11 17:16:15 [INFO]	[TRAIN] Epoch=270/270, Step=64/74, loss=14.813768, lr=1e-06, time_each_step=0.22s, eta=0:0:22
    2021-08-11 17:16:15 [INFO]	[TRAIN] Epoch=270/270, Step=66/74, loss=10.640337, lr=1e-06, time_each_step=0.21s, eta=0:0:21
    2021-08-11 17:16:15 [INFO]	[TRAIN] Epoch=270/270, Step=68/74, loss=9.21396, lr=1e-06, time_each_step=0.21s, eta=0:0:21
    2021-08-11 17:16:16 [INFO]	[TRAIN] Epoch=270/270, Step=70/74, loss=9.63536, lr=1e-06, time_each_step=0.21s, eta=0:0:20
    2021-08-11 17:16:16 [INFO]	[TRAIN] Epoch=270/270, Step=72/74, loss=6.572531, lr=1e-06, time_each_step=0.2s, eta=0:0:20
    2021-08-11 17:16:17 [INFO]	[TRAIN] Epoch=270/270, Step=74/74, loss=42.738445, lr=1e-06, time_each_step=0.19s, eta=0:0:19
    2021-08-11 17:16:17 [INFO]	[TRAIN] Epoch 270 finished, loss=14.885777, lr=1e-06 .
    2021-08-11 17:16:17 [INFO]	Start to evaluating(total_samples=170, total_steps=22)...


    100%|██████████| 22/22 [00:09<00:00,  2.44it/s]


    2021-08-11 17:16:26 [INFO]	[EVAL] Finished, Epoch=270, bbox_map=80.818289 .
    2021-08-11 17:16:32 [INFO]	Model saved in output/yolov3_DarkNet53/epoch_270.
    2021-08-11 17:16:32 [INFO]	Current evaluated best model in eval_dataset is epoch_240, bbox_map=80.86807042180423





```python
#在test_dateset上选一张图片进行infer
model = pdx.load_model('output/yolov3_DarkNet53/best_model')
image_name = 'objDataset/facemask/JPEGImages/maksssksksss381.png'
result = model.predict(image_name)
pdx.det.visualize(image_name, result, threshold=0.5, save_dir='./output/yolov3_DarkNet53')#对
```

    2021-08-11 17:41:48 [INFO]	Model[YOLOv3] loaded.
    2021-08-11 17:41:48 [INFO]	The visualized result is saved as ./output/yolov3_DarkNet53/visualize_maksssksksss381.png


![](https://ai-studio-static-online.cdn.bcebos.com/781559f671e0430c8d6150cef01c0fb6a141b54c317341bb9d9281285c52e823)


# 五、作业提交

完成以上四个步骤，并且跑通后，恭喜你！你已经学会使用PaddleX训练目标检测模型了！

首先点击右上方的“**文件**”，点击“**导出Notebook为Markdown**”：
![](https://ai-studio-static-online.cdn.bcebos.com/0b502ce690274eee89c285c702a26a72ff012df791eb45b79b7866a97f549e33)

导出后的文件用于上传至GitHub。

下面请点击左侧“版本”，在出现的新界面中点击“生成新版本”，按如下操作即可生成版本，用于提交作业：
![](https://ai-studio-static-online.cdn.bcebos.com/4e58b131f2db45289119dfa90081eea43b781d6894254069b4f02cffa067ffe1)

生成版本后，回到项目主页，点击“**设置为公开**”：
![](https://ai-studio-static-online.cdn.bcebos.com/4a9229e9eb8146169c848327ccb1f85ea30af431eee14aea9c2e6c9aeb364926)



