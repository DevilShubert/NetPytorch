# 简介

使用springBoot + redis +Faster-RCNN的网站-目标检测工具



## 使用方法

**1、搭建Java环境**

[打入SpringBoot项目方法](https://blog.csdn.net/IT_Boy_/article/details/114886703?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163870557216780366533244%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=163870557216780366533244&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-114886703.first_rank_v2_pc_rank_v29&utm_term=%E5%A6%82%E4%BD%95%E5%AF%BC%E5%85%A5springBoot&spm=1018.2226.3001.4187)

在本项目中，对于Java前后端逻辑部分，导入`JavaNetPytorch`文件即可



**2、搭建redis环境**

[下载并安装Reds](https://www.jianshu.com/p/6b5eca8d908b)

按照config文件开启本地redis服务端

```bash
redis-server /your_location/redis.conf
# 开启服务端
```

再用本机作为客户端去进行连接，之后即可进行查看（redis默认端口为6379）

```bash
redis-cli -p 6379
# 本机作为客户端再去连本机的服务端

# 连接成功后出现如下提示
(base) J-Lian-MacAir:~ liuzheran$ redis-server /usr/local/redis-6.0.6/redis.conf
2855:C 05 Dec 2021 20:08:48.797 # oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
2855:C 05 Dec 2021 20:08:48.797 # Redis version=6.0.6, bits=64, commit=00000000, modified=0, pid=2855, just started
2855:C 05 Dec 2021 20:08:48.797 # Configuration loaded
(base) J-Lian-MacAir:~ liuzheran$ redis-cli -p 6379
127.0.0.1:6379> 
```

本项目中默认使用数据库1存储，切换到数据库1

```bash
127.0.0.1:6379> select 1
OK
127.0.0.1:6379[1]> 
```



**3、搭建python环境**

- **conda + 虚拟环境 + pychram导入**

安装conda，新建虚拟环境，并在pycharm中引入该虚拟环境，参考博客如下

[Mac系统下安装Anaconda并搭建Pytorch进行深度学习](https://blog.csdn.net/weixin_44263973/article/details/120585017?ops_request_misc=&request_id=&biz_id=102&utm_term=Mac%E9%83%A8%E7%BD%B2pytorch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-120585017.first_rank_v2_pc_rank_v29&spm=1018.2226.3001.4187)

[【Mac OS】Anaconda+PyCharm搭建PyTorch环境](https://blog.csdn.net/libing_zeng/article/details/96615716?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3.nonecase)



其中，新建虚拟环境需要的常见指令如下

```bash
创建虚拟环境 conda create -n [虚拟环境名称] python=3.x
激活虚拟环境 source activate[虚拟环境名称]
退出当前虚拟环境 conda deactivate
查看所有环境 conda info --envs
查看当前环境中的所有包 conda list
安装包 conda/pip install [包名称]
```

```bash
# 本文创建的虚拟环境名称为 py39
conda create -py39 python=3.9
```



*注意如果是使用的M1芯片则需要用另一种conda*

[M1芯片Mac上Anaconda的暂时替代：miniforge](https://blog.csdn.net/yc11tentgy/article/details/113469988?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163784363916780269893962%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163784363916780269893962&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-113469988.first_rank_v2_pc_rank_v29&utm_term=miniforge3%E6%98%AF%E4%BB%80%E4%B9%88&spm=1018.2226.3001.4187)



- **导入核心第三库**

**PyTorch 1.8.0、Torvision 0.9.1、flask**

如果是第一次pytorch，还可能踩`torchaudio`的坑；例如下图是pytorch官方给的安装指令

```bash
conda install pytorch torchvision torchaudio -c pytorch
```

但一般会由于镜像包缺少`torchaudio`而被卡住，所以我们直接不下载`torchaudio`，因为本项目目标检测用不到，所以将指令改为如下即可

```bash
conda install pytorch torchvision  -c pytorch
```

另外还有一些第三库按照缺少的一个个用conda导入即可，不问不在赘述



在配置好环境之后，将下文资源中的权重放在`save_weights`文件夹中，运行`app.py`脚本文件即可开启开启python目标检测部分的web应用



**4、使用**

- 打开浏览器输入`localhost:8080`进入页面

![image-20211205203436876.png](https://github.com/DevilShubert/NetPytorch/blob/master/IamgesFile/image-20211205203436876.png?raw=true)

- 选中需要访问的图片

![image-20211205204516200.png](https://github.com/DevilShubert/NetPytorch/blob/master/IamgesFile/image-20211205204516200.png?raw=true)

- 得到检测结果

![image-20211205204705282.png](https://github.com/DevilShubert/NetPytorch/blob/master/IamgesFile/image-20211205204705282.png?raw=true)



# 资源

- fasterRCNN权重：百度网盘连接https://pan.baidu.com/s/1WqLlbYCEIiTU5LJl0PuNvw 提取码: khln 
  - 下载好之后，放入save_weights文件夹之下，PythonNetPytorch.py文件会读取权重并加载进模型



# 参考

[使用Redis加速深度学习模型（Spring/Python/Redis）](https://blog.csdn.net/m0_46503651/article/details/108555082)

本项目是在此项目的基础上进行修改和优化的，唯一不同的是将目标检测算法换成了fasterRCNN

仅作为学习使用
