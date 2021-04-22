# 预测代码规范

### 环境

- 只支持python>=3.6，如有特殊情况，请在readme中里备注
- 测试机器：内存10G，CPU10核，显卡1080Ti（11G显存）一张，硬盘10G



### 运行

- 依赖包写在requirements.txt中，每个包指定版本号

  ```
  环境安装执行命令 python -m pip install -r requirements.txt
  ```

- 预测代码规范

  - 参考predictor.py中的类Predictor，必须实现predict函数，输入输出不能修改，每次预测一条数据
  - 代码中不要指定显卡等其他环境变量，如有特殊情况，请在readme中备注
  - 最小的模型超过了传输限制，采用云盘的形式保存[url](https://drive.google.com/file/d/1DhASp98LY_0JI7NOVxxgnQ8RQkUQz9eB/view?usp=sharing)
  	
- 其他文件、代码格式不限

  ​