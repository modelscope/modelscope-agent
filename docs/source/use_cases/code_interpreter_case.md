# 代码解释器应用案例

本文介绍Agent自带的Code Interpreter有哪些高级而实用的能力
## Code Interpreter（代码解释器）是什么？
简单来说，它赋予了语言模型运行Python代码的能力，用户只需用自然语言告诉模型任务是什么，模型就能编写相对应的Python代码并执行，来解决任务。

作为Agent的内置工具，当Agent会写代码又会执行代码，想象力的边界将被无限扩展，即使不会代码也能让大模型+代码高效快捷地完成我们想要的工作。

## Code Interpreter（代码解释器）可以做什么？
- 生成二维码，将链接地址秒转二维码图片
- 图片处理，如图片分割并转gif
- 文件类型转换，如pdf转txt
- 视频生成，将图片生成视频
- 数据分析及可视化，excel技能拉满
- 数学计算，解答高级数学问题
- 等等……

甚至可以作为自定义API调用的controller，只有你想不到没有做不到！

## 应用案例

### Agent配置示例

![agent配置示例](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_b7f0bdaa4dd340f594fe7ac327c64d6d.png)

### Agent构建的推荐Prompt

Name: Python编程专家

Description: 使用python解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。你可以处理用户上传到电脑的文件。

Instructions:

1. 你会数学解题；
2. 你会数据分析和可视化；
3. 你会转化文件格式，生成视频等；
4. 用户上传文件时，你必须先了解文件内容再进行下一步操作；如果没有上传文件但要求画图，则编造示例数据画图；
5. 调用工具前你需要说明理由；Think step by step；
6. 代码出错时你需要反思并改进。

注意在configure中勾选上code interpreter

![示例2](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_af6736e308b54542a9fe689473f11e2f.png)

### 功能演示
1. 生成二维码
   ![生成二维码](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_a573b4a3b3bc4e0497ec26c087be014e.png)
2. 文件类型转换：pdf转txt
   ![文件类型转换](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_f29abc51d1ef44d3a1c4693b9d95f469.png)
3. 图片处理

   输入：

   ![输入](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_8de20b3a96884ce89d2756ee90e8933c.png)

   ![输入2](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_753ead5841374c7daec285f039e5fe9c.png)

   输出：

   ![输出3](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_4ce39afce521416c9e30eef876c87439.gif)

   完整链路视频：

   ![完整链路视频](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_6f23077837954543adaca82bb267c8a2.gif)

4. 视频生成

   输入图片：

   ![输入图片](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_51d58fc24f054104ad450227a023f5d3.png)

   ![操作演示](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_3b1ed01c56ee4d8b8947f08e72a8d478.png)

   输出视频：

   ![输出视频](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_961bf8436bac4f5cbfb2ef8cc739c22a.gif)

5. 数据可视化

   ![数据可视化](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_486b54cc2ec64671be860257c321d0cf.png)

6. 数据分析

   ![数据分析](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_d5200094fd324e95b6ffa86456e26861.png)

7. 图表生成

   ![图表生成1](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_b2eba97461f34c7480ccfd174845b092.png)

   ![图表生成2](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_cd010573a4a64d5f9f58af2fbe5fe926.png)

8. 编程教学

   ![编程教学](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_d66fd3d300f0469985477feaf53270fb.png)

9. 高级数学计算

   不使用工具纯文本计算数学题容易出错：

   ![不使用工具](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_140b7869062749da86d46d2ce482b478.png)

   使用Code Interpreter计算后答案正确：

   ![使用工具](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_564d98fa23404c02b87e79a005b18917.png)
