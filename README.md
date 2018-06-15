# TinyMind_OCR
参加Tinymind书法识别大赛的源码
***
## 本代码参考[TinyMind-start-with-0](https://github.com/Link2Link/TinyMind-start-with-0)    
***
## 结构    
本代码在参考基础上，修改了网络结构，并增加了VGG16网络的预训练模型    
## 结果    
在只有CPU的基础上，由于时间原因，只训练了两个epoch，取得了82.79%的成绩，如果继续优化，结果应该会更好    
后续自由练习赛中成绩为94.66%    
## 改进    
数据未得到增强，由于在CENTOS虚拟机上运行，内存也限制到了5G，图片预处理后的大小为128*128，如果使用224*224的图片，载入预训练的VGG网络会更好    
## [参赛地址](https://www.tinymind.cn/competitions/41)
