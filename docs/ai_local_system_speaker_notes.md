# AI Local System 演讲稿 / Speaker Notes

## Slide 1: AI Local System

**中文演讲稿：**  
大家好，今天汇报的是 AI Local System，也就是面向现场自维护的本地 AI 视觉检测系统。这个方案的核心目标不是替代现有标准视觉系统，而是在现场快速新增检测项、快速换型、低成本国产化硬件适配方面，提供一套更容易被车间人员使用和维护的软件平台。

**English Script:**  
Hello everyone. Today I will introduce the AI Local System, a local AI vision inspection system designed for production self-maintenance. The goal is not to replace the existing standard vision system, but to provide an easier-to-use platform for fast inspection setup, quick changeover, and low-cost hardware integration on site.

## Slide 2: Background - On-site Maintenance Barrier

**中文演讲稿：**  
目前现有的 Camera System 标准视觉方案非常成熟，特别适合几何测量、强规则检测和长期稳定的检测项目。但对于现场临时新增错漏装检测、换批次、换工装这类需求，车间人员通常不希望打开代码，也不希望理解复杂的算法流程。他们更需要类似智能传感器 Teach-in 的操作方式：采样、框 ROI、学习、检测。因此我们希望补充一套车间可自助配置的 AI 检测工具。

**English Script:**  
The existing Camera System standard vision solution is mature and reliable, especially for geometric measurement, rule-based inspection, and long-term stable inspection items. However, when the production needs to add missing or wrong assembly checks, change batches, or change fixtures, users usually do not want to open code or understand complex algorithm flows. They need a Teach-in style workflow: sample, mark ROI, learn, and inspect. This is why we propose a self-configurable AI inspection tool for production use.

## Slide 3: Solution Overview

**中文演讲稿：**  
整体方案由低成本 CCD 硬件平台和小样本注册软件组成。前端使用国产相机和光源完成图像采集，软件中配置 ROI 区域并添加 OK/NG 样本，系统自动生成注册模型，最后输出产线检测结果。这个方案的特点是低成本、Teach-in 操作方式和现场自维护能力，适合把错漏装、反装、有无检测这类外观判断快速上线。

**English Script:**  
The overall solution combines a low-cost CCD hardware platform with small-sample registration software. Domestic cameras and lights are used for image acquisition. In the software, users set the ROI and add OK/NG samples. The system then builds the registration model and outputs inspection results to the production line. The key advantages are low cost, Teach-in style operation, and production maintainability, especially for missing, wrong, reversed, or presence inspection tasks.

## Slide 4: Core Algorithm

**中文演讲稿：**  
核心算法可以分成三步。第一步，ROI 图像输入到预训练 CNN 的 features 部分，提取边缘、纹理、形状等通用视觉特征。第二步，不使用原来的 ImageNet 分类头，因为它回答的是猫、狗、车等 1000 类类别，而不是当前零件 OK 还是 NG。第三步，通过 GAP 和 L2 归一化得到稳定的 embedding，再和当前产品的 OK/NG 原型计算相似度，判断新图像更像 OK 还是 NG。

**English Script:**  
The core algorithm has three steps. First, the ROI image is passed through the feature extraction part of a pretrained CNN to extract general visual features such as edges, textures, and shapes. Second, the original ImageNet classifier head is not used, because it predicts categories such as cats, dogs, and cars, not whether the current part is OK or NG. Third, Global Average Pooling and L2 normalization generate a stable embedding. The new embedding is then compared with the OK/NG prototypes to judge whether the image is closer to OK or NG.

## Slide 5: Training Difference - CNN Training vs Sample Registration

**中文演讲稿：**  
基于上一页的算法原理，这里再澄清“训练”的含义。本系统里的训练不是传统意义上的 CNN 网络训练。传统深度学习训练需要大量 OK/NG 图片，通过反向传播更新模型权重，最后得到一个重新训练后的分类模型。而我们这里使用的是官方预训练权重，CNN 权重固定不变，只进行前向推理提取 embedding 特征，然后根据少量 OK/NG 样本建立特征库和原型。所以这个过程更准确地说是“小样本注册”，不是重新训练 CNN。

**English Script:**  
Based on the algorithm flow from the previous slide, we can clarify what “training” means here. It is not conventional CNN training. Traditional deep learning training usually requires many OK/NG images and updates model weights through backpropagation to produce a retrained classifier. In our system, the official pretrained CNN weights remain fixed. We only run forward inference to extract embeddings, then build an OK/NG feature bank and prototypes from a small number of samples. So technically, this is sample registration, not CNN retraining.

## Slide 6: Hardware Configuration and Test Speed

**中文演讲稿：**  
这一页展示的是硬件配置和现场测试速度。硬件上采用国产相机、镜头、光源和 PC 控制器，双相机方案可以覆盖多个检测视角，同时降低整体成本。右侧是现场速度数据的重绘，单相机检测时间大约在 500ms 级别，耗时主要由采图、定位或匹配、推理组成。这个速度对于错漏装和外观注册检测的节拍验证是可接受的。

**English Script:**  
This slide shows the hardware configuration and on-site test speed. The system uses domestic cameras, lenses, lights, and a PC controller. The dual-camera setup can cover multiple inspection views while reducing total hardware cost. On the right, the timing data is shown. The single-camera inspection time is around 500 milliseconds, mainly including image capture, localization or matching, and inference. This cycle time is acceptable for missing assembly and appearance-based registration inspection.

## Slide 7: Overall Appearance

**中文演讲稿：**  
这里是系统的整体外观设计。设备采用半透明外框，内部包含相机、镜头、光源等光学元件，并支持多相机扩展。夹具部分可以兼容多种产品的快换需求，外部通过状态灯和检测结果指示灯显示运行状态。整体设计目标是让 AI 视觉检测系统在手工线SKA客户上先执行。

**English Script:**  
This slide shows the overall system appearance. The device uses a translucent housing and contains optical components such as cameras, lenses, and lights. It also supports multi-camera expansion. The fixture area is designed for quick changeover between different products, and status lights show the running and inspection results. The goal is to first deploy the AI vision inspection system on the manual line for the SKA customer.

## Slide 8: UI Overview

**中文演讲稿：**  
这一页展示的是软件界面。界面包含菜单栏、调试界面、样本添加区域、工具类型选择、注册和测试等功能。设计重点是让车间人员能够通过界面完成配置，而不是通过修改代码完成维护。用户可以添加 OK/NG 样本，选择检测工具，配置 ROI，然后进行注册和测试。

**English Script:**  
This slide shows the software interface. It includes the menu bar, debug interface, sample image area, tool type selection, registration, and testing functions. The design focus is to let production users configure and maintain the inspection through the UI instead of modifying code. Users can add OK/NG samples, select the inspection tool, configure ROI, and then run registration and testing.

## Slide 9: Operation Flow

**中文演讲稿：**  
实际操作流程非常简单。第一步，车间人员添加合格和不合格样本。第二步，选择匹配区域和学习 ROI。第三步，点击学习，系统自动提取 embedding 并生成特征库。第四步，开始检测。整个过程中，车间人员主要负责采样、框 ROI、补充边界样本和触发重新学习，系统自动完成特征提取、原型建立和相似度判定。

**English Script:**  
The actual operation flow is simple. First, production users add OK and NG samples. Second, they select the matching area and learning ROI. Third, they click Learn, and the system automatically extracts embeddings and builds the feature bank. Fourth, inspection starts. In this workflow, users mainly handle sampling, ROI marking, edge-case sample collection, and relearning. The system automatically handles feature extraction, prototype building, and similarity judgment.

## Slide 10: Traditional Camera System vs Registration Learning System

**中文演讲稿：**  
这页说明传统 Camera System 和小样本注册系统的定位区别。传统视觉系统更适合几何测量、强规则、长期稳定项目，通常由工程师配置流程、阈值和算法分支。小样本注册系统更适合需要频繁维护、难以规则化的外观判断，例如错装、漏装和反装。它不是完全替代传统视觉，而是把频繁变化、现场希望自维护的部分交给注册学习系统。

**English Script:**  
This slide explains the positioning difference between the traditional Camera System and the small-sample registration system. The traditional vision system is better for geometric measurement, strong rules, and long-term stable inspection items, usually configured by engineers. The registration learning system is better for frequently maintained and hard-to-rule-code appearance judgments, such as wrong, missing, or reversed assembly. It is not a full replacement, but a complementary tool for self-maintained inspection tasks.

## Slide 11: Closing

**中文演讲稿：**  
总结来说，AI Local System 的价值在于低成本、快上线和现场可维护。它基于固定的预训练 CNN 特征提取能力，通过小样本注册建立当前产品的 OK/NG 特征原型，让车间人员可以用界面完成维护。后续这套平台还可以继续扩展尺寸测量、定位检测、IO 联动等功能，形成更完整的本地 AI 视觉检测平台。

**English Script:**  
In summary, the value of the AI Local System is low cost, fast launch, and production maintainability. It uses fixed pretrained CNN feature extraction and builds OK/NG prototypes through small-sample registration, allowing users to maintain inspection items through the UI. In the future, the same platform can also be extended with dimensional measurement, localization, and IO integration to form a more complete local AI vision inspection platform.

