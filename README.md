# Voice QA Assistant

Voice QA Assistant 是一个基于语音识别和AI问答的智能助手应用，支持实时音频转文字和基于知识库的智能回答。

## 功能特点

- **自动音频采集**：支持从输入设备（麦克风）或输出设备（扬声器）采集音频
- **实时音频转文字**：使用Vosk或Whisper模型进行语音识别，支持中英文
- **知识库检索**：使用TF-IDF向量化器从PDF和Word文档中检索相关信息
- **AI智能回答**：集成OpenAI API，根据检索到的知识生成精确回答
- **配置界面**：可配置音频源、OpenAI API Key和Base URL等参数
- **实时显示**：在应用窗口中实时显示转录结果和AI回答

## 技术栈

- **前端**：PyQt5 GUI库
- **音频处理**：sounddevice和pyaudiowpatch
- **语音识别**：Vosk（默认）或Whisper（可选）
- **知识检索**：scikit-learn的TF-IDF和KNN算法
- **AI集成**：OpenAI API

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/voice-qa-assistant.git
cd voice-qa-assistant
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装可选依赖（用于Whisper模型）

```bash
pip install openai-whisper
```

## 配置

### 1. 知识库准备

- 在 `knowledge_base` 文件夹中放入PDF和Word文档
- 支持多层级文件夹结构，应用会递归扫描所有文件

### 2. 配置OpenAI API

- 启动应用，切换到"配置"选项卡
- 输入OpenAI API Key和Base URL
- 选择知识库文件夹路径（默认为`knowledge_base`）
- 点击"保存配置"

### 3. 选择音频源

- 在主界面选择"输入设备"或"输出设备"
- 从下拉列表中选择具体的设备

## 使用方法

1. **启动应用**：
   ```bash
   python main.py
   ```

2. **开始自动采集**：
   - 点击"启动自动采集"按钮
   - 开始说话或播放音频
   - 当检测到停顿时，应用会自动处理音频并生成AI回答

3. **查看结果**：
   - 在"音频转文字"区域查看识别的文字
   - 在"AI回答"区域查看生成的回答

## 模型选择

### Vosk模型（默认）
- 轻量级，适合资源有限的设备
- 完全离线运行
- 支持中英文单语言模型

### Whisper模型（可选）
- 更准确的识别效果
- 支持中英文混合识别
- 模型大小较大，需要更多资源

## 常见问题

### 1. ASR模型未加载
- 确保模型文件已正确下载
- 检查模型路径是否正确
- 对于Vosk模型，确保模型文件夹存在且包含必要文件

### 2. 转录效果不好
- 尝试使用Whisper模型
- 确保在安静的环境中使用
- 保持适当的距离和音量

### 3. AI回答失败
- 确保OpenAI API Key正确设置
- 检查网络连接
- 确保API端点正确

## 故障排除

### 1. 内存使用过高
- 减少知识库文件大小
- 选择更轻量级的ASR模型
- 关闭其他占用内存的应用

### 2. 音频设备无法选择
- 确保音频设备已正确连接
- 尝试刷新设备列表
- 检查系统权限

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 联系方式

如有问题，请联系项目维护者。