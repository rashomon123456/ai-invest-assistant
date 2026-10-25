# 🤖 AI 投研助手

输入 A 股代码，抓取近一年行情 → 计算技术指标 → 调用大模型生成结构化行情分析。

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.50-red)

## 功能

- 支持沪深主板、创业板、科创板、ETF/场内基金（自动识别代码前缀）
- 提交前先算好 MA5/10/20/60、量比、20 日年化波动率、距高点回撤等指标，再喂给模型
- 输出「趋势 / 量能 / 关键位 / 风险」四段分析，强制引用具体数字
- 数据缓存 30 分钟；报告可一键导出 Markdown

## 本地运行

```bash
pip install -r requirements.txt

# 方式一：环境变量
set MOONSHOT_API_KEY=sk-xxxx          # Windows CMD
# export MOONSHOT_API_KEY=sk-xxxx     # macOS / Linux

# 方式二：secrets 文件
copy .streamlit\secrets.toml.example .streamlit\secrets.toml

streamlit run app.py
```

也可以不配置——页面左侧会给出临时输入框（只存内存，不落盘）。

## 部署到 Streamlit Community Cloud

1. 仓库推到 GitHub（`secrets.toml` 已被 `.gitignore` 排除）
2. share.streamlit.io → New app → 选仓库，Main file path 填 `app.py`
3. App settings → **Secrets**，粘贴：
   ```toml
   MOONSHOT_API_KEY = "sk-xxxx"
   ```
4. Deploy

## 说明

- 数据源：[akshare](https://github.com/akfamily/akshare) 腾讯行情接口
- 模型：Moonshot Kimi（`kimi-k2.6` / `kimi-k3` / `kimi-k3.1-pro`）
- 仅供研究学习，不构成投资建议
