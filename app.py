# -*- coding: utf-8 -*-
"""
AI 投研助手 v2  —— 由小龙虾 🦞 在 app.py 基础上升级
主要改动：
1. API Key 不再硬编码（环境变量 / .streamlit/secrets.toml / 侧边栏临时输入）
2. 股票代码前缀覆盖 ETF、指数、B股，并做输入校验
3. 数据缓存 30 分钟，重复点击不重复抓数、不重复烧 token
4. 先算好 MA/量比/波动率/回撤等指标再喂给模型，减少瞎编空间
5. 请求超时 + 错误分类提示 + 报告可下载
"""
import os

# —— 直连：A股数据源和 Moonshot 都在国内，走系统代理反而失败 ——
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
for _k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(_k, None)

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import akshare as ak
from openai import OpenAI

BASE_URL = "https://api.moonshot.cn/v1"
MODELS = ["kimi-k2.6", "kimi-k3", "kimi-k3.1-pro"]

st.set_page_config(page_title="AI 投研助手", page_icon="🤖", layout="centered")


# ---------------- API Key ----------------
def get_api_key() -> str:
    key = os.getenv("MOONSHOT_API_KEY", "").strip()
    if key:
        return key
    try:
        key = str(st.secrets.get("MOONSHOT_API_KEY", "")).strip()
        if key:
            return key
    except Exception:
        pass
    return st.session_state.get("user_key", "").strip()


# ---------------- 代码 -> 行情前缀 ----------------
def to_symbol(code: str):
    """返回 (腾讯接口symbol, 市场名)。不支持的市场抛 ValueError。"""
    code = code.strip()
    if not (len(code) == 6 and code.isdigit()):
        raise ValueError("请输入 6 位数字股票代码，例如 600519 / 000001 / 510300")
    head = code[0]
    if head in "69":
        return f"sh{code}", "沪市"
    if head == "5":
        return f"sh{code}", "沪市基金/ETF"
    if head in "023":
        return f"sz{code}", "深市"
    if head == "1":
        return f"sz{code}", "深市基金/ETF"
    if head in "48":
        raise ValueError("北交所（4/8 开头）数据源暂不支持，换沪深代码试试")
    raise ValueError(f"无法识别的代码：{code}")


# ---------------- 拉数据（缓存 30 分钟） ----------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_hist(symbol: str, start: str, end: str) -> pd.DataFrame:
    return ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust="qfq")


# ---------------- 指标计算 ----------------
def compute_metrics(df: pd.DataFrame) -> tuple:
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    close = pd.to_numeric(d["close"], errors="coerce").astype(float)
    vol = pd.to_numeric(d["volume"], errors="coerce").astype(float)
    ret = close.pct_change().dropna()

    ma = {n: close.rolling(n).mean().iloc[-1] for n in (5, 10, 20, 60) if len(close) >= n}
    last = close.iloc[-1]
    lines = [
        f"最新收盘价：{last:.2f}",
        f"近5日涨跌幅：{close.iloc[-1] / close.iloc[-6] - 1:+.2%}" if len(close) >= 6 else "",
        f"近20日涨跌幅：{close.iloc[-1] / close.iloc[-21] - 1:+.2%}" if len(close) >= 21 else "",
    ]
    for n, v in ma.items():
        lines.append(f"MA{n}：{v:.2f}（现价{'高于' if last > v else '低于'}均线 {abs(last / v - 1):.2%}）")

    hi, lo = close.max(), close.min()
    lines += [
        f"近一年最高/最低：{hi:.2f} / {lo:.2f}，现价距高点回撤 {last / hi - 1:+.2%}",
        f"5日均量：{vol.tail(5).mean() / 1e4:.0f} 万股；20日均量：{vol.tail(20).mean() / 1e4:.0f} 万股"
        f"（量比 {vol.tail(5).mean() / vol.tail(20).mean():.2f}）",
        f"近20日年化波动率：{ret.tail(20).std() * np.sqrt(252):.2%}",
    ]

    tail = d.tail(5).copy()
    rename = {"date": "日期", "open": "开盘", "close": "收盘", "high": "最高",
              "low": "最低", "volume": "成交量(股)", "amount": "成交额(元)"}
    tail = tail.rename(columns=rename)
    keep = [c for c in ["日期", "开盘", "最高", "最低", "收盘", "成交量(股)", "成交额(元)"] if c in tail.columns]
    tail_txt = tail[keep].to_string(index=False)

    return "\n".join([x for x in lines if x]), tail_txt


# ---------------- Prompt ----------------
def build_prompt(code: str, market: str, end_date: str, metrics: str, tail_txt: str) -> str:
    return f"""你是一名有 10 年经验的 A 股卖方分析师。以下是 {market} 股票 {code} 截至 {end_date} 的行情数据。

【技术指标摘要】
{metrics}

【最近 5 个交易日原始行情】
{tail_txt}

请写一段 350 字以内的行情分析，分四小节：
1. 趋势：价格与均线、近 5/20 日涨跌幅的关系；
2. 量能：成交量/量比变化说明了什么；
3. 关键位：由数据和均线推断的支撑位与压力位；
4. 风险：波动率、回撤、量价背离等值得警惕的点。

要求：
- 每个结论都必须引用上面给出的具体数字，禁止编造数据或杜撰消息面、财报、政策；
- 不写"建议买入/卖出/持有"等投资建议措辞；
- 结尾单独一行写：以上分析仅供参考，不构成投资建议。"""


# ---------------- 页面 ----------------
st.title("🤖 智能 AI 投研助手")
st.caption("输入 A 股代码，一键生成行情分析报告 · 小龙虾升级版 v2")

with st.sidebar:
    st.subheader("设置")
    model = st.selectbox("模型", MODELS, index=0)
    # 【修改点】：永远显示输入框
    st.text_input("Moonshot API Key", type="password", key="user_key",
                  help="也可写入环境变量 MOONSHOT_API_KEY 或 .streamlit/secrets.toml")
    st.caption("数据源：akshare（腾讯行情）· 仅供研究，不构成投资建议")

col1, col2 = st.columns([3, 1])
with col1:
    stock_code = st.text_input("请输入 6 位股票代码", "600519")
with col2:
    days = st.selectbox("回溯天数", [365, 730], index=0, label_visibility="visible")

submit = st.button("生成分析报告", type="primary", use_container_width=True)

if submit:
    try:
        symbol, market = to_symbol(stock_code)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    api_key = get_api_key()
    if not api_key:
        st.warning("还没配置 API Key，请先在左侧填写（或设置环境变量 MOONSHOT_API_KEY）")
        st.stop()

    today = datetime.now()
    start, end = (today - timedelta(days=days)).strftime("%Y%m%d"), today.strftime("%Y%m%d")

    with st.spinner("抓取行情数据..."):
        try:
            df = fetch_hist(symbol, start, end)
        except Exception as e:
            st.error(f"行情接口出错：{type(e).__name__} - {str(e)[:120]}")
            st.stop()

    if df is None or df.empty:
        st.error("没取到数据。核对一下代码，或稍后重试（接口偶尔抽风）。")
        st.stop()
    if len(df) < 20:
        st.warning(f"只有 {len(df)} 个交易日数据，可能停牌或次新股，指标参考性有限。")

    metrics, tail_txt = compute_metrics(df)
    with st.expander("📈 技术指标与近 5 日行情（喂给模型的原始输入）"):
        st.text(metrics)
        st.code(tail_txt, language="text")

    prompt = build_prompt(stock_code, market, today.strftime("%Y-%m-%d"), metrics, tail_txt)

    with st.spinner("大模型分析中..."):
        try:
            client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=90, max_retries=1)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            report = resp.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "Authentication" in msg or "401" in msg:
                st.error("API Key 无效或已失效，请更换。")
            elif "429" in msg:
                st.error("调用太频繁 / 额度不足，稍后再试。")
            else:
                st.error(f"模型调用失败：{type(e).__name__} - {msg[:160]}")
            st.stop()

    st.success(f"分析完成（{model}）")
    st.markdown(f"### 📊 {stock_code} · {market} 行情分析")
    st.info(report)

    md = (f"# {stock_code} {market} 行情分析\n\n生成时间：{datetime.now():%Y-%m-%d %H:%M}　模型：{model}\n\n"
          f"## AI 分析\n\n{report}\n\n## 技术指标\n\n```\n{metrics}\n```\n\n## 近5日行情\n\n```\n{tail_txt}\n```\n")
    st.download_button("⬇️ 下载报告 (Markdown)", md,
                       file_name=f"{stock_code}_{today:%Y%m%d}.md", mime="text/markdown")
